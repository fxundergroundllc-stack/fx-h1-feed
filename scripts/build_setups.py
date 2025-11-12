#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
On-time H1 signal generator:
- Base signal = EMA20/EMA50 cross, confirmed on the bar AFTER the cross.
- Emits ONLY if that signal bar is the most recent CLOSED bar (no late emits),
  unless EMIT_PAST_CROSS=true.
- Adds non-blocking atoms (MACD sign, EMA200 side, stack, near EMA20/50, near Fibo 38.2/61.8,
  swing breakout) to rule_score.
- SL = 1.5*ATR14, TP = 2R.
Outputs:
  signals/mt4.csv (legacy first 10 cols + rule_score,rules)
  signals/latest.json
  signals/debug_report.txt (why/why not for every pair)
"""
import os, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

PAIRS = ["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD","XAUUSD"]
DATA_DIR = "data"
OUT_DIR  = "signals"
os.makedirs(OUT_DIR, exist_ok=True)

# params (overridable via env)
ATR_PERIOD      = 14
ATR_SL_MULT     = 1.5
TP_R_MULT       = 2.0
SWING_LOOKBACK  = 30
FIBO_LOOKBACK   = 80
NEAR_PIPS       = 10.0
FRESH_MAX_HOURS = float(os.getenv("FRESH_MAX_HOURS","2"))   # data freshness gate
EMIT_PAST       = os.getenv("EMIT_PAST_CROSS","false").lower() == "true"

def pip_size(pair: str) -> float:
    pair = pair.upper()
    if pair.endswith("JPY"): return 0.01
    if pair == "XAUUSD":     return 0.10
    return 0.0001

def load_df(pair: str) -> pd.DataFrame:
    p = os.path.join(DATA_DIR, f"{pair}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("timestamp") or cols.get("time") or cols.get("datetime") or list(df.columns)[0]
    ocol = cols.get("open","open"); hcol = cols.get("high","high"); lcol = cols.get("low","low"); ccol = cols.get("close","close")
    df = df.rename(columns={tcol:"timestamp", ocol:"open", hcol:"high", lcol:"low", ccol:"close"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").reset_index(drop=True)
    return df

def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()
def atr(df: pd.DataFrame, n=14) -> np.ndarray:
    c,h,l = df["close"].values, df["high"].values, df["low"].values
    prev_c = np.roll(c,1); prev_c[0]=c[0]
    tr = np.maximum(h-l, np.maximum(np.abs(h-prev_c), np.abs(l-prev_c)))
    return pd.Series(tr).rolling(n, min_periods=n).mean().values

def macd_hist(series: pd.Series, fast=8, slow=17, sig=9) -> pd.Series:
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    signal= macd.ewm(span=sig, adjust=False).mean()
    return macd - signal

def window_swing(df: pd.DataFrame, lookback: int, idx: int):
    lo = df["low"].iloc[max(0, idx+1-lookback): idx+1].min()
    hi = df["high"].iloc[max(0, idx+1-lookback): idx+1].max()
    return float(hi), float(lo)

def fibo_levels(df: pd.DataFrame, lookback: int, idx: int, dir_long: bool):
    hi, lo = window_swing(df, lookback, idx)
    if dir_long:   # up
        r382 = hi - 0.382*(hi-lo); r618 = hi - 0.618*(hi-lo)
    else:          # down
        r382 = lo + 0.382*(hi-lo); r618 = lo + 0.618*(hi-lo)
    return r382, r618

def maybe_signal(df: pd.DataFrame, pair: str):
    # FRESHNESS GATE
    last_ts = df["timestamp"].iloc[-1]
    age_h = (datetime.now(timezone.utc) - last_ts).total_seconds()/3600.0
    if age_h > FRESH_MAX_HOURS:
        return None, f"stale data ({age_h:.1f}h> {FRESH_MAX_HOURS}h)"

    # indicators
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"]= ema(df["close"], 200)
    df["macd_hist"] = macd_hist(df["close"], 8,17,9)
    df["atr"] = atr(df, ATR_PERIOD)

    diff = (df["ema20"] - df["ema50"]).to_numpy()
    sign = np.sign(np.nan_to_num(diff))
    idx  = np.where(np.diff(sign)!=0)[0]
    if len(idx)==0:
        return None, "no cross in history"

    i_cross = int(idx[-1])
    i_sig   = min(i_cross+1, len(df)-1)      # confirmed bar
    if (i_sig != len(df)-1) and (not EMIT_PAST):
        return None, f"cross happened at {df['timestamp'].iloc[i_sig].isoformat()}, but late emit blocked"

    row     = df.iloc[i_sig]
    dir_long= bool(row.ema20 > row.ema50)
    side    = "long" if dir_long else "short"
    entry   = float(row.close)
    ps      = pip_size(pair)

    a = float(row.atr)
    if not np.isfinite(a) or a<=0:
        return None, "no ATR"

    if dir_long:
        sl = entry - ATR_SL_MULT*a
        tp = entry + TP_R_MULT*abs(entry-sl)
    else:
        sl = entry + ATR_SL_MULT*a
        tp = entry - TP_R_MULT*abs(entry-sl)
    stop_pips = int(round(abs(entry-sl)/ps))

    # atoms
    atoms = {}
    atoms["A1_MACD_SIGN"]   = (row.macd_hist>0) if dir_long else (row.macd_hist<0)
    atoms["A2_SIDE_EMA200"] = (entry>row.ema200) if dir_long else (entry<row.ema200)
    atoms["A3_EMA_STACK"]   = (row.ema20>row.ema50>row.ema200) if dir_long else (row.ema20<row.ema50<row.ema200)
    atoms["A4_NEAR_EMA20"]  = (abs(entry-row.ema20)/ps <= NEAR_PIPS)
    atoms["A5_NEAR_EMA50"]  = (abs(entry-row.ema50)/ps <= NEAR_PIPS)
    r382, r618              = fibo_levels(df, FIBO_LOOKBACK, i_sig, dir_long)
    atoms["A6_NEAR_FIBO"]   = (abs(entry-r382)/ps<=NEAR_PIPS) or (abs(entry-r618)/ps<=NEAR_PIPS)
    swing_hi, swing_lo      = window_swing(df, SWING_LOOKBACK, i_sig)
    atoms["A7_SWING_BREAK"] = (entry>swing_hi) if dir_long else (entry<swing_lo)

    rule_score = 1 + sum(1 for v in atoms.values() if v)
    rules_text = "CROSS " + " ".join([k for k,v in atoms.items() if v])

    sig = {
        "pair": pair,
        "direction": side,
        "entry": round(entry, 5),
        "sl":    round(sl, 5),
        "tp":    round(tp, 5),
        "stop_pips": int(stop_pips),
        "lot_size": 0.01,
        "risk_pct": "",
        "risk_usd": "",
        "issued_at": row.timestamp.isoformat(),
        "rule_score": int(rule_score),
        "rules": rules_text
    }
    dbg = {
        "time_signal": row.timestamp.isoformat(),
        "dir": side,
        "ema20": float(row.ema20),
        "ema50": float(row.ema50),
        "ema200": float(row.ema200),
        "macd_hist": float(row.macd_hist),
        "atr": float(a),
        "swing_high": float(swing_hi),
        "swing_low": float(swing_lo),
        "r382": float(r382),
        "r618": float(r618),
        "atoms": atoms
    }
    return (sig, dbg), "ok"

def main():
    items = []
    lines = []
    for pair in PAIRS:
        try:
            df = load_df(pair)
            out, why = maybe_signal(df, pair)
            if out is None:
                lines.append(f"[{pair}] SKIP: {why}")
                continue
            sig, dbg = out
            items.append(sig)
            hit = sum(1 for v in dbg["atoms"].values() if v)
            lines.append(f"[{pair}] OK {sig['direction']} @ {dbg['time_signal']} score={sig['rule_score']} atoms={hit} -> {sig['rules']}")
        except Exception as e:
            lines.append(f"[{pair}] ERROR: {e}")

    # CSV (legacy + rule fields)
    csv_path = os.path.join(OUT_DIR, "mt4.csv")
    hdr = ['pair','direction','entry','sl','tp','stop_pips','lot_size','risk_pct','risk_usd','issued_at','rule_score','rules']
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(hdr)+"\n")
        for r in items:
            f.write(",".join(str(r.get(k,"")) for k in hdr)+"\n")

    with open(os.path.join(OUT_DIR,"latest.json"),"w",encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "items": items}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR,"debug_report.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")

    print(f"OK: {len(items)} signals -> {csv_path}")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
