#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator sygnałów H1:
- Sygnał bazowy = ostatnie przecięcie EMA20/EMA50 (potwierdzone bar po przecięciu)
- Atomy (nie blokują, tylko zwiększają rule_score):
  A1 MACD(8,17,9) histogram po stronie kierunku
  A2 Close po właściwej stronie EMA200
  A3 Stack: EMA20>EMA50>EMA200 (long) lub odwrotnie (short)
  A4 Blisko EMA20 (<= near_pips)
  A5 Blisko EMA50 (<= near_pips)
  A6 Blisko Fibo 38.2%/61.8% ostatniego swingu (lookback)
  A7 Wybicie swingu (close > swingHigh lub close < swingLow)
- SL/TP: SL = 1.5*ATR14, TP = 2R (R = |entry-SL|)
Zapis:
  signals/mt4.csv  (pierwsze 10 kolumn kompatybilne ze starym EA)
  signals/latest.json
  signals/debug_report.txt
"""
import os, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# ======= KONFIG =======
PAIRS = ["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD","XAUUSD"]
DATA_DIR = "data"
OUT_DIR  = "signals"
os.makedirs(OUT_DIR, exist_ok=True)

ATR_PERIOD      = 14
ATR_SL_MULT     = 1.5
TP_R_MULT       = 2.0
SWING_LOOKBACK  = 30
FIBO_LOOKBACK   = 80
NEAR_PIPS       = 10.0
# ======================

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

def window_swing(df: pd.DataFrame, lookback: int, shift: int):
    lo = df["low"].iloc[max(0, shift+1): shift+1+lookback].min()
    hi = df["high"].iloc[max(0, shift+1): shift+1+lookback].max()
    if np.isnan(lo): lo = df["low"].min()
    if np.isnan(hi): hi = df["high"].max()
    return float(hi), float(lo)

def fibo_levels(df: pd.DataFrame, lookback: int, shift: int, dir_long: bool):
    hi, lo = window_swing(df, lookback, shift)
    if dir_long:   # ruch w górę
        r382 = hi - 0.382*(hi-lo); r618 = hi - 0.618*(hi-lo)
    else:          # ruch w dół
        r382 = lo + 0.382*(hi-lo); r618 = lo + 0.618*(hi-lo)
    return r382, r618

def last_cross_signal(df: pd.DataFrame, pair: str):
    # EMA
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"]= ema(df["close"], 200)
    # MACD hist
    df["macd_hist"] = macd_hist(df["close"], 8,17,9)
    # ATR
    df["atr"] = atr(df, ATR_PERIOD)

    diff = (df["ema20"] - df["ema50"]).to_numpy()
    sign = np.sign(np.nan_to_num(diff))
    idx  = np.where(np.diff(sign)!=0)[0]
    if len(idx)==0: return None

    i_cross = int(idx[-1])                  # bar z przecięciem
    i_sig   = min(i_cross+1, len(df)-1)     # bar po przecięciu (zamknięty)
    row     = df.iloc[i_sig]
    dir_long= bool(row.ema20 > row.ema50)
    side    = "long" if dir_long else "short"
    entry   = float(row.close)
    ps      = pip_size(pair)

    # SL/TP na bazie ATR
    a = float(row.atr)
    if not np.isfinite(a) or a<=0: return None
    if dir_long:
        sl = entry - ATR_SL_MULT*a
        tp = entry + TP_R_MULT*abs(entry-sl)
    else:
        sl = entry + ATR_SL_MULT*a
        tp = entry - TP_R_MULT*abs(entry-sl)
    stop_pips = int(round(abs(entry-sl)/ps))

    # ===== Atomy (boolean) =====
    atoms = {}
    # A1: MACD histogram po stronie kierunku
    atoms["A1_MACD_SIGN"] = (row.macd_hist>0) if dir_long else (row.macd_hist<0)
    # A2: strona EMA200
    atoms["A2_SIDE_EMA200"] = (entry>row.ema200) if dir_long else (entry<row.ema200)
    # A3: stack
    atoms["A3_EMA_STACK"] = (row.ema20>row.ema50>row.ema200) if dir_long else (row.ema20<row.ema50<row.ema200)
    # A4/A5: blisko EMA20/EMA50
    atoms["A4_NEAR_EMA20"] = (abs(entry-row.ema20)/ps <= NEAR_PIPS)
    atoms["A5_NEAR_EMA50"] = (abs(entry-row.ema50)/ps <= NEAR_PIPS)
    # A6: blisko Fibo 38.2/61.8
    r382, r618 = fibo_levels(df, FIBO_LOOKBACK, i_sig, dir_long)
    atoms["A6_NEAR_FIBO"] = (abs(entry-r382)/ps<=NEAR_PIPS) or (abs(entry-r618)/ps<=NEAR_PIPS)
    # A7: wybicie swingu z okna
    swing_hi, swing_lo = window_swing(df, SWING_LOOKBACK, i_sig)
    atoms["A7_SWING_BREAK"] = (entry>swing_hi) if dir_long else (entry<swing_lo)

    # score = 1 (cross) + sum(atoms)
    rule_score = 1 + sum(1 for v in atoms.values() if v)
    rules_text = "CROSS " + " ".join([k for k,v in atoms.items() if v])

    return {
        "pair": pair,
        "direction": side,
        "entry": round(entry, 5),
        "sl":    round(sl, 5),
        "tp":    round(tp, 5),
        "stop_pips": int(stop_pips),
        "lot_size": 0.01,      # EA może liczyć własny lot na bazie rule_score
        "risk_pct": "",
        "risk_usd": "",
        "issued_at": row.timestamp.isoformat(),
        "rule_score": int(rule_score),
        "rules": rules_text
    }, {
        "i_cross": int(i_cross),
        "i_signal": int(i_sig),
        "time_signal": row.timestamp.isoformat(),
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

def main():
    items = []
    debug_lines = []
    for pair in PAIRS:
        try:
            df = load_df(pair)
            out = last_cross_signal(df, pair)
            if out is None:
                debug_lines.append(f"[{pair}] no-cross or insufficient ATR/data")
                continue
            sig, dbg = out
            items.append(sig)
            hits = sum(1 for v in dbg["atoms"].values() if v)
            debug_lines.append(
                f"[{pair}] {sig['direction']} time={dbg['time_signal']} "
                f"entry={sig['entry']:.5f} SL={sig['sl']:.5f} TP={sig['tp']:.5f} "
                f"score={sig['rule_score']} atoms={hits} -> {sig['rules']}"
            )
        except Exception as e:
            debug_lines.append(f"[{pair}] ERROR: {e}")

    # CSV dla EA (legacy + dodatkowe kolumny na końcu)
    csv_path = os.path.join(OUT_DIR, "mt4.csv")
    hdr = ['pair','direction','entry','sl','tp','stop_pips','lot_size','risk_pct','risk_usd','issued_at','rule_score','rules']
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(hdr) + "\n")
        for r in items:
            f.write(",".join(str(r.get(k,"")) for k in hdr) + "\n")

    # JSON meta
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(), "items": items}
    with open(os.path.join(OUT_DIR, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Raport tekstowy
    with open(os.path.join(OUT_DIR, "debug_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(debug_lines) + "\n")

    print(f"OK: generated {len(items)} signals → {csv_path}")
    print("\n".join(debug_lines))

if __name__ == "__main__":
    main()
