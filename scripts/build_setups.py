# scripts/build_setups.py
import os, json, math
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUT_DIR = "signals"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

CAPITAL   = float(os.getenv("CAPITAL", "500"))      # USD
RISK_PCT  = float(os.getenv("RISK_PCT", "0.01"))    # 1%/trade
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.5"))
TP_R_MULT   = float(os.getenv("TP_R_MULT",  "2.0"))
CONTRACT_SIZE_FX = 100_000

def pip_size(pair: str) -> float:
    if pair == "BTCUSD": return 1.0
    if pair == "XAUUSD": return 0.01
    if pair.endswith("JPY"): return 0.01
    return 0.0001

def pip_value_usd_per_lot(pair: str, price: float) -> float:
    if pair == "XAUUSD": return 1.0           # 1 lot = 100 oz, 0.01 USD pip -> 1 USD/pip/lot
    if pair == "BTCUSD": return 1.0           # 1 lot = 1 BTC, 1 USD pip -> 1 USD/pip/lot
    if pair.endswith("USD"): return 10.0      # EURUSD, GBPUSD, AUDUSD, NZDUSD
    return (CONTRACT_SIZE_FX * pip_size(pair)) / float(price)  # USD-baza (USDJPY/CHF/CAD)

def round_price(pair: str, price: float) -> float:
    if pair in ("BTCUSD","XAUUSD"): nd = 2
    elif pair.endswith("JPY"): nd = 3
    else: nd = 5
    return round(float(price), nd)

def load_pair_df(pair: str) -> pd.DataFrame:
    p = Path(DATA_DIR) / f"{pair}.csv"
    if not p.exists(): raise FileNotFoundError(p)
    df = pd.read_csv(p)
    tcol = next((c for c in ["time","timestamp","datetime","Date","date","datetime_utc"] if c in df.columns), None)
    if not tcol: raise KeyError(f"{p}: brak kolumny czasu")
    df["time"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.rename(columns={c:c.lower() for c in df.columns})
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        else: df[c] = np.nan
    now_full = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    df = df[df["time"] < now_full]
    df = df.dropna(subset=["open","high","low","close"])
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    return df[["time","open","high","low","close","volume"]]

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def atr_wilder(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def last_cross_setup(df: pd.DataFrame, pair: str):
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["atr14"] = atr_wilder(df["high"], df["low"], df["close"], 14)
    if len(df) < 60 or pd.isna(df["ema20"].iloc[-1]) or pd.isna(df["ema50"].iloc[-1]): return None

    e20p,e20n = df["ema20"].iloc[-2], df["ema20"].iloc[-1]
    e50p,e50n = df["ema50"].iloc[-2], df["ema50"].iloc[-1]
    c = df["close"].iloc[-1]; atr_now = float(df["atr14"].iloc[-1])

    direction = None
    if e20p <= e50p and e20n > e50n and c > e20n: direction = "long"
    elif e20p >= e50p and e20n < e50n and c < e20n: direction = "short"
    if not direction: return None

    ps = pip_size(pair)
    stop_dist = atr_now * ATR_MULT_SL
    if stop_dist <= 0 or not math.isfinite(stop_dist): return None
    stop_pips = max(1, int(round(stop_dist / ps)))

    risk_usd = CAPITAL * RISK_PCT
    pip_val = pip_value_usd_per_lot(pair, price=c)
    lot_size = max(0.01, round(risk_usd / (stop_pips * pip_val), 2))

    entry = c
    if direction == "long":
        sl = entry - stop_dist; tp = entry + stop_dist * TP_R_MULT
    else:
        sl = entry + stop_dist; tp = entry - stop_dist * TP_R_MULT

    return {
        "status": "setup", "pair": pair, "direction": direction,
        "entry": round_price(pair, entry), "sl": round_price(pair, sl), "tp": round_price(pair, tp),
        "stop_pips": stop_pips, "atr14": round(atr_now, 6),
        "risk_pct": RISK_PCT, "risk_usd": round(risk_usd, 2), "lot_size": lot_size,
    }

def autodiscover_pairs():
    pairs = [f.stem.upper() for f in Path(DATA_DIR).glob("*.csv")]
    if "PAIRS" in os.environ:
        wl = [s.strip().upper() for s in os.environ["PAIRS"].split(",") if s.strip()]
        pairs = [p for p in pairs if p in wl]
    return sorted(pairs)

def main():
    items=[]
    for pair in autodiscover_pairs():
        try:
            df = load_pair_df(pair); setup = last_cross_setup(df, pair)
            if setup:
                print(f"[{pair}] SETUP: {setup['direction']} @ {setup['entry']} (SL {setup['sl']}, TP {setup['tp']})")
                items.append(setup)
            else:
                print(f"[{pair}] brak setupu")
        except Exception as e:
            print(f"[{pair}] ERROR: {e}")
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capital": CAPITAL, "risk_pct": RISK_PCT, "items": items,
    }
    out = Path(OUT_DIR) / "latest.json"
    with open(out,"w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,indent=2)
    print(f"Zapisano {out} (items: {len(items)})")

if __name__ == "__main__":
    main()
