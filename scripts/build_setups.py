import os, json, math
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
    ZONE = ZoneInfo("Europe/Warsaw")
except Exception:
    ZONE = timezone.utc

import pandas as pd
import numpy as np

DATA_DIR = "data"
OUT_DIR = "signals"
os.makedirs(OUT_DIR, exist_ok=True)

# --- parametry z ENV ---
ENTRY_MODE       = os.getenv("ENTRY_MODE", "EMA_CROSS").upper()
TREND_MIN_BARS   = int(os.getenv("TREND_MIN_BARS", "3"))
SWING_LOOKBACK   = int(os.getenv("SWING_LOOKBACK", "80"))
DONCHIAN_N       = int(os.getenv("DONCHIAN_N", "20"))
ATR_MIN_MULT     = float(os.getenv("ATR_MIN_MULT", "0.8"))

ATR_MULT_SL      = float(os.getenv("ATR_MULT_SL", "1.5"))
TP_R_MULT        = float(os.getenv("TP_R_MULT", "2.0"))

PAIRS            = [p.strip().upper() for p in os.getenv("PAIRS","EURUSD,GBPUSD,USDJPY").split(",")]

CAPITAL          = float(os.getenv("CAPITAL", "10000"))   # opcjonalnie do risk_usd (info)
RISK_PCT         = float(os.getenv("RISK_PCT", "0.01"))   # 1% = 0.01

# --- utils ---
def pip_size(pair: str) -> float:
    if pair == "BTCUSD": return 1.0
    if pair == "XAUUSD": return 0.01
    return 0.01 if pair.endswith("JPY") else 0.0001

def load_pair_df(pair: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{pair}.csv")
    df = pd.read_csv(path)
    # elastyczne nazwy kolumn
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("timestamp") or cols.get("time") or cols.get("datetime") or list(df.columns)[0]
    ocol = cols.get("open") or "open"
    hcol = cols.get("high") or "high"
    lcol = cols.get("low")  or "low"
    ccol = cols.get("close") or "close"
    df = df.rename(columns={tcol:"timestamp", ocol:"open", hcol:"high", lcol:"low", ccol:"close"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").reset_index(drop=True)
    return df

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema20"]  = ema(d["close"], 20)
    d["ema50"]  = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    # Wilder ATR(14)
    tr1 = d["high"] - d["low"]
    tr2 = (d["high"] - d["close"].shift(1)).abs()
    tr3 = (d["low"]  - d["close"].shift(1)).abs()
    d["tr"] = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    d["atr14"] = d["tr"].rolling(14, min_periods=1).mean()
    # DMACD (8,17,9)
    fast, slow, sig = 8,17,9
    macd = ema(d["close"], fast) - ema(d["close"], slow)
    d["macd_hist"] = macd - ema(macd, sig)
    # Donchian
    d["don_hi"] = d["high"].rolling(DONCHIAN_N, min_periods=1).max()
    d["don_lo"] = d["low"].rolling(DONCHIAN_N, min_periods=1).min()
    return d

def last_swing(d: pd.DataFrame, lookback: int):
    w = d.iloc[-lookback:].copy()
    return float(w["high"].max()), float(w["low"].min())

def in_trend_long(d, i=None):
    if i is None: i = len(d)-1
    sl = slice(max(0, i-TREND_MIN_BARS+1), i+1)
    return (d["close"].iloc[sl] > d["ema200"].iloc[sl]).all()

def in_trend_short(d, i=None):
    if i is None: i = len(d)-1
    sl = slice(max(0, i-TREND_MIN_BARS+1), i+1)
    return (d["close"].iloc[sl] < d["ema200"].iloc[sl]).all()

# --- logika wejść ---
def signal_ema_cross(d: pd.DataFrame):
    diff = d["ema20"] - d["ema50"]
    sign = np.sign(diff.fillna(0.0).values)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx)==0: return None
    i = int(idx[-1]+1)  # świeca po przecięciu
    if i >= len(d): i = len(d)-1
    if d["ema20"].iloc[i] > d["ema50"].iloc[i] and d["close"].iloc[i] > d["ema20"].iloc[i]:
        return ("long", i)
    if d["ema20"].iloc[i] < d["ema50"].iloc[i] and d["close"].iloc[i] < d["ema20"].iloc[i]:
        return ("short", i)
    return None

def signal_ema_cross_trend_macd(d: pd.DataFrame):
    out = signal_ema_cross(d)
    if not out: return None
    side, i = out
    if side=="long":
        if not in_trend_long(d, i): return None
        if d["macd_hist"].iloc[i] <= 0: return None
    else:
        if not in_trend_short(d, i): return None
        if d["macd_hist"].iloc[i] >= 0: return None
    return (side, i)

def signal_pullback_fibo(d: pd.DataFrame):
    i = len(d)-1
    # trend
    if (d["ema20"].iloc[i] > d["ema50"].iloc[i] > d["ema200"].iloc[i]).all() if hasattr(d["ema20"].iloc[i],"all") else (d["ema20"].iloc[i] > d["ema50"].iloc[i] > d["ema200"].iloc[i]):
        side="long"
    elif (d["ema20"].iloc[i] < d["ema50"].iloc[i] < d["ema200"].iloc[i]).all() if hasattr(d["ema20"].iloc[i],"all") else (d["ema20"].iloc[i] < d["ema50"].iloc[i] < d["ema200"].iloc[i]):
        side="short"
    else:
        return None
    hi, lo = last_swing(d, SWING_LOOKBACK)
    if side=="long":
        top, base = hi, lo
        r382 = top - 0.382*(top-base)
        r618 = top - 0.618*(top-base)
        in_zone = r618 <= d["close"].iloc[i] <= r382
        momentum_ok = d["macd_hist"].iloc[i] > 0
    else:
        base, top = hi, lo  # tu "hi" > "lo"
        r382 = top + 0.382*(base-top)
        r618 = top + 0.618*(base-top)
        in_zone = r382 <= d["close"].iloc[i] <= r618
        momentum_ok = d["macd_hist"].iloc[i] < 0
    if in_zone and momentum_ok:
        return (side, i)
    return None

def signal_donchian_breakout(d: pd.DataFrame):
    i = len(d)-1
    # atr filter: zmienność powyżej mediany*N
    atr = d["atr14"].iloc[-DONCHIAN_N:]
    if len(atr) < 5: return None
    if d["atr14"].iloc[i] < ATR_MIN_MULT * float(np.median(atr.values)): return None
    # wybicie
    hi = float(d["don_hi"].iloc[i-1])  # poprzednie N
    lo = float(d["don_lo"].iloc[i-1])
    c = float(d["close"].iloc[i])
    if c > hi: return ("long", i)
    if c < lo: return ("short", i)
    return None

def choose_signal(d: pd.DataFrame):
    mode = ENTRY_MODE
    if mode == "EMA_CROSS":                  return signal_ema_cross(d)
    if mode == "EMA_CROSS_TREND_MACD":       return signal_ema_cross_trend_macd(d)
    if mode == "PULLBACK_FIBO":              return signal_pullback_fibo(d)
    if mode == "DONCHIAN_BREAKOUT":          return signal_donchian_breakout(d)
    # fallback
    return signal_ema_cross(d)

def build_item(pair: str, d: pd.DataFrame):
    sig = choose_signal(d)
    if not sig: return None
    side, i = sig
    entry = float(d["close"].iloc[i])
    atr   = float(d["atr14"].iloc[i])
    if atr <= 0: return None
    if side=="long":
        sl = entry - ATR_MULT_SL*atr
        tp = entry + TP_R_MULT*(entry - sl)
    else:
        sl = entry + ATR_MULT_SL*atr
        tp = entry - TP_R_MULT*(sl - entry)
    ps = pip_size(pair)
    stop_pips = abs(entry - sl)/ps
    return {
        "pair": pair,
        "status": "setup",
        "direction": side,
        "entry": round(entry, 5),
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "stop_pips": int(round(stop_pips)),
        "risk_pct": RISK_PCT,
        "risk_usd": round(CAPITAL * RISK_PCT, 2),
        "atr14": round(atr, 6),
        "ema20": round(float(d["ema20"].iloc[i]), 6),
        "ema50": round(float(d["ema50"].iloc[i]), 6),
        "ema200": round(float(d["ema200"].iloc[i]), 6),
        "signal_bar": str(d["timestamp"].iloc[i]),
        "entry_mode": ENTRY_MODE
    }

def main():
    items = []
    for pair in PAIRS:
        try:
            df = load_pair_df(pair)
            df = indicators(df)
            item = build_item(pair, df)
            if item: items.append(item)
        except Exception as e:
            print(f"[{pair}] ERROR: {e}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
        "entry_mode": ENTRY_MODE,
        "params": {
            "TREND_MIN_BARS": TREND_MIN_BARS,
            "SWING_LOOKBACK": SWING_LOOKBACK,
            "DONCHIAN_N": DONCHIAN_N,
            "ATR_MULT_SL": ATR_MULT_SL,
            "TP_R_MULT": TP_R_MULT
        }
    }
    with open(os.path.join(OUT_DIR, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"latest.json written with {len(items)} item(s), ENTRY_MODE={ENTRY_MODE}")

if __name__ == "__main__":
    main()
