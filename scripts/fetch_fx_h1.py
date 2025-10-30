# scripts/fetch_fx_h1.py
# H1 OHLC z Yahoo Finance dla: BTCUSD, XAUUSD (GOLD), EURUSD, GBPUSD, USDJPY
# Zapis do data/<SYMBOL>.csv, deduplikacja po timestamp. Bez kluczy API.

import os, time, sys
import pandas as pd
import yfinance as yf

PAIRS = {
    "BTCUSD": "BTC-USD",     # Bitcoin (USD)
    "XAUUSD": "XAUUSD=X",    # Złoto spot (Gold vs USD)
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}

INTERVAL = "60m"                           # H1
PERIOD = os.environ.get("YF_PERIOD", "90d")# zakres historii (np. 60–120d)
OUTDIR = "data"
PAUSE_S = 2                                # krótka pauza między zapytaniami

os.makedirs(OUTDIR, exist_ok=True)

def fetch_one(name: str, ticker: str) -> bool:
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(
                interval=INTERVAL, period=PERIOD, auto_adjust=False
            )
            if df is None or df.empty:
                print(f"[WARN] {name}: empty frame (attempt {attempt+1}).")
                time.sleep(1.5)
                continue

            # Index -> UTC-naive
            idx = df.index
            try:
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_convert("UTC")
            except Exception:
                pass
            df = df.copy()
            df.index = pd.to_datetime(idx).tz_localize(None)

            out = df[["Open","High","Low","Close"]].rename(columns=str.lower)
            out.reset_index(inplace=True)
            out.rename(columns={"index":"timestamp","Datetime":"timestamp"}, inplace=True)
            out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            fn = os.path.join(OUTDIR, f"{name}.csv")
            if os.path.exists(fn):
                old = pd.read_csv(fn)
                merged = pd.concat([old, out], ignore_index=True)
                merged.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
                merged.sort_values("timestamp", inplace=True)
                out = merged

            out.to_csv(fn, index=False)
            print(f"[OK] {name} saved {len(out)} rows → {fn}")
            return True

        except Exception as e:
            print(f"[ERR] {name}: {e.__class__.__name__}: {e}")
            time.sleep(2)

    return False

ok = 0
for name, tick in PAIRS.items():
    if fetch_one(name, tick):
        ok += 1
    time.sleep(PAUSE_S)

print(f"Done. Success: {ok}, Failed: {len(PAIRS) - ok}")
sys.exit(0 if ok > 0 else 1)
