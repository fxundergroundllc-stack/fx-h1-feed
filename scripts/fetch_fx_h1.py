import os
from datetime import datetime
import pandas as pd
import yfinance as yf

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

PERIOD = "60d"   # ~60 days back
INTERVAL = "1h"  # hourly
MAX_ROWS = 400   # keep last ~200-400 rows

for pair, ticker in PAIRS.items():
    try:
        df = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=False, progress=False, threads=False)
        if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
            print(f"[WARN] Missing OHLC for {pair} ({ticker}). Skipping.")
            continue

        # Ensure UTC and flatten index
        if df.index.tz is not None:
            df = df.tz_convert("UTC")
        df = df.tail(MAX_ROWS).copy()
        df.reset_index(inplace=True)
        df.rename(columns={
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
        }, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
        out_path = os.path.join(OUT_DIR, f"{pair}.csv")
        df[["timestamp", "open", "high", "low", "close"]].to_csv(out_path, index=False)
        print(f"[OK] Wrote {out_path} with {len(df)} rows")
    except Exception as e:
        print(f"[ERR] {pair} ({ticker}): {e}")