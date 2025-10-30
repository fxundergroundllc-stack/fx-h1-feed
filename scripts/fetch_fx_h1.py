# scripts/fetch_fx_h1.py
# Pobiera H1 OHLC dla głównych par FX z AlphaVantage i zapisuje do data/*.csv

import os
import sys
import time
import json
import requests
import pandas as pd

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
if not API_KEY:
    print("ERROR: Missing ALPHAVANTAGE_API_KEY (repo Settings → Secrets → Actions).", file=sys.stderr)
    sys.exit(1)

# Lista par do pobrania
PAIRS = [
    {"base": "EUR", "quote": "USD", "symbol": "EURUSD"},
    {"base": "GBP", "quote": "USD", "symbol": "GBPUSD"},
    {"base": "USD", "quote": "JPY", "symbol": "USDJPY"},
    {"base": "USD", "quote": "CHF", "symbol": "USDCHF"},
    {"base": "USD", "quote": "CAD", "symbol": "USDCAD"},
    {"base": "AUD", "quote": "USD", "symbol": "AUDUSD"},
    {"base": "NZD", "quote": "USD", "symbol": "NZDUSD"},
]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

WAIT_BETWEEN_CALLS = 20  # AlphaVantage free: ok. 5 zapytań/min → 20s przerwy

def fetch_pair(base: str, quote: str, symbol: str) -> bool:
    url = (
        "https://www.alphavantage.co/query"
        f"?function=FX_INTRADAY&from_symbol={base}&to_symbol={quote}"
        "&interval=60min&outputsize=compact&datatype=json"
        f"&apikey={API_KEY}"
    )

    try:
        r = requests.get(url, timeout=30)
    except Exception as e:
        print(f"[ERR] {symbol}: request error: {e}")
        return False

    if r.status_code != 200:
        print(f"[ERR] {symbol}: HTTP {r.status_code}")
        return False

    txt = r.text
    # Sprawdź limity / błędy API
    try:
        js = json.loads(txt)
    except json.JSONDecodeError:
        preview = txt[:120].replace("\n", " ").replace("\r", " ")
        print(f"[WARN] {symbol}: non-JSON response: {preview}")
        return False

    for k in ("Note", "Information", "Error Message"):
        if k in js:
            preview = str(js.get(k, ""))[:200].replace("\n", " ").replace("\r", " ")
            print(f"[WARN] {symbol}: API {k}: {preview}")
            return False

    # Znajdź klucz z danymi H1 (zwykle: "Time Series FX (60min)")
    ts_key = None
    for key in js.keys():
        if "Time Series FX" in key:
            ts_key = key
            break

    if not ts_key or not isinstance(js.get(ts_key), dict):
        print(f"[WARN] {symbol}: Missing time series in response")
        return False

    ts = js[ts_key]
    rows = []
    for ts_str, ohlc in ts.items():
        try:
            rows.append(
                {
                    "timestamp": ts_str,
                    "open": float(ohlc["1. open"]),
                    "high": float(ohlc["2. high"]),
                    "low": float(ohlc["3. low"]),
                    "close": float(ohlc["4. close"]),
                }
            )
        except Exception:
            continue

    if not rows:
        print(f"[WARN] {symbol}: empty time series")
        return False

    df = pd.DataFrame(rows)
    # sort od najstarszych do najnowszych
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] {symbol}: saved {len(df)} rows -> {out_path}")
    return True


def main():
    ok, fail = 0, 0
    for p in PAIRS:
        success = fetch_pair(p["base"], p["quote"], p["symbol"])
        if success:
            ok += 1
        else:
            fail += 1
        # przerwa między wywołaniami, by nie złapać limitów
        time.sleep(WAIT_BETWEEN_CALLS)
    print(f"Done. Success: {ok}, Failed: {fail}")


if __name__ == "__main__":
    main()
