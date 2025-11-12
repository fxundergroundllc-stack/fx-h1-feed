#!/usr/bin/env python3
import os, sys, time, csv
from datetime import datetime
import requests

PAIRS = ["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD","XAUUSD"]
OUTDIR = "data"; os.makedirs(OUTDIR, exist_ok=True)
API_KEY = os.getenv("TWELVE_DATA_KEY", "")
BASE = "https://api.twelvedata.com/time_series"

def fetch_pair(pair):
    if not API_KEY:
        print(f"[{pair}] No TWELVE_DATA_KEY — skip fetch.")
        return False
    params = dict(symbol=pair, interval="1h", outputsize=5000, format="JSON",
                  apikey=API_KEY, timezone="UTC")
    r = requests.get(BASE, params=params, timeout=30)
    if r.status_code != 200:
        print(f"[{pair}] HTTP {r.status_code} -> {r.text[:200]}")
        return False
    data = r.json()
    if "values" not in data:
        print(f"[{pair}] API error: {data}")
        return False
    path = os.path.join(OUTDIR, f"{pair}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["timestamp","open","high","low","close"])
        for v in reversed(data["values"]):
            w.writerow([v["datetime"], v["open"], v["high"], v["low"], v["close"]])
    print(f"[{pair}] saved -> {path} ({len(data['values'])} rows)")
    return True

def main():
    ok=0
    for p in PAIRS:
        try:
            if fetch_pair(p): ok+=1
            time.sleep(0.2)
        except Exception as e:
            print(f"[{p}] ERROR:", e)
    print(f"Fetch done: {ok}/{len(PAIRS)}")
if __name__ == "__main__":
    main()
