#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pobiera H1 dla wybranych instrumentów z Twelve Data (REST) i zapisuje CSV w 'data/'.
Obsługa: majors, XAUUSD, BTCUSD. Filtrowanie tylko zamkniętych świec, dedup po czasie,
łączenie z istniejącymi CSV. Czyta listę z ENV PAIRS albo używa domyślnej.
ENV (w workflow):
  TWELVE_DATA_KEY=<klucz>
  OUT_DIR=data
  PAIRS=EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD,BTCUSD
"""

import os, sys, time, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

OUT_DIR = os.getenv("OUT_DIR", "data")
API_KEY = os.environ["TWELVE_DATA_KEY"]  # wymagany
PAIRS = [s.strip().upper() for s in os.getenv(
    "PAIRS",
    "EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD,BTCUSD"
).split(",") if s.strip()]

UA = {"User-Agent": "Mozilla/5.0 (fx-h1-feed/1.0)"}

def pair_to_td_symbol(pair: str) -> str:
    # EURUSD -> EUR/USD, XAUUSD -> XAU/USD, BTCUSD -> BTC/USD
    return pair[:3] + "/" + pair[3:]

def fetch_td(pair: str) -> pd.DataFrame:
    symbol = pair_to_td_symbol(pair)
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval=1h&outputsize=5000&timezone=UTC&order=ASC&apikey={API_KEY}"
    )
    r = requests.get(url, headers=UA, timeout=40)
    r.raise_for_status()
    j = r.json()
    if "values" not in j:
        raise RuntimeError(f"Twelve Data error for {pair}: {j.get('message') or 'no values'}")
    vals = j["values"]
    df = pd.DataFrame(vals)
    df.rename(columns={"datetime":"time"}, inplace=True)
    for col in ["open","high","low","close","volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    df["time"] = pd.to_datetime(df["time"], utc=True)
    # tylko zamknięte H1 (do pełnej bieżącej godziny)
    now_full = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    df = df[df["time"] < now_full]
    df = df.dropna(subset=["open","high","low","close"])
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    return df[["time","open","high","low","close","volume"]]

def merge_with_existing(pair: str, df_new: pd.DataFrame) -> pd.DataFrame:
    out = Path(OUT_DIR) / f"{pair}.csv"
    if out.exists():
        try:
            old = pd.read_csv(out, parse_dates=["time"])
            if old["time"].dtype.tz is None:
                old["time"] = pd.to_datetime(old["time"], utc=True)
            combo = pd.concat([old, df_new], ignore_index=True)
            combo = combo.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
            return combo
        except Exception as e:
            print(f"[{pair}] Warning: istniejący CSV uszkodzony ({e}) — nadpisuję.")
    return df_new

def save_csv(pair: str, df: pd.DataFrame) -> Path:
    out = Path(OUT_DIR) / f"{pair}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = df.copy()
    tmp["time"] = tmp["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp.to_csv(out, index=False)
    return out

def run_for_pair(pair: str) -> dict:
    df = fetch_td(pair)
    df = merge_with_existing(pair, df)
    p = save_csv(pair, df)
    print(f"[{pair}] {len(df)} wierszy -> {p}")
    return {"pair": pair, "rows": int(len(df)), "path": str(p)}

def main(pairs):
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    ok, results = True, []
    for i, pair in enumerate(pairs):
        try:
            results.append(run_for_pair(pair))
            # delikatny throttle vs. rate-limit
            time.sleep(1.2)
        except Exception as e:
            ok = False
            print(f"[{pair}] ERROR: {e}")
    print(json.dumps({"ok": ok, "results": results}, ensure_ascii=False))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main(PAIRS))
