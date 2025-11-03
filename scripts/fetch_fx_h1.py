#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pobiera H1 z Yahoo Finance i zapisuje CSV w katalogu 'data/'.
- Obsługa majors: EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD
- Flatten tablic z YF (naprawia 'Data must be 1-dimensional')
- Odrzuca bieżącą (niezamkniętą) świecę
- Dedup po czasie i łączenie z istniejącym CSV

Uruchamianie:
  python scripts/fetch_fx_h1.py
  python scripts/fetch_fx_h1.py EURUSD USDJPY
  PAIRS="EURUSD,USDJPY" python scripts/fetch_fx_h1.py

ENV (opcjonalnie):
  OUT_DIR=data
  INTERVAL=60m
  RANGE=10d
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

OUT_DIR = os.getenv("OUT_DIR", "data")
INTERVAL = os.getenv("INTERVAL", "60m")
RANGE = os.getenv("RANGE", "10d")

# Mapowanie par -> symbole Yahoo
SYMBOL_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
}

UA = {"User-Agent": "Mozilla/5.0 (compatible; fx-h1-feed/1.0)"}


def _flat(x):
    a = np.asarray(x)
    if a.ndim > 1:
        a = a.reshape(-1)
    return a


def fetch_yf_chart(symbol: str, interval: str = INTERVAL, range_: str = RANGE) -> dict:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_}&includePrePost=false"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    j = r.json()

    ok = j.get("chart", {}).get("result", [])
    if ok and ok[0].get("timestamp"):
        return j

    # fallback: ostatnie 14 dni przez period1/period2
    now = int(time.time())
    p1 = now - 14 * 24 * 3600
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&period1={p1}&period2={now}&includePrePost=false"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    j = r.json()

    ok = j.get("chart", {}).get("result", [])
    if not ok or not ok[0].get("timestamp"):
        raise RuntimeError(f"Brak danych z YF dla {symbol}")
    return j


def chart_to_df(chart_json: dict) -> pd.DataFrame:
    res = chart_json["chart"]["result"][0]
    q = res["indicators"]["quote"][0]

    ts = _flat(res["timestamp"])
    o = _flat(q.get("open"))
    h = _flat(q.get("high"))
    l = _flat(q.get("low"))
    c = _flat(q.get("close"))
    v = _flat(q.get("volume", np.zeros_like(ts)))

    n = min(len(ts), len(o), len(h), len(l), len(c), len(v))
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(ts[:n], unit="s", utc=True),
            "open": pd.to_numeric(o[:n], errors="coerce"),
            "high": pd.to_numeric(h[:n], errors="coerce"),
            "low": pd.to_numeric(l[:n], errors="coerce"),
            "close": pd.to_numeric(c[:n], errors="coerce"),
            "volume": pd.to_numeric(v[:n], errors="coerce").fillna(0),
        }
    )

    # tylko zamknięte świece (pełna poprzednia godzina UTC)
    now_full_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    df = df[df["time"] < now_full_hour]

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    return df


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
            print(f"[{pair}] Warning: nie mogę wczytać istniejącego CSV ({e}) — nadpisuję.")
    return df_new


def save_csv(pair: str, df: pd.DataFrame) -> Path:
    out = Path(OUT_DIR) / f"{pair}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    to_save = df.copy()
    to_save["time"] = to_save["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    to_save.to_csv(out, index=False)
    return out


def run_for_pair(pair: str) -> dict:
    key = pair.upper()
    yf_symbol = SYMBOL_MAP.get(key)
    if not yf_symbol:
        raise ValueError(f"Brak mapowania symbolu YF dla {pair}")
    j = fetch_yf_chart(yf_symbol, INTERVAL, RANGE)
    df = chart_to_df(j)
    df = merge_with_existing(key, df)
    path = save_csv(key, df)
    return {"pair": key, "rows": int(len(df)), "path": str(path)}


def main(pairs: list[str]) -> int:
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    results, ok = [], True
    for p in pairs:
        try:
            r = run_for_pair(p)
            print(f"[{r['pair']}] zapisano {r['rows']} wierszy -> {r['path']}")
            results.append(r)
        except Exception as e:
            ok = False
            print(f"[{p}] ERROR: {e}")
    print(json.dumps({"ok": ok, "results": results}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    if "PAIRS" in os.environ:
        pairs = [s.strip() for s in os.environ["PAIRS"].split(",") if s.strip()]
    elif len(sys.argv) > 1:
        pairs = [s.strip() for s in sys.argv[1:] if s.strip()]
    else:
        # Domyślnie: pełny zestaw majors
        pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
    sys.exit(main(pairs))
