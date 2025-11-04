#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twelve Data H1 -> data/<PAIR>.csv
- throttling: max 8 zapytań/min (domyślnie)
- auto-retry po rate limit (sleep i ponów)
- bezpieczne łączenie z istniejącymi CSV (różne nazwy kolumn czasu)
ENV:
  TWELVE_DATA_KEY  (w Secrets)
  OUT_DIR=data
  PAIRS=EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD,BTCUSD
  TD_RATE_PER_MIN=8   # jeśli Twój plan ma inny limit, ustaw tutaj
"""

import os, sys, time, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

OUT_DIR = os.getenv("OUT_DIR", "data")
API_KEY = os.environ["TWELVE_DATA_KEY"]
PAIRS = [s.strip().upper() for s in os.getenv(
    "PAIRS",
    "EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD,BTCUSD"
).split(",") if s.strip()]
RATE_PER_MIN = int(os.getenv("TD_RATE_PER_MIN", "8"))  # limit/min wg planu

UA = {"User-Agent": "Mozilla/5.0 (fx-h1-feed/1.0)"}

def pair_to_td_symbol(pair: str) -> str:
    return pair[:3] + "/" + pair[3:]

def _only_closed(df: pd.DataFrame) -> pd.DataFrame:
    now_full = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    df = df[df["time"] < now_full]
    df = df.dropna(subset=["open","high","low","close"])
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

def fetch_td_once(pair: str) -> pd.DataFrame:
    symbol = pair_to_td_symbol(pair)
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval=1h&outputsize=5000&timezone=UTC&order=ASC&apikey={API_KEY}"
    )
    r = requests.get(url, headers=UA, timeout=40)
    # 429 od razu oddajemy do retry
    if r.status_code == 429:
        raise RuntimeError("RATE_LIMIT")
    r.raise_for_status()
    j = r.json()
    msg = (j.get("message") or "").lower()
    if "api credits" in msg or "rate limit" in msg:
        raise RuntimeError("RATE_LIMIT")
    if "values" not in j:
        raise RuntimeError(f"Twelve Data error for {pair}: {j.get('message') or 'no values'}")

    df = pd.DataFrame(j["values"]).rename(columns={"datetime": "time"})
    for col in ["open","high","low","close","volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return _only_closed(df[["time","open","high","low","close","volume"]])

def fetch_td(pair: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries + 1):
        try:
            return fetch_td_once(pair)
        except RuntimeError as e:
            if "RATE_LIMIT" in str(e):
                # poczekaj do następnej minuty i spróbuj ponownie
                time.sleep(65)
                continue
            raise
        except requests.RequestException:
            # sieć/HTTP – krótki backoff
            time.sleep(5)
    raise RuntimeError(f"{pair}: rate limit / network after retries")

def load_existing_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # wykryj kolumnę czasu niezależnie od starego formatu
    tcol = next((c for c in ["time","timestamp","datetime","Date","date","datetime_utc"] if c in df.columns), None)
    if not tcol:
        raise ValueError("no time column")
    df["time"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.rename(columns={c: c.lower() for c in df.columns})
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        else: df[c] = np.nan
    return _only_closed(df[["time","open","high","low","close","volume"]])

def merge_with_existing(pair: str, df_new: pd.DataFrame) -> pd.DataFrame:
    out = Path(OUT_DIR) / f"{pair}.csv"
    if out.exists():
        try:
            old = load_existing_csv(out)
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
    for i, pair in enumerate(pairs, start=1):
        try:
            results.append(run_for_pair(pair))
        except Exception as e:
            ok = False
            print(f"[{pair}] ERROR: {e}")
        # throttling: po każdych RATE_PER_MIN parach poczekaj do kolejnej minuty
        if i % RATE_PER_MIN == 0 and i < len(pairs):
            time.sleep(65)
    print(json.dumps({"ok": ok, "results": results}, ensure_ascii=False))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main(PAIRS))
