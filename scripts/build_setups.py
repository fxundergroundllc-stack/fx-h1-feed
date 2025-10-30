# scripts/build_setups.py
import os, json, math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

DATA_DIR = "data"
OUT_DIR = "signals"
os.makedirs(OUT_DIR, exist_ok=True)

# Użyjemy tych plików (złoto pomijamy)
PAIRS = ["BTCUSD", "EURUSD", "GBPUSD", "USDJPY"]

CAPITAL = 500.0
CONTRACT_SIZE = 100_000

def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001

def pip_value_usd_per_lot(pair: str, price: float) -> float:
    """
    Dla par z USD jako walutą kwotowaną (EURUSD, GBPUSD) -> 10 USD/pip/lot
    Dla JPY-kwotowanych (USDJPY) -> (100000 * 0.01) / price = 1000/price USD/pip/lot
    Dla krypto – brak standardu lotowego 100k, zwracamy None (nie liczymy pozycji).
    """
    if pair in ("EURUSD", "GBPUSD"):
        return CONTRACT_SIZE * pip_size(pair)  # = 10
    if pair == "USDJPY":
        return (CONTRACT_SIZE * pip_size(pair)) / price  # ≈ 1000/price
    return None  # BTC itp.

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr14(df: pd.DataFrame) -> pd.Series:
    # True Range
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def prior_day_levels(df: pd.DataFrame):
    """Zwraca (prev_high, prev_low, prev_pivot) z poprzedniego dnia kalendarzowego."""
    d = df.copy()
    d["date"] = d["timestamp"].dt.date
    last_two = d["date"].dropna().unique()[-2:]  # w tym ostatni pełny dzień + bieżący
    if len(last_two) < 2:
        return None, None, None
    prev_day = last_two[0]
    prev_rows = d[d["date"] == prev_day]
    if prev_rows.empty:
        return None, None, None
    ph = prev_rows["high"].max()
    pl = prev_rows["low"].min()
    pc = prev_rows["close"].iloc[-1]
    pivot = (ph + pl + pc) / 3.0
    return ph, pl, pivot

def swing_levels(df: pd.DataFrame, lookback=5):
    """Ostatnie lokalne high/low w ostatnich N świecach przed bieżącą."""
    recent = df.iloc[-(lookback+1):-1]  # 5 świec przed ostatnią
    return recent["high"].max(), recent["low"].min()

def load_pair_df(pair: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{pair}.csv")
    df = pd.read_csv(path)
    # Kolumny: timestamp, open, high, low, close
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).reset_index(drop=True)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["atr14"] = atr14(df)
    return df

def decide_setup(pair: str, df: pd.DataFrame):
    if len(df) < 60:
        return {"pair": pair, "status": "brak setupu", "reason": "za mało świec"}
    last = df.iloc[-1]
    price = float(last["close"])
    e20 = float(last["ema20"])
    e50 = float(last["ema50"])
    atr = float(last["atr14"])
    ps = pip_size(pair)

    prev_high, prev_low, prev_pivot = prior_day_levels(df)
    if prev_high is None:
        return {"pair": pair, "status": "brak setupu", "reason": "brak poprzedniego dnia"}

    # Kierunek wg filtra EMA + potwierdzenie (PDH/PDL lub Pivot)
    long_ok = (e20 > e50) and (price > max(prev_high, prev_pivot))
    short_ok = (e20 < e50) and (price < min(prev_low, prev_pivot))

    if not long_ok and not short_ok:
        return {"pair": pair, "status": "brak setupu"}

    # Entry = pullback do EMA20 (dla zlecenia limit – raportujemy jako cena wejścia)
    entry = e20

    # Swingi do SL
    swing_high, swing_low = swing_levels(df, lookback=5)

    # ATR w pipach + minimum 15 pips
    atr_pips = atr / ps if not np.isnan(atr) else 0.0
    min_pips = 15.0
    # Odległość do swingu w pipach (z 1 pip bufferem)
    if long_ok:
        swing_dist_pips = max(0.0, (entry - swing_low) / ps) + 1.0
        stop_pips = max(atr_pips, min_pips, swing_dist_pips)
        sl = entry - stop_pips * ps
        tp = entry + 2.0 * stop_pips * ps
        direction = "long"
        invalidation = "H1 close poniżej EMA50 lub poniżej wczorajszego pivotu"
    else:
        swing_dist_pips = max(0.0, (swing_high - entry) / ps) + 1.0
        stop_pips = max(atr_pips, min_pips, swing_dist_pips)
        sl = entry + stop_pips * ps
        tp = entry - 2.0 * stop_pips * ps
        direction = "short"
        invalidation = "H1 close powyżej EMA50 lub powyżej wczorajszego pivotu"

    # Pozycjonowanie (pomijamy BTC – brak standardu lotowego 100k/pips)
    if pair == "BTCUSD":
        lot_size = None
        risk_pct = None
        risk_usd = None
    else:
        pip_val = pip_value_usd_per_lot(pair, price)
        # domyślnie 2% ryzyka
        risk_pct = 0.02
        risk_usd = CAPITAL * risk_pct
        lot_size = risk_usd / (stop_pips * pip_val) if pip_val and stop_pips > 0 else 0.0
        # zaokrąglenie do 0.01 lota
        lot_size = round(lot_size + 1e-9, 2)

    return {
        "pair": pair,
        "status": "setup",
        "direction": direction,
        "entry": round(entry, 5 if not pair.endswith("JPY") else 3),
        "sl": round(sl, 5 if not pair.endswith("JPY") else 3),
        "tp": round(tp, 5 if not pair.endswith("JPY") else 3),
        "stop_pips": round(float(stop_pips), 1),
        "risk_pct": risk_pct,
        "risk_usd": round(risk_usd, 2) if risk_usd is not None else None,
        "lot_size": lot_size,
        "rationale": (
            f"Trend H1 {('EMA20>EMA50' if direction=='long' else 'EMA20<EMA50')}, "
            f"cena vs wczorajsze poziomy (PDH/PDL/pivot). Wejście przy EMA20."
        ),
        "invalidation": invalidation,
        "links": {}  # uzupełnimy niżej
    }

def raw_link(repo: str, pair: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/main/data/{pair}.csv"

def main():
    repo = os.getenv("GITHUB_REPOSITORY", "").strip()  # org/repo – dostępne na Actions
    results = []
    for pair in PAIRS:
        path = os.path.join(DATA_DIR, f"{pair}.csv")
        if not os.path.exists(path):
            results.append({"pair": pair, "status": "brak setupu", "reason": "brak pliku"})
            continue
        df = load_pair_df(pair)
        res = decide_setup(pair, df)
        if repo:
            res["links"] = {"csv_raw": raw_link(repo, pair)}
        results.append(res)

    # Zapis JSON
    with open(os.path.join(OUT_DIR, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": datetime.now(ZoneInfo("Europe/Warsaw")).isoformat(),
             "capital": CAPITAL, "items": results},
            f, ensure_ascii=False, indent=2
        )

    # Zapis markdown (krótka lista)
    ts = datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M Europe/Warsaw")
    lines = [f"### Setupy H1 — {ts}", ""]
    for r in results:
        if r.get("status") != "setup":
            lines.append(f"- {r['pair']}: **brak setupu**")
            continue
        risk_txt = "—" if r["risk_pct"] is None else f"{int(r['risk_pct']*100)}% (${r['risk_usd']})"
        lot_txt = "—" if r["lot_size"] is None else f"{r['lot_size']:.2f} lot"
        lines.append(
            f"- **{r['pair']}** | {r['direction']} | "
            f"entry {r['entry']} | SL {r['sl']} | TP {r['tp']} | "
            f"stop {r['stop_pips']} pips | {lot_txt} | ryzyko {risk_txt} "
            f"| *{r['rationale']}* | Invalidation: {r['invalidation']} "
            f"| [CSV]({r['links'].get('csv_raw','')})"
        )
    with open(os.path.join(OUT_DIR, "latest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
