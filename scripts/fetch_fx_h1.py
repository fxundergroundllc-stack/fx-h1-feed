# scripts/fetch_fx_h1.py
import os, sys, time, io, requests, pandas as pd

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
if not API_KEY:
    print("ERROR: Missing ALPHAVANTAGE_API_KEY", file=sys.stderr)
    sys.exit(1)

PAIRS = [
    ("EUR","USD","EURUSD"),
    ("GBP","USD","GBPUSD"),
    ("USD","JPY","USDJPY"),
    ("USD","CHF","USDCHF"),
    ("USD","CAD","USDCAD"),
    ("AUD","USD","AUDUSD"),
    ("NZD","USD","NZDUSD"),
]

os.makedirs("data", exist_ok=True)

def fetch_pair(base, quote, symbol):
    url = (
        "https://www.alphavantage.co/query"
        f"?function=FX_INTRADAY&from_symbol={base}&to_symbol={quote}"
        "&interval=60min&outputsize=compact&datatype=csv"
        f"&apikey={API_KEY}"
    )
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print(f"[ERR] {symbol}: HTTP {r.status_code}")
        return False
    txt = r.text.strip()
    if txt.startswith("Note") or txt.startswith("{"):
        print(f"[WARN] {symbol}: API note/limit: {txt[:120].replace('\\n',' ')}")
        return False
    df = pd.read_csv(io.StringIO(txt))
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = df[["timestamp","open","high","low","close"]]
    out = f"data/{symbol}.csv"
    df.to_csv(out, index=False)
    print(f"[OK] {symbol}: saved {len(df)} rows -> {out}")
    return True

PAUSE_S = 15  # limit 5 req/min → pauza między parami
for i, (base, quote, sym) in enumerate(PAIRS):
    fetch_pair(base, quote, sym)
    if i < len(PAIRS)-1:
        time.sleep(PAUSE_S)
