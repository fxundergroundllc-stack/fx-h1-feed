# fx-h1-feed
Free, no-key hourly FX H1 CSV feed (7 major pairs) using GitHub Actions + yfinance.

## What this gives you
- 7 CSVs in `data/` (EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD)
- ~200–400 H1 candles each (last ~60 days)
- Auto-updates **every 30 minutes**

## Quick start
1. Create a new **public** repo on GitHub (e.g., `fx-h1-feed`).
2. Upload the contents of this folder to the repo (keep the same structure).
3. Go to **Actions** tab → enable workflows for this repo if prompted.
4. In **Actions**, run the workflow `update-fx-h1` once manually (green button).
5. After it finishes, open `data/` and click each CSV → **Raw** → copy the URL.

Send me those 7 raw URLs here so I can compute EMA20/EMA50 + ATR(14) and generate setups.

## Notes
- Data source: Yahoo Finance via `yfinance` (intraday coverage ~60–90 days).
- Timestamps are UTC; I will convert to Europe/Warsaw on my side.