# scripts/fetch_fx_h1.py
import os
import pandas as pd
import yfinance as yf

os.makedirs("data", exist_ok=True)

# Mapowanie: PARA -> ticker Yahoo
TICKERS = {
    "BTCUSD": "BTC-USD",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}

def save_csv(pair: str, ticker: str):
    # 60 dni H1 daje sporo świec do EMA/ATR
    df = yf.download(tickers=ticker, period="60d", interval="1h", auto_adjust=False, progress=False)
    if df is None or df.empty:
        print(f"[WARN] Brak danych dla {pair}/{ticker}")
        return
    df = df.reset_index()
    # Standaryzacja kolumn
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["Datetime"]).dt.tz_convert("UTC"),
        "open": df["Open"].astype(float),
        "high": df["High"].astype(float),
        "low": df["Low"].astype(float),
        "close": df["Close"].astype(float),
    }).dropna()

    path = os.path.join("data", f"{pair}.csv")
    out.to_csv(path, index=False)
    print(f"[OK] Zapisano {path}: {len(out)} wierszy")

def main():
    for pair, tic in TICKERS.items():
        save_csv(pair, tic)

if __name__ == "__main__":
    main()
