"""
data_loader_real.py
--------------------
Loads real historical daily price data for a hand-picked, sector-diverse
6-stock S&P 500 universe, sourced from the Kaggle dataset: "S&P 500 Stocks - Daily Historical Data (10 Years)"
(https://www.kaggle.com/datasets/innacampo/s-and-p-500-stocks-daily-historical-data-10-years)

Selected tickers (one per sector, chosen for full 10-year coverage with
zero missing values and realistic cross-sector correlation):

    AAPL  - Technology
    XOM   - Energy
    JNJ   - Healthcare
    JPM   - Financials
    PG    - Consumer Staples
    CAT   - Industrials

This module is a drop-in replacement for data_gen.generate_price_data(): it returns
a DataFrame with the exact same shape/column convention

"""
import os
import numpy as np
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "real_data")
TICKERS = ["AAPL", "XOM", "JNJ", "JPM", "PG", "CAT"]
SECTORS = {
    "AAPL": "Technology", "XOM": "Energy", "JNJ": "Healthcare",
    "JPM": "Financials", "PG": "Consumer Staples", "CAT": "Industrials",
}


def _load_single_ticker(ticker):
    """Parses the Kaggle/yfinance-style CSV (3 header rows) for one ticker."""
    path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(
        path, skiprows=3, header=None,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Close", "High", "Low", "Open", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    return df[["Date", "Close", "Volume"]]


def load_real_price_data(tickers=TICKERS):
    """Loads and aligns all tickers on a common trading-day calendar."""
    frames = {t: _load_single_ticker(t) for t in tickers}

    # inner-join on Date so every asset has a value on every row (no NaNs)
    common_dates = None
    for t, df in frames.items():
        dates = set(df["Date"])
        common_dates = dates if common_dates is None else (common_dates & dates)
    common_dates = sorted(common_dates)

    out = pd.DataFrame({"Date": common_dates})
    for t in tickers:
        df = frames[t].set_index("Date")
        out[t] = df.loc[common_dates, "Close"].values
        out[f"{t}_VOL"] = df.loc[common_dates, "Volume"].values

    out = out.drop(columns=["Date"]).reset_index(drop=True)
    out.index.name = "day"
    return out


def train_test_split(df, test_fraction=0.2):
    n = len(df)
    split = int(n * (1 - test_fraction))
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


if __name__ == "__main__":
    data = load_real_price_data()
    print(data.head())
    print(data.shape)
    print("Date-aligned rows (trading days common to all 6 tickers):", len(data))
