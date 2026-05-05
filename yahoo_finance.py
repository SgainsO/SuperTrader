from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"

# Historical periods of interest

def download(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance for the given tickers and date range."""
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(tickers, list) and len(tickers) == 1:
        raw.columns = raw.columns.droplevel(1)
    return raw


def save(df: pd.DataFrame, filename: str) -> Path:
    """Write a DataFrame to DATA_DIR as a CSV and return the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    df.to_csv(path)
    return path


def fetch_all(tickers: List[str], interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Download and save all three historical periods; return a mapping of period name → DataFrame."""
    results: dict[str, pd.DataFrame] = {}
    for period_name, (start, end) in _PERIODS.items():
        df = download(tickers, start=start, end=end, interval=interval)
        ticker_label = "_".join(tickers)
        path = save(df, f"{ticker_label}_{period_name}.csv")
        print(f"Saved {period_name} → {path}")
        results[period_name] = df
    return results


def load(filename: str) -> pd.DataFrame:
    """Load a previously saved CSV from DATA_DIR."""
    return pd.read_csv(DATA_DIR / filename, index_col=0, parse_dates=True)


if __name__ == "__main__":
    fetch_all(["SPY"])
