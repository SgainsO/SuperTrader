from __future__ import annotations

import os
from enum import Enum
from typing import List

import pandas as pd


class Period(Enum):
    CRISIS_2008     = "crisis_2008"
    COVID_ERA       = "covid_era"
    RANGE_2014_2018 = "range_2014_2018"


_PERIOD_DATES: dict[str, tuple[str, str]] = {
    Period.CRISIS_2008.value:     ("2007-01-01", "2009-12-31"),
    Period.COVID_ERA.value:       ("2020-01-01", "2021-12-31"),
    Period.RANGE_2014_2018.value: ("2014-01-01", "2018-12-31"),
}


def symbol_to_path(symbol: str, period: Period, base_dir: str | None = None) -> str:
    """Return CSV file path given ticker symbol and period."""
    if base_dir is None:
        base_dir = os.environ.get("MARKET_DATA_DIR", "../data/")
    return os.path.join(base_dir, f"{symbol}_{period.value}.csv")


def get_data(
    symbols: List[str],
    period: Period,
    columns: List[str],
    addSPY: bool = True,
) -> dict[str, pd.DataFrame]:
    """Read stock data for given symbols from CSV files for the chosen period.

    Returns a mapping of ticker → DataFrame containing only the requested columns.
    """
    start, end = _PERIOD_DATES[period.value]

    if addSPY and "SPY" not in symbols:
        symbols = ["SPY"] + list(symbols)

    results: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = pd.read_csv(
            symbol_to_path(symbol, period),
            index_col="Date",
            parse_dates=True,
            usecols=["Date"] + columns,
            na_values=["nan"],
        )
        results[symbol] = df.loc[start:end, columns]

    return results
