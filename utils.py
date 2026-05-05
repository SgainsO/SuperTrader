from __future__ import annotations

from yahoo_finance import download, save
from enum import Enum
from pathlib import Path
from typing import List

import pandas as pd


DATA_DIR = Path(__file__).parent / "data"


class Period(Enum):
    CRISIS_2008     = "crisis_2008"
    POST_2008       = "post_2008"
    COVID_ERA       = "covid_era"
    POST_COVID_ERA  = "post_covid_era"
    RANGE_2014_2016 = "range_2014_2016"
    RANGE_2017_2019 = "range_2017_2019"


_PERIODS: dict[str, tuple[str, str]] = {
    "crisis_2008":     ("2007-01-01", "2009-12-31"),
    "post_2008":       ("2010-01-31", "2012-12-31"),
    "covid_era":       ("2020-01-01", "2021-12-31"),
    "post_covid_era":  ("2022-01-01", "2023-12-31"), 
    "range_2014_2016": ("2014-01-01", "2016-12-31"),
    "range_2017_2019": ("2017-01-01", "2019-12-31")
}


def symbol_to_path(symbol: str, period: Period) -> Path:
    return DATA_DIR / f"{symbol}_{period.value}.csv"


def get_data(
    symbols: List[str],
    period: Period,
    columns: List[str],
    addSPY: bool = True,
) -> dict[str, pd.DataFrame]:
    # Deferred import to avoid circular dependency (yahoo_finance does not import utils)

    start, end = _PERIODS[period.value]

    if addSPY and "SPY" not in symbols:
        symbols = ["SPY"] + list(symbols)

    results: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = symbol_to_path(symbol, period)
        if not path.exists():
            print(f"Downloading {symbol} ({period.value})...")
            df = download([symbol], start=start, end=end)
            save(df, path.name)

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        results[symbol] = df.loc[start:end, columns]

    return results
