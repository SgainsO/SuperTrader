from math import sqrt

import numpy as np
import pandas as pd


def correct_index(sort_index):
    return pd.date_range(sort_index[0], sort_index[-1])


def compute_stats(position_total):
    daily_rets = position_total.pct_change()
    cr   = (position_total.iloc[-1] / position_total.iloc[0]) - 1
    adr  = daily_rets.iloc[1:].mean()
    sddr = daily_rets.iloc[1:].std()
    sr   = (adr / sddr) * sqrt(252.0)
    return cr, adr, sddr, sr


def compute_portvals_full(
    symbol,
    orders_file,
    prices_df,
    start_val=100_000,
    commission=9.95,
    impact=0.005,
):
    """
    Simulate portfolio value from a trade log.

    :param symbol: ticker string
    :param orders_file: DataFrame with datetime index and a "Trade" column
    :param prices_df: DataFrame with datetime index and a column named ``symbol`` (Close prices)
    :param start_val: starting cash
    :param commission: flat commission per trade
    :param impact: per-share market impact as fraction of price
    :return: DataFrame with an "ans" column containing daily portfolio value
    """
    san_inputs = orders_file

    prices = prices_df[[symbol]].copy().dropna()
    prices.insert(0, "Cash", 1.0)

    totalPort = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype=float)

    for date, row in san_inputs.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        tr = row["Trade"]
        if tr == 0:
            continue
        price = prices.loc[date_str, symbol]
        cash_delta = -tr * price - abs(tr) * price * impact - commission
        totalPort.loc[date_str, symbol] += tr
        totalPort.loc[date_str, "Cash"] += cash_delta

    makeHolding = pd.DataFrame(0, index=prices.index, columns=prices.columns, dtype=float)
    makeHolding.iloc[0] = totalPort.iloc[0]
    makeHolding.iloc[0, 0] += float(start_val)

    prev = totalPort.index[0].strftime("%Y-%m-%d")
    for date in totalPort.index[1:]:
        date_str = date.strftime("%Y-%m-%d")
        makeHolding.loc[date_str] = makeHolding.loc[prev] + totalPort.loc[date_str]
        prev = date_str

    portvals = (prices * makeHolding).sum(axis=1)
    return pd.DataFrame({"ans": portvals}, index=prices.index)
