from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd

from marketsimcode import compute_portvals_full, compute_stats
from strategy_learner import StrategyLearner
from utils import Period, get_data

# ── Configure stocks and starting value here ──────────────────────────────────
SYMBOLS = ["AAPL", "MSFT", "JPM"]
SV      = 100_000.0
# ─────────────────────────────────────────────────────────────────────────────

GRAPHS_DIR = Path(__file__).parent / "graphs"

EXPERIMENTS = [
    (Period.CRISIS_2008,     Period.POST_2008),
    (Period.COVID_ERA,       Period.POST_COVID_ERA),
    (Period.RANGE_2014_2016, Period.RANGE_2017_2019),
]


def run_experiment(symbol: str, train: Period, test: Period):
    learner = StrategyLearner(verbose=True)
    learner.add_evidence(symbol=symbol, period=train, sv=SV)

    trades   = learner.testPolicy(symbol=symbol, period=test, sv=SV)
    data     = get_data([symbol], period=test, columns=["Close", "High", "Low", "Volume"])
    close_df = data[symbol][["Close"]].rename(columns={"Close": symbol})
    close    = close_df[symbol]

    trades_input = trades.rename(columns={symbol: "Trade"})
    portvals     = compute_portvals_full(symbol, trades_input, close_df, start_val=SV)["ans"]
    benchmark    = close / close.iloc[0] * SV

    return portvals, benchmark, trades[symbol], close


def save_portfolio_plot(portvals: pd.Series, benchmark: pd.Series,
                        symbol: str, train: Period, test: Period) -> None:
    norm_port = portvals  / portvals.iloc[0]
    norm_bah  = benchmark / benchmark.iloc[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(norm_port, label="Q-Learner",  color="steelblue")
    ax.plot(norm_bah,  label="Buy & Hold", color="gray", linestyle="--")
    ax.set_title(f"{symbol} — Portfolio: train={train.value}  test={test.value}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = GRAPHS_DIR / symbol
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"portfolio_{test.value}.png", dpi=150)
    plt.close(fig)


def save_signals_plot(close: pd.Series, trades: pd.Series,
                      symbol: str, train: Period, test: Period) -> None:
    buy_dates  = trades.index[trades > 0]
    sell_dates = trades.index[trades < 0]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(close, color="black", linewidth=0.8, label=symbol)
    ax.scatter(buy_dates,  close[buy_dates],  marker="^", color="green", s=60, zorder=5, label="Buy")
    ax.scatter(sell_dates, close[sell_dates], marker="v", color="red",   s=60, zorder=5, label="Sell")
    ax.set_title(f"{symbol} — Signals: train={train.value}  test={test.value}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = GRAPHS_DIR / symbol
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"signals_{test.value}.png", dpi=150)
    plt.close(fig)


def print_stats(portvals: pd.Series, symbol: str, train: Period, test: Period) -> None:
    cr, adr, sddr, sr = compute_stats(portvals)
    print(f"\n── {symbol}  {train.value} → {test.value} ──")
    print(f"  Cumulative Return : {cr:.4f}")
    print(f"  Avg Daily Return  : {adr:.6f}")
    print(f"  Std Daily Return  : {sddr:.6f}")
    print(f"  Sharpe Ratio      : {sr:.4f}")
    print(f"  Final Value       : ${portvals.iloc[-1]:,.2f}")


def main():
    GRAPHS_DIR.mkdir(exist_ok=True)

    for symbol in SYMBOLS:
        for train, test in EXPERIMENTS:
            portvals, benchmark, trades, close = run_experiment(symbol, train, test)
            print_stats(portvals, symbol, train, test)
            save_portfolio_plot(portvals, benchmark, symbol, train, test)
            save_signals_plot(close, trades, symbol, train, test)

    print(f"\nGraphs saved to {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
