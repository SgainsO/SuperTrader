"""StrategyLearner — IndicatorCollective features + tabular Q-learning (numpy)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import IndicatorCollective
from utils import Period, get_data


# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------

def _disc(v: float, lo: float, hi: float, n: int = 5) -> int:
    """Map v into n equal-width bins over [lo, hi], clipping at boundaries."""
    return min(int((max(lo, min(hi, v)) - lo) / (hi - lo) * n), n - 1)

def _state(row: np.ndarray) -> int:
    """
    Combine indicator bins into one integer (base-10 encoding).
    IndicatorCollective columns with configure(n_reversion=2, n_volatility=2):
      0 = RSI  (0–100)
      1 = Williams %R  (-100–0)
      2 = ATR  (price units, skipped)
      3 = Bollinger %B  (0–100)
    """
    return (
        _disc(row[0], 0, 100)        +   # RSI       → 0-4
        _disc(row[1], -100, 0) * 10  +   # Williams  → 0-4
        _disc(row[3], 0, 100)  * 100     # Bollinger → 0-4
    )


# ---------------------------------------------------------------------------
# Tabular Q-learner
# ---------------------------------------------------------------------------

class _QTable:
    """Minimal tabular Q-learner backed by a numpy array."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.2,
        gamma: float = 0.9,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.993,
    ) -> None:
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self._s: int = 0
        self._a: int = 0

    def querysetstate(self, s: int) -> int:
        self._s = s
        if np.random.random() < self.epsilon:
            self._a = np.random.randint(self.Q.shape[1])
        else:
            self._a = int(np.argmax(self.Q[s]))
        return self._a

    def query(self, s: int, r: float) -> int:
        self.Q[self._s, self._a] += self.alpha * (
            r + self.gamma * float(np.max(self.Q[s])) - self.Q[self._s, self._a]
        )
        self.epsilon *= self.epsilon_decay
        return self.querysetstate(s)

    def greedy(self, s: int) -> int:
        return int(np.argmax(self.Q[s]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Max state: (n-1) + (n-1)*10 + (n-1)*100 = 4+40+400 = 444
_N_STATES  = 500
_N_ACTIONS = 3      # 0 = short, 1 = neutral, 2 = long
_EPOCHS    = 175

def _portfolio(cash: float, hold: int, price: float) -> float:
    return cash + hold * 1000 * price

def _reward(prev: float, curr: float) -> float:
    return 0.0 if prev == 0 else ((curr - prev) / prev) * 100 * 10


# ---------------------------------------------------------------------------
# StrategyLearner
# ---------------------------------------------------------------------------

class StrategyLearner:
    """
    Tabular Q-learner using RSI, Williams %R, and Bollinger %B via IndicatorCollective.

    :param verbose: Print debug info when True.
    :param impact: Per-share market impact as a fraction of price.
    :param commission: Flat commission per trade.
    :param epochs: Training sweeps over the price series.
    """

    def __init__(
        self,
        verbose: bool = False,
        impact: float = 0.0,
        commission: float = 0.0,
        epochs: int = _EPOCHS,
    ) -> None:
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.epochs = epochs
        self._qtable = _QTable(_N_STATES, _N_ACTIONS)

    def _run_episode(
        self,
        prices: np.ndarray,
        states: np.ndarray,
        sv: float,
        is_training: bool,
    ) -> np.ndarray:
        cash = sv
        hold = 0
        prev_val = sv
        trades = np.zeros(len(prices), dtype=int)
        first = True

        for t in range(len(prices)):
            price = prices[t]
            s = states[t]
            curr_val = _portfolio(cash, hold, price)

            if first:
                action = self._qtable.querysetstate(s)
                first = False
            elif is_training:
                action = self._qtable.query(s, _reward(prev_val, curr_val))
            else:
                action = self._qtable.greedy(s)

            prev_val = curr_val

            target = action - 1          # 0→-1, 1→0, 2→+1
            trade = (target - hold) * 1000
            if trade != 0:
                cash -= trade * price
                cash -= abs(trade) * price * self.impact + self.commission
                hold = target

            trades[t] = trade

        if self.verbose:
            print(f"  final portfolio: {_portfolio(cash, hold, prices[-1]):.2f}")

        return trades

    def _load(self, symbol: str, period: Period):
        data = get_data([symbol], period, columns=["Close", "High", "Low", "Volume"])
        ohlcv = data[symbol]

        ic = IndicatorCollective(
            close=ohlcv["Close"],
            high=ohlcv["High"],
            low=ohlcv["Low"],
            volume=ohlcv["Volume"],
        )
        ic.configure(n_reversion=2, n_volatility=2, n_participation=0)

        feat = np.nan_to_num(ic.values().astype(float), nan=0.0)
        states = np.array([_state(feat[t]) for t in range(len(feat))])

        return ohlcv, ohlcv["Close"].to_numpy(dtype=float), states

    def add_evidence(
        self,
        symbol: str = "SPY",
        period: Period = Period.CRISIS_2008,
        sv: float = 10_000.0,
    ) -> None:
        """Train the Q-table over the given period."""
        _, prices, states = self._load(symbol, period)
        self._qtable = _QTable(_N_STATES, _N_ACTIONS)

        for epoch in range(self.epochs):
            if self.verbose:
                print(f"epoch {epoch}")
            self._run_episode(prices, states, sv, is_training=True)

    def testPolicy(
        self,
        symbol: str = "SPY",
        period: Period = Period.COVID_ERA,
        sv: float = 10_000.0,
    ) -> pd.DataFrame:
        """
        Apply the trained policy to an out-of-sample period.

        Returns a DataFrame of daily trades (+1000, -1000, +2000, -2000, or 0).
        """
        ohlcv, prices, states = self._load(symbol, period)
        trades = self._run_episode(prices, states, sv, is_training=False)
        return pd.DataFrame({symbol: trades}, index=ohlcv.index)


if __name__ == "__main__":
    learner = StrategyLearner(verbose=True)
    learner.add_evidence(symbol="SPY", period=Period.CRISIS_2008)
    trades = learner.testPolicy(symbol="SPY", period=Period.COVID_ERA)
    print(trades["SPY"].value_counts().sort_index())
