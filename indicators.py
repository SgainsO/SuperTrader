from __future__ import annotations

from enum import IntEnum
from typing import List

import numpy as np
import pandas as pd

# trend direction, mean-reversion, volatility regime or Participation

def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    """Trend direction — MACD histogram (fast EMA − slow EMA − signal line)."""
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return (macd_line - signal_line).round().fillna(0).astype("int64")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Mean-reversion — Relative Strength Index (0–100)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).round().fillna(0).astype("int64")


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Mean-reversion — Williams %R (-100–0); values near -100 oversold, near 0 overbought."""
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    return wr.round().fillna(0).astype("int64")


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Volatility regime — Average True Range (same units as price)."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean().round().fillna(0).astype("int64")


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """Volatility regime — %B position scaled 0–100 (0 = lower band, 100 = upper band)."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = 100 * (close - lower) / (upper - lower).replace(0, np.nan)
    return pct_b.round().fillna(0).astype("int64")


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Participation — On-Balance Volume cumulative."""
    direction = np.sign(close.diff()).fillna(0).astype(int)
    return (direction * volume).cumsum().astype("int64")


def volume_ratio(volume: pd.Series, period: int = 14) -> pd.Series:
    """Participation — current volume vs rolling average, scaled to integer percent (100 = average)."""
    rolling_mean = volume.rolling(period).mean().replace(0, np.nan)
    return (100 * volume / rolling_mean).round().astype("int64")


# ---------------------------------------------------------------------------
# Indicator enum
# ---------------------------------------------------------------------------


class Indicator(IntEnum):
    MACD = 1
    RSI = 2
    WILLIAMS_R = 3
    ATR = 4
    BOLLINGER_BANDS = 5
    OBV = 6
    VOLUME_RATIO = 7


# ---------------------------------------------------------------------------
# Indicator collective
# ---------------------------------------------------------------------------

_REVERSION_POOL: List[Indicator] = [Indicator.RSI, Indicator.WILLIAMS_R]
_VOLATILITY_POOL: List[Indicator] = [Indicator.ATR, Indicator.BOLLINGER_BANDS]
_PARTICIPATION_POOL: List[Indicator] = [Indicator.OBV, Indicator.VOLUME_RATIO]


class IndicatorCollective:
    """Stores raw OHLCV data and computes only the indicators that are configured."""

    def __init__(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
    ) -> None:
        self._close = close
        self._high = high
        self._low = low
        self._volume = volume
        self._computed: dict[Indicator, pd.Series] = {}
        self._active_reversion: List[Indicator] = []
        self._active_volatility: List[Indicator] = []
        self._active_participation: List[Indicator] = []

    def _compute(self, indicator: Indicator) -> pd.Series:
        if indicator not in self._computed:
            if indicator == Indicator.MACD:
                self._computed[indicator] = macd(self._close)
            elif indicator == Indicator.RSI:
                self._computed[indicator] = rsi(self._close)
            elif indicator == Indicator.WILLIAMS_R:
                self._computed[indicator] = williams_r(self._high, self._low, self._close)
            elif indicator == Indicator.ATR:
                self._computed[indicator] = atr(self._high, self._low, self._close)
            elif indicator == Indicator.BOLLINGER_BANDS:
                self._computed[indicator] = bollinger_bands(self._close)
            elif indicator == Indicator.OBV:
                self._computed[indicator] = obv(self._close, self._volume)
            elif indicator == Indicator.VOLUME_RATIO:
                self._computed[indicator] = volume_ratio(self._volume)
        return self._computed[indicator]

    def configure(
        self,
        n_reversion: int,
        n_volatility: int,
        n_participation: int,
    ) -> None:
        """Activate and compute n indicators per category (selected in enum order)."""
        if not 0 <= n_reversion <= len(_REVERSION_POOL):
            raise ValueError(f"n_reversion must be 0–{len(_REVERSION_POOL)}")
        if not 0 <= n_volatility <= len(_VOLATILITY_POOL):
            raise ValueError(f"n_volatility must be 0–{len(_VOLATILITY_POOL)}")
        if not 0 <= n_participation <= len(_PARTICIPATION_POOL):
            raise ValueError(f"n_participation must be 0–{len(_PARTICIPATION_POOL)}")

        self._active_reversion = _REVERSION_POOL[:n_reversion]
        self._active_volatility = _VOLATILITY_POOL[:n_volatility]
        self._active_participation = _PARTICIPATION_POOL[:n_participation]

        for ind in self.retrieve():
            self._compute(ind)

    def retrieve(self) -> List[Indicator]:
        """Return the currently active indicators in category order."""
        active: List[Indicator] = []
        active.extend(self._active_reversion)
        active.extend(self._active_volatility)
        active.extend(self._active_participation)
        return active

    def values(self) -> np.ndarray:
        """Return a 2-D int64 array (rows=time, cols=indicators) ordered Reversion → Volatility → Participation."""
        active = self.retrieve()
        if not active:
            return np.empty((0, 0), dtype=np.int64)
        return np.column_stack(
            [self._computed[ind].to_numpy(dtype=np.int64) for ind in active]
        )
