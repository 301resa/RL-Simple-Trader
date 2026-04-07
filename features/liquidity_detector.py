"""
features/liquidity_detector.py
================================
Liquidity sweep detection from intraday swing levels.

Liquidity sweeps are a key confluence factor in the Order Zone strategy:
  1. Identify swing highs and lows (local price extremes)
  2. Detect when price briefly violates a swing level (wick beyond it)
     and then closes back on the other side
  3. UP_SWEEP: wick above a swing high, closes back below → bearish signal
  4. DOWN_SWEEP: wick below a swing low, closes back above → bullish signal
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


class SweepDirection(Enum):
    UP_SWEEP   = "up_sweep"   # Swept above swing high (bearish)
    DOWN_SWEEP = "down_sweep" # Swept below swing low (bullish)
    NONE       = "none"


@dataclass
class LiquidityState:
    """
    Liquidity feature state at a given bar.

    Attributes
    ----------
    swing_highs : List[float]
        Confirmed swing high price levels (most recent first).
    swing_lows : List[float]
        Confirmed swing low price levels (most recent first).
    recent_swept_high : Optional[float]
        Most recently swept high level (if any).
    recent_swept_low : Optional[float]
        Most recently swept low level (if any).
    sweep_bar_idx : Optional[int]
        Bar index where the most recent sweep occurred.
    sweep_direction : SweepDirection
        Direction of the most recent sweep.
    """
    swing_highs: List[float]
    swing_lows: List[float]
    recent_swept_high: Optional[float]
    recent_swept_low: Optional[float]
    sweep_bar_idx: Optional[int]
    sweep_direction: SweepDirection

    def as_feature_dict(self) -> dict:
        """Return normalised features for the observation vector."""
        return {
            "has_recent_sweep": float(self.sweep_direction != SweepDirection.NONE),
            "sweep_up":   float(self.sweep_direction == SweepDirection.UP_SWEEP),
            "sweep_down": float(self.sweep_direction == SweepDirection.DOWN_SWEEP),
            "n_swing_highs": float(min(len(self.swing_highs), 5) / 5.0),
            "n_swing_lows":  float(min(len(self.swing_lows),  5) / 5.0),
        }


class LiquidityDetector:
    """
    Detects swing high/low levels and liquidity sweeps.

    Parameters
    ----------
    swing_lookback : int
        Bars required on each side to confirm a swing high/low.
    proximity_atr_pct : float
        Fraction of ATR defining "close to a level".
    sweep_wick_min_atr_pct : float
        Minimum wick penetration beyond a level (as % of ATR) to call a sweep.
    sweep_lookback_bars : int
        How many bars back to search for sweeps.
    """

    def __init__(
        self,
        swing_lookback: int = 3,
        proximity_atr_pct: float = 0.10,
        sweep_wick_min_atr_pct: float = 0.05,
        sweep_lookback_bars: int = 10,
    ) -> None:
        self.swing_lookback = swing_lookback
        self.proximity_atr_pct = proximity_atr_pct
        self.sweep_wick_min_atr_pct = sweep_wick_min_atr_pct
        self.sweep_lookback_bars = sweep_lookback_bars

    def compute_state(
        self,
        bars: pd.DataFrame,
        atr_series: pd.Series,
        current_bar_idx: int,
    ) -> LiquidityState:
        """
        Compute the liquidity state as of current_bar_idx.

        Only uses data up to and including current_bar_idx (no lookahead).

        Parameters
        ----------
        bars : pd.DataFrame
        atr_series : pd.Series
        current_bar_idx : int

        Returns
        -------
        LiquidityState
        """
        atr = float(atr_series.iloc[current_bar_idx]) if atr_series is not None else 100.0

        # Confirmed swings from bars strictly before current bar
        # (the current bar hasn't had right-side confirmation yet)
        bars_confirmed = bars.iloc[:current_bar_idx + 1]
        swing_highs = self._find_swing_highs(bars_confirmed, self.swing_lookback)
        swing_lows  = self._find_swing_lows(bars_confirmed, self.swing_lookback)

        # Detect the most recent sweep in the lookback window
        sweep_dir = SweepDirection.NONE
        recent_swept_high: Optional[float] = None
        recent_swept_low: Optional[float]  = None
        sweep_bar_idx: Optional[int] = None

        search_start = max(0, current_bar_idx - self.sweep_lookback_bars)

        for bar_i in range(search_start, current_bar_idx + 1):
            bar = bars.iloc[bar_i]
            high  = float(bar["high"])
            low   = float(bar["low"])
            close = float(bar["close"])

            # UP_SWEEP: wick above a swing high, closed back below it
            for sh in swing_highs:
                if (high - sh) > atr * self.sweep_wick_min_atr_pct and close < sh:
                    if sweep_bar_idx is None or bar_i >= sweep_bar_idx:
                        sweep_dir = SweepDirection.UP_SWEEP
                        recent_swept_high = sh
                        sweep_bar_idx = bar_i
                    break

            # DOWN_SWEEP: wick below a swing low, closed back above it
            for sl in swing_lows:
                if (sl - low) > atr * self.sweep_wick_min_atr_pct and close > sl:
                    if sweep_bar_idx is None or bar_i >= sweep_bar_idx:
                        sweep_dir = SweepDirection.DOWN_SWEEP
                        recent_swept_low = sl
                        sweep_bar_idx = bar_i
                    break

        return LiquidityState(
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            recent_swept_high=recent_swept_high,
            recent_swept_low=recent_swept_low,
            sweep_bar_idx=sweep_bar_idx,
            sweep_direction=sweep_dir,
        )

    # ── Static helpers ────────────────────────────────────────

    @staticmethod
    def _find_swing_highs(bars: pd.DataFrame, lookback: int) -> List[float]:
        """
        Find confirmed swing highs: bars with the highest high among
        `lookback` bars on both sides.

        Returns price levels (not indices), most recent first.
        """
        highs = bars["high"].values
        n = len(highs)
        result = []
        for i in range(lookback, n - lookback):
            center = highs[i]
            if (all(center > highs[i - j] for j in range(1, lookback + 1)) and
                    all(center > highs[i + j] for j in range(1, lookback + 1))):
                result.append(float(center))
        result.reverse()  # most recent first
        return result

    @staticmethod
    def _find_swing_lows(bars: pd.DataFrame, lookback: int) -> List[float]:
        """
        Find confirmed swing lows: bars with the lowest low among
        `lookback` bars on both sides.

        Returns price levels (not indices), most recent first.
        """
        lows = bars["low"].values
        n = len(lows)
        result = []
        for i in range(lookback, n - lookback):
            center = lows[i]
            if (all(center < lows[i - j] for j in range(1, lookback + 1)) and
                    all(center < lows[i + j] for j in range(1, lookback + 1))):
                result.append(float(center))
        result.reverse()  # most recent first
        return result
