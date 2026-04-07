"""
features/trend_classifier.py
==============================
Trend classification using swing high/low structure.

Classification logic (from the video):
  UPTREND    : Higher Highs (HH) + Higher Lows (HL) confirmed
  DOWNTREND  : Lower Lows (LL) + Lower Highs (LH) confirmed
  M_UPTREND  : M-pattern — partial bullish structure (HH present, not full)
  W_DOWNTREND: W-pattern — partial bearish structure (LL present, not full)
  RANGING    : Mixed signals, neither trend confirmed
  UNDEFINED  : Not enough swing data to classify
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class TrendState(Enum):
    UPTREND     = "uptrend"
    DOWNTREND   = "downtrend"
    RANGING     = "ranging"
    M_UPTREND   = "m_uptrend"
    W_DOWNTREND = "w_downtrend"
    UNDEFINED   = "undefined"


@dataclass
class TrendSnapshot:
    """
    Trend classification snapshot at a given bar.

    Attributes
    ----------
    state : TrendState
    last_swing_high : Optional[float]
    last_swing_low : Optional[float]
    hh_count : int  — consecutive Higher Highs
    hl_count : int  — consecutive Higher Lows
    ll_count : int  — consecutive Lower Lows
    lh_count : int  — consecutive Lower Highs
    trend_strength : float  — 0.0 (ranging/weak) to 1.0 (strong trend)
    """
    state: TrendState
    last_swing_high: Optional[float]
    last_swing_low: Optional[float]
    hh_count: int
    hl_count: int
    ll_count: int
    lh_count: int
    trend_strength: float

    @property
    def is_bullish(self) -> bool:
        return self.state in (TrendState.UPTREND, TrendState.M_UPTREND)

    @property
    def is_bearish(self) -> bool:
        return self.state in (TrendState.DOWNTREND, TrendState.W_DOWNTREND)

    def as_feature_dict(self) -> dict:
        """Return normalised features for the observation vector."""
        max_count = 5.0
        return {
            "trend_uptrend":   float(self.state in (TrendState.UPTREND,   TrendState.M_UPTREND)),
            "trend_downtrend": float(self.state in (TrendState.DOWNTREND, TrendState.W_DOWNTREND)),
            "trend_ranging":   float(self.state == TrendState.RANGING),
            "trend_undefined": float(self.state == TrendState.UNDEFINED),
            "trend_strength":  float(np.clip(self.trend_strength, 0.0, 1.0)),
            "hh_count_norm":   float(min(self.hh_count, max_count) / max_count),
            "hl_count_norm":   float(min(self.hl_count, max_count) / max_count),
            "ll_count_norm":   float(min(self.ll_count, max_count) / max_count),
            "lh_count_norm":   float(min(self.lh_count, max_count) / max_count),
        }


class TrendClassifier:
    """
    Classifies market trend using swing high/low structure.

    Parameters
    ----------
    swing_lookback : int
        Bars required on each side to confirm a swing point.
    min_hh_hl_for_uptrend : int
        Minimum HH + HL count each to classify as UPTREND.
    min_ll_lh_for_downtrend : int
        Minimum LL + LH count each to classify as DOWNTREND.
    """

    def __init__(
        self,
        swing_lookback: int = 3,
        min_hh_hl_for_uptrend: int = 2,
        min_ll_lh_for_downtrend: int = 2,
        reversal_requires_breaks: int = 2,
        strength_lookback_bars: int = 40,
    ) -> None:
        self.swing_lookback = swing_lookback
        self.min_hh_hl_for_uptrend = min_hh_hl_for_uptrend
        self.min_ll_lh_for_downtrend = min_ll_lh_for_downtrend
        self.reversal_requires_breaks = reversal_requires_breaks
        self.strength_lookback_bars = strength_lookback_bars

    def classify(self, bars: pd.DataFrame, current_bar_idx: int) -> TrendSnapshot:
        """
        Classify the trend at current_bar_idx using only prior bar data.

        Parameters
        ----------
        bars : pd.DataFrame
        current_bar_idx : int

        Returns
        -------
        TrendSnapshot
        """
        bars_so_far = bars.iloc[:current_bar_idx + 1]
        lb = self.swing_lookback

        # Need at least 2 * lb + 1 bars to confirm any swing
        if len(bars_so_far) < 2 * lb + 2:
            return TrendSnapshot(
                state=TrendState.UNDEFINED,
                last_swing_high=None, last_swing_low=None,
                hh_count=0, hl_count=0, ll_count=0, lh_count=0,
                trend_strength=0.0,
            )

        swings = self._find_swings(bars_so_far, lb)

        if len(swings) < 2:
            return TrendSnapshot(
                state=TrendState.UNDEFINED,
                last_swing_high=None, last_swing_low=None,
                hh_count=0, hl_count=0, ll_count=0, lh_count=0,
                trend_strength=0.0,
            )

        swing_highs: List[Tuple[int, float]] = [(i, p) for i, p, t in swings if t == "high"]
        swing_lows:  List[Tuple[int, float]] = [(i, p) for i, p, t in swings if t == "low"]

        last_swing_high = swing_highs[-1][1] if swing_highs else None
        last_swing_low  = swing_lows[-1][1]  if swing_lows  else None

        # Count sequential HH / LH among swing highs
        hh_count = lh_count = 0
        if len(swing_highs) >= 2:
            for k in range(1, len(swing_highs)):
                if swing_highs[k][1] > swing_highs[k - 1][1]:
                    hh_count += 1
                else:
                    lh_count += 1

        # Count sequential HL / LL among swing lows
        hl_count = ll_count = 0
        if len(swing_lows) >= 2:
            for k in range(1, len(swing_lows)):
                if swing_lows[k][1] > swing_lows[k - 1][1]:
                    hl_count += 1
                else:
                    ll_count += 1

        # Classify
        min_up   = self.min_hh_hl_for_uptrend
        min_down = self.min_ll_lh_for_downtrend
        bullish_ev = hh_count + hl_count
        bearish_ev = ll_count + lh_count
        total_ev   = bullish_ev + bearish_ev

        if total_ev == 0:
            state    = TrendState.UNDEFINED
            strength = 0.0
        elif hh_count >= min_up and hl_count >= min_up:
            state    = TrendState.UPTREND
            strength = float(min(1.0, bullish_ev / (2.0 * min_up * 2.0)))
        elif ll_count >= min_down and lh_count >= min_down:
            state    = TrendState.DOWNTREND
            strength = float(min(1.0, bearish_ev / (2.0 * min_down * 2.0)))
        elif hh_count >= min_up or hl_count >= min_up:
            # Partial bullish — M-pattern (uptrend in progress)
            state    = TrendState.M_UPTREND if hh_count > 0 else TrendState.RANGING
            strength = float(bullish_ev / max(total_ev, 1))
        elif ll_count >= min_down or lh_count >= min_down:
            # Partial bearish — W-pattern (downtrend in progress)
            state    = TrendState.W_DOWNTREND if ll_count > 0 else TrendState.RANGING
            strength = float(bearish_ev / max(total_ev, 1))
        else:
            state    = TrendState.RANGING
            strength = float(max(bullish_ev, bearish_ev) / max(total_ev, 1))

        return TrendSnapshot(
            state=state,
            last_swing_high=last_swing_high,
            last_swing_low=last_swing_low,
            hh_count=hh_count,
            hl_count=hl_count,
            ll_count=ll_count,
            lh_count=lh_count,
            trend_strength=float(np.clip(strength, 0.0, 1.0)),
        )

    # ── Private helpers ───────────────────────────────────────

    @staticmethod
    def _find_swings(
        bars: pd.DataFrame, lookback: int
    ) -> List[Tuple[int, float, str]]:
        """
        Find confirmed swing highs and lows.

        A swing high at index i is confirmed when the next `lookback`
        bars all have a lower high.  Same logic for swing lows.

        Returns list of (bar_idx, price, "high"|"low") sorted by bar_idx.
        No lookahead: only bars already present in `bars` are used.
        """
        highs = bars["high"].values
        lows  = bars["low"].values
        n = len(highs)
        result = []

        for i in range(lookback, n - lookback):
            # Swing high
            center_high = highs[i]
            if (all(center_high > highs[i - j] for j in range(1, lookback + 1)) and
                    all(center_high > highs[i + j] for j in range(1, lookback + 1))):
                result.append((i, float(center_high), "high"))

            # Swing low
            center_low = lows[i]
            if (all(center_low < lows[i - j] for j in range(1, lookback + 1)) and
                    all(center_low < lows[i + j] for j in range(1, lookback + 1))):
                result.append((i, float(center_low), "low"))

        result.sort(key=lambda x: x[0])
        return result
