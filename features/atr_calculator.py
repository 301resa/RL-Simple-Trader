"""
features/atr_calculator.py
===========================
ATR (Average True Range) computation and session exhaustion tracking.

Used for:
  - Daily ATR filter: only trade if session hasn't moved more than X% of ATR
  - Profit target computation: target = ATR remaining × pct
  - Action masking: block entries when ATR is exhausted
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ATRState:
    """
    ATR feature state for the current session bar.

    Attributes
    ----------
    atr_daily : float
        Daily ATR value for this session date.
    prior_day_high : float
    prior_day_low : float
    prior_day_range : float
    session_open : float
        Opening price of the current session (first bar open).
    session_high : float
        Highest high seen so far in the current session.
    session_low : float
        Lowest low seen so far in the current session.
    current_daily_range : float
        Range covered so far (session_high - session_low).
    atr_pct_used : float
        current_daily_range / atr_daily.
    atr_remaining_pts : float
        ATR not yet consumed in price points.
    atr_short_exhausted : bool
        True when downward move (session_open - session_low) >= threshold × atr_daily.
        Blocks new SHORT entries — no point selling into an already-exhausted down move.
    atr_long_exhausted : bool
        True when upward move (session_high - session_open) >= threshold × atr_daily.
        Blocks new LONG entries — no point buying into an already-exhausted up move.
        The OPPOSITE direction is always still allowed.
    """
    atr_daily: float
    prior_day_high: float
    prior_day_low: float
    prior_day_range: float
    session_open: float
    session_high: float
    session_low: float
    current_daily_range: float
    atr_pct_used: float
    atr_remaining_pts: float
    atr_short_exhausted: bool
    atr_long_exhausted: bool

    def as_feature_dict(self) -> dict:
        """Return normalised feature values for the observation vector.

        Four ATR features (obs dimension unchanged):
          atr_pct_used        — total session range as fraction of daily ATR
          atr_remaining_norm  — remaining ATR room (0=exhausted, 1=full)
          atr_short_exhausted — 1.0 if downward move >= 85% of ATR (block SHORTs)
          atr_long_exhausted  — 1.0 if upward move >= 85% of ATR (block LONGs)
        """
        return {
            "atr_pct_used": float(np.clip(self.atr_pct_used, 0.0, 2.0)),
            "atr_remaining_norm": float(
                np.clip(self.atr_remaining_pts / max(self.atr_daily, 1.0), 0.0, 1.0)
            ),
            "atr_short_exhausted": float(self.atr_short_exhausted),
            "atr_long_exhausted":  float(self.atr_long_exhausted),
        }


class ATRCalculator:
    """
    Computes daily ATR values from historical daily bars, then provides
    session-level exhaustion state for each intraday bar.

    Parameters
    ----------
    atr_period : int
        Number of days for ATR computation (Wilder's smoothing).
    exhaustion_threshold : float
        Fraction of daily ATR at which session is considered exhausted
        in either direction (blocks new entries).
    profit_target_atr_pct : float
        Default fraction of ATR remaining used as profit target.
    """

    def __init__(
        self,
        atr_period: int = 14,
        exhaustion_threshold: float = 0.85,
        profit_target_atr_pct: float = 0.75,
    ) -> None:
        self.atr_period = atr_period
        self.exhaustion_threshold = exhaustion_threshold
        self.profit_target_atr_pct = profit_target_atr_pct

        self._atr_series: Optional[pd.Series] = None   # date_str → atr_value
        self._daily_bars: Optional[pd.DataFrame] = None  # date_str indexed
        # Fast O(log n) lookup structures built in fit()
        self._sorted_dates: List[str] = []
        self._sorted_atrs: List[float] = []

    # ── Public API ────────────────────────────────────────────

    def fit(self, daily_bars: pd.DataFrame) -> None:
        """
        Compute ATR for every date in the daily bars DataFrame.

        Uses Wilder's ATR (exponential smoothing, com = period - 1).

        Parameters
        ----------
        daily_bars : pd.DataFrame
            Must have columns: open, high, low, close.
            Index: DatetimeIndex (or string-indexed, YYYY-MM-DD).
        """
        df = daily_bars.copy()

        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's smoothing: EWM with com = period - 1
        atr = tr.ewm(com=self.atr_period - 1, min_periods=self.atr_period).mean()

        # Build YYYY-MM-DD indexed series for lookups
        if hasattr(df.index, "strftime"):
            date_strs = df.index.strftime("%Y-%m-%d")
        else:
            date_strs = pd.Index([str(d)[:10] for d in df.index])

        self._atr_series = pd.Series(atr.values, index=date_strs)
        self._daily_bars = df.copy()
        self._daily_bars.index = date_strs

        # Build sorted arrays for O(log n) date lookups
        self._sorted_dates = list(date_strs)
        self._sorted_atrs  = list(atr.values.astype(float))

    def get_atr_for_date(self, date_str: str) -> Optional[float]:
        """
        Return the ATR value to use for a given trading date.

        Returns the last ATR computed strictly before `date_str` to
        avoid lookahead bias (we can't know today's ATR at day open).

        Parameters
        ----------
        date_str : str
            YYYY-MM-DD string.

        Returns
        -------
        float or None if not enough history.
        """
        if not self._sorted_dates:
            return None

        # O(log n) bisect lookup — find last date strictly before date_str
        idx = bisect.bisect_left(self._sorted_dates, date_str)
        if idx > 0:
            return self._sorted_atrs[idx - 1]

        # Fallback: allow the date itself (e.g. first day in dataset)
        if idx < len(self._sorted_dates) and self._sorted_dates[idx] == date_str:
            return self._sorted_atrs[idx]

        return None

    def compute_session_state(
        self,
        date: str,
        session_bars: pd.DataFrame,
        current_bar_idx: int,
    ) -> Optional[ATRState]:
        """
        Compute the ATRState for the current intraday bar.

        Uses only data up to and including `current_bar_idx`.

        Parameters
        ----------
        date : str
            Trading date (YYYY-MM-DD).
        session_bars : pd.DataFrame
            All intraday bars for this session.
        current_bar_idx : int
            Index of the current bar within session_bars.

        Returns
        -------
        ATRState or None if ATR is unavailable.
        """
        atr_val = self.get_atr_for_date(date)
        if atr_val is None or atr_val <= 0:
            return None

        # O(log n) bisect lookup for prior day — avoids full-DataFrame boolean filter
        prior_high = prior_low = prior_range = 0.0
        if self._daily_bars is not None:
            idx = bisect.bisect_left(self._sorted_dates, date)
            if idx > 0:
                last_day    = self._daily_bars.iloc[idx - 1]
                prior_high  = float(last_day["high"])
                prior_low   = float(last_day["low"])
                prior_range = prior_high - prior_low

        # Session range so far (no lookahead)
        bars_so_far = session_bars.iloc[:current_bar_idx + 1]
        if bars_so_far.empty:
            ref = float(session_bars.iloc[0]["close"])
            session_open = session_high = session_low = ref
        else:
            session_open = float(bars_so_far.iloc[0]["open"])   # first bar's open
            session_high = float(bars_so_far["high"].max())
            session_low  = float(bars_so_far["low"].min())

        current_range = session_high - session_low
        atr_pct_used  = current_range / atr_val
        atr_remaining = max(0.0, atr_val - current_range)

        # Directional exhaustion: how much of ATR has been consumed in each direction
        # from the session open price.
        move_down = max(0.0, session_open - session_low)   # total downward move
        move_up   = max(0.0, session_high - session_open)  # total upward move
        thresh    = self.exhaustion_threshold * atr_val

        return ATRState(
            atr_daily=atr_val,
            prior_day_high=prior_high,
            prior_day_low=prior_low,
            prior_day_range=prior_range,
            session_open=session_open,
            session_high=session_high,
            session_low=session_low,
            current_daily_range=current_range,
            atr_pct_used=atr_pct_used,
            atr_remaining_pts=atr_remaining,
            atr_short_exhausted=move_down >= thresh,  # already moved down 85% of ATR
            atr_long_exhausted=move_up >= thresh,     # already moved up 85% of ATR
        )

    def compute_all_session_states(
        self,
        date: str,
        session_bars: pd.DataFrame,
    ) -> List[ATRState]:
        """
        Vectorised version of compute_session_state — computes ATRState for
        every bar in the session in a single O(n) pass using running max/min.

        Replaces calling compute_session_state() n times (which is O(n²) due
        to .max()/.min() on growing slices).

        Returns
        -------
        List[ATRState] with one entry per row in session_bars.
        """
        atr_val = self.get_atr_for_date(date)
        if atr_val is None or atr_val <= 0:
            return []

        # O(log n) bisect lookup for prior day — avoids full-DataFrame boolean filter
        prior_high = prior_low = prior_range = 0.0
        if self._daily_bars is not None:
            idx = bisect.bisect_left(self._sorted_dates, date)
            if idx > 0:
                last_day    = self._daily_bars.iloc[idx - 1]
                prior_high  = float(last_day["high"])
                prior_low   = float(last_day["low"])
                prior_range = prior_high - prior_low

        highs_np = session_bars["high"].to_numpy(dtype=np.float64)
        lows_np  = session_bars["low"].to_numpy(dtype=np.float64)
        opens_np = session_bars["open"].to_numpy(dtype=np.float64)
        n = len(highs_np)
        if n == 0:
            return []

        session_open = float(opens_np[0])
        thresh = self.exhaustion_threshold * atr_val

        # Vectorised O(n) running max/min — replaces the Python loop entirely
        running_highs  = np.maximum.accumulate(highs_np)
        running_lows   = np.minimum.accumulate(lows_np)
        current_ranges = running_highs - running_lows
        atr_pct_used_v = current_ranges / atr_val
        atr_remaining_v = np.maximum(0.0, atr_val - current_ranges)
        move_down_v    = np.maximum(0.0, session_open - running_lows)
        move_up_v      = np.maximum(0.0, running_highs - session_open)
        short_exh_v    = move_down_v >= thresh
        long_exh_v     = move_up_v   >= thresh

        return [
            ATRState(
                atr_daily=atr_val,
                prior_day_high=prior_high,
                prior_day_low=prior_low,
                prior_day_range=prior_range,
                session_open=session_open,
                session_high=float(running_highs[i]),
                session_low=float(running_lows[i]),
                current_daily_range=float(current_ranges[i]),
                atr_pct_used=float(atr_pct_used_v[i]),
                atr_remaining_pts=float(atr_remaining_v[i]),
                atr_short_exhausted=bool(short_exh_v[i]),
                atr_long_exhausted=bool(long_exh_v[i]),
            )
            for i in range(n)
        ]

    @staticmethod
    def compute_atr_target_price(
        current_price: float,
        direction: int,
        atr_state: ATRState,
        pct: float = 0.75,
    ) -> float:
        """
        Compute a profit target based on ATR remaining.

        Parameters
        ----------
        current_price : float
        direction : int
            1 = LONG, -1 = SHORT.
        atr_state : ATRState
        pct : float
            Fraction of ATR remaining to use as target distance.
        """
        target_pts = atr_state.atr_remaining_pts * pct
        # Ensure a minimum target of 10% of daily ATR
        target_pts = max(target_pts, atr_state.atr_daily * 0.10)
        return current_price + direction * target_pts
