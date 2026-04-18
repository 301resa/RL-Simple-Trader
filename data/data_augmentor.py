"""
data/data_augmentor.py
=======================
Per-episode OHLCV augmentation for training diversity.

Three complementary techniques are applied in order:

  1. Session-level return scaling
     A scale factor f ~ U(1 - trend_scale, 1 + trend_scale) is sampled once
     per episode.  All OHLC values are re-expressed as ratios relative to the
     session open, those ratios are raised to the power f, then converted back
     to prices.  This compresses or expands intraday moves (e.g. ±15%) so the
     agent encounters different volatility regimes on every replay.

  2. Bar-level OHLC jitter (continuous uniform ±max_jitter_pts)
     Each bar's O/H/L/C receives an independent random offset drawn from
     U(-max_jitter_pts, +max_jitter_pts).  Default: ±2.0 pts (8 ticks for ES).

  3. Volume multiplicative scaling
     Volume is multiplied by a factor drawn from U(1 - volume_scale, 1 + volume_scale).
     Scales the entire session uniformly so relative bar volumes are preserved.

OHLC integrity is enforced after every step:
  high  ≥ max(open, close)
  low   ≤ min(open, close)

Because every parallel environment worker is seeded differently, each worker
sees its own augmentation sequence — the agent cannot memorise price-to-outcome
mappings.

Usage:
    augmentor = OHLCVAugmentor(rng=np.random.default_rng(42))
    augmented_bars = augmentor.apply(session_bars)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class OHLCVAugmentor:
    """
    Applies three-stage augmentation to a session's OHLCV bars.

    Parameters
    ----------
    rng : numpy Generator, optional
        Random number generator.  Created with default_rng() if None.
    max_jitter_pts : float
        Half-range of bar-level OHLC jitter in price points.
        Default 2.0 = ±8 ticks for ES/MES (0.25 pt/tick).
    trend_scale : float
        Half-range of session-level return scaling.
        0.15 → scale factor drawn from U(0.85, 1.15).
        Set to 0.0 to disable.
    volume_scale : float
        Half-range of session-level volume multiplicative scaling.
        0.30 → factor drawn from U(0.70, 1.30).
        Set to 0.0 to disable.
    """

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        max_jitter_pts: float = 2.0,
        trend_scale:    float = 0.15,
        volume_scale:   float = 0.30,
    ) -> None:
        self._rng           = rng if rng is not None else np.random.default_rng()
        self.max_jitter_pts = max_jitter_pts
        self.trend_scale    = trend_scale
        self.volume_scale   = volume_scale

    # ── Public API ────────────────────────────────────────────

    def seed(self, seed: int) -> None:
        """Re-seed (call from TradingEnv.reset when the env is seeded)."""
        self._rng = np.random.default_rng(seed)

    def apply(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Return an augmented copy of *bars*.  Original is never mutated.

        Parameters
        ----------
        bars : pd.DataFrame
            Must contain columns: open, high, low, close.
            Column 'volume' is optional — left untouched if absent.

        Returns
        -------
        pd.DataFrame  (same index and columns as input)
        """
        n = len(bars)
        if n == 0:
            return bars.copy()

        df = bars.copy()

        # ── 1. Session-level return scaling ───────────────────
        if self.trend_scale > 0.0:
            scale_f = float(self._rng.uniform(
                1.0 - self.trend_scale, 1.0 + self.trend_scale
            ))
            if scale_f != 1.0:
                open0 = float(df["open"].iloc[0])
                if open0 > 0.0:
                    for col in ("open", "high", "low", "close"):
                        ratios = df[col].values / open0
                        # Clamp to avoid log(≤0); ES prices are always >> 0
                        ratios = np.maximum(ratios, 1e-6)
                        df[col] = open0 * (ratios ** scale_f)

        # ── 2. Bar-level OHLC jitter ───────────────────────────
        if self.max_jitter_pts > 0.0:
            offsets = self._rng.uniform(
                -self.max_jitter_pts, self.max_jitter_pts, size=(n, 4)
            ).astype(np.float32)
            df["open"]  = df["open"].values  + offsets[:, 0]
            df["high"]  = df["high"].values  + offsets[:, 1]
            df["low"]   = df["low"].values   + offsets[:, 2]
            df["close"] = df["close"].values + offsets[:, 3]

        # ── 3. Enforce OHLC consistency ───────────────────────
        open_arr  = df["open"].values
        close_arr = df["close"].values
        df["high"] = np.maximum(df["high"].values,  np.maximum(open_arr, close_arr))
        df["low"]  = np.minimum(df["low"].values,   np.minimum(open_arr, close_arr))

        # ── 4. Volume multiplicative scaling ──────────────────
        if "volume" in df.columns and self.volume_scale > 0.0:
            vol_factor = float(self._rng.uniform(
                1.0 - self.volume_scale, 1.0 + self.volume_scale
            ))
            df["volume"] = np.maximum(0.0, df["volume"].values * vol_factor)

        return df
