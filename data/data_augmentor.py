"""
data/data_augmentor.py
=======================
Per-episode OHLC augmentation for training diversity.

Two complementary techniques are applied in order:

  1. Session-level return scaling
     A scale factor f ~ U(1 - trend_scale, 1 + trend_scale) is sampled once
     per episode.  All OHLC values are re-expressed as ratios relative to the
     session open, those ratios are raised to the power f, then converted back
     to prices.  This compresses or expands intraday moves (e.g. ±15%) so the
     agent encounters different volatility regimes on every replay.

  2. Bar-level OHLC jitter (continuous uniform ±max_jitter_pts)
     Each bar's O/H/L/C receives an independent random offset drawn from
     U(-max_jitter_pts, +max_jitter_pts).  Default: ±2.0 pts (8 ticks for ES).

Volume is intentionally excluded — the strategy is price-structure based and
volume is not part of the observation vector.

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
    Applies two-stage OHLC augmentation to a session's bars.

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
    """

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        max_jitter_pts: float = 2.0,
        trend_scale:    float = 0.15,
    ) -> None:
        self._rng           = rng if rng is not None else np.random.default_rng()
        self.max_jitter_pts = max_jitter_pts
        self.trend_scale    = trend_scale

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

        # Extract all OHLC as one (n, 4) float64 array — O=0, H=1, L=2, C=3.
        ohlc = df[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)

        # ── 1. Session-level return scaling ───────────────────
        if self.trend_scale > 0.0:
            scale_f = float(self._rng.uniform(
                1.0 - self.trend_scale, 1.0 + self.trend_scale
            ))
            if scale_f != 1.0:
                open0 = ohlc[0, 0]
                if open0 > 0.0:
                    ratios = np.maximum(ohlc / open0, 1e-6)
                    np.multiply(open0, ratios ** scale_f, out=ohlc)

        # ── 2. Bar-level OHLC jitter ───────────────────────────
        if self.max_jitter_pts > 0.0:
            ohlc += self._rng.uniform(
                -self.max_jitter_pts, self.max_jitter_pts, size=(n, 4)
            )

        # ── 3. Enforce OHLC consistency ───────────────────────
        np.maximum(ohlc[:, 1], np.maximum(ohlc[:, 0], ohlc[:, 3]), out=ohlc[:, 1])
        np.minimum(ohlc[:, 2], np.minimum(ohlc[:, 0], ohlc[:, 3]), out=ohlc[:, 2])

        df[["open", "high", "low", "close"]] = ohlc
        return df
