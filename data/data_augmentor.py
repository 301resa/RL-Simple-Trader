"""
data/data_augmentor.py
=======================
Per-episode OHLCV discrete jitter for training diversity.

Each bar's O/H/L/C receives an independent random offset drawn from the
discrete set  {-0.5, -0.25, 0.0, +0.25, +0.5}  with equal probability.
Volume receives an independent offset from the same set scaled by 20
(i.e. {-10, -5, 0, +5, +10}), then clipped to [0, ∞).

Because every parallel environment worker is seeded differently, each
worker sees its own jitter sequence → the agent cannot memorise
price-to-outcome mappings.

OHLC integrity is enforced after jittering:
  high  ≥ max(open, close)   (widen upward if needed)
  low   ≤ min(open, close)   (widen downward if needed)

Usage:
    augmentor = OHLCVAugmentor(rng=np.random.default_rng(42))
    augmented_bars = augmentor.apply(session_bars)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Discrete jitter values (ticks)
_OHLC_STEPS   = np.array([-0.50, -0.25, 0.0, +0.25, +0.50], dtype=np.float32)
_VOLUME_STEPS = np.array([-10.0, -5.0,  0.0, +5.0,  +10.0], dtype=np.float32)


class OHLCVAugmentor:
    """
    Applies discrete random jitter to a session's OHLCV bars.

    Each bar gets an independently sampled offset from
    {-0.5, -0.25, 0, +0.25, +0.5} for every OHLC column and
    {-10, -5, 0, +5, +10} for volume.

    Parameters
    ----------
    rng : numpy Generator, optional
        Random number generator.  Created with default_rng() if None.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    # ── Public API ────────────────────────────────────────────

    def seed(self, seed: int) -> None:
        """Re-seed (call from TradingEnv.reset when the env is seeded)."""
        self._rng = np.random.default_rng(seed)

    def apply(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Return a jittered copy of *bars*.  Original is never mutated.

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
        n_steps = len(_OHLC_STEPS)

        # ── Sample discrete offsets for each column independently ──
        # Shape: (n,) index into _OHLC_STEPS
        o_idx = self._rng.integers(0, n_steps, size=n)
        h_idx = self._rng.integers(0, n_steps, size=n)
        l_idx = self._rng.integers(0, n_steps, size=n)
        c_idx = self._rng.integers(0, n_steps, size=n)

        df["open"]  = df["open"].values  + _OHLC_STEPS[o_idx]
        df["close"] = df["close"].values + _OHLC_STEPS[c_idx]
        df["high"]  = df["high"].values  + _OHLC_STEPS[h_idx]
        df["low"]   = df["low"].values   + _OHLC_STEPS[l_idx]

        # ── Enforce OHLC consistency ───────────────────────────
        open_arr  = df["open"].values
        close_arr = df["close"].values
        df["high"] = np.maximum(df["high"].values,  np.maximum(open_arr, close_arr))
        df["low"]  = np.minimum(df["low"].values,   np.minimum(open_arr, close_arr))

        # ── Volume jitter ──────────────────────────────────────
        if "volume" in df.columns:
            v_idx = self._rng.integers(0, len(_VOLUME_STEPS), size=n)
            df["volume"] = np.maximum(0.0, df["volume"].values + _VOLUME_STEPS[v_idx])

        return df
