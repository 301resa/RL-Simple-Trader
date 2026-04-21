"""
features/harmonic_detector.py
==============================
W (double-bottom) and M (double-top) pattern detector.

A W pattern consists of two comparable swing lows separated by a higher
peak — price forms a bullish reversal / continuation structure at support.
An M pattern is the mirror: two comparable swing highs with a lower trough
between them — bearish structure at resistance.

These align naturally with the Order Zone strategy:
  W at demand zone  → extra confluence for LONG entry
  M at supply zone  → extra confluence for SHORT entry

Detection uses local pivot points (swing highs / lows) scanned over a
configurable lookback window.  Pattern quality is scored [0, 1] based on:
  symmetry  — how close the two lows (W) or highs (M) are in price
  prominence — how pronounced the middle peak / trough is relative to ATR

A score of 0 means no pattern was found; higher = better quality setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class HarmonicState:
    """Output of HarmonicDetector for the current bar."""
    w_score: float   # [0, 1] quality of W (double-bottom) pattern; 0 = not detected
    m_score: float   # [0, 1] quality of M (double-top)    pattern; 0 = not detected

    @property
    def w_detected(self) -> bool:
        return self.w_score > 0.0

    @property
    def m_detected(self) -> bool:
        return self.m_score > 0.0


# Sentinel returned when detection cannot run (not enough bars, bad ATR, etc.)
HARMONIC_NONE = HarmonicState(w_score=0.0, m_score=0.0)


class HarmonicDetector:
    """
    Detects W (double-bottom) and M (double-top) patterns from OHLC high/low arrays.

    Parameters
    ----------
    lookback_bars : int
        Bars to scan behind the current bar for pattern legs.
        At 5-min bars: 40 bars ≈ 3.3 hours of structure.
    pivot_window : int
        Half-width of local pivot confirmation window (bars each side).
        3 → a swing low must be the lowest bar in a 7-bar window.
    symmetry_tol_atr_pct : float
        Max allowed price difference between the two lows (W) or highs (M)
        expressed as a fraction of daily ATR.  0.12 → within 12% of ATR.
    min_peak_atr_pct : float
        Minimum height of the middle peak (W) or depth of trough (M)
        relative to ATR.  0.15 → at least 15% of ATR.
    min_separation_bars : int
        Minimum bars separating the two pivot lows / highs.
    recency_bars : int
        The second pivot must fall within this many bars of current_bar_idx.
    """

    def __init__(
        self,
        lookback_bars: int = 40,
        pivot_window: int = 3,
        symmetry_tol_atr_pct: float = 0.12,
        min_peak_atr_pct: float = 0.15,
        min_separation_bars: int = 5,
        recency_bars: int = 6,
    ) -> None:
        self.lookback_bars         = lookback_bars
        self.pivot_window          = pivot_window
        self.symmetry_tol_atr_pct  = symmetry_tol_atr_pct
        self.min_peak_atr_pct      = min_peak_atr_pct
        self.min_separation_bars   = min_separation_bars
        self.recency_bars          = recency_bars

    # ── Public API ────────────────────────────────────────────

    def detect(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        current_bar_idx: int,
        atr: float,
    ) -> HarmonicState:
        """
        Detect W and M patterns ending at or near current_bar_idx.

        Parameters
        ----------
        highs : np.ndarray
            Bar highs for the full combined (history + session) array.
        lows : np.ndarray
            Bar lows for the full combined array.
        current_bar_idx : int
            Index of the current bar in the combined arrays.
        atr : float
            Daily ATR for this session — used to scale all thresholds.
        """
        min_bars_needed = self.lookback_bars + self.pivot_window
        if atr <= 0 or current_bar_idx < min_bars_needed:
            return HARMONIC_NONE

        start = current_bar_idx - self.lookback_bars
        h = highs[start:current_bar_idx + 1]
        l = lows[start:current_bar_idx + 1]
        n = len(h)

        # ── Find local pivot highs and lows ───────────────────
        w = self.pivot_window
        pivot_highs: List[Tuple[int, float]] = []
        pivot_lows:  List[Tuple[int, float]] = []

        for i in range(w, n - w):
            window_h = h[i - w: i + w + 1]
            window_l = l[i - w: i + w + 1]
            if h[i] >= window_h.max():
                pivot_highs.append((i, float(h[i])))
            if l[i] <= window_l.min():
                pivot_lows.append((i, float(l[i])))

        sym_tol          = atr * self.symmetry_tol_atr_pct
        min_peak         = atr * self.min_peak_atr_pct
        recent_cutoff    = n - 1 - self.recency_bars   # second pivot must be >= this local index

        w_score = self._scan_w(pivot_lows, pivot_highs, sym_tol, min_peak, recent_cutoff)
        m_score = self._scan_m(pivot_highs, pivot_lows, sym_tol, min_peak, recent_cutoff)

        return HarmonicState(w_score=w_score, m_score=m_score)

    # ── Private helpers ───────────────────────────────────────

    def _scan_w(
        self,
        pivot_lows:  List[Tuple[int, float]],
        pivot_highs: List[Tuple[int, float]],
        sym_tol: float,
        min_peak: float,
        recent_cutoff: int,
    ) -> float:
        """Return the best W (double-bottom) score in the pivot lists."""
        best = 0.0
        n_lows = len(pivot_lows)
        for j in range(n_lows):
            idx2, p2 = pivot_lows[j]
            if idx2 < recent_cutoff:
                continue            # second low must be recent
            for i in range(j):
                idx1, p1 = pivot_lows[i]
                if idx2 - idx1 < self.min_separation_bars:
                    continue
                if abs(p2 - p1) > sym_tol:
                    continue        # lows too far apart in price
                # Peak between the two lows
                peak = self._max_between(pivot_highs, idx1, idx2)
                if peak is None:
                    continue
                low_avg = (p1 + p2) / 2.0
                if peak - low_avg < min_peak:
                    continue        # middle peak not pronounced enough
                sym_score  = 1.0 - abs(p2 - p1) / max(sym_tol, 1e-6)
                peak_score = min(1.0, (peak - low_avg) / max(min_peak * 3.0, 1e-6))
                score = float(np.clip(sym_score * 0.5 + peak_score * 0.5, 0.0, 1.0))
                if score > best:
                    best = score
        return best

    def _scan_m(
        self,
        pivot_highs: List[Tuple[int, float]],
        pivot_lows:  List[Tuple[int, float]],
        sym_tol: float,
        min_peak: float,
        recent_cutoff: int,
    ) -> float:
        """Return the best M (double-top) score in the pivot lists."""
        best = 0.0
        n_highs = len(pivot_highs)
        for j in range(n_highs):
            idx2, p2 = pivot_highs[j]
            if idx2 < recent_cutoff:
                continue
            for i in range(j):
                idx1, p1 = pivot_highs[i]
                if idx2 - idx1 < self.min_separation_bars:
                    continue
                if abs(p2 - p1) > sym_tol:
                    continue
                trough = self._min_between(pivot_lows, idx1, idx2)
                if trough is None:
                    continue
                high_avg = (p1 + p2) / 2.0
                if high_avg - trough < min_peak:
                    continue
                sym_score   = 1.0 - abs(p2 - p1) / max(sym_tol, 1e-6)
                depth_score = min(1.0, (high_avg - trough) / max(min_peak * 3.0, 1e-6))
                score = float(np.clip(sym_score * 0.5 + depth_score * 0.5, 0.0, 1.0))
                if score > best:
                    best = score
        return best

    @staticmethod
    def _max_between(
        pivots: List[Tuple[int, float]], idx_lo: int, idx_hi: int
    ) -> Optional[float]:
        vals = [p for k, p in pivots if idx_lo < k < idx_hi]
        return max(vals) if vals else None

    @staticmethod
    def _min_between(
        pivots: List[Tuple[int, float]], idx_lo: int, idx_hi: int
    ) -> Optional[float]:
        vals = [p for k, p in pivots if idx_lo < k < idx_hi]
        return min(vals) if vals else None
