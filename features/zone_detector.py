"""
features/zone_detector.py
==========================
Supply and Demand zone detection from intraday bar sequences.

Detection logic:
  - A SUPPLY zone forms when:
      1. Tight consolidation (2–8 bars, range < X% of ATR)
      2. Followed by a strong BEARISH impulse bar (body > Y% of ATR)
  - A DEMAND zone forms when:
      1. Tight consolidation
      2. Followed by a strong BULLISH impulse bar

Zone boundaries are the high and low of the consolidation base.
Zones are invalidated when price breaks through them or touches exceed max_touches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


class ZoneType(Enum):
    SUPPLY = "supply"
    DEMAND = "demand"


@dataclass
class Zone:
    """A supply or demand zone with validity tracking."""
    top: float
    bottom: float
    zone_type: ZoneType
    is_valid: bool = True
    touches: int = 0
    bar_formed_idx: int = 0
    # Extreme of the impulse bar that created this zone.
    # For DEMAND zones: impulse_extreme = impulse bar high  → natural LONG target.
    # For SUPPLY zones: impulse_extreme = impulse bar low   → natural SHORT target.
    # Falls back to midpoint if not set (pre-existing zones / no-zone fallback).
    impulse_extreme: float = 0.0
    # Liquidity sweep flag — True once price has traded through the zone's liquidity level.
    # Supply: swept when price >= zone.top  (consolidation high is the liquidity level).
    # Demand: swept when price <= zone.bottom (consolidation low is the liquidity level).
    # Entry is only allowed after the sweep occurs and price re-enters the zone.
    was_swept: bool = False

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass
class ZoneState:
    """Nearest valid supply and demand zones at a given bar."""
    nearest_supply: Optional[Zone] = None
    nearest_demand: Optional[Zone] = None

    def as_feature_dict(self, current_price: float, atr: float) -> dict:
        """Return normalised zone features for the observation vector."""
        def dist_norm(zone: Optional[Zone]) -> float:
            if zone is None or not zone.is_valid:
                return 0.0
            return float(np.clip(abs(current_price - zone.midpoint) / max(atr, 1.0), 0.0, 5.0))

        in_supply = 1.0 if (
            self.nearest_supply and self.nearest_supply.is_valid
            and self.nearest_supply.bottom <= current_price <= self.nearest_supply.top
        ) else 0.0

        in_demand = 1.0 if (
            self.nearest_demand and self.nearest_demand.is_valid
            and self.nearest_demand.bottom <= current_price <= self.nearest_demand.top
        ) else 0.0

        return {
            "supply_zone_dist_norm": dist_norm(self.nearest_supply),
            "demand_zone_dist_norm": dist_norm(self.nearest_demand),
            "in_supply_zone": in_supply,
            "in_demand_zone": in_demand,
        }


class ZoneDetector:
    """
    Detects Supply and Demand zones from intraday bar sequences.

    Parameters
    ----------
    consolidation_min_bars : int
        Minimum bars in a consolidation base.
    consolidation_max_bars : int
        Maximum bars in a consolidation base.
    consolidation_range_atr_pct : float
        Max high-to-low range of the base relative to ATR (tight coil).
    impulse_min_body_atr_pct : float
        Min body size of the impulse bar relative to ATR.
        Must exceed consolidation_range_atr_pct to ensure jump > base.
    max_zone_touches : int
        Invalidate a zone after this many price entries.
    max_zones_per_side : int
        Keep only the N most recent valid zones per side.
    max_zone_age_bars : int
        Expire a zone after this many bars regardless of touches.
    zone_buffer_atr_pct : float
        Buffer beyond zone boundary required for a breakout to invalidate.
    """

    def __init__(
        self,
        consolidation_min_bars: int = 2,
        consolidation_max_bars: int = 8,
        consolidation_range_atr_pct: float = 0.05,
        impulse_min_body_atr_pct: float = 0.12,
        max_zone_touches: int = 3,
        max_zones_per_side: int = 5,
        max_zone_age_bars: int = 200,
        zone_buffer_atr_pct: float = 0.02,
    ) -> None:
        self.consolidation_min_bars = consolidation_min_bars
        self.consolidation_max_bars = consolidation_max_bars
        self.consolidation_range_atr_pct = consolidation_range_atr_pct
        self.impulse_min_body_atr_pct = impulse_min_body_atr_pct
        self.max_zone_touches = max_zone_touches
        self.max_zones_per_side = max_zones_per_side
        self.max_zone_age_bars = max_zone_age_bars
        self.zone_buffer_atr_pct = zone_buffer_atr_pct

        self._supply_zones: List[Zone] = []
        self._demand_zones: List[Zone] = []
        self._last_scanned_idx: int = -1

        # Numpy array cache — set via set_bars_numpy() before the precompute loop.
        # Using numpy indexing instead of pandas .iloc is ~50x faster in tight loops.
        self._np_open:  Optional[np.ndarray] = None
        self._np_high:  Optional[np.ndarray] = None
        self._np_low:   Optional[np.ndarray] = None
        self._np_close: Optional[np.ndarray] = None
        self._np_atr:   Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset zone lists for a new episode."""
        self._supply_zones = []
        self._demand_zones = []
        self._last_scanned_idx = -1
        # Numpy cache is invalidated each episode — caller must call set_bars_numpy()
        # again after reset() if they want fast path for the new episode's bars.
        self._np_open = self._np_high = self._np_low = self._np_close = self._np_atr = None

    def set_bars_numpy(
        self,
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
        atr_arr: Optional[np.ndarray] = None,
    ) -> None:
        """
        Cache numpy arrays extracted from the bars DataFrame.

        Call this once per episode (after reset) with the combined-bars arrays
        so that _try_detect_zone and scan_and_update use direct array indexing
        instead of pandas .iloc row access (~50x faster in tight loops).
        """
        self._np_open  = open_arr
        self._np_high  = high_arr
        self._np_low   = low_arr
        self._np_close = close_arr
        self._np_atr   = atr_arr

    def scan_and_update(
        self,
        bars: pd.DataFrame,
        atr_series: pd.Series,
        current_bar_idx: int,
    ) -> ZoneState:
        """
        Incrementally scan for new zones up to current_bar_idx,
        then update validity of all known zones.

        No lookahead — only bars up to current_bar_idx are used.
        """
        if self._np_close is not None:
            current_price = float(self._np_close[current_bar_idx])
            atr = float(self._np_atr[current_bar_idx]) if self._np_atr is not None else 100.0
        else:
            current_price = float(bars.iloc[current_bar_idx]["close"])
            atr = float(atr_series.iloc[current_bar_idx]) if atr_series is not None else 100.0

        for i in range(max(0, self._last_scanned_idx + 1), current_bar_idx + 1):
            self._try_detect_zone(bars, atr_series, i)
        self._last_scanned_idx = current_bar_idx

        self._update_zone_validity(current_bar_idx, current_price, atr)
        self._prune_invalid_zones()
        return self._build_state(current_price)

    # ── Private helpers ───────────────────────────────────────

    def _update_zone_validity(
        self, current_bar_idx: int, current_price: float, atr: float
    ) -> None:
        """Invalidate zones that are broken through, over-touched, or too old."""
        buffer = atr * self.zone_buffer_atr_pct

        for zone in self._supply_zones:
            if not zone.is_valid:
                continue
            if current_bar_idx - zone.bar_formed_idx > self.max_zone_age_bars:
                zone.is_valid = False
                continue
            # Sweep detection: supply liquidity level = zone.top (consolidation high).
            # Swept when price reaches or exceeds zone.top before a full breakout.
            if not zone.was_swept and current_price >= zone.top:
                zone.was_swept = True
            if current_price > zone.top + buffer:       # sustained breakout — zone dead
                zone.is_valid = False
            elif zone.bottom <= current_price <= zone.top:
                zone.touches += 1
                if zone.touches > self.max_zone_touches:
                    zone.is_valid = False

        for zone in self._demand_zones:
            if not zone.is_valid:
                continue
            if current_bar_idx - zone.bar_formed_idx > self.max_zone_age_bars:
                zone.is_valid = False
                continue
            # Sweep detection: demand liquidity level = zone.bottom (consolidation low).
            # Swept when price reaches or goes below zone.bottom before a full breakout.
            if not zone.was_swept and current_price <= zone.bottom:
                zone.was_swept = True
            if current_price < zone.bottom - buffer:    # sustained breakout — zone dead
                zone.is_valid = False
            elif zone.bottom <= current_price <= zone.top:
                zone.touches += 1
                if zone.touches > self.max_zone_touches:
                    zone.is_valid = False

    def _try_detect_zone(
        self,
        bars: pd.DataFrame,
        atr_series: pd.Series,
        impulse_idx: int,
    ) -> None:
        """
        Try to detect a zone whose impulse bar is at impulse_idx.

        Looks back for the tightest valid consolidation base (shortest window
        that fits within the ATR threshold) preceding a strong impulse bar.
        """
        if impulse_idx < self.consolidation_min_bars:
            return

        # Use numpy cache if available (set via set_bars_numpy) — ~50x faster than
        # pandas .iloc in tight loops.
        if self._np_close is not None:
            atr = float(self._np_atr[impulse_idx]) if self._np_atr is not None else 100.0
            if atr <= 0:
                return
            close_i = float(self._np_close[impulse_idx])
            open_i  = float(self._np_open[impulse_idx])
            body = abs(close_i - open_i)
            if body < atr * self.impulse_min_body_atr_pct:
                return
            is_bearish = close_i < open_i
            impulse_extreme = float(self._np_low[impulse_idx]) if is_bearish else float(self._np_high[impulse_idx])

            for n in range(self.consolidation_min_bars, self.consolidation_max_bars + 1):
                start = impulse_idx - n
                if start < 0:
                    break
                base_high_max = float(self._np_high[start:impulse_idx].max())
                base_low_min  = float(self._np_low[start:impulse_idx].min())
                base_range = base_high_max - base_low_min
                if base_range > atr * self.consolidation_range_atr_pct:
                    break
                zone = Zone(
                    top=base_high_max,
                    bottom=base_low_min,
                    zone_type=ZoneType.SUPPLY if is_bearish else ZoneType.DEMAND,
                    bar_formed_idx=impulse_idx,
                    impulse_extreme=impulse_extreme,
                )
                if is_bearish:
                    self._supply_zones.append(zone)
                    self._prune(self._supply_zones)
                else:
                    self._demand_zones.append(zone)
                    self._prune(self._demand_zones)
                return
            return  # no consolidation window found

        # ── Pandas fallback (used when numpy cache is not set) ────────────────
        atr = float(atr_series.iloc[impulse_idx]) if atr_series is not None else 100.0
        if atr <= 0:
            return

        bar = bars.iloc[impulse_idx]
        body = abs(float(bar["close"]) - float(bar["open"]))
        if body < atr * self.impulse_min_body_atr_pct:
            return

        is_bearish = float(bar["close"]) < float(bar["open"])
        # The impulse bar's extreme becomes the natural profit target:
        #   bearish impulse (supply zone)  → short target = impulse bar LOW
        #   bullish impulse (demand zone)  → long target  = impulse bar HIGH
        impulse_extreme = float(bar["low"]) if is_bearish else float(bar["high"])

        for n in range(self.consolidation_min_bars, self.consolidation_max_bars + 1):
            start = impulse_idx - n
            if start < 0:
                break
            base = bars.iloc[start:impulse_idx]
            base_range = float(base["high"].max()) - float(base["low"].min())
            if base_range > atr * self.consolidation_range_atr_pct:
                break   # wider windows only get larger — stop

            # Tightest valid base found — register the zone
            zone = Zone(
                top=float(base["high"].max()),
                bottom=float(base["low"].min()),
                zone_type=ZoneType.SUPPLY if is_bearish else ZoneType.DEMAND,
                bar_formed_idx=impulse_idx,
                impulse_extreme=impulse_extreme,
            )
            if is_bearish:
                self._supply_zones.append(zone)
                self._prune(self._supply_zones)
            else:
                self._demand_zones.append(zone)
                self._prune(self._demand_zones)
            return

    def _prune(self, zone_list: List[Zone]) -> None:
        """Expire the oldest valid zone when the per-side limit is exceeded."""
        valid = [z for z in zone_list if z.is_valid]
        if len(valid) > self.max_zones_per_side:
            min(valid, key=lambda z: z.bar_formed_idx).is_valid = False

    def _prune_invalid_zones(self) -> None:
        """Remove fully-invalid zones once lists exceed 3× the per-side cap."""
        cap = self.max_zones_per_side * 3
        if len(self._supply_zones) > cap:
            self._supply_zones = [z for z in self._supply_zones if z.is_valid]
        if len(self._demand_zones) > cap:
            self._demand_zones = [z for z in self._demand_zones if z.is_valid]

    def _build_state(self, current_price: float) -> ZoneState:
        """Return the nearest valid supply and demand zones to current price.

        Selection is edge-proximity based:
        - Supply: zone whose *top* edge is closest to current price (SHORT entry
          triggers at supply.top, so proximity at the top edge matters).
        - Demand: zone whose *bottom* edge is closest to current price (LONG entry
          triggers at demand.bottom).
        """
        valid_supply = [z for z in self._supply_zones if z.is_valid]
        valid_demand = [z for z in self._demand_zones if z.is_valid]
        return ZoneState(
            nearest_supply=min(valid_supply, key=lambda z: abs(z.top - current_price), default=None),
            nearest_demand=min(valid_demand, key=lambda z: abs(z.bottom - current_price), default=None),
        )
