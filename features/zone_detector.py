"""
features/zone_detector.py
==========================
Supply and Demand zone detection from intraday bar sequences.

Detection logic:
  - A SUPPLY zone forms when:
      1. Tight consolidation (2–8 bars, range < X% of ATR)
      2. Followed by a strong BEARISH impulse bar (large body)
  - A DEMAND zone forms when:
      1. Tight consolidation
      2. Followed by a strong BULLISH impulse bar

Zone boundaries are set at the consolidation price range.
Zones are invalidated when price closes through them or touches exceed max_touches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


class ZoneType(Enum):
    SUPPLY = "supply"
    DEMAND = "demand"
    NONE   = "none"


@dataclass
class Zone:
    """A supply or demand zone with validity tracking."""
    top: float
    bottom: float
    zone_type: ZoneType
    is_valid: bool = True
    touches: int = 0
    bar_formed_idx: int = 0

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def height(self) -> float:
        return self.top - self.bottom


@dataclass
class ZoneState:
    """Current nearest valid supply and demand zones."""
    nearest_supply: Optional[Zone] = None
    nearest_demand: Optional[Zone] = None

    def as_feature_dict(self, current_price: float, atr: float) -> dict:
        """Return normalised zone features for the observation vector."""
        def dist_norm(zone: Optional[Zone]) -> float:
            if zone is None or not zone.is_valid:
                return 0.0
            return float(np.clip(
                abs(current_price - zone.midpoint) / max(atr, 1.0), 0.0, 5.0
            ))

        in_supply = 0.0
        in_demand = 0.0
        if self.nearest_supply and self.nearest_supply.is_valid:
            if self.nearest_supply.bottom <= current_price <= self.nearest_supply.top:
                in_supply = 1.0
        if self.nearest_demand and self.nearest_demand.is_valid:
            if self.nearest_demand.bottom <= current_price <= self.nearest_demand.top:
                in_demand = 1.0

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
        Minimum bars for a consolidation base.
    consolidation_max_bars : int
        Maximum bars for a consolidation base.
    consolidation_range_atr_pct : float
        Max range of consolidation bars relative to ATR.
    impulse_min_body_atr_pct : float
        Minimum impulse bar body size relative to ATR.
    max_zone_touches : int
        Invalidate zone after this many price touches.
    max_zones_per_side : int
        Keep only N most recent valid zones per side.
    """

    def __init__(
        self,
        consolidation_min_bars: int = 2,
        consolidation_max_bars: int = 8,
        consolidation_range_atr_pct: float = 0.30,
        impulse_min_body_atr_pct: float = 0.15,
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

    def reset(self) -> None:
        """Reset zone lists for a new episode/session."""
        self._supply_zones = []
        self._demand_zones = []
        self._last_scanned_idx = -1

    def scan_and_update(
        self,
        bars: pd.DataFrame,
        atr_series: pd.Series,
        current_bar_idx: int,
    ) -> ZoneState:
        """
        Scan bars up to current_bar_idx for new zones, update validity.

        Uses only data up to current_bar_idx (no lookahead).

        Parameters
        ----------
        bars : pd.DataFrame
        atr_series : pd.Series
            ATR value per bar (same index as bars).
        current_bar_idx : int

        Returns
        -------
        ZoneState with nearest valid supply and demand zones.
        """
        current_price = float(bars.iloc[current_bar_idx]["close"])
        atr = float(atr_series.iloc[current_bar_idx]) if atr_series is not None else 100.0

        # Incrementally scan only new bars
        scan_start = max(0, self._last_scanned_idx + 1)
        for i in range(scan_start, current_bar_idx + 1):
            self._try_detect_zone(bars, atr_series, i)
        self._last_scanned_idx = current_bar_idx

        # Update validity of all known zones
        self._update_zone_validity(current_bar_idx, current_price, atr)

        return self._build_state(current_price)

    # ── Private helpers ───────────────────────────────────────

    def _update_zone_validity(
        self, current_bar_idx: int, current_price: float, atr: float
    ) -> None:
        """Invalidate zones that have been broken through or over-touched."""
        break_buffer = atr * max(self.zone_buffer_atr_pct, 0.02)

        for zone in self._supply_zones:
            if not zone.is_valid:
                continue
            # Age-based expiry
            if current_bar_idx - zone.bar_formed_idx > self.max_zone_age_bars:
                zone.is_valid = False
                continue
            # Price closes above supply zone top
            if current_price > zone.top + break_buffer:
                zone.is_valid = False
                continue
            # Count touches (price enters zone)
            if zone.bottom <= current_price <= zone.top:
                zone.touches += 1
                if zone.touches > self.max_zone_touches:
                    zone.is_valid = False

        for zone in self._demand_zones:
            if not zone.is_valid:
                continue
            # Age-based expiry
            if current_bar_idx - zone.bar_formed_idx > self.max_zone_age_bars:
                zone.is_valid = False
                continue
            # Price closes below demand zone bottom
            if current_price < zone.bottom - break_buffer:
                zone.is_valid = False
                continue
            # Count touches
            if zone.bottom <= current_price <= zone.top:
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
        Try to detect a supply or demand zone ending at impulse_idx.

        Looks back for a tight consolidation preceding a strong impulse bar.
        """
        if impulse_idx < self.consolidation_min_bars:
            return

        atr = float(atr_series.iloc[impulse_idx]) if atr_series is not None else 100.0
        if atr <= 0:
            return

        impulse_bar = bars.iloc[impulse_idx]
        imp_open = float(impulse_bar["open"])
        imp_close = float(impulse_bar["close"])
        body = abs(imp_close - imp_open)

        if body < atr * self.impulse_min_body_atr_pct:
            return  # Too small to be an impulse

        is_bearish = imp_close < imp_open
        is_bullish = imp_close > imp_open

        if not (is_bearish or is_bullish):
            return

        # Search for a preceding consolidation
        for n_consol in range(self.consolidation_min_bars, self.consolidation_max_bars + 1):
            consol_start = impulse_idx - n_consol
            if consol_start < 0:
                break

            consol_bars = bars.iloc[consol_start:impulse_idx]
            consol_high = float(consol_bars["high"].max())
            consol_low = float(consol_bars["low"].min())
            consol_range = consol_high - consol_low

            if consol_range > atr * self.consolidation_range_atr_pct:
                break  # Wider windows will only be larger — stop searching

            # Valid tight consolidation found
            if is_bearish:
                zone = Zone(
                    top=consol_high,
                    bottom=consol_low,
                    zone_type=ZoneType.SUPPLY,
                    bar_formed_idx=impulse_idx,
                )
                self._supply_zones.append(zone)
                self._prune_zones(self._supply_zones)
            else:
                zone = Zone(
                    top=consol_high,
                    bottom=consol_low,
                    zone_type=ZoneType.DEMAND,
                    bar_formed_idx=impulse_idx,
                )
                self._demand_zones.append(zone)
                self._prune_zones(self._demand_zones)
            break  # Only need the first (widest) valid consolidation

    def _prune_zones(self, zone_list: List[Zone]) -> None:
        """Invalidate oldest valid zones beyond max_zones_per_side."""
        valid = [z for z in zone_list if z.is_valid]
        if len(valid) > self.max_zones_per_side:
            # Invalidate oldest (lowest bar_formed_idx)
            oldest = min(valid, key=lambda z: z.bar_formed_idx)
            oldest.is_valid = False

    def _build_state(self, current_price: float) -> ZoneState:
        """Find the nearest valid supply and demand zones."""
        valid_supply = [z for z in self._supply_zones if z.is_valid]
        valid_demand = [z for z in self._demand_zones if z.is_valid]

        nearest_supply = (
            min(valid_supply, key=lambda z: abs(z.midpoint - current_price))
            if valid_supply else None
        )
        nearest_demand = (
            min(valid_demand, key=lambda z: abs(z.midpoint - current_price))
            if valid_demand else None
        )

        return ZoneState(nearest_supply=nearest_supply, nearest_demand=nearest_demand)
