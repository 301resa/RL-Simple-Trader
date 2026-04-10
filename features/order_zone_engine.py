"""
features/order_zone_engine.py
==============================
Confluence scoring engine — determines if the current setup warrants a trade.

The Order Zone concept requires ONE pillar:
  1. Supply or Demand zone (price at a key structural level)

Additional filter:
  2. ATR room remaining (directional)

All factors are weighted and combined into a [0, 1] confluence score.
Liquidity sweep (Pillar 2) has been removed — the LSTM learns sweep context
from the raw observation features directly.
Rejection candle (Pillar 3) was previously removed for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd

from features.atr_calculator import ATRState
from features.zone_detector import ZoneState, ZoneType
from features.trend_classifier import TrendSnapshot

# Fixed stop buffer in points (placed beyond zone boundary)
FIXED_STOP_BUFFER_PTS: float = 1.5


class OrderZoneType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NONE    = "none"


@dataclass
class OrderZoneState:
    """
    Full output of the OrderZoneEngine for the current bar.

    Attributes
    ----------
    zone_type : OrderZoneType
    confluence_score : float        0.0–1.0 weighted score.
    in_bearish_order_zone : bool    Price inside a supply (bearish) zone.
    in_bullish_order_zone : bool    Price inside a demand (bullish) zone.
    rr_ratio : float                Estimated risk:reward ratio.
    trade_worthwhile : bool         True when all thresholds are met.
    component_scores : Dict[str, float]
    stop_pts_bearish : float        Stop distance in points for a short entry.
    stop_pts_bullish : float        Stop distance in points for a long entry.
    """
    zone_type: OrderZoneType
    confluence_score: float
    in_bearish_order_zone: bool
    in_bullish_order_zone: bool
    rr_ratio: float
    trade_worthwhile: bool
    component_scores: Dict[str, float]
    stop_pts_bearish: float = 1.5
    stop_pts_bullish: float = 1.5


class OrderZoneEngine:
    """
    Computes the Order Zone confluence score for each bar.

    Two factors:
      1. Supply/Demand zone membership  (zone_weight  = 0.90)
      2. ATR room remaining             (atr_weight   = 0.10)

    Parameters
    ----------
    min_confluence_score : float
        Minimum score required for trade_worthwhile = True.
    min_rr_ratio : float
        Minimum R:R ratio required for trade_worthwhile = True.
    weights : dict, optional
        Keys: "zone", "atr"
    """

    def __init__(
        self,
        min_confluence_score: float = 0.35,
        min_rr_ratio: float = 1.5,
        weights: Optional[dict] = None,
        zone_weight: float = 0.90,
        atr_weight: float = 0.10,
        # Kept for API compatibility — no longer used
        sweep_weight: float = 0.0,
        rejection_weight: float = 0.0,
        pin_bar_wick_ratio: float = 2.0,
        engulfing_body_ratio: float = 0.70,
    ) -> None:
        self.min_confluence_score = min_confluence_score
        self.min_rr_ratio = min_rr_ratio

        if weights is not None:
            self.zone_weight = float(weights.get("zone", zone_weight))
            self.atr_weight  = float(weights.get("atr",  atr_weight))
        else:
            self.zone_weight = zone_weight
            self.atr_weight  = atr_weight

        self._total_weight = self.zone_weight + self.atr_weight

    def compute(
        self,
        bars: pd.DataFrame,
        current_bar_idx: int,
        atr_state: ATRState,
        zone_state: Optional[ZoneState],
        liquidity_state=None,          # kept for API compatibility — ignored
        trend_snapshot: Optional[TrendSnapshot] = None,  # kept for API compatibility — ignored
    ) -> OrderZoneState:
        """
        Score the current setup and return an OrderZoneState.
        """
        bar = bars.iloc[current_bar_idx]
        current_price = float(bar["close"])
        atr = atr_state.atr_daily

        # ── 1. Zone proximity / membership ───────────────────────────────────
        in_supply = False
        in_demand = False
        zone_score_bearish = 0.0
        zone_score_bullish = 0.0
        # Stop is placed FIXED_STOP_BUFFER_PTS beyond the zone boundary
        stop_pts_bearish = FIXED_STOP_BUFFER_PTS
        stop_pts_bullish = FIXED_STOP_BUFFER_PTS

        if zone_state is not None:
            if zone_state.nearest_supply and zone_state.nearest_supply.is_valid:
                s = zone_state.nearest_supply
                if s.bottom - atr * 0.05 <= current_price <= s.top + atr * 0.05:
                    in_supply = True
                    zone_score_bearish = 1.0
                    # Stop: zone top + 1.5 pts (fixed buffer)
                    stop_pts_bearish = (s.top - current_price) + FIXED_STOP_BUFFER_PTS
                    stop_pts_bearish = max(stop_pts_bearish, FIXED_STOP_BUFFER_PTS)
                else:
                    prox = max(0.0, 1.0 - abs(current_price - s.midpoint) / max(atr, 1.0))
                    zone_score_bearish = prox * 0.5

            if zone_state.nearest_demand and zone_state.nearest_demand.is_valid:
                d = zone_state.nearest_demand
                if d.bottom - atr * 0.05 <= current_price <= d.top + atr * 0.05:
                    in_demand = True
                    zone_score_bullish = 1.0
                    # Stop: zone bottom - 1.5 pts (fixed buffer)
                    stop_pts_bullish = (current_price - d.bottom) + FIXED_STOP_BUFFER_PTS
                    stop_pts_bullish = max(stop_pts_bullish, FIXED_STOP_BUFFER_PTS)
                else:
                    prox = max(0.0, 1.0 - abs(current_price - d.midpoint) / max(atr, 1.0))
                    zone_score_bullish = prox * 0.5

        # ── 2. ATR room (directional) ─────────────────────────────────────────
        # ATR score reflects remaining room; zeroed if that direction is exhausted.
        atr_base = max(0.0, 1.0 - atr_state.atr_pct_used)
        atr_score_bearish = 0.0 if atr_state.atr_short_exhausted else atr_base
        atr_score_bullish = 0.0 if atr_state.atr_long_exhausted  else atr_base

        # ── 3. Weighted confluence scores ─────────────────────────────────────
        raw_bearish = (
            zone_score_bearish * self.zone_weight
            + atr_score_bearish * self.atr_weight
        )
        raw_bullish = (
            zone_score_bullish * self.zone_weight
            + atr_score_bullish * self.atr_weight
        )

        bearish_score = float(np.clip(raw_bearish / self._total_weight, 0.0, 1.0))
        bullish_score = float(np.clip(raw_bullish / self._total_weight, 0.0, 1.0))

        # ── 4. R:R estimate ───────────────────────────────────────────────────
        target_pts = atr_state.atr_remaining_pts * 0.75
        target_pts = max(target_pts, atr * 0.10)

        rr_ratio = 0.0
        if in_supply and stop_pts_bearish > 0:
            rr_ratio = target_pts / stop_pts_bearish
        elif in_demand and stop_pts_bullish > 0:
            rr_ratio = target_pts / stop_pts_bullish

        # ── 5. Primary direction and trade worthiness ─────────────────────────
        if bearish_score >= bullish_score and bearish_score > 0:
            zone_type = OrderZoneType.BEARISH
            confluence_score = bearish_score
        elif bullish_score > bearish_score and bullish_score > 0:
            zone_type = OrderZoneType.BULLISH
            confluence_score = bullish_score
        else:
            zone_type = OrderZoneType.NONE
            confluence_score = max(bearish_score, bullish_score)

        # Block trade if ATR exhausted in the identified direction
        dir_atr_ok = True
        if zone_type == OrderZoneType.BEARISH and atr_state.atr_short_exhausted:
            dir_atr_ok = False
        elif zone_type == OrderZoneType.BULLISH and atr_state.atr_long_exhausted:
            dir_atr_ok = False

        trade_worthwhile = (
            dir_atr_ok
            and confluence_score >= self.min_confluence_score
            and rr_ratio >= self.min_rr_ratio
        )

        component_scores = {
            "zone_bearish":    zone_score_bearish,
            "zone_bullish":    zone_score_bullish,
            "atr_room_bearish": atr_score_bearish,
            "atr_room_bullish": atr_score_bullish,
        }

        return OrderZoneState(
            zone_type=zone_type,
            confluence_score=confluence_score,
            in_bearish_order_zone=in_supply,
            in_bullish_order_zone=in_demand,
            rr_ratio=float(np.clip(rr_ratio, 0.0, 50.0)),
            trade_worthwhile=trade_worthwhile,
            component_scores=component_scores,
            stop_pts_bearish=stop_pts_bearish,
            stop_pts_bullish=stop_pts_bullish,
        )
