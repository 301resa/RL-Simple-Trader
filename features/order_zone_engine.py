"""
features/order_zone_engine.py
==============================
Confluence scoring engine — determines if the current setup warrants a trade.

The Order Zone concept requires TWO pillars:
  1. Supply or Demand zone (price at a key structural level)
  2. Liquidity sweep (prior highs/lows swept, stops cleared)

Additional filter:
  3. ATR room remaining

All factors are weighted and combined into a [0, 1] confluence score.
Pillar 3 (rejection candle) has been removed — the LSTM learns candle
context from the raw observation features directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd

from features.atr_calculator import ATRState
from features.zone_detector import ZoneState, ZoneType
from features.liquidity_detector import LiquidityState, SweepDirection
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

    Two pillars:
      1. Supply/Demand zone membership  (zone_weight)
      2. Liquidity sweep recency        (sweep_weight)
    Plus ATR room filter               (atr_weight)

    Parameters
    ----------
    min_confluence_score : float
        Minimum score required for trade_worthwhile = True.
    min_rr_ratio : float
        Minimum R:R ratio required for trade_worthwhile = True.
    weights : dict, optional
        Keys: "zone", "sweep", "atr"
    """

    def __init__(
        self,
        min_confluence_score: float = 0.35,
        min_rr_ratio: float = 1.5,
        weights: Optional[dict] = None,
        zone_weight: float = 0.55,
        sweep_weight: float = 0.35,
        atr_weight: float = 0.10,
        # Kept for API compatibility — no longer used
        rejection_weight: float = 0.0,
        pin_bar_wick_ratio: float = 2.0,
        engulfing_body_ratio: float = 0.70,
    ) -> None:
        self.min_confluence_score = min_confluence_score
        self.min_rr_ratio = min_rr_ratio

        if weights is not None:
            self.zone_weight  = float(weights.get("zone",  zone_weight))
            self.sweep_weight = float(weights.get("sweep", sweep_weight))
            self.atr_weight   = float(weights.get("atr",   atr_weight))
        else:
            self.zone_weight  = zone_weight
            self.sweep_weight = sweep_weight
            self.atr_weight   = atr_weight

        self._total_weight = self.zone_weight + self.sweep_weight + self.atr_weight

    def compute(
        self,
        bars: pd.DataFrame,
        current_bar_idx: int,
        atr_state: ATRState,
        zone_state: Optional[ZoneState],
        liquidity_state: Optional[LiquidityState],
        trend_snapshot: Optional[TrendSnapshot],   # unused — kept for API compatibility
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

        # ── 2. Liquidity sweep recency ────────────────────────────────────────
        sweep_score_bearish = 0.0
        sweep_score_bullish = 0.0

        if liquidity_state is not None and liquidity_state.sweep_bar_idx is not None:
            age = current_bar_idx - liquidity_state.sweep_bar_idx
            recency = max(0.0, 1.0 - age / 10.0)
            if liquidity_state.sweep_direction == SweepDirection.UP_SWEEP:
                sweep_score_bearish = recency
            elif liquidity_state.sweep_direction == SweepDirection.DOWN_SWEEP:
                sweep_score_bullish = recency

        # ── 3. ATR room ───────────────────────────────────────────────────────
        atr_score = 0.0
        if not atr_state.atr_exhausted:
            atr_score = max(0.0, 1.0 - atr_state.atr_pct_used)

        # ── 4. Weighted confluence scores ─────────────────────────────────────
        raw_bearish = (
            zone_score_bearish  * self.zone_weight
            + sweep_score_bearish * self.sweep_weight
            + atr_score           * self.atr_weight
        )
        raw_bullish = (
            zone_score_bullish  * self.zone_weight
            + sweep_score_bullish * self.sweep_weight
            + atr_score           * self.atr_weight
        )

        bearish_score = float(np.clip(raw_bearish / self._total_weight, 0.0, 1.0))
        bullish_score = float(np.clip(raw_bullish / self._total_weight, 0.0, 1.0))

        # ── 5. R:R estimate ───────────────────────────────────────────────────
        target_pts = atr_state.atr_remaining_pts * 0.75
        target_pts = max(target_pts, atr * 0.10)

        rr_ratio = 0.0
        if in_supply and stop_pts_bearish > 0:
            rr_ratio = target_pts / stop_pts_bearish
        elif in_demand and stop_pts_bullish > 0:
            rr_ratio = target_pts / stop_pts_bullish

        # ── 6. Primary direction and trade worthiness ─────────────────────────
        if bearish_score >= bullish_score and bearish_score > 0:
            zone_type = OrderZoneType.BEARISH
            confluence_score = bearish_score
        elif bullish_score > bearish_score and bullish_score > 0:
            zone_type = OrderZoneType.BULLISH
            confluence_score = bullish_score
        else:
            zone_type = OrderZoneType.NONE
            confluence_score = max(bearish_score, bullish_score)

        trade_worthwhile = (
            not atr_state.atr_exhausted
            and confluence_score >= self.min_confluence_score
            and rr_ratio >= self.min_rr_ratio
        )

        component_scores = {
            "zone_bearish":  zone_score_bearish,
            "zone_bullish":  zone_score_bullish,
            "sweep_bearish": sweep_score_bearish,
            "sweep_bullish": sweep_score_bullish,
            "atr_room":      atr_score,
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
