"""
features/order_zone_engine.py
==============================
Confluence scoring engine — determines if the current setup warrants a trade.

The "Order Zone" concept requires all three pillars:
  1. Supply or Demand zone (price at a key structural level)
  2. Liquidity sweep (prior highs/lows swept, stops cleared)
  3. Rejection candle (pin bar, engulfing, or displacement bar)

Additional filters:
  4. Trend alignment
  5. ATR room remaining
  6. Minimum R:R ratio

All factors are weighted and combined into a [0, 1] confluence score.
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


class OrderZoneType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NONE    = "none"


@dataclass
class RejectionCandle:
    """
    Result of rejection candle pattern detection.

    Attributes
    ----------
    detected : bool
    pattern_name : str
        E.g. "pin_bar_bearish", "pin_bar_bullish", "displacement_bearish".
    direction : int
        1 = bullish rejection, -1 = bearish rejection, 0 = none.
    strength : float
        0.0–1.0 signal strength.
    """
    detected: bool
    pattern_name: str
    direction: int
    strength: float


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
    rejection_candle : RejectionCandle
    rr_ratio : float                Estimated risk:reward ratio.
    trade_worthwhile : bool         True when all thresholds are met.
    component_scores : Dict[str, float]
    """
    zone_type: OrderZoneType
    confluence_score: float
    in_bearish_order_zone: bool
    in_bullish_order_zone: bool
    rejection_candle: RejectionCandle
    rr_ratio: float
    trade_worthwhile: bool
    component_scores: Dict[str, float]


class OrderZoneEngine:
    """
    Computes the Order Zone confluence score for each bar.

    Parameters
    ----------
    min_confluence_score : float
        Minimum score required for trade_worthwhile = True.
    min_rr_ratio : float
        Minimum R:R ratio required for trade_worthwhile = True.
    zone_weight : float
    sweep_weight : float
    rejection_weight : float
    trend_weight : float
    atr_weight : float
    """

    def __init__(
        self,
        min_confluence_score: float = 0.60,
        min_rr_ratio: float = 1.5,
        weights: Optional[dict] = None,
        zone_weight: float = 0.35,
        sweep_weight: float = 0.30,
        rejection_weight: float = 0.25,
        atr_weight: float = 0.10,
        pin_bar_wick_ratio: float = 2.0,
        engulfing_body_ratio: float = 0.70,
    ) -> None:
        self.min_confluence_score = min_confluence_score
        self.min_rr_ratio = min_rr_ratio
        self.pin_bar_wick_ratio = pin_bar_wick_ratio
        self.engulfing_body_ratio = engulfing_body_ratio

        # Trend weight removed — no lagging trend indicators.
        # Weights: zone=0.35, sweep=0.30, rejection=0.25, atr=0.10 (sum=1.0)
        if weights is not None:
            self.zone_weight       = float(weights.get("zone",       zone_weight))
            self.sweep_weight      = float(weights.get("sweep",      sweep_weight))
            self.rejection_weight  = float(weights.get("rejection",  rejection_weight))
            self.atr_weight        = float(weights.get("atr",        atr_weight))
        else:
            self.zone_weight      = zone_weight
            self.sweep_weight     = sweep_weight
            self.rejection_weight = rejection_weight
            self.atr_weight       = atr_weight

        self._total_weight = (
            self.zone_weight + self.sweep_weight
            + self.rejection_weight + self.atr_weight
        )

    def compute(
        self,
        bars: pd.DataFrame,
        current_bar_idx: int,
        atr_state: ATRState,
        zone_state: Optional[ZoneState],
        liquidity_state: Optional[LiquidityState],
        trend_snapshot: Optional[TrendSnapshot],
    ) -> OrderZoneState:
        """
        Score the current setup and return an OrderZoneState.

        Parameters
        ----------
        bars : pd.DataFrame
        current_bar_idx : int
        atr_state : ATRState
        zone_state : ZoneState or None
        liquidity_state : LiquidityState or None
        trend_snapshot : TrendSnapshot or None
        """
        bar = bars.iloc[current_bar_idx]
        current_price = float(bar["close"])
        atr = atr_state.atr_daily

        # ── 1. Zone proximity / membership ───────────────────
        in_supply = False
        in_demand = False
        zone_score_bearish = 0.0
        zone_score_bullish = 0.0
        stop_pts_bearish = atr * 0.10  # fallback stop distance
        stop_pts_bullish = atr * 0.10

        if zone_state is not None:
            if zone_state.nearest_supply and zone_state.nearest_supply.is_valid:
                s = zone_state.nearest_supply
                # Inside the zone (or within a small buffer)
                if s.bottom - atr * 0.05 <= current_price <= s.top + atr * 0.05:
                    in_supply = True
                    zone_score_bearish = 1.0
                    stop_pts_bearish = (s.top - current_price) + atr * 0.03
                else:
                    prox = max(0.0, 1.0 - abs(current_price - s.midpoint) / max(atr, 1.0))
                    zone_score_bearish = prox * 0.5

            if zone_state.nearest_demand and zone_state.nearest_demand.is_valid:
                d = zone_state.nearest_demand
                if d.bottom - atr * 0.05 <= current_price <= d.top + atr * 0.05:
                    in_demand = True
                    zone_score_bullish = 1.0
                    stop_pts_bullish = (current_price - d.bottom) + atr * 0.03
                else:
                    prox = max(0.0, 1.0 - abs(current_price - d.midpoint) / max(atr, 1.0))
                    zone_score_bullish = prox * 0.5

        # ── 2. Liquidity sweep recency ────────────────────────
        sweep_score_bearish = 0.0
        sweep_score_bullish = 0.0

        if liquidity_state is not None and liquidity_state.sweep_bar_idx is not None:
            age = current_bar_idx - liquidity_state.sweep_bar_idx
            recency = max(0.0, 1.0 - age / 10.0)
            if liquidity_state.sweep_direction == SweepDirection.UP_SWEEP:
                sweep_score_bearish = recency   # swept above highs → bearish
            elif liquidity_state.sweep_direction == SweepDirection.DOWN_SWEEP:
                sweep_score_bullish = recency   # swept below lows → bullish

        # ── 3. Rejection candle ───────────────────────────────
        rejection = self._detect_rejection_candle(bar, atr)
        rejection_score_bearish = 0.0
        rejection_score_bullish = 0.0
        if rejection.detected:
            if rejection.direction == -1:
                rejection_score_bearish = rejection.strength
            elif rejection.direction == 1:
                rejection_score_bullish = rejection.strength

        # ── 4. ATR room ───────────────────────────────────────
        atr_score = 0.0
        if not atr_state.atr_exhausted:
            atr_score = max(0.0, 1.0 - atr_state.atr_pct_used)

        # ── 5. Weighted confluence scores ─────────────────────
        raw_bearish = (
            zone_score_bearish        * self.zone_weight
            + sweep_score_bearish     * self.sweep_weight
            + rejection_score_bearish * self.rejection_weight
            + atr_score               * self.atr_weight
        )
        raw_bullish = (
            zone_score_bullish        * self.zone_weight
            + sweep_score_bullish     * self.sweep_weight
            + rejection_score_bullish * self.rejection_weight
            + atr_score               * self.atr_weight
        )

        # Normalise to [0, 1]
        bearish_score = float(np.clip(raw_bearish / self._total_weight, 0.0, 1.0))
        bullish_score = float(np.clip(raw_bullish / self._total_weight, 0.0, 1.0))

        # ── 7. R:R estimate ───────────────────────────────────
        target_pts = atr_state.atr_remaining_pts * 0.75
        target_pts = max(target_pts, atr * 0.10)

        rr_ratio = 0.0
        if in_supply and stop_pts_bearish > 0:
            rr_ratio = target_pts / stop_pts_bearish
        elif in_demand and stop_pts_bullish > 0:
            rr_ratio = target_pts / stop_pts_bullish

        # ── 8. Primary direction and trade worthiness ─────────
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
            "zone_bearish":       zone_score_bearish,
            "zone_bullish":       zone_score_bullish,
            "sweep_bearish":      sweep_score_bearish,
            "sweep_bullish":      sweep_score_bullish,
            "rejection_bearish":  rejection_score_bearish,
            "rejection_bullish":  rejection_score_bullish,
            "atr_room":           atr_score,
        }

        return OrderZoneState(
            zone_type=zone_type,
            confluence_score=confluence_score,
            in_bearish_order_zone=in_supply,
            in_bullish_order_zone=in_demand,
            rejection_candle=rejection,
            rr_ratio=float(np.clip(rr_ratio, 0.0, 50.0)),
            trade_worthwhile=trade_worthwhile,
            component_scores=component_scores,
        )

    # ── Private helpers ───────────────────────────────────────

    def _detect_rejection_candle(self, bar: pd.Series, atr: float) -> RejectionCandle:
        """
        Detect pin bars and displacement candles.

        Pin bar criteria:
          - Body < 40% of full range
          - Dominant wick > 60% of full range
          - Dominant wick at least 2× the opposite wick

        Displacement criteria:
          - Body > 70% of full range (strong directional bar)
        """
        high  = float(bar["high"])
        low   = float(bar["low"])
        open_ = float(bar["open"])
        close = float(bar["close"])

        full_range = high - low
        if full_range < atr * 0.03:
            return RejectionCandle(False, "none", 0, 0.0)

        body        = abs(close - open_)
        upper_wick  = high - max(close, open_)
        lower_wick  = min(close, open_) - low
        body_ratio  = body / full_range
        upper_ratio = upper_wick / full_range
        lower_ratio = lower_wick / full_range

        # Pin bar: dominant wick must be pin_bar_wick_ratio× the opposite wick
        if body_ratio < 0.40:
            if upper_ratio > 0.50 and upper_ratio > lower_ratio * self.pin_bar_wick_ratio:
                return RejectionCandle(True, "pin_bar_bearish", -1, float(np.clip(upper_ratio, 0.0, 1.0)))
            if lower_ratio > 0.50 and lower_ratio > upper_ratio * self.pin_bar_wick_ratio:
                return RejectionCandle(True, "pin_bar_bullish", 1, float(np.clip(lower_ratio, 0.0, 1.0)))

        # Displacement / engulfing bar: body > engulfing_body_ratio of full range
        if body_ratio > self.engulfing_body_ratio:
            if close > open_:
                return RejectionCandle(True, "displacement_bullish", 1, float(np.clip(body_ratio, 0.0, 1.0)))
            else:
                return RejectionCandle(True, "displacement_bearish", -1, float(np.clip(body_ratio, 0.0, 1.0)))

        return RejectionCandle(False, "none", 0, 0.0)
