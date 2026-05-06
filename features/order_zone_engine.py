"""
features/order_zone_engine.py
==============================
Confluence scoring engine — determines if the current setup warrants a trade.

Three weighted components:
  1. Zone quality     — was price inside a valid supply/demand zone?
  2. ATR room         — directional room remaining before exhaustion
  3. Sweep freshness  — how recently was the zone's liquidity swept?

All factors normalised to [0, 1] and combined into a single confluence score.
Weights are configurable in features_config.yaml → order_zone.weights.
Default: zone=0.50, atr=0.30, sweep=0.20.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd

from features.atr_calculator import ATRState
from features.zone_detector import ZoneState, ZoneType

# Zone quality scoring constants
_ZONE_MAX_AGE_BARS: int  = 300   # matches max_zone_age_bars in features_config.yaml
_ZONE_MAX_TOUCHES: int   = 3     # matches max_zone_touches in features_config.yaml
_SWEEP_DECAY_BARS: float = 40.0  # sweep score decays from 1.0 → 0.0 over this many bars


def _sweep_freshness_score(zone, current_bar_idx: int, decay_bars: float = _SWEEP_DECAY_BARS) -> float:
    """
    Sweep recency weight.

    Returns 1.0 if the zone was swept very recently, decaying linearly to 0.0
    at ``decay_bars`` bars after formation.  Returns 0.0 if the zone was
    never swept.

    For Sonarlab zones, was_swept=True at creation and bar_formed_idx is the
    ROC crossover bar — so this naturally rewards fresh institutional setups.
    ``decay_bars`` is timeframe-aware (passed from OrderZoneEngine instance).
    """
    if zone is None or not zone.is_valid or not zone.was_swept:
        return 0.0
    bars_since = current_bar_idx - zone.bar_formed_idx
    return float(max(0.0, 1.0 - bars_since / max(decay_bars, 1.0)))


def _zone_quality_score(zone, current_bar_idx: int, max_zone_width_pts: float) -> float:
    """
    Graduate the in-zone confluence score based on zone quality.

    Components (all in [0, 1], higher = better):
      width_score  — narrow zones are more precise entry points
      age_score    — fresh zones react more cleanly than old ones
      touch_score  — fewer prior touches means more untapped structure

    Returns a score in [0.55, 1.00] when price is inside a valid zone.
    The 0.55 floor ensures in-zone setups always beat the minimum
    confluence threshold (0.45) while still being sized conservatively.
    """
    width   = zone.top - zone.bottom
    age     = current_bar_idx - zone.bar_formed_idx
    touches = getattr(zone, "touches", 0)

    width_score = max(0.0, 1.0 - width  / max(max_zone_width_pts, 1e-6))
    age_score   = max(0.0, 1.0 - age    / _ZONE_MAX_AGE_BARS)
    touch_score = max(0.0, 1.0 - touches / max(_ZONE_MAX_TOUCHES, 1))

    quality = width_score * 0.40 + age_score * 0.35 + touch_score * 0.25
    return float(np.clip(0.55 + quality * 0.45, 0.55, 1.00))


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
    stop_pts_bearish: float = 0.0
    stop_pts_bullish: float = 0.0


class OrderZoneEngine:
    """
    Computes the Order Zone confluence score for each bar.

    Three weighted components:
      1. Zone quality    (zone_weight  = 0.50) — supply/demand membership + quality
      2. ATR room        (atr_weight   = 0.30) — directional room remaining
      3. Sweep freshness (sweep_weight = 0.20) — how recently the zone was swept

    Configurable via features_config.yaml → order_zone.weights.
    """

    def __init__(
        self,
        min_confluence_score: float = 0.35,
        min_rr_ratio: float = 1.5,
        weights: Optional[dict] = None,
        zone_weight: float = 0.50,
        atr_weight: float = 0.30,
        sweep_weight: float = 0.20,
        max_zone_pts: float = 10.0,
        stop_buffer_pts: float = 1.5,
        fallback_stop_pts: float = 3.0,
        sweep_decay_bars: int = 40,
    ) -> None:
        self.min_confluence_score = min_confluence_score
        self.min_rr_ratio         = min_rr_ratio
        self.max_zone_pts         = max_zone_pts
        self.stop_buffer_pts      = stop_buffer_pts
        self.fallback_stop_pts    = fallback_stop_pts
        self._sweep_decay_bars    = float(sweep_decay_bars)

        if weights is not None:
            self.zone_weight  = float(weights.get("zone",  zone_weight))
            self.atr_weight   = float(weights.get("atr",   atr_weight))
            self.sweep_weight = float(weights.get("sweep", sweep_weight))
        else:
            self.zone_weight  = zone_weight
            self.atr_weight   = atr_weight
            self.sweep_weight = sweep_weight

        self._total_weight = self.zone_weight + self.atr_weight + self.sweep_weight

    def compute(
        self,
        bars: pd.DataFrame,
        current_bar_idx: int,
        atr_state: ATRState,
        zone_state: Optional[ZoneState],
        current_price: Optional[float] = None,
        harmonic_state=None,   # kept for call-site compat — unused
    ) -> OrderZoneState:
        """Score the current setup and return an OrderZoneState."""
        if current_price is None:
            current_price = float(bars.iloc[current_bar_idx]["close"])
        atr = atr_state.atr_daily

        # Tick-based geometry from instrument profile
        max_zone_width_pts = self.max_zone_pts
        stop_buffer_pts    = self.stop_buffer_pts
        fallback_stop_pts  = self.fallback_stop_pts

        # ── 1. Zone proximity / membership ───────────────────────────────────
        in_supply = False
        in_demand = False
        zone_score_bearish = 0.0
        zone_score_bullish = 0.0
        stop_pts_bearish = fallback_stop_pts
        stop_pts_bullish = fallback_stop_pts

        if zone_state is not None:
            if zone_state.nearest_supply and zone_state.nearest_supply.is_valid:
                s = zone_state.nearest_supply
                if s.bottom - atr * 0.05 <= current_price <= s.top + atr * 0.05:
                    in_supply = True
                    zone_score_bearish = _zone_quality_score(s, current_bar_idx, max_zone_width_pts)
                    # True risk = full zone width + buffer (stop lives beyond far edge)
                    stop_pts_bearish = (s.top - s.bottom) + stop_buffer_pts
                else:
                    prox = max(0.0, 1.0 - abs(current_price - s.midpoint) / max(atr, 1.0))
                    zone_score_bearish = prox * 0.5

            if zone_state.nearest_demand and zone_state.nearest_demand.is_valid:
                d = zone_state.nearest_demand
                if d.bottom - atr * 0.05 <= current_price <= d.top + atr * 0.05:
                    in_demand = True
                    zone_score_bullish = _zone_quality_score(d, current_bar_idx, max_zone_width_pts)
                    stop_pts_bullish = (d.top - d.bottom) + stop_buffer_pts
                else:
                    prox = max(0.0, 1.0 - abs(current_price - d.midpoint) / max(atr, 1.0))
                    zone_score_bullish = prox * 0.5

        # ── 2. ATR room (directional) ─────────────────────────────────────────
        # ATR score reflects remaining room; zeroed if that direction is exhausted.
        atr_base = max(0.0, 1.0 - atr_state.atr_pct_used)
        atr_score_bearish = 0.0 if atr_state.atr_short_exhausted else atr_base
        atr_score_bullish = 0.0 if atr_state.atr_long_exhausted  else atr_base

        # ── 3. Sweep freshness (directional) ──────────────────────────────────
        # 1.0 = zone swept very recently; decays to 0.0 over _SWEEP_DECAY_BARS.
        # Unswept zones score 0.0 — forces the confluence floor to be lower,
        # making trade_worthwhile harder to achieve without a confirmed sweep.
        s_zone = zone_state.nearest_supply if zone_state else None
        d_zone = zone_state.nearest_demand if zone_state else None
        sweep_score_bearish = _sweep_freshness_score(s_zone, current_bar_idx, self._sweep_decay_bars)
        sweep_score_bullish = _sweep_freshness_score(d_zone, current_bar_idx, self._sweep_decay_bars)

        # ── 4. Weighted confluence scores ─────────────────────────────────────
        raw_bearish = (
            zone_score_bearish  * self.zone_weight
            + atr_score_bearish * self.atr_weight
            + sweep_score_bearish * self.sweep_weight
        )
        raw_bullish = (
            zone_score_bullish  * self.zone_weight
            + atr_score_bullish * self.atr_weight
            + sweep_score_bullish * self.sweep_weight
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
            "atr_bearish":     atr_score_bearish,
            "atr_bullish":     atr_score_bullish,
            "sweep_bearish":   sweep_score_bearish,
            "sweep_bullish":   sweep_score_bullish,
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
