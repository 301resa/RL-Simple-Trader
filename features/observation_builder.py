"""
features/observation_builder.py
=================================
Assembles the neural network observation vector from all feature states.

The observation is a flat float32 numpy array containing:
  1. Recent OHLC price history — 20-bar sliding window (log-returns)     [80]
  2. ATR features (exhaustion, remaining room)                            [4]
  3. Zone features (distance, in-zone, width, age, sweep_weight)         [10]
  4. Order zone / confluence features (score, R:R, pending order state)  [10]
  5. Portfolio state (position, P&L, drawdown, trade counts)             [8]
  6. Session timing + market context                                      [6]
       session_time_pct, bars_remaining_pct, is_rth, rth_time_pct,
       sin_time, cos_time  ← cyclical time encoding (no discontinuity)
  7. Engineered features — 15 pre-computed signals:                      [15]
       Price location (5): session drift, dist from session high/low, prior day high/low
       Momentum (2): 5-bar, 12-bar (1h) log-returns
       Volatility regime (2): short/long vol ratio, avg close-in-range
       Bar character (1): avg candle body/range ratio
       HTF context (5): 1h close-in-range, 2h close-in-range, prior day range position,
                        multi-TF momentum coherence, HTF vol expansion

Note: volume excluded (price-structure strategy).
Harmonic patterns removed (zero signal when disabled; LSTM handles sequence).
r15/r30 momentum removed (LSTM already sees raw bars and carries sequence memory).

Sweep weight replaces binary sweep flag:
  1.0 = zone swept very recently (fresh setup)
  0.0 = zone not swept OR sweep is stale (>40 bars ago)
  Decays linearly from 1→0 over 40 bars after formation.

sin/cos time encoding:
  t = session_time_pct ∈ [0, 1]
  sin_time = sin(2πt),  cos_time = cos(2πt)
  Gives the agent cyclical regime awareness (open/midday/close) without discontinuity.

Total fixed features: 38 + 15 = 53
Observation vector size: 20 × 4 + 53 = 133
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from features.atr_calculator import ATRState
from features.zone_detector import ZoneState
from features.order_zone_engine import OrderZoneState

# Structured feature counts
_N_STRUCTURED  = 38   # ATR(4) + Zone(10) + OrderZone(10) + Portfolio(8) + Session(6)
_N_ENGINEERED  = 15   # PriceLocation(5) + Momentum(2) + VolRegime(2) + BarCharacter(1) + HTFContext(5)

# Sweep weight decays to zero after this many bars
_SWEEP_DECAY_BARS = 40


class ObservationBuilder:
    """
    Assembles the flat observation vector fed to the policy network.

    Parameters
    ----------
    clip_value : float
        All observation components clipped to [-clip_value, clip_value].
    lookback_bars : int
        Number of most recent bars as OHLC log-returns (default 20).
    """

    def __init__(
        self,
        normalize_observations: bool = False,
        clip_value: float = 10.0,
        lookback_bars: int = 20,
        max_zone_age_bars: int = 300,
        max_zone_pts: float = 10.0,
        min_zone_pts: float = 1.0,
    ) -> None:
        self.normalize_observations = normalize_observations
        self.clip_value = clip_value
        self.price_history_len = lookback_bars
        self._max_zone_age = float(max_zone_age_bars)
        self._max_zone_pts = float(max_zone_pts)
        self._min_zone_pts = float(min_zone_pts)
        self._cached_opens:  Optional[np.ndarray] = None
        self._cached_highs:  Optional[np.ndarray] = None
        self._cached_lows:   Optional[np.ndarray] = None
        self._cached_closes: Optional[np.ndarray] = None
        self._lr_open:  Optional[np.ndarray] = None
        self._lr_high:  Optional[np.ndarray] = None
        self._lr_low:   Optional[np.ndarray] = None
        self._lr_close: Optional[np.ndarray] = None

    @property
    def obs_dim(self) -> int:
        """Total length of the observation vector."""
        return self.price_history_len * 4 + _N_STRUCTURED + _N_ENGINEERED

    def prepare_episode(self, bars: pd.DataFrame) -> None:
        """Pre-extract numpy arrays once per episode reset."""
        self._cached_opens  = bars["open"].to_numpy(dtype=np.float64)
        self._cached_highs  = bars["high"].to_numpy(dtype=np.float64)
        self._cached_lows   = bars["low"].to_numpy(dtype=np.float64)
        self._cached_closes = bars["close"].to_numpy(dtype=np.float64)
        n  = len(self._cached_closes)
        cv = self.clip_value
        prev_c = np.empty(n, dtype=np.float64)
        prev_c[0]  = self._cached_closes[0]
        prev_c[1:] = self._cached_closes[:-1]
        np.maximum(prev_c, 1e-6, out=prev_c)
        self._lr_close = np.clip(np.log(np.maximum(self._cached_closes, 1e-6) / prev_c), -cv, cv)
        self._lr_open  = np.clip(np.log(np.maximum(self._cached_opens,  1e-6) / prev_c), -cv, cv)
        self._lr_high  = np.clip(np.log(np.maximum(self._cached_highs,  1e-6) / prev_c), -cv, cv)
        self._lr_low   = np.clip(np.log(np.maximum(self._cached_lows,   1e-6) / prev_c), -cv, cv)
        self._lr_close[0] = self._lr_open[0] = self._lr_high[0] = self._lr_low[0] = 0.0

    def build(
        self,
        bars: pd.DataFrame,
        current_bar_idx: int,
        atr_state: ATRState,
        zone_state: ZoneState,
        order_zone_state: OrderZoneState,
        portfolio_state: dict,
        session_info: dict,
        pending_order: dict | None = None,
        harmonic_state=None,   # kept for signature compat — ignored
    ) -> np.ndarray:
        """Build and return the observation vector for the current bar."""
        cv  = self.clip_value
        atr = max(atr_state.atr_daily, 1.0)

        if self._cached_closes is not None:
            opens_np  = self._cached_opens
            highs_np  = self._cached_highs
            lows_np   = self._cached_lows
            closes_np = self._cached_closes
        else:
            opens_np  = bars["open"].to_numpy(dtype=np.float64)
            highs_np  = bars["high"].to_numpy(dtype=np.float64)
            lows_np   = bars["low"].to_numpy(dtype=np.float64)
            closes_np = bars["close"].to_numpy(dtype=np.float64)

        current_close = float(closes_np[current_bar_idx])
        features: list = []

        # ── 1. Recent price history (log returns) ─────────────────────────────
        n = self.price_history_len
        if self._lr_close is not None:
            start = current_bar_idx - n + 1
            if start >= 0:
                price_block = np.column_stack([
                    self._lr_open[start:current_bar_idx + 1],
                    self._lr_high[start:current_bar_idx + 1],
                    self._lr_low[start:current_bar_idx + 1],
                    self._lr_close[start:current_bar_idx + 1],
                ])
            else:
                pad = -start
                price_block = np.column_stack([
                    np.concatenate([np.zeros(pad), self._lr_open[:current_bar_idx + 1]]),
                    np.concatenate([np.zeros(pad), self._lr_high[:current_bar_idx + 1]]),
                    np.concatenate([np.zeros(pad), self._lr_low[:current_bar_idx + 1]]),
                    np.concatenate([np.zeros(pad), self._lr_close[:current_bar_idx + 1]]),
                ])
            features.extend(price_block.ravel().tolist())
        else:
            indices    = np.arange(current_bar_idx - n + 1, current_bar_idx + 1)
            valid_mask = indices >= 0
            safe_idx   = np.maximum(indices, 0)
            prev_idx   = np.maximum(indices - 1, 0)
            prev_closes = np.where(
                valid_mask & (indices > 0), closes_np[prev_idx], closes_np[safe_idx]
            )
            prev_closes = np.maximum(prev_closes, 1e-6)
            def _lr_vec(prices: np.ndarray) -> np.ndarray:
                return np.where(
                    valid_mask,
                    np.clip(np.log(np.maximum(prices[safe_idx], 1e-6) / prev_closes), -cv, cv),
                    0.0,
                )
            price_block = np.column_stack(
                [_lr_vec(opens_np), _lr_vec(highs_np), _lr_vec(lows_np), _lr_vec(closes_np)]
            )
            features.extend(price_block.ravel().tolist())

        # ── 2. ATR features ───────────────────────────────────────────────────
        atr_dict = atr_state.as_feature_dict()
        features.extend([
            atr_dict["atr_pct_used"],
            atr_dict["atr_remaining_norm"],
            atr_dict["atr_short_exhausted"],
            atr_dict["atr_long_exhausted"],
        ])

        # ── 3. Zone features ──────────────────────────────────────────────────
        supply = zone_state.nearest_supply if zone_state else None
        demand = zone_state.nearest_demand if zone_state else None

        max_zone_pts = self._max_zone_pts
        min_zone_pts = self._min_zone_pts
        if supply is not None:
            w = supply.top - supply.bottom
            if w > max_zone_pts or w < min_zone_pts:
                supply = None
        if demand is not None:
            w = demand.top - demand.bottom
            if w > max_zone_pts or w < min_zone_pts:
                demand = None

        def _dist_norm(z) -> float:
            if z is None or not z.is_valid:
                return 0.0
            return float(np.clip(abs(current_close - z.midpoint) / max(atr, 1.0), 0.0, 5.0))

        in_supply = 1.0 if (supply and supply.is_valid
                            and supply.bottom <= current_close <= supply.top) else 0.0
        in_demand = 1.0 if (demand and demand.is_valid
                            and demand.bottom <= current_close <= demand.top) else 0.0

        def _width_norm(z) -> float:
            if z is None or not z.is_valid:
                return 0.0
            return float(np.clip((z.top - z.bottom) / max(atr, 1.0), 0.0, 2.0))

        _max_zone_age = self._max_zone_age

        def _age_norm(z, current_idx: int) -> float:
            if z is None or not z.is_valid:
                return 1.0
            return float(np.clip((current_idx - z.bar_formed_idx) / _max_zone_age, 0.0, 1.0))

        def _sweep_weight(z, current_idx: int) -> float:
            """Sweep recency weight: 1.0 = just swept, decays to 0.0 over _SWEEP_DECAY_BARS."""
            if z is None or not z.is_valid or not z.was_swept:
                return 0.0
            bars_since = current_idx - z.bar_formed_idx
            return float(max(0.0, 1.0 - bars_since / _SWEEP_DECAY_BARS))

        features.extend([
            _dist_norm(supply),
            _dist_norm(demand),
            in_supply,
            in_demand,
            _width_norm(supply),
            _width_norm(demand),
            _age_norm(supply, current_bar_idx),
            _age_norm(demand, current_bar_idx),
            _sweep_weight(supply, current_bar_idx),
            _sweep_weight(demand, current_bar_idx),
        ])

        # ── 4. Order zone / confluence features ──────────────────────────────
        po_active    = 0.0
        po_direction = 0.0
        po_dist_norm = 0.0
        if pending_order is not None:
            po_active    = 1.0
            po_direction = float(pending_order["direction"])
            dist = (pending_order["limit_price"] - current_close) * pending_order["direction"]
            po_dist_norm = float(np.clip(dist / atr, -1.0, 1.0))

        features.extend([
            float(np.clip(order_zone_state.confluence_score, 0.0, 1.0)),
            float(order_zone_state.in_bearish_order_zone),
            float(order_zone_state.in_bullish_order_zone),
            float(order_zone_state.zone_type.value == "bearish"),
            float(order_zone_state.zone_type.value == "bullish"),
            float(np.clip(order_zone_state.rr_ratio / 10.0, 0.0, 1.0)),
            float(order_zone_state.trade_worthwhile),
            po_active,
            po_direction,
            po_dist_norm,
        ])

        # ── 5. Portfolio state ────────────────────────────────────────────────
        pos_dir = portfolio_state.get("position_direction", "FLAT")
        features.extend([
            float(portfolio_state.get("position_open", False)),
            float(pos_dir == "LONG"),
            float(pos_dir == "SHORT"),
            float(np.clip(portfolio_state.get("current_pnl_r", 0.0)         /  5.0, -1.0, 1.0)),
            float(np.clip(portfolio_state.get("daily_pnl_r", 0.0)           / 10.0, -1.0, 1.0)),
            float(np.clip(portfolio_state.get("max_drawdown_remaining", 1.0) /  5.0,  0.0, 1.0)),
            float(np.clip(portfolio_state.get("trades_today", 0)             /  5.0,  0.0, 1.0)),
            float(np.clip(portfolio_state.get("consecutive_losses", 0)       /  3.0,  0.0, 1.0)),
        ])

        # ── 6. Session timing + market context ───────────────────────────────
        t = float(np.clip(session_info.get("session_time_pct", 0.5), 0.0, 1.0))
        features.extend([
            t,
            float(np.clip(session_info.get("bars_remaining_pct", 0.5), 0.0, 1.0)),
            float(session_info.get("is_rth", 0.0)),
            float(np.clip(session_info.get("rth_time_pct", 0.0), 0.0, 1.0)),
            math.sin(2.0 * math.pi * t),
            math.cos(2.0 * math.pi * t),
        ])

        # ── 7. Engineered features ────────────────────────────────────────────

        # — Price location (5)
        features.extend([
            float(np.clip((current_close - atr_state.session_open)   / atr, -2.0, 2.0)),
            float(np.clip((atr_state.session_high - current_close)    / atr,  0.0, 2.0)),
            float(np.clip((current_close - atr_state.session_low)     / atr,  0.0, 2.0)),
            float(np.clip((atr_state.prior_day_high - current_close)  / atr, -2.0, 2.0)),
            float(np.clip((current_close - atr_state.prior_day_low)   / atr, -2.0, 2.0)),
        ])

        # — Momentum (2): 5-bar and 12-bar (1h) only — LSTM handles longer-horizon
        def _multi_bar_return(lookback: int) -> float:
            past_idx = max(current_bar_idx - lookback, 0)
            past_close = float(closes_np[past_idx])
            if past_close < 1e-6:
                return 0.0
            return float(np.clip(np.log(max(current_close, 1e-6) / past_close), -cv, cv))

        r5  = _multi_bar_return(5)
        r12 = _multi_bar_return(12)
        features.extend([r5, r12])

        # — Volatility regime (2)
        i = current_bar_idx
        w5_start  = max(i - 4,  0)
        w20_start = max(i - 19, 0)

        if self._lr_close is not None:
            returns_5  = self._lr_close[w5_start + 1:i + 1]
            returns_20 = self._lr_close[w20_start + 1:i + 1]
        else:
            returns_5  = np.diff(np.log(np.maximum(closes_np[w5_start:i + 1],  1e-6)))
            returns_20 = np.diff(np.log(np.maximum(closes_np[w20_start:i + 1], 1e-6)))
        vol_5  = float(np.std(returns_5))  if len(returns_5)  > 1 else 0.0
        vol_20 = float(np.std(returns_20)) if len(returns_20) > 1 else 0.0
        vol_expansion = float(np.clip(vol_5 / max(vol_20, 1e-8), 0.0, 4.0)) / 4.0

        w5_h = highs_np[w5_start:i + 1]
        w5_l = lows_np[w5_start:i + 1]
        w5_c = closes_np[w5_start:i + 1]
        bar_range = np.maximum(w5_h - w5_l, 1e-6)
        close_in_range = float(np.mean((w5_c - w5_l) / bar_range))

        features.extend([vol_expansion, close_in_range])

        # — Bar character (1)
        w5_o = opens_np[w5_start:i + 1]
        body_ratio = float(np.mean(np.abs(w5_c - w5_o) / np.maximum(w5_h - w5_l, 1e-6)))
        features.append(float(np.clip(body_ratio, 0.0, 1.0)))

        # — HTF context (5)
        w12_start = max(i - 11, 0)
        w12_h = highs_np[w12_start:i + 1].max()
        w12_l = lows_np[w12_start:i + 1].min()
        cir_1h = float(np.clip((current_close - w12_l) / max(w12_h - w12_l, 1e-6), 0.0, 1.0))

        w24_start = max(i - 23, 0)
        w24_h = highs_np[w24_start:i + 1].max()
        w24_l = lows_np[w24_start:i + 1].min()
        cir_2h = float(np.clip((current_close - w24_l) / max(w24_h - w24_l, 1e-6), 0.0, 1.0))

        prior_range = max(atr_state.prior_day_range, 1e-6)
        prior_range_pos = float(np.clip(
            (current_close - atr_state.prior_day_low) / prior_range, 0.0, 1.0
        ))

        signs = np.sign([r5, r12])
        coherence = float(np.sum(signs)) / 2.0   # range [-1, 1]

        if self._lr_close is not None:
            returns_12 = self._lr_close[w12_start + 1:i + 1]
            returns_24 = self._lr_close[w24_start + 1:i + 1]
        else:
            returns_12 = np.diff(np.log(np.maximum(closes_np[w12_start:i + 1], 1e-6)))
            returns_24 = np.diff(np.log(np.maximum(closes_np[w24_start:i + 1], 1e-6)))
        htf_vol_12 = float(np.std(returns_12)) if len(returns_12) > 1 else 0.0
        htf_vol_24 = float(np.std(returns_24)) if len(returns_24) > 1 else 0.0
        htf_vol_expansion = float(np.clip(htf_vol_12 / max(htf_vol_24, 1e-8), 0.0, 4.0)) / 4.0

        features.extend([cir_1h, cir_2h, prior_range_pos, coherence, htf_vol_expansion])

        # ── Assemble and sanitise ─────────────────────────────────────────────
        obs = np.array(features, dtype=np.float32)
        obs = np.clip(obs, -cv, cv)
        obs = np.nan_to_num(obs, nan=0.0, posinf=cv, neginf=-cv)

        return obs
