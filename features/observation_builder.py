"""
features/observation_builder.py
=================================
Assembles the neural network observation vector from all feature states.

The observation is a flat float32 numpy array containing:
  1. Recent OHLC price history — 60-bar sliding window (log-returns, no volume)
  2. ATR features (exhaustion, remaining room)                             [4]
  3. Zone features (distance, in-zone, width, age, swept — supply+demand) [10]
  4. Order zone / confluence features (score, R:R, pending order state)   [10]
  5. Portfolio state (position, P&L, drawdown, trade counts)              [8]
  6. Session timing + market context (elapsed, remaining, RTH flag, RTH time) [4]
  7. Engineered features — 17 pre-computed signals:                       [17]
       Price location (5): session drift, dist from session high/low, prior day high/low
       Momentum (4): 5-bar, 12-bar (1h), 15-bar, 30-bar log-returns
       Volatility regime (2): short/long vol ratio, avg close-in-range
       Bar character (1): avg candle body/range ratio
       HTF context (5): 1h close-in-range, 2h close-in-range, prior day range position,
                        multi-TF momentum coherence, HTF vol expansion

Note: volume is intentionally excluded — the strategy is price-structure based
and volume data was found to add noise rather than signal.

Total fixed features: 36 + 17 = 53
Observation vector size: 60 × 4 + 53 = 293

Zone width/age features give the agent context on zone quality — tight fresh zones
trade differently from wide stale ones.  Wide zones (>10 pts) are zeroed out in
the order-zone observation so the agent never learns to chase untradeable signals.
The RTH features let the agent distinguish pre-market drift from regular-hours structure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from features.atr_calculator import ATRState
from features.zone_detector import ZoneState
from features.order_zone_engine import MAX_ZONE_WIDTH_PTS, OrderZoneState

# Structured feature counts
_N_STRUCTURED  = 36   # ATR(4) + Zone(10) + OrderZone(10) + Portfolio(8) + Session(4)
_N_ENGINEERED  = 17   # PriceLocation(5) + Momentum(4) + VolRegime(2) + BarCharacter(1) + HTFContext(5)


class ObservationBuilder:
    """
    Assembles the flat observation vector fed to the policy network.

    Parameters
    ----------
    normalize_observations : bool
        Reserved for future running-normalisation integration.
    clip_value : float
        All observation components are clipped to [-clip_value, clip_value].
        Also used by TradingEnv as the observation_space bound.
    lookback_bars : int
        Number of most recent bars to include as OHLCV price features.
        Zone detector continues to use its own 500-bar history context.
    """

    def __init__(
        self,
        normalize_observations: bool = False,
        clip_value: float = 10.0,
        lookback_bars: int = 60,
        max_zone_age_bars: int = 300,
    ) -> None:
        self.normalize_observations = normalize_observations
        self.clip_value = clip_value
        self.price_history_len = lookback_bars
        self._max_zone_age = float(max_zone_age_bars)
        # Per-episode numpy cache — populated by prepare_episode()
        self._cached_opens:  Optional[np.ndarray] = None
        self._cached_highs:  Optional[np.ndarray] = None
        self._cached_lows:   Optional[np.ndarray] = None
        self._cached_closes: Optional[np.ndarray] = None

    @property
    def obs_dim(self) -> int:
        """Total length of the observation vector (deterministic from config).

        Structured features (36):
          ATR(4) + Zone(10) + OrderZone(10) + Portfolio(8) + Session(4)
        Engineered features (17):
          PriceLocation(5) + Momentum(4) + VolRegime(2) + BarCharacter(1) + HTFContext(5)
        Price window:
          lookback_bars × 4 OHLC log-returns (volume excluded)
        """
        return self.price_history_len * 4 + _N_STRUCTURED + _N_ENGINEERED

    def prepare_episode(self, bars: pd.DataFrame) -> None:
        """
        Pre-extract numpy arrays for an episode's bar DataFrame.  Call once per
        episode (after reset) so build() does not repeat .to_numpy() each step.

        Parameters
        ----------
        bars : pd.DataFrame
            The combined history+session DataFrame (_combined_bars) for this episode.
        """
        self._cached_opens  = bars["open"].to_numpy(dtype=np.float64)
        self._cached_highs  = bars["high"].to_numpy(dtype=np.float64)
        self._cached_lows   = bars["low"].to_numpy(dtype=np.float64)
        self._cached_closes = bars["close"].to_numpy(dtype=np.float64)

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
    ) -> np.ndarray:
        """
        Build and return the observation vector for the current bar.

        All values are finite float32, clipped to [-clip_value, clip_value].

        Returns
        -------
        np.ndarray, dtype=float32
        """
        cv  = self.clip_value
        atr = max(atr_state.atr_daily, 1.0)

        # ── Use per-episode cached arrays when available ──────────────────────
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
        indices    = np.arange(current_bar_idx - n + 1, current_bar_idx + 1)
        valid_mask = indices >= 0
        safe_idx   = np.maximum(indices, 0)
        prev_idx   = np.maximum(indices - 1, 0)

        prev_closes = np.where(
            valid_mask & (indices > 0),
            closes_np[prev_idx],
            closes_np[safe_idx],
        )
        prev_closes = np.maximum(prev_closes, 1e-6)

        def _lr_vec(prices: np.ndarray) -> np.ndarray:
            return np.where(
                valid_mask,
                np.clip(np.log(np.maximum(prices[safe_idx], 1e-6) / prev_closes), -cv, cv),
                0.0,
            )

        lr_open  = _lr_vec(opens_np)
        lr_high  = _lr_vec(highs_np)
        lr_low   = _lr_vec(lows_np)
        lr_close = _lr_vec(closes_np)

        price_block = np.column_stack([lr_open, lr_high, lr_low, lr_close])
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

        if supply and (supply.top - supply.bottom) > MAX_ZONE_WIDTH_PTS:
            supply = None
        if demand and (demand.top - demand.bottom) > MAX_ZONE_WIDTH_PTS:
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

        _max_zone_age = self._max_zone_age  # local capture avoids repeated attr lookup

        def _age_norm(z, current_idx: int) -> float:
            if z is None or not z.is_valid:
                return 1.0  # unknown age → treat as stale
            return float(np.clip((current_idx - z.bar_formed_idx) / _max_zone_age, 0.0, 1.0))

        supply_swept = 1.0 if (supply and supply.is_valid and supply.was_swept) else 0.0
        demand_swept = 1.0 if (demand and demand.is_valid and demand.was_swept) else 0.0

        features.extend([
            _dist_norm(supply),
            _dist_norm(demand),
            in_supply,
            in_demand,
            _width_norm(supply),    # B1: tight zone = smaller value = better
            _width_norm(demand),
            _age_norm(supply, current_bar_idx),   # B1: 0=fresh, 1=expired
            _age_norm(demand, current_bar_idx),
            supply_swept,           # 1.0 once price has swept supply zone.top
            demand_swept,           # 1.0 once price has swept demand zone.bottom
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
        # is_rth: 1 during Regular Trading Hours (09:30–16:00 ET), 0 in Globex/pre-market.
        # rth_time_pct: 0→1 across the RTH session; 0 outside RTH.
        # These let the agent distinguish pre-market range-building from RTH price action.
        features.extend([
            float(np.clip(session_info.get("session_time_pct",   0.5), 0.0, 1.0)),
            float(np.clip(session_info.get("bars_remaining_pct", 0.5), 0.0, 1.0)),
            float(session_info.get("is_rth", 0.0)),
            float(np.clip(session_info.get("rth_time_pct", 0.0), 0.0, 1.0)),
        ])

        # ── 7. Engineered features (11) ───────────────────────────────────────
        # These replace the information previously carried by the 500-bar raw
        # window.  The zone detector still uses 500 bars internally — these
        # features give the network the same structural context in compact form.

        # — Price location (5): where is price relative to key reference levels?
        #   All normalised by daily ATR, clipped to [-2, 2].
        features.extend([
            float(np.clip((current_close - atr_state.session_open)   / atr, -2.0, 2.0)),
            float(np.clip((atr_state.session_high - current_close)    / atr,  0.0, 2.0)),
            float(np.clip((current_close - atr_state.session_low)     / atr,  0.0, 2.0)),
            float(np.clip((atr_state.prior_day_high - current_close)  / atr, -2.0, 2.0)),
            float(np.clip((current_close - atr_state.prior_day_low)   / atr, -2.0, 2.0)),
        ])

        # — Momentum (4): multi-timeframe log-returns (5 / 12(1h) / 15 / 30 bars back)
        def _multi_bar_return(lookback: int) -> float:
            past_idx = max(current_bar_idx - lookback, 0)
            past_close = float(closes_np[past_idx])
            if past_close < 1e-6:
                return 0.0
            return float(np.clip(np.log(max(current_close, 1e-6) / past_close), -cv, cv))

        r5  = _multi_bar_return(5)
        r12 = _multi_bar_return(12)   # 1h (12 × 5min)
        r15 = _multi_bar_return(15)
        r30 = _multi_bar_return(30)
        features.extend([r5, r12, r15, r30])

        # — Volatility regime (2): short-vol / long-vol expansion ratio + close-in-range
        i = current_bar_idx
        w5_start  = max(i - 4,  0)
        w20_start = max(i - 19, 0)

        returns_5  = np.diff(np.log(np.maximum(closes_np[w5_start:i + 1],  1e-6)))
        returns_20 = np.diff(np.log(np.maximum(closes_np[w20_start:i + 1], 1e-6)))
        vol_5  = float(np.std(returns_5))  if len(returns_5)  > 1 else 0.0
        vol_20 = float(np.std(returns_20)) if len(returns_20) > 1 else 0.0
        vol_expansion = float(np.clip(vol_5 / max(vol_20, 1e-8), 0.0, 4.0)) / 4.0

        # Average close-in-range over last 5 bars (1.0 = always closing at top)
        w5_h = highs_np[w5_start:i + 1]
        w5_l = lows_np[w5_start:i + 1]
        w5_c = closes_np[w5_start:i + 1]
        bar_range = np.maximum(w5_h - w5_l, 1e-6)
        close_in_range = float(np.mean((w5_c - w5_l) / bar_range))

        features.extend([vol_expansion, close_in_range])

        # — Bar character (1): avg candle body/range ratio over last 5 bars
        #   1.0 = full-body decisive candles, 0.0 = all doji/wicks
        w5_o = opens_np[w5_start:i + 1]
        body_ratio = float(np.mean(np.abs(w5_c - w5_o) / np.maximum(w5_h - w5_l, 1e-6)))
        features.append(float(np.clip(body_ratio, 0.0, 1.0)))

        # — HTF context (5): higher-timeframe structure derived from 5-min bars ──
        #   Gives the agent multi-hour context without requiring separate HTF data.

        # 1h close-in-range: where is price in the last 12 bars' H-L range?
        #   1.0 = at top (bullish), 0.0 = at bottom (bearish)
        w12_start = max(i - 11, 0)
        w12_h = highs_np[w12_start:i + 1].max()
        w12_l = lows_np[w12_start:i + 1].min()
        cir_1h = float((current_close - w12_l) / max(w12_h - w12_l, 1e-6))
        cir_1h = float(np.clip(cir_1h, 0.0, 1.0))

        # 2h close-in-range: same over last 24 bars
        w24_start = max(i - 23, 0)
        w24_h = highs_np[w24_start:i + 1].max()
        w24_l = lows_np[w24_start:i + 1].min()
        cir_2h = float((current_close - w24_l) / max(w24_h - w24_l, 1e-6))
        cir_2h = float(np.clip(cir_2h, 0.0, 1.0))

        # Prior day range position: where is price relative to yesterday's H-L?
        #   0.5 = midpoint, >0.5 = upper half (bullish bias), <0.5 = lower half
        prior_range = max(atr_state.prior_day_range, 1e-6)
        prior_range_pos = float(np.clip(
            (current_close - atr_state.prior_day_low) / prior_range, 0.0, 1.0
        ))

        # Multi-TF momentum coherence: are 5-bar, 12-bar and 30-bar returns aligned?
        #   +1 = all three pointing same direction (strong trend)
        #    0 = mixed signals
        #   -1 = all three pointing same direction (strong counter-trend context)
        signs = np.sign([r5, r12, r30])
        coherence = float(np.sum(signs)) / 3.0   # range [-1, 1]

        # HTF volatility expansion: 1h vol vs 2h vol (is the market accelerating?)
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
