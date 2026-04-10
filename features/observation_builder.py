"""
features/observation_builder.py
=================================
Assembles the neural network observation vector from all feature states.

The observation is a flat float32 numpy array containing:
  1. Recent OHLCV price history (price normalised by ATR, volume by avg)
  2. ATR features (exhaustion, remaining room)
  3. Zone features (distance to supply/demand, in-zone flags)
  4. Order zone / confluence features (score, R:R, rejection candle)
  5. Portfolio state (position, P&L, drawdown, trade counts)
  6. Session timing (fraction of session elapsed / remaining)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from features.atr_calculator import ATRState
from features.zone_detector import ZoneState
from features.order_zone_engine import OrderZoneState


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
    price_history_len : int
        Number of most recent bars to include as OHLCV price features.
    """

    def __init__(
        self,
        normalize_observations: bool = False,
        clip_value: float = 10.0,
        lookback_bars: int = 8,
    ) -> None:
        self.normalize_observations = normalize_observations
        self.clip_value = clip_value
        self.price_history_len = lookback_bars  # internal alias

    @property
    def obs_dim(self) -> int:
        """Total length of the observation vector (deterministic from config).

        Fixed features (28):
          ATR(4) + Zone(4) + OrderZone(10) + Portfolio(8) + Session(2)
        Liquidity/sweep features removed — LSTM learns sweep context from raw price.
        Trend features removed — LSTM carries directional memory across candles.
        """
        return self.price_history_len * 5 + 28

    def build(
        self,
        bars: pd.DataFrame,
        current_bar_idx: int,
        atr_state: ATRState,
        zone_state: ZoneState,
        order_zone_state: OrderZoneState,
        portfolio_state: dict,
        session_info: dict,
        liquidity_state=None,   # kept for API compatibility — ignored
    ) -> np.ndarray:
        """
        Build and return the observation vector for the current bar.

        All values are finite float32, clipped to [-clip_value, clip_value].

        Returns
        -------
        np.ndarray, dtype=float32
        """
        cv   = self.clip_value
        atr  = max(atr_state.atr_daily, 1.0)
        current_bar  = bars.iloc[current_bar_idx]
        current_close = float(current_bar["close"])
        features: list = []

        # ── 1. Recent price history (log returns) ────────────
        # Log returns: log(price_t / price_{t-1}) are stationary,
        # zero-centred, and scale-invariant — neural nets learn faster
        # than ATR-normalised absolute offsets.
        # OHLC: log return relative to previous bar's close.
        # Volume: ratio vs 20-bar rolling average (unchanged).
        for i in range(self.price_history_len):
            idx = current_bar_idx - (self.price_history_len - 1 - i)
            if idx < 0:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            b    = bars.iloc[idx]
            # Previous bar's close as the base for log returns
            prev_close = float(bars.iloc[idx - 1]["close"]) if idx > 0 else float(b["close"])
            prev_close = max(prev_close, 1e-6)  # guard against zero

            def _lr(price: float) -> float:
                return float(np.clip(np.log(max(price, 1e-6) / prev_close), -cv, cv))

            features.append(_lr(float(b["open"])))
            features.append(_lr(float(b["high"])))
            features.append(_lr(float(b["low"])))
            features.append(_lr(float(b["close"])))
            # Volume ratio (unchanged — already a ratio, no log needed)
            vol = float(b["volume"]) if "volume" in b.index else 0.0
            vol_window = bars.iloc[max(0, idx - 20): idx]
            if "volume" in bars.columns and len(vol_window) > 0:
                avg_vol = float(vol_window["volume"].mean())
            else:
                avg_vol = max(vol, 1.0)
            features.append(float(np.clip(vol / max(avg_vol, 1.0), 0.0, 5.0)))

        # ── 2. ATR features ───────────────────────────────────
        atr_dict = atr_state.as_feature_dict()
        features.extend([
            atr_dict["atr_pct_used"],
            atr_dict["atr_remaining_norm"],
            atr_dict["atr_short_exhausted"],   # 1.0 = down move >= 85% ATR → no shorts
            atr_dict["atr_long_exhausted"],    # 1.0 = up move >= 85% ATR → no longs
        ])

        # ── 3. Zone features ──────────────────────────────────
        zone_dict = zone_state.as_feature_dict(current_close, atr)
        features.extend([
            zone_dict["supply_zone_dist_norm"],
            zone_dict["demand_zone_dist_norm"],
            zone_dict["in_supply_zone"],
            zone_dict["in_demand_zone"],
        ])

        # ── 4. Order zone / confluence features ──────────────
        # Pillar 2 (liquidity sweep) removed — LSTM learns sweep context from raw price.
        # Pillar 3 (rejection candle) removed — 3 features kept as zeros.
        features.extend([
            float(np.clip(order_zone_state.confluence_score, 0.0, 1.0)),
            float(order_zone_state.in_bearish_order_zone),
            float(order_zone_state.in_bullish_order_zone),
            float(order_zone_state.zone_type.value == "bearish"),
            float(order_zone_state.zone_type.value == "bullish"),
            float(np.clip(order_zone_state.rr_ratio / 10.0, 0.0, 1.0)),
            float(order_zone_state.trade_worthwhile),
            0.0,   # rc.detected   — removed
            0.0,   # rc.strength   — removed
            0.5,   # rc_dir_norm   — removed (neutral value)
        ])

        # ── 5. Portfolio state ────────────────────────────────
        pos_dir = portfolio_state.get("position_direction", "FLAT")
        features.extend([
            float(portfolio_state.get("position_open", False)),
            float(pos_dir == "LONG"),
            float(pos_dir == "SHORT"),
            float(np.clip(portfolio_state.get("current_pnl_r", 0.0)        /  5.0, -1.0, 1.0)),
            float(np.clip(portfolio_state.get("daily_pnl_r", 0.0)          / 10.0, -1.0, 1.0)),
            float(np.clip(portfolio_state.get("max_drawdown_remaining", 1.0) / 5.0,  0.0, 1.0)),
            float(np.clip(portfolio_state.get("trades_today", 0)            /  5.0,  0.0, 1.0)),
            float(np.clip(portfolio_state.get("consecutive_losses", 0)      /  3.0,  0.0, 1.0)),
        ])

        # ── 6. Session timing ─────────────────────────────────
        features.extend([
            float(np.clip(session_info.get("session_time_pct",    0.5), 0.0, 1.0)),
            float(np.clip(session_info.get("bars_remaining_pct",  0.5), 0.0, 1.0)),
        ])

        # ── Assemble and sanitise ─────────────────────────────
        obs = np.array(features, dtype=np.float32)
        obs = np.clip(obs, -cv, cv)
        obs = np.nan_to_num(obs, nan=0.0, posinf=cv, neginf=-cv)

        return obs
