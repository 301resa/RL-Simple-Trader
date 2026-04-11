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
        # Per-episode numpy cache — populated by prepare_episode()
        self._cached_opens:   Optional[np.ndarray] = None
        self._cached_highs:   Optional[np.ndarray] = None
        self._cached_lows:    Optional[np.ndarray] = None
        self._cached_closes:  Optional[np.ndarray] = None
        self._cached_vol_ratios: Optional[np.ndarray] = None  # rolling-20 avg per bar

    @property
    def obs_dim(self) -> int:
        """Total length of the observation vector (deterministic from config).

        Fixed features (28):
          ATR(4) + Zone(4) + OrderZone(10) + Portfolio(8) + Session(2)
        Liquidity/sweep features removed — LSTM learns sweep context from raw price.
        Trend features removed — LSTM carries directional memory across candles.
        """
        return self.price_history_len * 5 + 28

    def prepare_episode(self, bars: pd.DataFrame) -> None:
        """
        Pre-extract numpy arrays and precompute rolling volume averages for an
        episode's bar DataFrame.  Call once per episode (after reset) so that
        build() does not repeat .to_numpy() and rolling-average work each step.

        Parameters
        ----------
        bars : pd.DataFrame
            The combined history+session DataFrame (_combined_bars) for this episode.
        """
        self._cached_opens  = bars["open"].to_numpy(dtype=np.float64)
        self._cached_highs  = bars["high"].to_numpy(dtype=np.float64)
        self._cached_lows   = bars["low"].to_numpy(dtype=np.float64)
        self._cached_closes = bars["close"].to_numpy(dtype=np.float64)

        # Vectorised rolling 20-bar average volume (for bars BEFORE each position).
        # Uses cumsum trick: O(n) instead of O(n × window) per episode.
        if "volume" in bars.columns:
            vols = bars["volume"].to_numpy(dtype=np.float64)
            cumvol = np.concatenate([[0.0], np.cumsum(vols)])
            n = len(vols)
            end_idx   = np.arange(n, dtype=np.int64)          # exclusive end = idx
            start_idx = np.maximum(end_idx - 20, 0)
            window_sum = cumvol[end_idx] - cumvol[start_idx]
            window_len = np.maximum(end_idx - start_idx, 1)
            avg_before = np.where(end_idx > start_idx, window_sum / window_len, vols)
            self._cached_vol_ratios = np.clip(vols / np.maximum(avg_before, 1.0), 0.0, 5.0)
        else:
            n = len(bars)
            self._cached_vol_ratios = np.zeros(n, dtype=np.float64)

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
        cv  = self.clip_value
        atr = max(atr_state.atr_daily, 1.0)

        # ── Use per-episode cached arrays when available ──────
        # prepare_episode() pre-extracts these once per reset(); falling back
        # to on-the-fly extraction if called without prior prepare_episode().
        if self._cached_closes is not None:
            opens_np      = self._cached_opens
            highs_np      = self._cached_highs
            lows_np       = self._cached_lows
            closes_np     = self._cached_closes
            cached_vr     = self._cached_vol_ratios
        else:
            opens_np  = bars["open"].to_numpy(dtype=np.float64)
            highs_np  = bars["high"].to_numpy(dtype=np.float64)
            lows_np   = bars["low"].to_numpy(dtype=np.float64)
            closes_np = bars["close"].to_numpy(dtype=np.float64)
            cached_vr = None

        current_close = float(closes_np[current_bar_idx])
        features: list = []

        # ── 1. Recent price history (log returns) ────────────
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

        # Volume ratios — use pre-computed per-episode array when available
        if cached_vr is not None:
            vol_ratios = np.where(valid_mask, cached_vr[safe_idx], 0.0)
        else:
            vols_np = bars["volume"].to_numpy(dtype=np.float64) if "volume" in bars.columns else None
            if vols_np is not None:
                vol_ratios = np.zeros(n, dtype=np.float64)
                for i, idx in enumerate(safe_idx):
                    if valid_mask[i]:
                        vol = vols_np[idx]
                        s   = max(0, idx - 20)
                        avg = float(np.mean(vols_np[s:idx])) if idx > s else float(vol)
                        vol_ratios[i] = np.clip(vol / max(avg, 1.0), 0.0, 5.0)
            else:
                vol_ratios = np.zeros(n, dtype=np.float64)

        price_block = np.column_stack([lr_open, lr_high, lr_low, lr_close, vol_ratios])
        features.extend(price_block.ravel().tolist())

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
