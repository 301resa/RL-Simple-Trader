"""
features/observation_builder.py
=================================
Assembles the neural network observation vector from all feature states.

The observation is a flat float32 numpy array containing:
  1. Recent OHLCV price history — 60-bar sliding window (log-returns + vol ratio)
  2. ATR features (exhaustion, remaining room)
  3. Zone features (distance to supply/demand, in-zone flags)
  4. Order zone / confluence features (score, R:R, pending order state)
  5. Portfolio state (position, P&L, drawdown, trade counts)
  6. Session timing (fraction of session elapsed / remaining)
  7. Engineered features — 11 pre-computed signals replacing raw 500-bar history:
       Price location (5): session drift, dist from session high/low, prior day high/low
       Momentum (3): 5-bar, 15-bar, 30-bar log-returns
       Volatility regime (2): short/long vol ratio, avg close-in-range
       Bar character (1): avg candle body/range ratio

Total fixed features: 28 + 11 = 39
Observation vector size: 60 × 5 + 39 = 339
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from features.atr_calculator import ATRState
from features.zone_detector import ZoneState
from features.order_zone_engine import OrderZoneState

# Number of engineered features appended after the 28 structured features
_N_ENGINEERED = 11


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
    ) -> None:
        self.normalize_observations = normalize_observations
        self.clip_value = clip_value
        self.price_history_len = lookback_bars
        # Per-episode numpy cache — populated by prepare_episode()
        self._cached_opens:      Optional[np.ndarray] = None
        self._cached_highs:      Optional[np.ndarray] = None
        self._cached_lows:       Optional[np.ndarray] = None
        self._cached_closes:     Optional[np.ndarray] = None
        self._cached_vol_ratios: Optional[np.ndarray] = None

    @property
    def obs_dim(self) -> int:
        """Total length of the observation vector (deterministic from config).

        Structured features (28):
          ATR(4) + Zone(4) + OrderZone(10) + Portfolio(8) + Session(2)
        Engineered features (11):
          PriceLocation(5) + Momentum(3) + VolRegime(2) + BarCharacter(1)
        Price window:
          lookback_bars × 5 OHLCV log-returns
        """
        return self.price_history_len * 5 + 28 + _N_ENGINEERED

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
            end_idx   = np.arange(n, dtype=np.int64)
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
        liquidity_state=None,       # kept for API compatibility — ignored
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
            cached_vr = self._cached_vol_ratios
        else:
            opens_np  = bars["open"].to_numpy(dtype=np.float64)
            highs_np  = bars["high"].to_numpy(dtype=np.float64)
            lows_np   = bars["low"].to_numpy(dtype=np.float64)
            closes_np = bars["close"].to_numpy(dtype=np.float64)
            cached_vr = None

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

        # ── 2. ATR features ───────────────────────────────────────────────────
        atr_dict = atr_state.as_feature_dict()
        features.extend([
            atr_dict["atr_pct_used"],
            atr_dict["atr_remaining_norm"],
            atr_dict["atr_short_exhausted"],
            atr_dict["atr_long_exhausted"],
        ])

        # ── 3. Zone features ──────────────────────────────────────────────────
        zone_dict = zone_state.as_feature_dict(current_close, atr)
        features.extend([
            zone_dict["supply_zone_dist_norm"],
            zone_dict["demand_zone_dist_norm"],
            zone_dict["in_supply_zone"],
            zone_dict["in_demand_zone"],
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

        # ── 6. Session timing ─────────────────────────────────────────────────
        features.extend([
            float(np.clip(session_info.get("session_time_pct",   0.5), 0.0, 1.0)),
            float(np.clip(session_info.get("bars_remaining_pct", 0.5), 0.0, 1.0)),
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

        # — Momentum (3): multi-timeframe log-returns (5 / 15 / 30 bars back)
        def _multi_bar_return(lookback: int) -> float:
            past_idx = max(current_bar_idx - lookback, 0)
            past_close = float(closes_np[past_idx])
            if past_close < 1e-6:
                return 0.0
            return float(np.clip(np.log(max(current_close, 1e-6) / past_close), -cv, cv))

        features.extend([
            _multi_bar_return(5),
            _multi_bar_return(15),
            _multi_bar_return(30),
        ])

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

        # ── Assemble and sanitise ─────────────────────────────────────────────
        obs = np.array(features, dtype=np.float32)
        obs = np.clip(obs, -cv, cv)
        obs = np.nan_to_num(obs, nan=0.0, posinf=cv, neginf=-cv)

        return obs
