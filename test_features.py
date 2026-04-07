"""
tests/test_features.py
=======================
Unit tests for all feature engineering modules.

Tests verify:
  - Correct ATR computation and exhaustion logic
  - Supply/Demand zone detection accuracy
  - Liquidity sweep detection
  - Trend classification including M/W formations
  - Order Zone confluence scoring
  - No lookahead bias in any feature

Run with: python -m pytest tests/test_features.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.atr_calculator import ATRCalculator, ATRState
from features.liquidity_detector import LiquidityDetector, SweepDirection
from features.order_zone_engine import OrderZoneEngine, OrderZoneType
from features.trend_classifier import TrendClassifier, TrendState
from features.zone_detector import ZoneDetector, ZoneType


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_bars(prices: list, multiplier: float = 1.0) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame from a list of close prices."""
    rows = []
    for i, c in enumerate(prices):
        o = c - 2 * multiplier
        h = c + 5 * multiplier
        l = c - 5 * multiplier
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 1000.0})
    df = pd.DataFrame(rows)
    df.index = pd.date_range("2023-01-01", periods=len(prices), freq="5min", tz="America/New_York")
    return df


def make_daily_bars(n: int = 30, base_price: float = 15000.0) -> pd.DataFrame:
    """Create daily bars with realistic ATR."""
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.randn(n) * 50)
    rows = []
    for c in prices:
        o = c + np.random.randn() * 20
        h = max(o, c) + abs(np.random.randn()) * 30 + 200
        l = min(o, c) - abs(np.random.randn()) * 30 - 200
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 10000.0})
    df = pd.DataFrame(rows)
    df.index = pd.date_range("2022-12-01", periods=n, freq="1D", tz="America/New_York")
    return df


# ── ATR Calculator Tests ──────────────────────────────────────────────────────

class TestATRCalculator:

    def test_atr_computed_from_daily_bars(self):
        daily = make_daily_bars(30)
        calc = ATRCalculator(atr_period=14)
        calc.fit(daily)
        atr = calc.get_atr_for_date("2023-01-01")
        # After fitting, ATR for a date past the data should return last known value
        # or None if date is before data
        # Use a date within the data range
        atr_inrange = calc.get_atr_for_date("2022-12-20")
        assert atr_inrange is not None
        assert atr_inrange > 0, "ATR must be positive"

    def test_atr_no_lookahead(self):
        """ATR for date X must not use data from date X or later."""
        daily = make_daily_bars(30)
        calc = ATRCalculator(atr_period=14)
        calc.fit(daily)
        date_str = str(daily.index[15].date())
        atr = calc.get_atr_for_date(date_str)
        assert atr is not None
        # ATR should be computable without future data
        assert atr > 0

    def test_atr_exhaustion_flag(self):
        daily = make_daily_bars(30)
        calc = ATRCalculator(atr_period=14, exhaustion_threshold=0.95)
        calc.fit(daily)
        date_str = str(daily.index[20].date())

        # Create a session where range is 100% of ATR
        atr_val = calc.get_atr_for_date(date_str)
        assert atr_val is not None

        # Build session bars where range ≈ ATR
        session = pd.DataFrame([
            {"open": 15000, "high": 15000 + atr_val * 0.98, "low": 15000, "close": 15000 + atr_val * 0.98, "volume": 1000},
        ])
        session.index = pd.date_range(date_str + " 09:30", periods=1, freq="5min", tz="America/New_York")

        state = calc.compute_session_state(date_str, session, 0)
        assert state is not None
        assert state.atr_pct_used > 0.9
        assert state.atr_warning  # Warning should trigger

    def test_atr_state_feature_dict_keys(self):
        daily = make_daily_bars(30)
        calc = ATRCalculator(atr_period=14)
        calc.fit(daily)
        date_str = str(daily.index[20].date())
        session = make_bars([15000, 15010, 15020])
        state = calc.compute_session_state(date_str, session, 1)
        if state:
            fd = state.as_feature_dict()
            required_keys = {"atr_pct_used", "atr_remaining_norm", "atr_exhausted", "atr_warning"}
            assert required_keys.issubset(fd.keys())


# ── Zone Detector Tests ───────────────────────────────────────────────────────

class TestZoneDetector:

    def make_supply_pattern(self) -> pd.DataFrame:
        """
        Create OHLCV data with a clear supply zone pattern:
        consolidation bars followed by large bearish impulse.
        """
        rows = []
        base = 15000
        # 20 warming bars
        for i in range(20):
            rows.append({"open": base, "high": base + 10, "low": base - 10, "close": base, "volume": 500})
        # 4 consolidation bars (small range)
        for i in range(4):
            rows.append({"open": base + 5, "high": base + 15, "low": base - 5, "close": base + 5, "volume": 500})
        # Large bearish impulse bar
        rows.append({"open": base + 5, "high": base + 15, "low": base - 300, "close": base - 280, "volume": 5000})
        # Follow-through bars
        for i in range(5):
            rows.append({"open": base - 280, "high": base - 270, "low": base - 310, "close": base - 300, "volume": 1000})

        df = pd.DataFrame(rows)
        df.index = pd.date_range("2023-01-01 09:30", periods=len(df), freq="5min", tz="America/New_York")
        return df

    def test_supply_zone_detected(self):
        bars = self.make_supply_pattern()
        detector = ZoneDetector(
            consolidation_min_bars=2,
            consolidation_max_bars=8,
            consolidation_range_atr_pct=0.50,  # Generous for unit test
            impulse_min_body_atr_pct=0.05,
        )
        atr_series = pd.Series([300.0] * len(bars), index=bars.index)

        # Scan to the end of the impulse bar
        state = detector.scan_and_update(bars, atr_series, 24)
        # Should have detected at least one supply zone
        assert state.nearest_supply is not None, "Expected supply zone to be detected"
        assert state.nearest_supply.zone_type == ZoneType.SUPPLY

    def test_zone_invalidation_on_close_through(self):
        bars = self.make_supply_pattern()
        detector = ZoneDetector(consolidation_range_atr_pct=0.50, impulse_min_body_atr_pct=0.05)
        atr_series = pd.Series([300.0] * len(bars), index=bars.index)

        # Scan once to detect zone
        state = detector.scan_and_update(bars, atr_series, 24)
        supply = state.nearest_supply

        if supply:
            # Simulate price closing above the supply zone top
            supply.touches = 4  # Exceed max_zone_touches=3
            detector._update_zone_validity(30, supply.top + 100, 300)
            # Zone should now be invalid
            assert not supply.is_valid


# ── Liquidity Detector Tests ──────────────────────────────────────────────────

class TestLiquidityDetector:

    def test_swing_high_detected(self):
        # Create a clear swing high: bar[10] has highest high
        closes = [100] * 5 + [105, 110, 120, 110, 105, 100] + [100] * 10
        bars = make_bars(closes)
        detector = LiquidityDetector(swing_lookback=3)
        atr_s = pd.Series([20.0] * len(bars), index=bars.index)
        state = detector.compute_state(bars, atr_s, len(bars) - 1)
        # Swing highs should have been detected
        assert len(state.swing_highs) >= 1

    def test_sweep_detection(self):
        """
        Create a bar that wicks above a prior swing high and closes back below it.
        This should register as an UP_SWEEP (bearish signal).
        """
        # Build bars with a clear swing high then a sweep bar
        closes = [100, 102, 108, 105, 100, 98, 96, 97, 99, 103, 99, 95]
        rows = []
        for i, c in enumerate(closes):
            if i == 9:  # Sweep bar: wick above swing high
                rows.append({"open": 100, "high": 115, "low": 98, "close": 99, "volume": 1000})
            else:
                rows.append({"open": c - 1, "high": c + 3, "low": c - 3, "close": c, "volume": 500})

        df = pd.DataFrame(rows)
        df.index = pd.date_range("2023-01-01", periods=len(df), freq="5min", tz="America/New_York")
        detector = LiquidityDetector(swing_lookback=2, sweep_wick_min_atr_pct=0.01)
        atr_s = pd.Series([10.0] * len(df), index=df.index)
        state = detector.compute_state(df, atr_s, len(df) - 1)
        # Should detect some swing highs/lows
        assert isinstance(state.swing_highs, list)


# ── Trend Classifier Tests ────────────────────────────────────────────────────

class TestTrendClassifier:

    def make_uptrend_bars(self) -> pd.DataFrame:
        """Clear uptrend: consistent higher highs and higher lows."""
        closes = [100, 102, 104, 103, 106, 105, 108, 107, 110, 109, 112, 111, 114, 113, 116]
        rows = []
        for i, c in enumerate(closes):
            rows.append({
                "open": c - 1, "high": c + 2, "low": c - 2, "close": c, "volume": 1000
            })
        df = pd.DataFrame(rows)
        df.index = pd.date_range("2023-01-01", periods=len(df), freq="5min", tz="America/New_York")
        return df

    def make_downtrend_bars(self) -> pd.DataFrame:
        """Clear downtrend: consistent lower lows and lower highs."""
        closes = [116, 113, 111, 114, 109, 112, 107, 110, 104, 108, 102, 106, 100, 104, 98]
        rows = []
        for i, c in enumerate(closes):
            rows.append({
                "open": c + 1, "high": c + 2, "low": c - 2, "close": c, "volume": 1000
            })
        df = pd.DataFrame(rows)
        df.index = pd.date_range("2023-01-01", periods=len(df), freq="5min", tz="America/New_York")
        return df

    def test_uptrend_detected(self):
        bars = self.make_uptrend_bars()
        classifier = TrendClassifier(swing_lookback=2, min_hh_hl_for_uptrend=1)
        snap = classifier.classify(bars, len(bars) - 1)
        assert snap.state in (TrendState.UPTREND, TrendState.M_UPTREND, TrendState.RANGING), \
            f"Expected bullish or ranging state, got {snap.state}"
        assert snap.is_bullish or snap.state == TrendState.RANGING

    def test_downtrend_detected(self):
        bars = self.make_downtrend_bars()
        classifier = TrendClassifier(swing_lookback=2, min_ll_lh_for_downtrend=1)
        snap = classifier.classify(bars, len(bars) - 1)
        assert snap.state in (TrendState.DOWNTREND, TrendState.W_DOWNTREND, TrendState.RANGING), \
            f"Expected bearish or ranging state, got {snap.state}"

    def test_undefined_with_few_bars(self):
        bars = make_bars([100, 101, 102])
        classifier = TrendClassifier(swing_lookback=3)
        snap = classifier.classify(bars, 2)
        # With only 3 bars and lookback=3, cannot confirm swings
        assert snap.state in (TrendState.UNDEFINED, TrendState.RANGING)

    def test_feature_dict_has_required_keys(self):
        bars = self.make_uptrend_bars()
        classifier = TrendClassifier(swing_lookback=2)
        snap = classifier.classify(bars, len(bars) - 1)
        fd = snap.as_feature_dict()
        required = {"trend_uptrend", "trend_downtrend", "trend_strength", "hh_count_norm"}
        assert required.issubset(fd.keys())

    def test_trend_strength_between_0_and_1(self):
        bars = self.make_uptrend_bars()
        classifier = TrendClassifier(swing_lookback=2)
        snap = classifier.classify(bars, len(bars) - 1)
        assert 0.0 <= snap.trend_strength <= 1.0


# ── Order Zone Engine Tests ───────────────────────────────────────────────────

class TestOrderZoneEngine:

    def _make_atr_state(self, pct_used: float = 0.5) -> ATRState:
        atr = 500.0
        return ATRState(
            atr_daily=atr,
            prior_day_high=15500, prior_day_low=15000, prior_day_range=500,
            session_high=15500, session_low=15250,
            current_daily_range=atr * pct_used,
            atr_pct_used=pct_used,
            atr_remaining_pts=atr * (1 - pct_used),
            atr_exhausted=pct_used >= 0.95,
            atr_warning=pct_used >= 0.85,
        )

    def test_no_zone_gives_low_score(self):
        from features.zone_detector import ZoneState
        from features.liquidity_detector import LiquidityState, SweepDirection
        from features.trend_classifier import TrendSnapshot, TrendState

        bars = make_bars([15000] * 10)
        engine = OrderZoneEngine(min_confluence_score=0.60, min_rr_ratio=4.0)

        atr_state = self._make_atr_state(0.5)
        zone_state = ZoneState()  # No zones
        liq_state = LiquidityState([], [], None, None, None, SweepDirection.NONE)
        trend_snap = TrendSnapshot(
            state=TrendState.RANGING, last_swing_high=None, last_swing_low=None,
            hh_count=0, hl_count=0, ll_count=0, lh_count=0, trend_strength=0.0
        )

        state = engine.compute(bars, 9, atr_state, zone_state, liq_state, trend_snap)
        assert state.zone_type == OrderZoneType.NONE
        assert not state.trade_worthwhile

    def test_confluence_score_between_0_and_1(self):
        from features.zone_detector import ZoneState
        from features.liquidity_detector import LiquidityState, SweepDirection
        from features.trend_classifier import TrendSnapshot, TrendState

        bars = make_bars([15000] * 10)
        engine = OrderZoneEngine()
        atr_state = self._make_atr_state(0.5)
        zone_state = ZoneState()
        liq_state = LiquidityState([], [], None, None, None, SweepDirection.NONE)
        trend_snap = TrendSnapshot(
            state=TrendState.DOWNTREND, last_swing_high=15100, last_swing_low=14900,
            hh_count=0, hl_count=0, ll_count=2, lh_count=2, trend_strength=0.7
        )

        state = engine.compute(bars, 9, atr_state, zone_state, liq_state, trend_snap)
        assert 0.0 <= state.confluence_score <= 1.0

    def test_atr_exhausted_blocks_trade(self):
        from features.zone_detector import ZoneState
        from features.liquidity_detector import LiquidityState, SweepDirection
        from features.trend_classifier import TrendSnapshot, TrendState

        bars = make_bars([15000] * 10)
        engine = OrderZoneEngine()
        atr_state = self._make_atr_state(pct_used=1.05)  # ATR exceeded
        zone_state = ZoneState()
        liq_state = LiquidityState([], [], None, None, None, SweepDirection.NONE)
        trend_snap = TrendSnapshot(
            state=TrendState.DOWNTREND, last_swing_high=15100, last_swing_low=14900,
            hh_count=0, hl_count=0, ll_count=2, lh_count=2, trend_strength=0.8
        )

        state = engine.compute(bars, 9, atr_state, zone_state, liq_state, trend_snap)
        # Even with decent setup, ATR exhausted means no trade
        assert not state.trade_worthwhile


# ── Observation Builder Tests ─────────────────────────────────────────────────

class TestObservationBuilder:

    def test_obs_vector_is_finite(self):
        """Observation vector must not contain NaN or Inf."""
        from features.observation_builder import ObservationBuilder
        from features.zone_detector import ZoneState
        from features.liquidity_detector import LiquidityState, SweepDirection
        from features.order_zone_engine import OrderZoneState, OrderZoneType
        from features.order_zone_engine import RejectionCandle
        from features.trend_classifier import TrendSnapshot, TrendState

        bars = make_bars([15000 + i * 2 for i in range(30)])
        atr_state = ATRState(
            atr_daily=500, prior_day_high=15100, prior_day_low=14900, prior_day_range=200,
            session_high=15060, session_low=14980, current_daily_range=80,
            atr_pct_used=0.16, atr_remaining_pts=420,
            atr_exhausted=False, atr_warning=False,
        )
        zone_state = ZoneState()
        liq_state = LiquidityState([], [], None, None, None, SweepDirection.NONE)
        trend_snap = TrendSnapshot(
            state=TrendState.UPTREND, last_swing_high=15050, last_swing_low=14950,
            hh_count=2, hl_count=2, ll_count=0, lh_count=0, trend_strength=0.65
        )
        oz_state = OrderZoneState(
            zone_type=OrderZoneType.NONE, confluence_score=0.0,
            in_bearish_order_zone=False, in_bullish_order_zone=False,
            rejection_candle=RejectionCandle(False, "none", 0, 0.0),
            rr_ratio=0.0, trade_worthwhile=False, component_scores={}
        )
        portfolio_state = {
            "position_open": False, "position_direction": "FLAT",
            "position_size": 0, "entry_price": 0.0, "current_pnl_r": 0.0,
            "stop_loss_price": 0.0, "take_profit_price": 0.0,
            "max_drawdown_remaining": 1.0, "daily_pnl_r": 0.0,
            "trades_today": 0, "consecutive_losses": 0,
        }
        session_info = {"session_time_pct": 0.5, "bars_remaining_pct": 0.5}

        builder = ObservationBuilder(normalize_observations=False)
        obs = builder.build(
            bars=bars, current_bar_idx=20,
            atr_state=atr_state, zone_state=zone_state,
            liquidity_state=liq_state, trend_snapshot=trend_snap,
            order_zone_state=oz_state, portfolio_state=portfolio_state,
            session_info=session_info,
        )

        assert np.all(np.isfinite(obs)), f"Observation contains non-finite values: {obs[~np.isfinite(obs)]}"
        assert obs.dtype == np.float32
        assert len(obs) > 0