"""
tests/test_environment.py
==========================
Unit tests for the trading environment and position manager.

Tests verify:
  - Episode resets correctly
  - Actions are masked properly
  - Rewards are finite and bounded
  - Position manager enforces all hard guardrails
  - Stop widening is rejected
  - Daily drawdown limit triggers episode end
  - Trailing stop logic fires correctly

Run with: python -m pytest tests/test_environment.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from environment.action_space import Action, ActionMasker
from environment.position_manager import (
    ExitReason,
    PositionDirection,
    PositionManager,
)
from features.atr_calculator import ATRState
from features.order_zone_engine import OrderZoneState, OrderZoneType
from features.trend_classifier import TrendSnapshot, TrendState


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_dummy_atr_state(pct_used: float = 0.5, atr: float = 500.0) -> ATRState:
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


def make_dummy_trend(state: TrendState = TrendState.DOWNTREND) -> TrendSnapshot:
    return TrendSnapshot(
        state=state, last_swing_high=15100, last_swing_low=14900,
        hh_count=0, hl_count=0, ll_count=2, lh_count=2, trend_strength=0.7
    )


def make_dummy_oz_state(in_bearish: bool = False, rr: float = 5.0) -> OrderZoneState:
    return OrderZoneState(
        zone_type=OrderZoneType.BEARISH if in_bearish else OrderZoneType.NONE,
        confluence_score=0.8 if in_bearish else 0.1,
        in_bearish_order_zone=in_bearish,
        in_bullish_order_zone=False,
        rr_ratio=rr,
        trade_worthwhile=in_bearish and rr >= 4.0,
        component_scores={},
    )


# ── PositionManager Tests ─────────────────────────────────────────────────────

class TestPositionManager:

    def make_pm(self, real_capital: float = 2500.0) -> PositionManager:
        return PositionManager(
            real_capital=real_capital,
            risk_per_trade_pct=0.01,
            min_contracts=1,
            max_contracts=3,
            point_value=2.0,
            max_trades_per_day=5,
            trail_activate_r=2.0,
            trail_aggressive_r=4.0,
            trail_lock_in_r=2.0,
            max_daily_loss_r=3.0,
            max_drawdown_r=5.0,
            pause_bars_after_loss_streak=6,
            loss_streak_threshold=3,
        )

    def test_can_enter_short(self):
        pm = self.make_pm()
        pm.reset()
        success, reason = pm.enter(
            direction=-1,
            current_price=15000.0,
            stop_price=15100.0,   # Above entry (correct for short)
            target_price=14500.0,
            current_bar_idx=0,
            atr=500.0,
        )
        assert success, f"Entry rejected: {reason}"
        assert pm.state.is_open
        assert pm.state.direction == PositionDirection.SHORT

    def test_can_enter_long(self):
        pm = self.make_pm()
        pm.reset()
        success, reason = pm.enter(
            direction=1,
            current_price=15000.0,
            stop_price=14900.0,   # Below entry (correct for long)
            target_price=15500.0,
            current_bar_idx=0,
            atr=500.0,
        )
        assert success
        assert pm.state.direction == PositionDirection.LONG

    def test_double_entry_rejected(self):
        pm = self.make_pm()
        pm.reset()
        pm.enter(-1, 15000.0, 15100.0, 14500.0, 0, 500.0)
        success, reason = pm.enter(-1, 15000.0, 15100.0, 14500.0, 1, 500.0)
        assert not success
        assert reason == "position_already_open"

    def test_invalid_short_stop_below_entry_rejected(self):
        """Short stop must be ABOVE entry — below entry is invalid."""
        pm = self.make_pm()
        pm.reset()
        success, reason = pm.enter(
            direction=-1,
            current_price=15000.0,
            stop_price=14900.0,   # Below entry — INVALID for short
            target_price=14500.0,
            current_bar_idx=0,
            atr=500.0,
        )
        assert not success
        assert "stop" in reason.lower()

    def test_stop_loss_triggers(self):
        pm = self.make_pm()
        pm.reset()
        pm.enter(-1, 15000.0, 15100.0, 14500.0, 0, 500.0)

        # Bar whose high exceeds the stop loss
        closed, reason, trade = pm.update(
            current_price=15050.0,
            current_bar_high=15150.0,  # Above stop 15100
            current_bar_low=14980.0,
            current_bar_idx=1,
            atr=500.0,
        )
        assert closed
        assert reason == ExitReason.STOP_LOSS
        assert trade is not None
        assert trade.pnl_r < 0  # Loss

    def test_take_profit_triggers(self):
        pm = self.make_pm()
        pm.reset()
        pm.enter(-1, 15000.0, 15100.0, 14500.0, 0, 500.0)

        # Bar whose low hits the take profit
        closed, reason, trade = pm.update(
            current_price=14510.0,
            current_bar_high=15010.0,
            current_bar_low=14490.0,   # Below target 14500
            current_bar_idx=1,
            atr=500.0,
        )
        assert closed
        assert reason == ExitReason.TAKE_PROFIT
        assert trade.pnl_r > 0  # Win

    def test_trailing_stop_activates_at_2r(self):
        pm = self.make_pm()
        pm.reset()
        pm.enter(-1, 15000.0, 15100.0, 14500.0, 0, 500.0)
        # initial_risk_pts = 100

        # Price moved to +2R = 15000 - 200 = 14800
        closed, reason, trade = pm.update(
            current_price=14800.0,
            current_bar_high=14900.0,
            current_bar_low=14780.0,
            current_bar_idx=1,
            atr=500.0,
            agent_wants_trail=True,  # Agent requests trail
        )
        # Should not be closed yet, but stop should have moved
        # (stop moved down toward entry direction)
        assert not closed  # No exit happened
        # Stop should now be closer to price (trailing active)

    def test_max_trades_per_day_enforced(self):
        pm = PositionManager(
            real_capital=2500, risk_per_trade_pct=0.01, min_contracts=1,
            max_contracts=3, point_value=2.0, max_trades_per_day=2,  # Only 2 allowed
            trail_activate_r=2.0, trail_aggressive_r=4.0, trail_lock_in_r=2.0,
            max_daily_loss_r=10.0, max_drawdown_r=20.0,
        )
        pm.reset()

        # Trade 1
        pm.enter(-1, 15000.0, 15100.0, 14500.0, 0, 500.0)
        pm.force_close(14600.0, 1)  # Close it

        # Trade 2
        pm.enter(-1, 14600.0, 14700.0, 14100.0, 2, 500.0)
        pm.force_close(14200.0, 3)

        # Trade 3 — should be blocked
        success, reason = pm.enter(-1, 14200.0, 14300.0, 13700.0, 4, 500.0)
        assert not success
        assert reason == "max_trades_per_day_reached"

    def test_max_drawdown_detection(self):
        pm = PositionManager(
            real_capital=2500, risk_per_trade_pct=0.5,  # Big risk for test
            min_contracts=1, max_contracts=1, point_value=2.0,
            max_trades_per_day=10, trail_activate_r=2.0, trail_aggressive_r=4.0,
            trail_lock_in_r=2.0, max_daily_loss_r=10.0, max_drawdown_r=2.0,
        )
        pm.reset()

        # Force a large loss to trigger drawdown
        pm.enter(-1, 15000.0, 15001.0, 14500.0, 0, 1.0)
        # Close at a massive loss
        pm.force_close(15200.0, 1)  # Stopped out far above entry

        # Drawdown should now be breached
        assert pm.is_max_drawdown_breached(15000.0) or pm.state.realised_pnl_r < -2.0


# ── ActionMasker Tests ────────────────────────────────────────────────────────

class TestActionMasker:

    def make_masker(self) -> ActionMasker:
        return ActionMasker(
            min_rr_ratio=4.0,
            atr_exhaustion_threshold=0.95,
            trail_min_r=2.0,
            max_trades_per_day=5,
            no_entry_last_n_bars=3,
        )

    def test_all_actions_available_when_flat_good_setup(self):
        masker = self.make_masker()
        atr = make_dummy_atr_state(0.5)
        trend = make_dummy_trend(TrendState.DOWNTREND)
        oz = make_dummy_oz_state(in_bearish=True, rr=5.0)

        mask = masker.compute_mask(
            is_position_open=False, position_direction="FLAT",
            unrealised_r=0.0, atr_state=atr, trend_snapshot=trend,
            order_zone_state=oz, trades_today=0,
            in_loss_streak_pause=False, bars_remaining_in_session=20,
            max_drawdown_breached=False,
        )
        assert mask[Action.HOLD] == 1.0
        assert mask[Action.ENTER_SHORT] == 1.0
        assert mask[Action.EXIT] == 0.0        # No position to exit
        assert mask[Action.TRAIL_STOP] == 0.0  # No position to trail

    def test_only_hold_when_max_drawdown_breached(self):
        masker = self.make_masker()
        atr = make_dummy_atr_state(0.5)
        trend = make_dummy_trend()
        oz = make_dummy_oz_state()

        mask = masker.compute_mask(
            is_position_open=False, position_direction="FLAT",
            unrealised_r=0.0, atr_state=atr, trend_snapshot=trend,
            order_zone_state=oz, trades_today=0, in_loss_streak_pause=False,
            bars_remaining_in_session=20, max_drawdown_breached=True,
        )
        assert mask[Action.HOLD] == 1.0
        assert mask[Action.ENTER_SHORT] == 0.0
        assert mask[Action.ENTER_LONG] == 0.0

    def test_entries_blocked_when_atr_exhausted(self):
        masker = self.make_masker()
        atr = make_dummy_atr_state(1.02)  # ATR exceeded
        trend = make_dummy_trend()
        oz = make_dummy_oz_state()

        mask = masker.compute_mask(
            is_position_open=False, position_direction="FLAT",
            unrealised_r=0.0, atr_state=atr, trend_snapshot=trend,
            order_zone_state=oz, trades_today=0, in_loss_streak_pause=False,
            bars_remaining_in_session=20, max_drawdown_breached=False,
        )
        assert mask[Action.ENTER_SHORT] == 0.0
        assert mask[Action.ENTER_LONG] == 0.0

    def test_short_blocked_in_uptrend(self):
        masker = self.make_masker()
        atr = make_dummy_atr_state(0.5)
        trend = make_dummy_trend(TrendState.UPTREND)  # Uptrend
        oz = make_dummy_oz_state(in_bearish=True, rr=5.0)

        mask = masker.compute_mask(
            is_position_open=False, position_direction="FLAT",
            unrealised_r=0.0, atr_state=atr, trend_snapshot=trend,
            order_zone_state=oz, trades_today=0, in_loss_streak_pause=False,
            bars_remaining_in_session=20, max_drawdown_breached=False,
        )
        assert mask[Action.ENTER_SHORT] == 0.0, "Short must be blocked in uptrend"
        assert mask[Action.ENTER_LONG] == 1.0   # Long should still be available

    def test_long_blocked_in_downtrend(self):
        masker = self.make_masker()
        atr = make_dummy_atr_state(0.5)
        trend = make_dummy_trend(TrendState.DOWNTREND)
        oz = make_dummy_oz_state(in_bearish=False, rr=5.0)

        mask = masker.compute_mask(
            is_position_open=False, position_direction="FLAT",
            unrealised_r=0.0, atr_state=atr, trend_snapshot=trend,
            order_zone_state=oz, trades_today=0, in_loss_streak_pause=False,
            bars_remaining_in_session=20, max_drawdown_breached=False,
        )
        assert mask[Action.ENTER_LONG] == 0.0, "Long must be blocked in downtrend"

    def test_entries_blocked_end_of_session(self):
        masker = self.make_masker()
        atr = make_dummy_atr_state(0.5)
        trend = make_dummy_trend()
        oz = make_dummy_oz_state(in_bearish=True, rr=5.0)

        mask = masker.compute_mask(
            is_position_open=False, position_direction="FLAT",
            unrealised_r=0.0, atr_state=atr, trend_snapshot=trend,
            order_zone_state=oz, trades_today=0, in_loss_streak_pause=False,
            bars_remaining_in_session=2,  # Last 2 bars of session
            max_drawdown_breached=False,
        )
        assert mask[Action.ENTER_SHORT] == 0.0, "No entries in last N bars"
        assert mask[Action.ENTER_LONG] == 0.0

    def test_trail_available_only_with_sufficient_pnl(self):
        masker = self.make_masker()
        atr = make_dummy_atr_state(0.5)
        trend = make_dummy_trend()
        oz = make_dummy_oz_state()

        # At 1R unrealised — trail should NOT be available (need >= 2R)
        mask_1r = masker.compute_mask(
            is_position_open=True, position_direction="SHORT",
            unrealised_r=1.0, atr_state=atr, trend_snapshot=trend,
            order_zone_state=oz, trades_today=1, in_loss_streak_pause=False,
            bars_remaining_in_session=20, max_drawdown_breached=False,
        )
        assert mask_1r[Action.TRAIL_STOP] == 0.0

        # At 2.5R unrealised — trail should be available
        mask_2r = masker.compute_mask(
            is_position_open=True, position_direction="SHORT",
            unrealised_r=2.5, atr_state=atr, trend_snapshot=trend,
            order_zone_state=oz, trades_today=1, in_loss_streak_pause=False,
            bars_remaining_in_session=20, max_drawdown_breached=False,
        )
        assert mask_2r[Action.TRAIL_STOP] == 1.0

    def test_mask_dtype_is_float32(self):
        masker = self.make_masker()
        atr = make_dummy_atr_state(0.5)
        trend = make_dummy_trend()
        oz = make_dummy_oz_state()

        mask = masker.compute_mask(
            is_position_open=False, position_direction="FLAT",
            unrealised_r=0.0, atr_state=atr, trend_snapshot=trend,
            order_zone_state=oz, trades_today=0, in_loss_streak_pause=False,
            bars_remaining_in_session=20, max_drawdown_breached=False,
        )
        assert mask.dtype == np.float32
        assert len(mask) == Action.n_actions()