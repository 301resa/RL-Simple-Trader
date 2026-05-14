"""
test_reward_fixes.py
====================

Unit tests for Items 3 & 4 audit fixes:
  - Item 3: Entry bonuses gate on order_placed (orders must be successfully placed)
  - Item 4: hold_flat_penalty guard on entry_is_masked (penalty suppressed when entries are masked)

Tests use minimal fixtures (ATRState, OrderZoneState dataclasses) and a bare RewardCalculator.
"""

import pytest

from features.atr_calculator import ATRState
from features.order_zone_engine import OrderZoneState, OrderZoneType
from environment.reward_calculator import RewardCalculator


def _make_rc() -> RewardCalculator:
    """Create a minimal RewardCalculator with known values for testing."""
    return RewardCalculator(
        core_scale=1.0,
        hold_flat_penalty=-0.01,
        entry_bonuses={
            "full_order_zone_confluence": 0.15,
            "in_supply_demand_zone": 0.05,
            "atr_has_room": 0.05,
            "high_rr_ratio": 0.10,
        },
        entry_penalties={
            "no_zone_present": -0.10,
            "atr_exhausted": -0.10,
            "rr_below_minimum": -0.05,
            "overtrading": -0.20,
        },
        exit_bonuses={
            "trailing_stop_correctly": 0.05,
            "aggressive_trail_at_4r": 0.10,
            "tp_hit_bonus": 0.25,
        },
        exit_penalties={},
        violations={},
        discipline={
            "loss_streak_threshold": 3,
            "re_entry_after_loss_streak_penalty": -0.10,
        },
        time_management={
            "penalty_start_bar": 6,
            "max_bars_before_penalty": 12,
            "penalty_per_bar": -0.01,
            "min_hold_bars": 3,
        },
        selectivity={
            "min_confluence_for_entry": 0.45,
            "no_trade_reward": 0.0,
            "weak_zone_penalty": -0.05,
        },
        entry_cost=-0.01,
    )


def _atr() -> ATRState:
    """ATR state: not exhausted in either direction."""
    return ATRState(
        atr_daily=20.0,
        prior_day_high=4500.0,
        prior_day_low=4480.0,
        prior_day_range=20.0,
        session_open=4490.0,
        session_high=4495.0,
        session_low=4485.0,
        current_daily_range=10.0,
        atr_pct_used=0.5,
        atr_remaining_pts=10.0,
        atr_short_exhausted=False,
        atr_long_exhausted=False,
    )


def _oz_in_bullish(score=0.9) -> OrderZoneState:
    """High-confluence bullish order zone."""
    return OrderZoneState(
        zone_type=OrderZoneType.BULLISH,
        confluence_score=score,
        in_bearish_order_zone=False,
        in_bullish_order_zone=True,
        rr_ratio=2.5,
        trade_worthwhile=True,
        component_scores={},
    )


def _oz_no_zone() -> OrderZoneState:
    """No order zone present."""
    return OrderZoneState(
        zone_type=OrderZoneType.NONE,
        confluence_score=0.0,
        in_bearish_order_zone=False,
        in_bullish_order_zone=False,
        rr_ratio=0.0,
        trade_worthwhile=False,
        component_scores={},
    )


_PORTFOLIO = {
    "trades_today": 0,
    "consecutive_losses": 0,
    "current_pnl_r": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Item 3 Tests: Entry Bonus Gate on order_placed
# ─────────────────────────────────────────────────────────────────────────────


def test_enter_long_no_order_placed_yields_no_entry_bonus():
    """
    When attempting ENTER_LONG with a valid zone but order_placed=False
    (order placement failed), entry bonuses should be 0.
    """
    rc = _make_rc()
    r = rc.step_reward(
        action=2,  # ENTER_LONG
        is_position_open=False,
        atr_state=_atr(),
        order_zone_state=_oz_in_bullish(),
        portfolio_state=_PORTFOLIO,
        order_placed=False,  # ← Order failed to place
    )
    assert r.entry_bonus == 0.0, "Entry bonus must be 0 when order_placed=False"


def test_enter_long_order_placed_yields_entry_bonus():
    """
    When attempting ENTER_LONG with a valid zone and order_placed=True,
    entry bonuses should be > 0 (confluence bonus applied).
    """
    rc = _make_rc()
    r = rc.step_reward(
        action=2,  # ENTER_LONG
        is_position_open=False,
        atr_state=_atr(),
        order_zone_state=_oz_in_bullish(score=0.9),
        portfolio_state=_PORTFOLIO,
        order_placed=True,  # ← Order placed successfully
    )
    assert r.entry_bonus > 0.0, "Entry bonus must be > 0 when order_placed=True with good zone"


def test_no_zone_penalty_fires_even_when_order_not_placed():
    """
    The no_zone_present penalty should fire unconditionally when entering
    outside a zone (regardless of order_placed). This penalizes the attempt, not the outcome.
    """
    rc = _make_rc()
    r = rc.step_reward(
        action=2,  # ENTER_LONG
        is_position_open=False,
        atr_state=_atr(),
        order_zone_state=_oz_no_zone(),  # ← No zone
        portfolio_state=_PORTFOLIO,
        order_placed=False,
    )
    assert (
        r.entry_penalty < 0.0
    ), "no_zone_present penalty must fire even when order not placed"
    assert r.entry_bonus == 0.0, "No bonuses when no zone and no order placed"


# ─────────────────────────────────────────────────────────────────────────────
# Item 4 Tests: hold_flat_penalty Guard on entry_is_masked
# ─────────────────────────────────────────────────────────────────────────────


def test_hold_in_zone_masked_no_penalty():
    """
    When HOLD is taken with a valid zone present, but entry_is_masked=True
    (entries are unavailable), hold_flat_penalty should NOT apply.
    """
    rc = _make_rc()
    r = rc.step_reward(
        action=0,  # HOLD
        is_position_open=False,
        atr_state=_atr(),
        order_zone_state=_oz_in_bullish(),
        portfolio_state=_PORTFOLIO,
        pending_order=None,
        entry_is_masked=True,  # ← Entries blocked (loss streak pause, etc.)
    )
    assert r.step_penalty == 0.0, "hold_flat_penalty must be 0 when entry_is_masked=True"


def test_hold_in_zone_no_mask_applies_penalty():
    """
    When HOLD is taken with a valid zone present and entry_is_masked=False,
    hold_flat_penalty should apply (agent could have entered but chose not to).
    """
    rc = _make_rc()
    r = rc.step_reward(
        action=0,  # HOLD
        is_position_open=False,
        atr_state=_atr(),
        order_zone_state=_oz_in_bullish(),
        portfolio_state=_PORTFOLIO,
        pending_order=None,
        entry_is_masked=False,  # ← Entries available
    )
    assert r.step_penalty == pytest.approx(
        rc.hold_flat_penalty
    ), "hold_flat_penalty must apply when entry_is_masked=False"


def test_hold_pending_order_no_penalty_regardless_of_mask():
    """
    When a pending order exists, HOLD is the correct action and should never
    incur hold_flat_penalty, regardless of entry_is_masked.
    """
    rc = _make_rc()
    r = rc.step_reward(
        action=0,  # HOLD
        is_position_open=False,
        atr_state=_atr(),
        order_zone_state=_oz_in_bullish(),
        portfolio_state=_PORTFOLIO,
        pending_order={"price": 4490.0},  # ← Pending order exists
        entry_is_masked=False,
    )
    assert (
        r.step_penalty == 0.0
    ), "hold_flat_penalty must be 0 when pending_order exists"


def test_failed_entry_combined():
    """
    When both order_placed=False AND entry_is_masked=True occur together:
    - Entry bonus should be 0 (no order placed)
    - Step penalty should be 0 (entries are masked, not a failure)
    """
    rc = _make_rc()
    r = rc.step_reward(
        action=2,  # ENTER_LONG
        is_position_open=False,
        atr_state=_atr(),
        order_zone_state=_oz_in_bullish(),
        portfolio_state=_PORTFOLIO,
        order_placed=False,
        entry_is_masked=True,
    )
    assert r.entry_bonus == 0.0, "No entry bonus when order not placed"
    assert r.step_penalty == 0.0, "No hold_flat_penalty when entry_is_masked"
