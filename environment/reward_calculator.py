"""
environment/reward_calculator.py
==================================
Reward Function — the complete signal that shapes agent behaviour.

All rewards are denominated in R-multiples (multiples of initial risk)
scaled by the episode's rolling win rate:

    core_reward = pnl_r × win_rate

  where win_rate is the smoothed episode win rate (Laplace prior → starts at
  0.5 before any trades, updates after each closed trade).

Design philosophy:
  - Core reward = pnl_r × win_rate  (outcome × consistency)
  - Shaping rewards = bonuses/penalties for PROCESS quality (following rules)
  - Violations = hard penalties for breaking risk guardrails

Every reward component is configurable via reward_config.yaml.
No magic numbers in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from environment.position_manager import ExitReason, Trade
from features.atr_calculator import ATRState
from features.order_zone_engine import OrderZoneState, OrderZoneType
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RewardBreakdown:
    """
    Detailed breakdown of reward components for a single step.
    Used for logging, debugging, and journal analysis.
    """
    total: float
    core_trade_r: float = 0.0
    entry_bonus: float = 0.0
    entry_penalty: float = 0.0
    exit_bonus: float = 0.0
    exit_penalty: float = 0.0
    step_penalty: float = 0.0
    violation_penalty: float = 0.0
    shaping_note: str = ""

    def __post_init__(self) -> None:
        # Sanity check — total should approximately equal sum of components
        computed = (
            self.core_trade_r
            + self.entry_bonus
            + self.entry_penalty
            + self.exit_bonus
            + self.exit_penalty
            + self.step_penalty
            + self.violation_penalty
        )
        if abs(computed - self.total) > 0.01:
            log.debug(
                "Reward breakdown mismatch",
                total=self.total,
                computed=round(computed, 4),
            )


class RewardCalculator:
    """
    Computes the shaped reward signal at each environment step.

    Parameters
    ----------
    core_scale : float
        Global reward scaling factor.
    hold_flat_penalty : float
        Per-bar penalty for doing nothing (anti-laziness).
    entry_bonuses : dict
        Bonuses awarded at trade entry for good setup quality.
    entry_penalties : dict
        Penalties at entry for rule violations.
    exit_bonuses : dict
        Bonuses for good exit management.
    exit_penalties : dict
        Penalties for poor exit management.
    violations : dict
        Hard penalties for guardrail breaches.
    discipline : dict
        Discipline-related shaping rewards.
    entry_cost : float
        Transaction cost proxy per entry.
    gave_back_profit_weight : float
        Weight applied to R given back from peak when exiting late.
    """

    def __init__(
        self,
        core_scale: float = 1.0,
        hold_flat_penalty: float = -0.001,
        entry_bonuses: Optional[dict] = None,
        entry_penalties: Optional[dict] = None,
        exit_bonuses: Optional[dict] = None,
        exit_penalties: Optional[dict] = None,
        violations: Optional[dict] = None,
        discipline: Optional[dict] = None,
        entry_cost: float = -0.05,
        gave_back_profit_weight: float = 0.30,
    ) -> None:
        self.core_scale = core_scale
        self.hold_flat_penalty = hold_flat_penalty
        self.entry_cost = entry_cost
        self.gave_back_profit_weight = gave_back_profit_weight

        # Shaping scale: 1.0 = full shaping, 0.0 = pure P&L only.
        # Decayed externally by ShapingDecayCallback during training.
        self.shaping_scale: float = 1.0

        # Episode win-rate tracking.  Reset each episode via reset_episode_stats().
        # win_rate = (wins + 1) / (trades + 2)  — Laplace smoothing gives a 0.5
        # prior before the first trade so the first reward is never zero.
        self._ep_wins:   int = 0
        self._ep_trades: int = 0

        # Default reward values (overridden by config)
        self.entry_bonuses = entry_bonuses or {
            "full_order_zone_confluence": 0.50,
            "in_supply_demand_zone": 0.25,
            "atr_has_room": 0.15,
            "high_rr_ratio": 0.20,
        }
        self.entry_penalties = entry_penalties or {
            "no_zone_present": -0.05,
            "atr_exhausted": -0.40,
            "rr_below_minimum": -0.08,
            "overtrading": 0.0,
        }
        self.exit_bonuses = exit_bonuses or {
            "trailing_stop_correctly": 0.30,
            "aggressive_trail_at_4r": 0.20,
            "exit_at_atr_target": 0.20,
        }
        self.exit_penalties = exit_penalties or {
            "held_past_4r_no_trail": -0.50,
        }
        self.violations = violations or {
            "max_drawdown_breach": -2.00,
            "daily_loss_limit_breach": -1.00,
        }
        self.discipline = discipline or {
            "re_entry_after_loss_streak_penalty": -0.20,
            "loss_streak_threshold": 3,
        }

    # ── Episode stats ─────────────────────────────────────────

    def reset_episode_stats(self) -> None:
        """Reset win-rate counters at the start of each episode."""
        self._ep_wins   = 0
        self._ep_trades = 0

    @property
    def _win_rate(self) -> float:
        """Smoothed episode win rate with Laplace prior (range ~0.1–0.9)."""
        return (self._ep_wins + 1) / (self._ep_trades + 2)

    # ── Step-level reward (called every bar) ─────────────────

    def step_reward(
        self,
        action: int,
        is_position_open: bool,
        atr_state: ATRState,
        order_zone_state: OrderZoneState,
        portfolio_state: dict,
        trend_snapshot=None,   # kept for API compatibility, unused
        min_rr_ratio: float = 2.0,
    ) -> RewardBreakdown:
        """
        Compute the per-step reward (excluding trade close rewards).

        Called on every bar where no trade was closed this step.

        Parameters
        ----------
        action : int
            Action taken this step (0=HOLD, 1=ENTER_SHORT, 2=ENTER_LONG,
            3=EXIT, 4=TRAIL_STOP).
        is_position_open : bool
            Whether a position is currently open.
        atr_state : ATRState
        order_zone_state : OrderZoneState
        portfolio_state : dict
        min_rr_ratio : float
            Minimum acceptable R:R for bonus/penalty assessment.
        """
        reward = 0.0
        entry_bonus = 0.0
        entry_penalty = 0.0
        step_penalty = 0.0
        note = ""

        # ── Hold / WAIT penalty ───────────────────────────────
        # Only penalise idling when a valid zone setup is present and the agent
        # is ignoring it.  Waiting patiently when no setup exists is correct
        # behaviour and should not be penalised.
        if action == 0 and not is_position_open:
            setup_present = (
                order_zone_state.in_bearish_order_zone
                or order_zone_state.in_bullish_order_zone
            )
            if setup_present:
                step_penalty += self.hold_flat_penalty
                note = "hold_flat_in_zone"

        # ── Entry bonuses & penalties ─────────────────────────
        elif action in (1, 2):  # ENTER_SHORT or ENTER_LONG
            direction = -1 if action == 1 else 1

            # Entry cost (transaction proxy)
            entry_penalty += self.entry_cost

            # Overtrading check
            if portfolio_state.get("trades_today", 0) >= 5:
                entry_penalty += self.entry_penalties["overtrading"]
                note += "overtrading "

            is_bearish_action = direction == -1
            is_bullish_action = direction == 1

            # ATR gate violation — directional: penalise only if entering in the exhausted direction
            dir_exhausted = (
                (is_bearish_action and atr_state.atr_short_exhausted)
                or (is_bullish_action and atr_state.atr_long_exhausted)
            )
            if dir_exhausted:
                entry_penalty += self.entry_penalties["atr_exhausted"]
                note += "atr_exhausted "

            # R:R check (shaping penalty — scaled)
            if order_zone_state.rr_ratio < min_rr_ratio:
                entry_penalty += self.entry_penalties["rr_below_minimum"] * self.shaping_scale
                note += "rr_too_low "

            # Zone presence check
            in_zone = (
                (is_bearish_action and order_zone_state.in_bearish_order_zone)
                or (is_bullish_action and order_zone_state.in_bullish_order_zone)
            )
            if not in_zone:
                if not order_zone_state.in_bearish_order_zone and not order_zone_state.in_bullish_order_zone:
                    entry_penalty += self.entry_penalties["no_zone_present"] * self.shaping_scale
                    note += "no_zone "
            else:
                # Bonuses for quality setup (all shaping — scaled to zero over time)
                score = order_zone_state.confluence_score

                if score >= 0.85:
                    entry_bonus += self.entry_bonuses["full_order_zone_confluence"] * self.shaping_scale
                    note += "full_confluence "
                else:
                    if order_zone_state.in_bearish_order_zone or order_zone_state.in_bullish_order_zone:
                        entry_bonus += self.entry_bonuses["in_supply_demand_zone"] * self.shaping_scale
                    # Pillar 3 (rejection candle) removed

                # ATR room bonus: neither direction exhausted = still has room
                atr_has_room = not atr_state.atr_short_exhausted and not atr_state.atr_long_exhausted
                if atr_has_room:
                    entry_bonus += self.entry_bonuses["atr_has_room"] * self.shaping_scale

                if order_zone_state.rr_ratio >= min_rr_ratio:
                    entry_bonus += self.entry_bonuses["high_rr_ratio"] * self.shaping_scale

            # Loss streak discipline (keep always — not shaping, it's a guardrail)
            if portfolio_state.get("consecutive_losses", 0) >= self.discipline["loss_streak_threshold"]:
                entry_penalty += self.discipline["re_entry_after_loss_streak_penalty"]
                note += "loss_streak_reentry "

        # ── Trail stop bonus (shaping — scaled) ──────────────
        elif action == 4 and is_position_open:  # TRAIL_STOP
            unrealised_r = portfolio_state.get("current_pnl_r", 0.0)
            if unrealised_r >= 4.0:
                entry_bonus += self.exit_bonuses["aggressive_trail_at_4r"] * self.shaping_scale
                note += "trail_at_4r "
            elif unrealised_r >= 2.0:
                entry_bonus += self.exit_bonuses["trailing_stop_correctly"] * self.shaping_scale
                note += "trail_at_2r "

        reward = (entry_bonus + entry_penalty + step_penalty) * self.core_scale

        return RewardBreakdown(
            total=round(reward, 6),
            entry_bonus=entry_bonus,
            entry_penalty=entry_penalty,
            step_penalty=step_penalty,
            shaping_note=note.strip(),
        )

    # ── Trade close reward (called when a trade completes) ────

    def trade_close_reward(
        self,
        trade: Trade,
        order_zone_state: OrderZoneState,
        atr_state: ATRState,
        was_trailing: bool,
        peak_unrealised_r: float,
    ) -> RewardBreakdown:
        """
        Compute the reward when a trade is closed.

        This is the PRIMARY reward signal — the actual trade outcome in R.

        Parameters
        ----------
        trade : Trade
            The completed trade object.
        order_zone_state : OrderZoneState
            State at entry (snapshot captured at time of entry).
        atr_state : ATRState
            ATR state at entry.
        was_trailing : bool
            Whether a trailing stop was active when the trade closed.
        peak_unrealised_r : float
            Maximum unrealised R reached during the trade (MFE in R).
        """
        # ── Core reward: pnl_r × win_rate ────────────────────
        # Update episode win-rate counters before computing the multiplier
        # so the current trade's outcome immediately influences its own reward.
        self._ep_trades += 1
        if trade.pnl_r > 0:
            self._ep_wins += 1

        core_r = trade.pnl_r * self._win_rate

        exit_bonus = 0.0
        exit_penalty = 0.0
        note = ""

        # ── Exit quality assessment ───────────────────────────
        if trade.exit_reason == ExitReason.TAKE_PROFIT:
            exit_bonus += self.exit_bonuses.get("exit_at_atr_target", 0.0)
            note += "hit_target "

        if was_trailing:
            exit_bonus += self.exit_bonuses.get("trailing_stop_correctly", 0.0) * self.shaping_scale
            note += "used_trail "

        # ── Greed penalty: gave back large unrealised profit (shaping — scaled) ──
        if peak_unrealised_r >= 4.0 and not was_trailing:
            exit_penalty += self.exit_penalties.get("held_past_4r_no_trail", 0.0) * self.shaping_scale
            note += "no_trail_at_4r "

        r_given_back = peak_unrealised_r - trade.pnl_r
        if r_given_back > 1.0 and trade.exit_reason in (ExitReason.STOP_LOSS, ExitReason.AGENT_EXIT):
            exit_penalty -= self.gave_back_profit_weight * r_given_back * self.shaping_scale
            note += f"gave_back_{r_given_back:.1f}r "

        # core_r is NEVER scaled — always full P&L signal
        total = (core_r + exit_bonus + exit_penalty) * self.core_scale

        log.debug(
            "Trade reward computed",
            pnl_r=round(core_r, 3),
            exit_bonus=round(exit_bonus, 3),
            exit_penalty=round(exit_penalty, 3),
            total=round(total, 3),
            note=note,
        )

        return RewardBreakdown(
            total=round(total, 6),
            core_trade_r=core_r,
            exit_bonus=exit_bonus,
            exit_penalty=exit_penalty,
            shaping_note=note.strip(),
        )

    # ── Violation penalties ───────────────────────────────────

    def violation_reward(self, violation_type: str) -> RewardBreakdown:
        """
        Return a penalty reward for a hard guardrail violation.

        Parameters
        ----------
        violation_type : str
            One of: "max_drawdown_breach", "daily_loss_limit_breach".
        """
        penalty = self.violations.get(violation_type, -1.0) * self.core_scale
        log.debug("Violation penalty applied", type=violation_type, penalty=penalty)
        return RewardBreakdown(
            total=round(penalty, 6),
            violation_penalty=penalty,
            shaping_note=violation_type,
        )

    # ── Class method: from config dict ───────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "RewardCalculator":
        """Construct from a parsed reward_config.yaml dict."""
        return cls(
            core_scale=cfg.get("core", {}).get("scale", 1.0),
            hold_flat_penalty=cfg.get("step", {}).get("hold_flat_penalty", -0.001),
            entry_bonuses=cfg.get("entry_bonuses", {}),
            entry_penalties=cfg.get("entry_penalties", {}),
            exit_bonuses=cfg.get("exit_rewards", {}),
            exit_penalties=cfg.get("exit_penalties", {}),
            violations=cfg.get("violations", {}),
            discipline=cfg.get("discipline", {}),
            entry_cost=cfg.get("entry_penalties", {}).get("entry_cost", -0.05),
            gave_back_profit_weight=cfg.get("exit_penalties", {}).get("gave_back_profit_pct", 0.30),
        )