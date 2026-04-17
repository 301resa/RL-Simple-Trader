"""
environment/action_space.py
============================
Action definitions and action masking logic.

Actions:
  0 — HOLD/WAIT    : Do nothing (stay flat or hold current position)
  1 — ENTER_SHORT  : Open a short position
  2 — ENTER_LONG   : Open a long position
  3 — EXIT         : Close current position immediately
  4 — TRAIL_STOP   : Move stop loss to trail price (protect profits)

Action Masking:
  Invalid actions are zeroed out BEFORE the agent sees probabilities.
  This is different from penalising bad actions — the agent never
  "tries" an invalid action; the environment simply removes it from
  the choice set. This is critical for training stability.

  Masking rules:
  - ENTER_SHORT/LONG masked if: position already open
  - ENTER_SHORT masked if: downward ATR move >= 85% (already sold down too far)
  - ENTER_LONG  masked if: upward ATR move >= 85% (already bought up too far)
  - Opposite direction is always still allowed when one side is exhausted
  - Trend-direction masking removed — LSTM learns direction from price
  - ENTER_SHORT/LONG masked if: R:R < minimum
  - ENTER_SHORT/LONG masked if: max trades per day reached
  - ENTER_SHORT/LONG masked if: in loss streak pause
  - ENTER_SHORT/LONG masked if: last N bars of session (no late entries)
  - EXIT masked if: no position open
  - TRAIL_STOP masked if: no position open OR unrealised PnL < 2R
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np

from features.atr_calculator import ATRState
from features.order_zone_engine import OrderZoneState
from utils.logger import get_logger

log = get_logger(__name__)


class Action(IntEnum):
    HOLD = 0
    ENTER_SHORT = 1
    ENTER_LONG = 2
    EXIT = 3
    TRAIL_STOP = 4

    @classmethod
    def n_actions(cls) -> int:
        return len(cls)


class ActionMasker:
    """
    Computes the binary action mask at each timestep.

    A mask value of 1 = action is available.
    A mask value of 0 = action is forbidden (will not be sampled).

    Parameters
    ----------
    atr_exhaustion_threshold : float
        ATR % used above which entries are blocked.
    trail_min_r : float
        Minimum unrealised R required to allow TRAIL_STOP.
    max_trades_per_day : int
        Block entries beyond this count.
    no_entry_last_n_bars : int
        Block entries in last N bars of session.
    """

    def __init__(
        self,
        atr_exhaustion_threshold: float = 0.85,
        trail_min_r: float = 2.0,
        max_trades_per_day: int = 5,
        no_entry_last_n_bars: int = 3,
    ) -> None:
        self.atr_exhaustion_threshold = atr_exhaustion_threshold
        self.trail_min_r = trail_min_r
        self.max_trades_per_day = max_trades_per_day
        self.no_entry_last_n_bars = no_entry_last_n_bars

    def compute_mask(
        self,
        is_position_open: bool,
        position_direction: str,
        unrealised_r: float,
        atr_state: ATRState,
        order_zone_state: OrderZoneState,
        trades_today: int,
        in_loss_streak_pause: bool,
        bars_remaining_in_session: int,
        max_drawdown_breached: bool,
    ) -> np.ndarray:
        """
        Compute the action mask for the current timestep.

        Parameters
        ----------
        is_position_open : bool
        position_direction : str
            "LONG", "SHORT", or "FLAT".
        unrealised_r : float
            Current unrealised PnL in R-multiples.
        atr_state : ATRState
        order_zone_state : OrderZoneState
        trades_today : int
        in_loss_streak_pause : bool
        bars_remaining_in_session : int
        max_drawdown_breached : bool

        Returns
        -------
        np.ndarray of shape (n_actions,) with dtype float32.
        Values: 1.0 = allowed, 0.0 = masked.
        """
        n = Action.n_actions()
        mask = np.ones(n, dtype=np.float32)

        # ── Always allow HOLD ─────────────────────────────────
        mask[Action.HOLD] = 1.0

        # ── If max drawdown breached — only HOLD allowed ──────
        if max_drawdown_breached:
            mask[:] = 0.0
            mask[Action.HOLD] = 1.0
            return mask

        # ── EXIT conditions ───────────────────────────────────
        if not is_position_open:
            mask[Action.EXIT] = 0.0       # Nothing to exit
            mask[Action.TRAIL_STOP] = 0.0  # Nothing to trail

        # ── TRAIL_STOP conditions ─────────────────────────────
        if is_position_open and unrealised_r < self.trail_min_r:
            mask[Action.TRAIL_STOP] = 0.0  # Not enough profit to trail yet

        # ── Entry blocking conditions (apply to both SHORT and LONG) ──
        entry_blocked = False
        block_reason = ""

        if is_position_open:
            entry_blocked = True
            block_reason = "position_open"
        elif trades_today >= self.max_trades_per_day:
            entry_blocked = True
            block_reason = "max_trades_reached"
        elif in_loss_streak_pause:
            entry_blocked = True
            block_reason = "loss_streak_pause"
        elif bars_remaining_in_session <= self.no_entry_last_n_bars:
            entry_blocked = True
            block_reason = "end_of_session"
        # ATR exhaustion is NOT a hard mask — after a volatile RTH session the
        # cumulative range exceeds the threshold, which would permanently block
        # all ETH/globex entries for the rest of the day.  The reward function
        # already penalises exhausted-ATR entries via entry_penalties["atr_exhausted"].

        if entry_blocked:
            mask[Action.ENTER_SHORT] = 0.0
            mask[Action.ENTER_LONG] = 0.0
            if block_reason not in ("position_open",):
                log.debug("Entry masked", reason=block_reason)
            return mask

        # ── Directional ATR exhaustion ────────────────────────
        # Block only the direction that has already consumed 85% of ATR.
        # The opposite direction is still allowed — if the market sold off
        # 85% of ATR, longs are fine; if it rallied 85%, shorts are fine.
        if atr_state.atr_short_exhausted:
            mask[Action.ENTER_SHORT] = 0.0
            log.debug("ENTER_SHORT masked", reason="atr_short_exhausted")
        if atr_state.atr_long_exhausted:
            mask[Action.ENTER_LONG] = 0.0
            log.debug("ENTER_LONG masked", reason="atr_long_exhausted")

        # R:R and zone-presence checks are handled by the reward function
        # (entry penalties), not the mask.  Hard-masking R:R collapses to
        # zero valid entries whenever no zone is detected, which prevents
        # the agent from ever exploring trade entries during early training.

        return mask

    def log_mask_state(self, mask: np.ndarray) -> None:
        """Log which actions are available (debug only)."""
        available = [Action(i).name for i, v in enumerate(mask) if v > 0]
        masked = [Action(i).name for i, v in enumerate(mask) if v == 0]
        log.debug("Action mask", available=available, masked=masked)