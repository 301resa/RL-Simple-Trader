"""
environment/position_manager.py
================================
Manages open and closed trading positions, P&L tracking, and risk controls.

Responsibilities:
  - Open/close positions with stop-loss and take-profit targets
  - Track unrealised and realised P&L in R-multiples
  - Enforce daily loss limits and max trades per day
  - Handle trailing stops
  - Trigger loss-streak pause
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

# Allowed position sizes in contracts (micro / fractional lots)
CONTRACT_TIERS: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5]

# Default confluence-score thresholds — one per tier boundary (n tiers = n-1 thresholds)
DEFAULT_CONFLUENCE_THRESHOLDS: List[float] = [0.50, 0.65, 0.75, 0.85]


class ExitReason(str, Enum):
    STOP_LOSS    = "stop_loss"
    TAKE_PROFIT  = "take_profit"
    AGENT_EXIT   = "agent_exit"
    SESSION_END  = "session_end"
    MAX_DRAWDOWN = "max_drawdown"
    TRAILING_STOP = "trailing_stop"


class PositionDirection(Enum):
    LONG  = 1
    SHORT = -1
    FLAT  = 0


@dataclass
class Trade:
    """Record of a completed trade."""
    direction: PositionDirection
    entry_price: float
    exit_price: float
    stop_price: float
    initial_target: float
    n_contracts: float          # fractional lots (0.5 increments)
    entry_bar_idx: int
    exit_bar_idx: int
    pnl_r: float
    pnl_points: float
    pnl_dollars: float
    is_win: bool
    exit_reason: ExitReason
    max_adverse_excursion: float = 0.0
    duration_bars: int = 0
    confluence_score: float = 0.0   # score at entry time


@dataclass
class PositionState:
    """Mutable state of the position manager."""
    is_open: bool = False
    direction: PositionDirection = PositionDirection.FLAT
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    n_contracts: float = 0.5    # fractional lots
    entry_bar_idx: int = 0
    trailing_active: bool = False
    trailing_stop_price: float = 0.0
    realised_pnl_r: float = 0.0
    unrealised_pnl_r: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    in_loss_streak_pause: bool = False
    max_drawdown_remaining: float = 3.0
    daily_pnl_r: float = 0.0
    daily_pnl_dollars: float = 0.0
    initial_risk_pts: float = 0.0
    mae_pts: float = 0.0
    confluence_score: float = 0.0


class PositionManager:
    """
    Manages position lifecycle and risk controls for a single trading session.

    Parameters
    ----------
    real_capital : float
        Account equity used for dynamic position sizing.
    risk_per_trade_pct : float
        Fraction of capital to risk per trade (e.g. 0.01 = 1%).
    min_contracts : int
        Minimum number of contracts per trade.
    max_contracts : int
        Maximum number of contracts per trade.
    point_value : float
        Dollar value per point per contract.
    max_trades_per_day : int
        Maximum number of trades per session.
    trail_activate_r : float
        Minimum unrealised R required to activate trailing stop.
    trail_aggressive_r : float
        Unrealised R level at which trailing becomes tighter.
    trail_lock_in_r : float
        Minimum R to lock in when trailing activates.
    max_daily_loss_r : float
        Stop trading after this many R lost on the day.
    max_drawdown_r : float
        Hard episode drawdown limit in R.
    loss_streak_threshold : int
        Consecutive losses before pause is triggered.
    pause_bars_after_loss_streak : int
        Reserved — pause duration after loss streak (not used in current impl).
    zone_buffer_atr_pct : float
        Reserved — buffer used externally for stop placement.
    trail_step_atr_pct : float
        Trail step size as fraction of ATR (used to move stop each bar).
    """

    def __init__(
        self,
        real_capital: float = 50000.0,
        risk_per_trade_pct: float = 0.01,
        min_contracts: float = 0.5,
        max_contracts: float = 2.5,
        point_value: float = 2.0,
        max_trades_per_day: int = 5,
        trail_activate_r: float = 1.2,
        trail_aggressive_r: float = 3.0,
        trail_lock_in_r: float = 1.1,
        max_daily_loss_r: float = 3.0,
        max_daily_loss_dollars: float = 1000.0,
        max_drawdown_r: float = 5.0,
        loss_streak_threshold: int = 3,
        pause_bars_after_loss_streak: int = 6,
        zone_buffer_atr_pct: float = 0.03,
        trail_step_atr_pct: float = 0.25,
        contract_tiers: Optional[List[float]] = None,
        confluence_tier_thresholds: Optional[List[float]] = None,
    ) -> None:
        self.real_capital = real_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.min_contracts = min_contracts
        self.max_contracts = max_contracts
        self.point_value = point_value
        self.max_trades_per_day = max_trades_per_day
        self.trail_activate_r = trail_activate_r
        self.trail_aggressive_r = trail_aggressive_r
        self.trail_lock_in_r = trail_lock_in_r
        self.max_daily_loss_r = max_daily_loss_r
        self.max_daily_loss_dollars = max_daily_loss_dollars
        self.max_drawdown_r = max_drawdown_r
        self.loss_streak_threshold = loss_streak_threshold
        self.pause_bars_after_loss_streak = pause_bars_after_loss_streak
        self.zone_buffer_atr_pct = zone_buffer_atr_pct
        self.trail_step_atr_pct = trail_step_atr_pct
        self._contract_tiers: List[float] = sorted(
            contract_tiers if contract_tiers else CONTRACT_TIERS
        )
        self._confluence_thresholds: List[float] = (
            confluence_tier_thresholds if confluence_tier_thresholds
            else DEFAULT_CONFLUENCE_THRESHOLDS
        )

        self._state = PositionState(
            n_contracts=min_contracts,
            max_drawdown_remaining=max_daily_loss_r,
        )
        self._completed_trades: List[Trade] = []

    # ── Public API ────────────────────────────────────────────

    @property
    def state(self) -> PositionState:
        return self._state

    @property
    def completed_trades(self) -> List[Trade]:
        return self._completed_trades

    def reset(self) -> None:
        """Reset for a new episode (new trading day)."""
        self._state = PositionState(
            n_contracts=self.min_contracts,
            max_drawdown_remaining=self.max_daily_loss_r,
        )
        self._completed_trades = []

    def _size_by_confluence(
        self,
        stop_pts: float,
        confluence_score: float,
    ) -> float:
        """
        Return the number of contracts (from CONTRACT_TIERS) for this trade.

        Two constraints are applied, and the smaller is taken:
          1. Capital constraint: 1% of equity / (stop_pts × point_value)
             — snapped DOWN to the nearest tier.
          2. Confluence constraint: confluence score maps to the highest
             allowed tier via the configured thresholds.
        """
        # Capital-based maximum (snap down to nearest tier that doesn't exceed risk)
        risk_dollars = self.real_capital * self.risk_per_trade_pct
        risk_per_contract = max(stop_pts * self.point_value, 1e-6)
        raw_n = risk_dollars / risk_per_contract

        # Find the highest tier that doesn't exceed raw_n
        capital_tier = self._contract_tiers[0]
        for t in self._contract_tiers:
            if t <= raw_n:
                capital_tier = t

        # Confluence-based maximum
        thresholds = self._confluence_thresholds
        if confluence_score < thresholds[0]:
            conf_tier = self._contract_tiers[0]
        elif len(thresholds) >= 2 and confluence_score < thresholds[1]:
            conf_tier = self._contract_tiers[min(1, len(self._contract_tiers) - 1)]
        elif len(thresholds) >= 3 and confluence_score < thresholds[2]:
            conf_tier = self._contract_tiers[min(2, len(self._contract_tiers) - 1)]
        elif len(thresholds) >= 4 and confluence_score < thresholds[3]:
            conf_tier = self._contract_tiers[min(3, len(self._contract_tiers) - 1)]
        else:
            conf_tier = self._contract_tiers[-1]

        # Take the smaller of capital and confluence constraints
        n = min(capital_tier, conf_tier)
        return max(self._contract_tiers[0], min(self._contract_tiers[-1], n))

    def enter(
        self,
        direction: int,
        current_price: float,
        stop_price: float,
        target_price: float,
        current_bar_idx: int,
        atr: float,
        confluence_score: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        Open a new position.

        Parameters
        ----------
        direction : int
            1 = LONG, -1 = SHORT.
        current_price : float
        stop_price : float
        target_price : float
        current_bar_idx : int
        atr : float
            Daily ATR (used for trailing stop initialisation).

        Returns
        -------
        (success: bool, reason: str)
        """
        s = self._state

        if s.is_open:
            return False, "position_already_open"
        if s.trades_today >= self.max_trades_per_day:
            return False, "max_trades_reached"
        if s.in_loss_streak_pause:
            return False, "loss_streak_pause"

        if direction == 1:  # LONG
            if stop_price >= current_price:
                return False, "invalid_stop_long"
            initial_risk = current_price - stop_price
        else:  # SHORT
            if stop_price <= current_price:
                return False, "invalid_stop_short"
            initial_risk = stop_price - current_price

        if initial_risk <= 0:
            return False, "zero_risk"

        # Confluence-graded fractional sizing (0.5 increments, capital-capped)
        n_contracts = self._size_by_confluence(initial_risk, confluence_score)

        pos_dir = PositionDirection.LONG if direction == 1 else PositionDirection.SHORT

        s.is_open = True
        s.direction = pos_dir
        s.n_contracts = n_contracts
        s.entry_price = current_price
        s.stop_price = stop_price
        s.target_price = target_price
        s.entry_bar_idx = current_bar_idx
        s.trailing_active = False
        s.trailing_stop_price = stop_price
        s.initial_risk_pts = initial_risk
        s.mae_pts = 0.0
        s.confluence_score = confluence_score

        return True, "ok"

    def update(
        self,
        current_price: float,
        current_bar_high: float,
        current_bar_low: float,
        current_bar_idx: int,
        atr: float,
        agent_wants_exit: bool = False,
        agent_wants_trail: bool = False,
    ) -> Tuple[bool, Optional[ExitReason], Optional[Trade]]:
        """
        Update position state for the current bar.

        Checks stops, targets, and trail requests in priority order.

        Returns
        -------
        (position_closed, exit_reason, trade) or (False, None, None).
        """
        s = self._state

        if not s.is_open:
            return False, None, None

        direction = s.direction
        dir_int = 1 if direction == PositionDirection.LONG else -1

        # ── Track MAE (max adverse excursion in pts) ──────────
        if direction == PositionDirection.LONG:
            adverse_move = s.entry_price - current_bar_low
        else:
            adverse_move = current_bar_high - s.entry_price
        s.mae_pts = max(s.mae_pts, adverse_move)

        # ── Activate trailing stop if requested and profitable ─
        if agent_wants_trail and not s.trailing_active:
            unrealised_r = self._compute_pnl_r(current_price)
            if unrealised_r >= self.trail_activate_r:
                s.trailing_active = True
                trail_dist = atr * self.trail_step_atr_pct
                if direction == PositionDirection.LONG:
                    s.trailing_stop_price = current_price - trail_dist
                else:
                    s.trailing_stop_price = current_price + trail_dist

        # ── Update trailing stop level ────────────────────────
        if s.trailing_active and agent_wants_trail:
            trail_dist = atr * self.trail_step_atr_pct
            if direction == PositionDirection.LONG:
                new_trail = current_price - trail_dist
                s.trailing_stop_price = max(s.trailing_stop_price, new_trail)
            else:
                new_trail = current_price + trail_dist
                s.trailing_stop_price = min(s.trailing_stop_price, new_trail)

        # ── Check exit conditions ─────────────────────────────
        active_stop = s.trailing_stop_price if s.trailing_active else s.stop_price
        stop_type   = ExitReason.TRAILING_STOP if s.trailing_active else ExitReason.STOP_LOSS
        exit_reason: Optional[ExitReason] = None
        exit_price  = current_price

        # ── Conservative same-bar rule ────────────────────────
        # When a single candle touches BOTH the stop loss and the take-profit
        # the order of fills is unknown.  We always assume the stop was hit
        # first (worst-case for the agent).
        #
        #   LONG : low <= stop  AND  high >= target  →  stop loss (loss)
        #   SHORT: high >= stop AND  low  <= target  →  stop loss (loss)
        #
        # This prevents the model from learning to exploit ambiguous candles
        # as free take-profits.
        if s.target_price > 0:
            if (direction == PositionDirection.LONG
                    and current_bar_low  <= active_stop
                    and current_bar_high >= s.target_price):
                exit_price  = active_stop
                exit_reason = stop_type

            elif (direction == PositionDirection.SHORT
                    and current_bar_high >= active_stop
                    and current_bar_low  <= s.target_price):
                exit_price  = active_stop
                exit_reason = stop_type

        # 1. Stop hit (single-sided — target NOT also hit)
        if exit_reason is None:
            if direction == PositionDirection.LONG and current_bar_low <= active_stop:
                exit_price  = active_stop
                exit_reason = stop_type
            elif direction == PositionDirection.SHORT and current_bar_high >= active_stop:
                exit_price  = active_stop
                exit_reason = stop_type

        # 2. Take-profit hit (only reached when stop was NOT hit)
        if exit_reason is None and s.target_price > 0:
            if direction == PositionDirection.LONG and current_bar_high >= s.target_price:
                exit_price  = s.target_price
                exit_reason = ExitReason.TAKE_PROFIT
            elif direction == PositionDirection.SHORT and current_bar_low <= s.target_price:
                exit_price  = s.target_price
                exit_reason = ExitReason.TAKE_PROFIT

        # 3. Agent requests exit
        if exit_reason is None and agent_wants_exit:
            exit_price  = current_price
            exit_reason = ExitReason.AGENT_EXIT

        if exit_reason is not None:
            trade = self._close_position(exit_price, current_bar_idx, exit_reason)
            return True, exit_reason, trade

        return False, None, None

    def force_close(
        self,
        current_price: float,
        current_bar_idx: int,
        exit_reason: ExitReason,
    ) -> Optional[Trade]:
        """Force-close any open position (e.g. session end, max drawdown)."""
        if not self._state.is_open:
            return None
        return self._close_position(current_price, current_bar_idx, exit_reason)

    def get_portfolio_state(self, current_price: float) -> dict:
        """Return a snapshot of the current portfolio state as a dict."""
        s = self._state
        unrealised_r = self._compute_pnl_r(current_price) if s.is_open else 0.0
        s.unrealised_pnl_r = unrealised_r
        active_stop = (s.trailing_stop_price if s.trailing_active else s.stop_price) if s.is_open else 0.0
        return {
            "position_open":          s.is_open,
            "position_direction":     s.direction.name,
            "position_size":          s.n_contracts if s.is_open else 0,
            "entry_price":            s.entry_price if s.is_open else 0.0,
            "current_pnl_r":          unrealised_r,
            "stop_loss_price":        active_stop,
            "take_profit_price":      s.target_price if s.is_open else 0.0,
            "max_drawdown_remaining": s.max_drawdown_remaining,
            "daily_pnl_r":            s.daily_pnl_r,
            "daily_pnl_dollars":      s.daily_pnl_dollars,
            "trades_today":           s.trades_today,
            "consecutive_losses":     s.consecutive_losses,
        }

    def is_max_drawdown_breached(self, current_price: float) -> bool:
        """Return True if either R or dollar daily loss limit is reached."""
        s = self._state
        unrealised_r = self._compute_pnl_r(current_price) if s.is_open else 0.0
        total_loss_r = -(s.daily_pnl_r + min(0.0, unrealised_r))
        if total_loss_r >= self.max_daily_loss_r:
            return True
        # Dollar limit: realised + unrealised (if negative)
        if s.is_open and s.initial_risk_pts > 0:
            unrealised_usd = (
                unrealised_r * s.initial_risk_pts * s.n_contracts * self.point_value
            )
        else:
            unrealised_usd = 0.0
        total_loss_usd = -(s.daily_pnl_dollars + min(0.0, unrealised_usd))
        return total_loss_usd >= self.max_daily_loss_dollars

    # ── Private helpers ───────────────────────────────────────

    def _compute_pnl_r(self, current_price: float) -> float:
        """Compute unrealised P&L in R-multiples."""
        s = self._state
        if not s.is_open or s.initial_risk_pts <= 0:
            return 0.0
        dir_int = 1 if s.direction == PositionDirection.LONG else -1
        return dir_int * (current_price - s.entry_price) / s.initial_risk_pts

    def _close_position(
        self,
        exit_price: float,
        current_bar_idx: int,
        exit_reason: ExitReason,
    ) -> Trade:
        """Close the open position and record a completed Trade."""
        s = self._state

        dir_int = 1 if s.direction == PositionDirection.LONG else -1
        pnl_points = dir_int * (exit_price - s.entry_price)
        pnl_r = pnl_points / s.initial_risk_pts if s.initial_risk_pts > 0 else 0.0
        pnl_dollars = pnl_points * s.n_contracts * self.point_value
        is_win = pnl_r > 0
        mae_r = s.mae_pts / s.initial_risk_pts if s.initial_risk_pts > 0 else 0.0

        trade = Trade(
            direction=s.direction,
            entry_price=s.entry_price,
            exit_price=exit_price,
            stop_price=s.stop_price,
            initial_target=s.target_price,
            n_contracts=s.n_contracts,
            entry_bar_idx=s.entry_bar_idx,
            exit_bar_idx=current_bar_idx,
            pnl_r=pnl_r,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            is_win=is_win,
            exit_reason=exit_reason,
            max_adverse_excursion=mae_r,
            duration_bars=current_bar_idx - s.entry_bar_idx,
            confluence_score=s.confluence_score,
        )

        # Update session-level accounting
        s.daily_pnl_r += pnl_r
        s.daily_pnl_dollars += pnl_dollars
        s.realised_pnl_r += pnl_r
        s.trades_today += 1
        s.max_drawdown_remaining = self.max_daily_loss_r + s.daily_pnl_r

        if is_win:
            s.consecutive_losses = 0
        else:
            s.consecutive_losses += 1
            if s.consecutive_losses >= self.loss_streak_threshold:
                s.in_loss_streak_pause = True

        # Clear position state
        s.is_open = False
        s.direction = PositionDirection.FLAT
        s.trailing_active = False
        s.entry_price = 0.0
        s.stop_price = 0.0
        s.target_price = 0.0
        s.initial_risk_pts = 0.0
        s.mae_pts = 0.0
        s.confluence_score = 0.0

        self._completed_trades.append(trade)
        return trade
