"""
environment/trading_env.py
===========================
Core Gymnasium trading environment implementing the full MDP.

Episode structure:
  - 1 episode = 1 RTH trading session (e.g. 09:30–16:00 EST)
  - 1 timestep = 1 intraday bar (e.g. 5-minute bar)
  - ~78 steps per episode for NQ/ES futures

At each step:
  1. Build observation from all feature modules
  2. Compute action mask (invalid actions zeroed)
  3. Agent selects action from masked distribution
  4. Execute action via PositionManager
  5. Compute reward via RewardCalculator
  6. Advance bar; check termination conditions

Compliant with gymnasium.Env API (stable-baselines3 compatible).
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from data.data_augmentor import OHLCVAugmentor
from data.data_loader import DataLoader
from environment.action_space import Action, ActionMasker
from environment.position_manager import ExitReason, PositionDirection, PositionManager
from environment.reward_calculator import RewardCalculator, RewardBreakdown
from features.atr_calculator import ATRCalculator, ATRState
from features.observation_builder import ObservationBuilder
from features.order_zone_engine import OrderZoneEngine, OrderZoneState
from features.trend_classifier import TrendClassifier
from features.zone_detector import ZoneDetector
from utils.logger import get_logger

log = get_logger(__name__)


_MAX_ZONE_WIDTH_ENV = 10.0  # matches observation_builder._MAX_ZONE_WIDTH


def _filter_wide_zones(zs) -> object:
    """
    Return a ZoneState with wide zones (> _MAX_ZONE_WIDTH_ENV points) set to None.
    Wide zones are never traded (skipped in _place_pending_order) so their signals
    must not influence confluence scoring or observation features.
    """
    from features.zone_detector import ZoneState
    supply = zs.nearest_supply if (
        zs.nearest_supply is not None
        and (zs.nearest_supply.top - zs.nearest_supply.bottom) <= _MAX_ZONE_WIDTH_ENV
    ) else None
    demand = zs.nearest_demand if (
        zs.nearest_demand is not None
        and (zs.nearest_demand.top - zs.nearest_demand.bottom) <= _MAX_ZONE_WIDTH_ENV
    ) else None
    return ZoneState(nearest_supply=supply, nearest_demand=demand)


def _freeze_zone_state(zs) -> object:
    """
    Return a snapshot of a ZoneState whose Zone objects cannot be mutated
    by future calls to _update_zone_validity().

    Uses dataclasses.replace() (shallow copy of scalar fields) which is
    ~10–50× faster than copy.deepcopy() for these small dataclasses.
    """
    supply = _dc_replace(zs.nearest_supply) if zs.nearest_supply is not None else None
    demand = _dc_replace(zs.nearest_demand) if zs.nearest_demand is not None else None
    from features.zone_detector import ZoneState
    return ZoneState(nearest_supply=supply, nearest_demand=demand)


class TradingEnv(gym.Env):
    """
    Gymnasium environment for the Order Zone RL trading agent.

    Parameters
    ----------
    data_loader : DataLoader
        Loaded data source (intraday + daily OHLCV).
    trading_days : List[str]
        List of date strings to sample episodes from.
    position_manager : PositionManager
        Pre-configured risk/position handler.
    reward_calculator : RewardCalculator
        Pre-configured reward function.
    observation_builder : ObservationBuilder
        Pre-configured observation assembler.
    atr_calculator : ATRCalculator
        Pre-configured ATR feature engine.
    zone_detector : ZoneDetector
        Pre-configured S/D zone detector.
    trend_classifier : TrendClassifier
        Pre-configured trend engine.
    order_zone_engine : OrderZoneEngine
        Pre-configured confluence scoring engine.
    action_masker : ActionMasker
        Pre-configured action masking logic.
    rth_start : str
        Session start time string, e.g. "09:30".
    rth_end : str
        Session end time string, e.g. "16:00".
    no_entry_last_n_bars : int
        Block entries in last N bars of session.
    early_terminate_on_max_dd : bool
        End episode immediately on max drawdown breach.
    point_value : float
        Dollar value per point per micro contract.
    curriculum_filter_fn : Optional[callable]
        If provided, called with (date, daily_bar) → bool.
        Returns True if this day should be included in the current
        curriculum stage.
    augmentor : OHLCVAugmentor, optional
        If provided, applied to session bars on every reset() call.
        Intended for training envs only — pass None for eval/test envs.
    session_type : str
        "RTH"    → Regular Trading Hours only  (rth_start – rth_end)
        "GLOBEX" → bars outside RTH in the day file (pre + post market)
        "FULL"   → all bars in the day file (no time filter)
    random_start : bool
        If True (default for training) the episode begins at a randomly
        sampled bar rather than the first bar.  The start is drawn from
        the env's seeded RNG so it is reproducible and consistent within
        a given env worker, but differs across workers.
    seed : Optional[int]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_loader: DataLoader,
        trading_days: List[str],
        position_manager: PositionManager,
        reward_calculator: RewardCalculator,
        observation_builder: ObservationBuilder,
        atr_calculator: ATRCalculator,
        zone_detector: ZoneDetector,
        trend_classifier: TrendClassifier,
        order_zone_engine: OrderZoneEngine,
        action_masker: ActionMasker,
        rth_start: str = "09:30",
        rth_end: str = "16:00",
        no_entry_last_n_bars: int = 3,
        early_terminate_on_max_dd: bool = True,
        point_value: float = 2.0,
        bar_minutes: int = 5,
        curriculum_filter_fn: Optional[Any] = None,
        augmentor: Optional[OHLCVAugmentor] = None,
        session_type: str = "RTH",
        random_start: bool = False,
        seed: Optional[int] = None,
        zone_lookback_bars: int = 500,
    ) -> None:
        super().__init__()

        self.data_loader = data_loader
        self.trading_days = trading_days
        self.position_manager = position_manager
        self.reward_calculator = reward_calculator
        self.observation_builder = observation_builder
        self.atr_calculator = atr_calculator
        self.zone_detector = zone_detector
        self.trend_classifier = trend_classifier
        self.order_zone_engine = order_zone_engine
        self.action_masker = action_masker
        self.rth_start = rth_start
        self.rth_end = rth_end
        self.no_entry_last_n_bars = no_entry_last_n_bars
        self.early_terminate_on_max_dd = early_terminate_on_max_dd
        self.point_value = point_value
        self.bar_minutes = bar_minutes
        self.curriculum_filter_fn = curriculum_filter_fn
        self.augmentor = augmentor
        self.session_type = session_type.upper()
        self.random_start = random_start
        self.zone_lookback_bars = zone_lookback_bars

        # State variables (initialised in reset())
        self._current_day: Optional[str] = None
        self._session_bars: Optional[pd.DataFrame] = None
        self._atr_series: Optional[pd.Series] = None
        # Combined history+session frames used by zone/liquidity detectors
        self._combined_bars: Optional[pd.DataFrame] = None
        self._combined_atr_series: Optional[pd.Series] = None
        self._combined_session_offset: int = 0  # index of first session bar in _combined_bars
        self._current_step: int = 0
        self._n_steps: int = 0
        self._episode_rewards: List[float] = []
        self._current_atr_state: Optional[ATRState] = None
        self._entry_order_zone_state: Optional[OrderZoneState] = None
        self._entry_atr_state: Optional[ATRState] = None
        self._peak_unrealised_r: float = 0.0
        # Pending limit order (placed at zone midpoint, filled when price reaches it)
        # Keys: direction, limit_price, stop_price, target_price,
        #       confluence_score, placed_bar_idx
        self._pending_order: Optional[Dict[str, Any]] = None

        # Define observation space eagerly using builder's known dimension
        self._obs_dim: int = observation_builder.obs_dim
        self._observation_space = spaces.Box(
            low=-observation_builder.clip_value,
            high=observation_builder.clip_value,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self._spaces_defined: bool = True

        # RNG
        self._rng = np.random.default_rng(seed)

        # Build filtered day list for curriculum
        self._available_days: List[str] = list(trading_days)

    # ── Gymnasium API ─────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        Randomly selects a trading day from the available pool.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Select episode day
        self._current_day = self._sample_episode_day()

        # Load session bars
        self._session_bars = self.data_loader.get_day_bars(self._current_day)
        if self._session_bars.empty:
            # Skip empty days — try another
            log.warning("Empty session, resampling", date=self._current_day)
            return self.reset(seed=seed, options=options)

        # Filter to the configured session window
        if self.session_type == "RTH":
            self._session_bars = self._filter_rth(self._session_bars)
        elif self.session_type == "GLOBEX":
            self._session_bars = self._filter_globex(self._session_bars)
        # FULL: no time filter — use all bars in the day file

        if len(self._session_bars) < 5:
            log.warning(
                "Too few session bars, resampling",
                date=self._current_day,
                session=self.session_type,
                bars=len(self._session_bars),
            )
            return self.reset(seed=seed, options=options)

        # Apply per-episode OHLCV jitter (training envs only)
        if self.augmentor is not None:
            self._session_bars = self.augmentor.apply(self._session_bars)

        # Precompute intraday ATR series (using daily ATR for this date)
        daily_atr = self.atr_calculator.get_atr_for_date(self._current_day)
        if daily_atr is None:
            log.warning("No ATR for date, resampling", date=self._current_day)
            return self.reset(seed=seed, options=options)

        # Build a constant ATR series for this session (all bars = daily ATR value)
        self._atr_series = pd.Series(
            [daily_atr] * len(self._session_bars),
            index=self._session_bars.index,
        )

        # ── Build history-extended context for zone/liquidity detection ─────────
        # Pull up to zone_lookback_bars of prior-session bars so the detectors
        # see meaningful supply/demand structure before the current session opens.
        history_bars = self.data_loader.get_bars_before(
            self._current_day, self.zone_lookback_bars
        )
        if not history_bars.empty:
            # Per-date daily ATR for history bars (falls back to today's ATR)
            history_atr_vals = []
            for ts in history_bars.index:
                d = ts.strftime("%Y-%m-%d")
                v = self.atr_calculator.get_atr_for_date(d)
                history_atr_vals.append(float(v) if v is not None else daily_atr)
            history_atr = pd.Series(history_atr_vals, index=history_bars.index)
            combined_raw   = pd.concat([history_bars, self._session_bars])
            combined_atr_raw = pd.concat([history_atr, self._atr_series])
        else:
            combined_raw     = self._session_bars.copy()
            combined_atr_raw = self._atr_series.copy()

        # Integer-indexed views — all detectors use .iloc internally
        self._combined_bars       = combined_raw.reset_index(drop=True)
        self._combined_atr_series = combined_atr_raw.reset_index(drop=True)
        history_len = len(history_bars)  # 0 when no prior data

        # Reset sub-components
        self.position_manager.reset()
        self.zone_detector.reset()
        self.reward_calculator.reset_episode_stats()

        # Random start: agent begins at a random bar within the session.
        # Drawn from the env's seeded RNG so it is deterministic and
        # consistent for a given worker, but differs across workers.
        n_bars = len(self._session_bars)
        start_offset = 0
        if self.random_start and n_bars > 5:
            # Allow start anywhere in the first 75% of the session so
            # there are always enough bars for a meaningful episode.
            max_offset = max(1, int(n_bars * 0.75))
            start_offset = int(self._rng.integers(0, max_offset))
            self._session_bars = self._session_bars.iloc[start_offset:].reset_index(drop=False)
            self._atr_series   = self._atr_series.iloc[start_offset:].reset_index(drop=True)

        # Offset into _combined_bars where the agent's episode begins
        self._combined_session_offset = history_len + start_offset

        self._current_step = 0
        self._n_steps = len(self._session_bars)
        self._episode_rewards = []
        self._entry_order_zone_state = None
        self._entry_atr_state = None
        self._peak_unrealised_r = 0.0
        self._pending_order = None

        # ── Pre-compute all market feature states for every bar ───────────────
        # Market features depend only on price history — never on agent actions.
        # One pass here eliminates repeated Pandas slicing inside step().

        # Vectorised ATR: O(n) running max/min instead of O(n²) growing slices
        atr_states = self.atr_calculator.compute_all_session_states(
            self._current_day, self._session_bars
        )
        if not atr_states:
            from features.atr_calculator import ATRState as _ATRState
            cp = float(self._session_bars.iloc[0]["close"])
            fallback = _ATRState(
                atr_daily=100.0, prior_day_high=cp + 50,
                prior_day_low=cp - 50, prior_day_range=100.0,
                session_open=cp, session_high=cp, session_low=cp,
                current_daily_range=0.0, atr_pct_used=0.0,
                atr_remaining_pts=100.0,
                atr_short_exhausted=False, atr_long_exhausted=False,
            )
            atr_states = [fallback] * self._n_steps

        # Cache per-episode numpy arrays in the observation builder (once per reset)
        self.observation_builder.prepare_episode(self._combined_bars)

        # Pre-extract numpy arrays from the combined bars once per episode.
        # Passing these to zone_detector avoids repeated pandas .iloc row access
        # inside the precompute loop (~50x faster than pandas in a tight Python loop).
        _np_open  = self._combined_bars["open"].to_numpy()
        _np_high  = self._combined_bars["high"].to_numpy()
        _np_low   = self._combined_bars["low"].to_numpy()
        _np_close = self._combined_bars["close"].to_numpy()
        _np_atr   = self._combined_atr_series.to_numpy() if self._combined_atr_series is not None else None
        self.zone_detector.set_bars_numpy(_np_open, _np_high, _np_low, _np_close, _np_atr)

        self._precomputed_states: list = []
        for bar_idx in range(self._n_steps):
            atr_s = atr_states[bar_idx] if bar_idx < len(atr_states) else atr_states[-1]

            # Zone detector uses full history-extended context
            combined_idx = self._combined_session_offset + bar_idx
            zone_s = self.zone_detector.scan_and_update(
                bars=self._combined_bars,
                atr_series=self._combined_atr_series,
                current_bar_idx=combined_idx,
            )
            # Filter wide zones before scoring — agent cannot trade them, so
            # confluence scores must not reflect their presence.
            current_price = float(_np_close[combined_idx])
            oz_s = self.order_zone_engine.compute(
                bars=self._combined_bars,
                current_bar_idx=combined_idx,
                atr_state=atr_s,
                zone_state=_filter_wide_zones(zone_s),
                trend_snapshot=None,
                current_price=current_price,
            )
            # Freeze ZoneState: Zone objects in the detector's lists are mutable
            # and will be updated by future bars — snapshot them cheaply with
            # dataclasses.replace() instead of copy.deepcopy() (~10-50× faster).
            # ATRState and OrderZoneState are created fresh each iteration → safe to
            # store directly with no copy.
            self._precomputed_states.append({
                "atr": atr_s,
                "zone": _freeze_zone_state(zone_s),
                "oz": oz_s,
            })

        # Build initial observation
        obs, info = self._build_obs_and_info()

        # Lazily define spaces after first observation is built
        if not self._spaces_defined:
            self._define_spaces(obs)

        log.debug("Episode reset", date=self._current_day, n_bars=self._n_steps)
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one timestep.

        Parameters
        ----------
        action : int
            Selected action (will be validated against mask).

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        assert self._session_bars is not None, "Call reset() before step()."

        # Single .iloc lookup — reuse values throughout the step
        current_bar   = self._session_bars.iloc[self._current_step]
        current_price = float(current_bar["close"])
        current_high  = float(current_bar["high"])
        current_low   = float(current_bar["low"])
        atr_state     = self._current_atr_state

        # ── Check pending limit order fill ────────────────────
        # Done before action processing so the fill reward is computed this bar.
        if self._pending_order is not None and not self.position_manager.state.is_open:
            if self._check_pending_fill(current_high, current_low):
                self._open_from_pending(current_bar_idx=self._current_step)

        # ── Validate action against mask ──────────────────────
        mask = self._compute_action_mask(atr_state)
        if mask[action] == 0.0:
            # Agent chose a masked action — treat as HOLD
            log.debug("Masked action overridden to HOLD", action=Action(action).name)
            action = Action.HOLD

        portfolio_state = self.position_manager.get_portfolio_state(current_price)
        is_open = self.position_manager.state.is_open

        # ── Execute action ────────────────────────────────────
        reward_breakdown = RewardBreakdown(total=0.0)
        trade_closed_this_step = False

        # Update peak unrealised R tracking
        unrealised_r = portfolio_state.get("current_pnl_r", 0.0)
        if is_open and unrealised_r > self._peak_unrealised_r:
            self._peak_unrealised_r = unrealised_r

        if action == Action.ENTER_SHORT:
            # Cancel any pending, then place new pending limit at zone midpoint
            self._pending_order = None
            self._place_pending_order(direction=-1, current_price=current_price, atr_state=atr_state)
        elif action == Action.ENTER_LONG:
            self._pending_order = None
            self._place_pending_order(direction=1, current_price=current_price, atr_state=atr_state)
        elif action == Action.EXIT:
            # Cancel pending order when flat, or close open position below
            if not self.position_manager.state.is_open and self._pending_order is not None:
                self._pending_order = None

        # ── Update position (check stops, targets, trail) ─────
        agent_wants_exit = (action == Action.EXIT)
        agent_wants_trail = (action == Action.TRAIL_STOP)

        pos_closed, exit_reason, closed_trade = self.position_manager.update(
            current_price=current_price,
            current_bar_high=current_high,
            current_bar_low=current_low,
            current_bar_idx=self._current_step,
            atr=atr_state.atr_daily,   # already loaded from precomputed states
            agent_wants_exit=agent_wants_exit,
            agent_wants_trail=agent_wants_trail,
        )

        if pos_closed and closed_trade is not None:
            trade_closed_this_step = True
            reward_breakdown = self.reward_calculator.trade_close_reward(
                trade=closed_trade,
                order_zone_state=self._entry_order_zone_state or self._get_current_order_zone_state(),
                atr_state=self._entry_atr_state or atr_state,
                was_trailing=self.position_manager.state.trailing_active,
                peak_unrealised_r=self._peak_unrealised_r,
            )
            self._peak_unrealised_r = 0.0  # Reset for next trade
        else:
            # Step-level reward (entry quality shaping, etc.)
            order_zone_state = self._get_current_order_zone_state()
            reward_breakdown = self.reward_calculator.step_reward(
                action=action,
                is_position_open=self.position_manager.state.is_open,
                atr_state=atr_state,
                order_zone_state=order_zone_state,
                trend_snapshot=None,
                portfolio_state=portfolio_state,
                pending_order=self._pending_order,
            )

        # ── Check termination conditions ──────────────────────
        terminated = False
        truncated = False

        # Max drawdown breach — close position first, then combine trade P&L + penalty
        if self.position_manager.is_max_drawdown_breached(current_price):
            dd_reward = self.reward_calculator.violation_reward("max_drawdown_breach")
            closed_trade_dd = self.position_manager.force_close(
                current_price, self._current_step, ExitReason.MAX_DRAWDOWN
            )
            if closed_trade_dd is not None:
                trade_reward_dd = self.reward_calculator.trade_close_reward(
                    trade=closed_trade_dd,
                    order_zone_state=self._entry_order_zone_state or self._get_current_order_zone_state(),
                    atr_state=self._entry_atr_state or atr_state,
                    was_trailing=False,
                    peak_unrealised_r=self._peak_unrealised_r,
                )
                reward_breakdown = RewardBreakdown(
                    total=reward_breakdown.total + dd_reward.total + trade_reward_dd.total,
                    violation_penalty=dd_reward.total,
                    core_trade_r=trade_reward_dd.core_trade_r,
                    shaping_note="max_drawdown_breach",
                )
                self._peak_unrealised_r = 0.0
            else:
                reward_breakdown = RewardBreakdown(
                    total=reward_breakdown.total + dd_reward.total,
                    violation_penalty=dd_reward.total,
                    shaping_note="max_drawdown_breach",
                )
            if self.early_terminate_on_max_dd:
                terminated = True

        # End of session
        self._current_step += 1
        if self._current_step >= self._n_steps:
            # Cancel any unfilled pending orders
            self._pending_order = None
            # Force close any open position at session end — reuse current_price
            if self.position_manager.state.is_open:
                self.position_manager.force_close(current_price, self._current_step - 1, ExitReason.SESSION_END)
            truncated = True

        # ── Build next observation ────────────────────────────
        if not (terminated or truncated):
            obs, info = self._build_obs_and_info()
        else:
            obs = self._last_obs if hasattr(self, "_last_obs") else np.zeros(self._obs_dim or 1, dtype=np.float32)
            info = self._episode_summary()

        self._episode_rewards.append(reward_breakdown.total)

        return obs, float(reward_breakdown.total), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Return the current action mask.
        Required by MaskablePPO from sb3-contrib.
        """
        if self._current_atr_state is None:
            return np.ones(Action.n_actions(), dtype=np.float32)
        return self._compute_action_mask(self._current_atr_state)

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            ps = self.position_manager.state
            print(
                f"Day: {self._current_day} | Bar: {self._current_step}/{self._n_steps} | "
                f"Position: {ps.direction.name} | "
                f"Daily PnL R: {ps.realised_pnl_r:.2f} | "
                f"Trades: {ps.trades_today}"
            )

    @property
    def observation_space(self) -> spaces.Box:
        if not self._spaces_defined:
            raise RuntimeError("Call reset() at least once to define observation_space.")
        return self._observation_space

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(Action.n_actions())

    # ── Private helpers ───────────────────────────────────────

    def _define_spaces(self, sample_obs: np.ndarray) -> None:
        self._obs_dim = len(sample_obs)
        self._observation_space = spaces.Box(
            low=-self.observation_builder.clip_value,
            high=self.observation_builder.clip_value,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self._spaces_defined = True
        log.info("Spaces defined", obs_dim=self._obs_dim, n_actions=Action.n_actions())

    def _build_obs_and_info(self) -> Tuple[np.ndarray, dict]:
        """Build the observation vector and info dict for the current step."""
        step = min(self._current_step, self._n_steps - 1)
        combined_idx  = self._combined_session_offset + step
        # Use cached numpy array when available — avoids pandas .iloc overhead
        if self.observation_builder._cached_closes is not None:
            current_price = float(self.observation_builder._cached_closes[combined_idx])
        else:
            current_price = float(self._session_bars.iloc[step]["close"])

        # ── O(1) lookup from pre-computed states ─────────────────────────────
        states           = self._precomputed_states[step]
        atr_state        = states["atr"]
        zone_state       = states["zone"]
        order_zone_state = states["oz"]

        self._current_atr_state     = atr_state
        self._last_zone_state       = zone_state
        self._last_order_zone_state = order_zone_state

        portfolio_state = self.position_manager.get_portfolio_state(current_price)

        # RTH context: tells the agent whether it is in Regular Trading Hours
        # (09:30–16:00 ET) vs Globex/pre-market.  RTH price action is structurally
        # different (higher volume, tighter spreads, cleaner zone reactions).
        is_rth = 0.0
        rth_time_pct = 0.0
        try:
            bar_ts = self._session_bars.index[step]
            bar_time = bar_ts.time()
            rth_s = pd.Timestamp(f"2000-01-01 {self.rth_start}").time()
            rth_e = pd.Timestamp(f"2000-01-01 {self.rth_end}").time()
            if rth_s <= bar_time <= rth_e:
                is_rth = 1.0
                rth_total_secs = (
                    pd.Timestamp(f"2000-01-01 {self.rth_end}")
                    - pd.Timestamp(f"2000-01-01 {self.rth_start}")
                ).total_seconds()
                elapsed_secs = (
                    pd.Timestamp(f"2000-01-01 {bar_time}")
                    - pd.Timestamp(f"2000-01-01 {self.rth_start}")
                ).total_seconds()
                rth_time_pct = float(np.clip(elapsed_secs / max(rth_total_secs, 1.0), 0.0, 1.0))
        except Exception:
            pass

        session_info = {
            "session_time_pct":   step / max(self._n_steps - 1, 1),
            "bars_remaining_pct": (self._n_steps - 1 - step) / max(self._n_steps - 1, 1),
            "is_rth":             is_rth,
            "rth_time_pct":       rth_time_pct,
        }

        # Use the history-extended bar context for the price lookback window.
        # combined_idx already computed above.
        obs = self.observation_builder.build(
            bars=self._combined_bars,
            current_bar_idx=combined_idx,
            atr_state=atr_state,
            zone_state=zone_state,
            order_zone_state=order_zone_state,
            portfolio_state=portfolio_state,
            session_info=session_info,
            pending_order=self._pending_order,
        )

        self._last_obs = obs
        self._last_zone_state = zone_state
        self._last_order_zone_state = order_zone_state

        info = {
            "date": self._current_day,
            "step": step,
            "confluence_score": order_zone_state.confluence_score,
            "in_order_zone": order_zone_state.in_bearish_order_zone or order_zone_state.in_bullish_order_zone,
            "atr_pct_used": atr_state.atr_pct_used,
            "trades_today": portfolio_state.get("trades_today", 0),
            "daily_pnl_r": portfolio_state.get("daily_pnl_r", 0.0),
            "action_mask": self._compute_action_mask(atr_state).tolist(),
        }
        return obs, info

    def _get_current_order_zone_state(self) -> OrderZoneState:
        step = min(self._current_step, self._n_steps - 1)
        if hasattr(self, "_precomputed_states") and self._precomputed_states:
            return self._precomputed_states[step]["oz"]
        return self._last_order_zone_state if hasattr(self, "_last_order_zone_state") else \
            self.order_zone_engine.compute(
                bars=self._session_bars,
                current_bar_idx=step,
                atr_state=self._current_atr_state,
                zone_state=None,
                trend_snapshot=None,
            )

    def _compute_action_mask(self, atr_state: ATRState) -> np.ndarray:
        step = min(self._current_step, self._n_steps - 1)
        portfolio_state = self.position_manager.get_portfolio_state(
            float(self._session_bars.iloc[step]["close"])
        )
        order_zone_state = (
            self._precomputed_states[step]["oz"]
            if (hasattr(self, "_precomputed_states") and self._precomputed_states)
            else self._get_current_order_zone_state()
        )

        bars_remaining = self._n_steps - 1 - self._current_step

        return self.action_masker.compute_mask(
            is_position_open=portfolio_state["position_open"],
            position_direction=portfolio_state["position_direction"],
            unrealised_r=portfolio_state.get("current_pnl_r", 0.0),
            atr_state=atr_state,
            order_zone_state=order_zone_state,
            trades_today=portfolio_state.get("trades_today", 0),
            in_loss_streak_pause=self.position_manager.state.in_loss_streak_pause,
            bars_remaining_in_session=bars_remaining,
            max_drawdown_breached=self.position_manager.is_max_drawdown_breached(
                float(self._session_bars.iloc[min(self._current_step, self._n_steps - 1)]["close"])
            ),
        )

    def _place_pending_order(
        self, direction: int, current_price: float, atr_state: ATRState
    ) -> None:
        """
        Compute limit price (zone edge), stop, target and store a pending limit
        order.

        Entry geometry:
          LONG  — limit at demand.top  (first touch of zone from above)
          SHORT — limit at supply.bottom (first touch of zone from below)

        Stop geometry:
          LONG  — stop at demand.bottom - 1.5 pts  (below entire zone)
          SHORT — stop at supply.top   + 1.5 pts  (above entire zone)
          stop_dist = zone_width + 1.5, floored at MIN_STOP_PTS

        Target: impulse_extreme (swing high/low of the bar that created the zone).

        Zones wider than 10 points are skipped — undefined risk.
        """
        from features.order_zone_engine import FIXED_STOP_BUFFER_PTS, MIN_STOP_PTS
        from features.atr_calculator import ATRCalculator

        zone_state = self._last_zone_state if hasattr(self, "_last_zone_state") else None
        oz_state   = self._last_order_zone_state if hasattr(self, "_last_order_zone_state") else None
        confluence = oz_state.confluence_score if oz_state is not None else 1.0
        atr        = atr_state.atr_daily

        _MAX_ZONE_WIDTH = 10.0
        if direction == -1 and zone_state and zone_state.nearest_supply:
            zone = zone_state.nearest_supply
            if zone.top - zone.bottom > _MAX_ZONE_WIDTH:
                log.debug("Pending order skipped — supply zone too wide", width=round(zone.top - zone.bottom, 2))
                return
            zone_width  = zone.top - zone.bottom
            stop_dist   = max(zone_width + FIXED_STOP_BUFFER_PTS, MIN_STOP_PTS)
            limit_price = zone.bottom          # SHORT: enter at zone bottom (first touch from below)
            stop_price  = zone.top + stop_dist  # stop above entire zone
        elif direction == 1 and zone_state and zone_state.nearest_demand:
            zone = zone_state.nearest_demand
            if zone.top - zone.bottom > _MAX_ZONE_WIDTH:
                log.debug("Pending order skipped — demand zone too wide", width=round(zone.top - zone.bottom, 2))
                return
            zone_width  = zone.top - zone.bottom
            stop_dist   = max(zone_width + FIXED_STOP_BUFFER_PTS, MIN_STOP_PTS)
            limit_price = zone.top             # LONG: enter at zone top (first touch from above)
            stop_price  = zone.bottom - stop_dist  # stop below entire zone
        else:
            # No zone: fall back to current price with ATR-based stop
            stop_dist   = max(atr * 0.08, MIN_STOP_PTS)
            limit_price = current_price
            stop_price  = current_price - direction * stop_dist

        # Target: use zone's impulse extreme if available, else ATR projection
        target_price = self._compute_target_price(direction, limit_price, zone_state, atr_state)

        self._pending_order = {
            "direction":       direction,
            "limit_price":     limit_price,
            "stop_price":      stop_price,
            "target_price":    target_price,
            "confluence_score": confluence,
            "placed_bar_idx":  self._current_step,
            "atr_state":       atr_state,
        }
        log.debug(
            "Pending order placed",
            direction="LONG" if direction == 1 else "SHORT",
            limit_price=limit_price,
        )

    def _check_pending_fill(self, bar_high: float, bar_low: float) -> bool:
        """Return True if the current bar's range touches the pending limit price."""
        po = self._pending_order
        if po is None:
            return False
        if po["direction"] == 1:    # LONG: price must dip to or below limit
            return bar_low <= po["limit_price"]
        else:                       # SHORT: price must rise to or above limit
            return bar_high >= po["limit_price"]

    def _open_from_pending(self, current_bar_idx: int) -> None:
        """Open a position at the pending limit price and clear the pending order."""
        po = self._pending_order
        if po is None:
            return

        success, reason = self.position_manager.enter(
            direction=po["direction"],
            current_price=po["limit_price"],   # fill at the limit price
            stop_price=po["stop_price"],
            target_price=po["target_price"],
            current_bar_idx=current_bar_idx,
            atr=po["atr_state"].atr_daily,
            confluence_score=po["confluence_score"],
        )

        if success:
            self._entry_order_zone_state = (
                self._last_order_zone_state
                if hasattr(self, "_last_order_zone_state") else None
            )
            self._entry_atr_state = po["atr_state"]
            self._peak_unrealised_r = 0.0
            log.debug(
                "Pending order filled",
                direction="LONG" if po["direction"] == 1 else "SHORT",
                fill_price=po["limit_price"],
            )
        else:
            log.debug("Pending order fill rejected by PositionManager", reason=reason)

        self._pending_order = None

    def _compute_target_price(
        self,
        direction: int,
        entry_price: float,
        zone_state,
        atr_state,
    ) -> float:
        """
        Determine the take-profit target.

        Priority:
          1. Zone impulse extreme — the high of the bullish impulse bar (LONG target)
             or the low of the bearish impulse bar (SHORT target).  This is the
             natural swing level the market already reached once from this zone,
             making it a realistic, strategy-faithful objective.
          2. ATR projection fallback — used when no zone or impulse_extreme is 0.

        The target is validated: for LONG it must be above entry; for SHORT below.
        If the impulse extreme fails validation (e.g. zone formed from a small bar),
        we fall back to the ATR projection.
        """
        from features.atr_calculator import ATRCalculator

        zone = None
        if direction == 1 and zone_state and zone_state.nearest_demand:
            zone = zone_state.nearest_demand
        elif direction == -1 and zone_state and zone_state.nearest_supply:
            zone = zone_state.nearest_supply

        if zone is not None and zone.impulse_extreme != 0.0:
            extreme = zone.impulse_extreme
            # Validate direction: LONG target must be above entry, SHORT below
            if direction == 1 and extreme > entry_price:
                return extreme
            if direction == -1 and extreme < entry_price:
                return extreme

        # Fallback: ATR remaining projection
        return ATRCalculator.compute_atr_target_price(entry_price, direction, atr_state)

    def _sample_episode_day(self) -> str:
        """Sample a trading day, applying curriculum filter if set."""
        candidates = self._available_days
        if self.curriculum_filter_fn is not None:
            filtered = []
            for d in candidates:
                daily_bar = self.data_loader.get_daily_bar(d)
                if daily_bar is not None and self.curriculum_filter_fn(d, daily_bar):
                    filtered.append(d)
            candidates = filtered if filtered else self._available_days

        idx = int(self._rng.integers(0, len(candidates)))
        return candidates[idx]

    def _set_shaping_scale(self, scale: float) -> None:
        """Called by ShapingDecayCallback to update reward shaping weight."""
        self.reward_calculator.shaping_scale = scale

    def _filter_rth(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Keep only Regular Trading Hours bars."""
        tz = bars.index.tz
        start = pd.Timestamp(f"{self._current_day} {self.rth_start}").tz_localize(tz)
        end   = pd.Timestamp(f"{self._current_day} {self.rth_end}").tz_localize(tz)
        return bars.loc[(bars.index >= start) & (bars.index <= end)]

    def _filter_globex(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Keep bars outside RTH (pre-market 00:00→rth_start and
        post-settlement rth_end→23:59).  These together constitute
        the Globex overnight window within a single calendar day's file.
        """
        tz = bars.index.tz
        rth_start = pd.Timestamp(f"{self._current_day} {self.rth_start}").tz_localize(tz)
        rth_end   = pd.Timestamp(f"{self._current_day} {self.rth_end}").tz_localize(tz)
        return bars.loc[(bars.index < rth_start) | (bars.index > rth_end)]

    def _episode_summary(self) -> dict:
        """Build full episode summary info dict used by callbacks and eval."""
        trades  = self.position_manager.completed_trades
        wins    = [t for t in trades if t.is_win]
        losses  = [t for t in trades if not t.is_win]

        n_trades = len(trades)
        n_wins   = len(wins)
        n_losses = len(losses)

        # ── P&L ───────────────────────────────────────────────
        total_r       = sum(t.pnl_r      for t in trades)
        total_dollars = sum(t.pnl_dollars for t in trades)
        win_rate      = n_wins / n_trades if n_trades else 0.0

        avg_win_r       = float(np.mean([t.pnl_r      for t in wins]))   if wins   else 0.0
        avg_loss_r      = float(abs(np.mean([t.pnl_r  for t in losses]))) if losses else 1.0
        avg_win_dollars = float(np.mean([t.pnl_dollars for t in wins]))   if wins   else 0.0
        avg_loss_dollars= float(np.mean([t.pnl_dollars for t in losses])) if losses else 0.0

        profit_factor = min(
            (sum(t.pnl_r for t in wins) / max(abs(sum(t.pnl_r for t in losses)), 1e-6))
            if trades else 0.0,
            99.99,
        )

        # ── Risk/reward & expected return ─────────────────────
        avg_rr          = avg_win_r / max(avg_loss_r, 1e-6)
        win_loss_ratio  = avg_win_r / max(avg_loss_r, 1e-6)
        expected_return = win_rate * avg_win_r - (1.0 - win_rate) * avg_loss_r

        # ── Sharpe (annualised, from per-trade returns) ───────
        # Formula: (mean_r / std_r) × √252  — same annualisation
        # used by the eval callback and V7 reference code.
        # Require ≥5 trades; clamp to [-9.99, 9.99] for column display.
        if n_trades >= 5:
            tr  = np.array([t.pnl_r for t in trades], dtype=np.float32)
            std = float(np.std(tr))
            raw = np.mean(tr) / std if std > 0.01 else 0.0
            sharpe_ratio = float(np.clip(raw * np.sqrt(252), -9.99, 9.99))
        else:
            sharpe_ratio = 0.0

        # ── Conservative max drawdown ─────────────────────────
        # For each trade the equity dipped to (equity_before - mae_r)
        # before recovering to (equity_before + pnl_r).
        # Using candle low (LONG) / candle high (SHORT) via the pre-recorded
        # max_adverse_excursion field gives the worst realistic intra-trade
        # equity trough — not just the closed-trade outcome.
        if trades:
            equity_before  = 0.0
            running_peak   = 0.0
            max_drawdown_r = 0.0
            for t in trades:
                worst = equity_before - t.max_adverse_excursion
                max_drawdown_r = max(max_drawdown_r, running_peak - worst)
                equity_before += t.pnl_r
                running_peak   = max(running_peak, equity_before)
            max_drawdown_pct = max_drawdown_r * self.position_manager.risk_per_trade_pct * 100.0
        else:
            max_drawdown_r   = 0.0
            max_drawdown_pct = 0.0

        # ── Duration ──────────────────────────────────────────
        if trades:
            dur_bars              = [t.duration_bars for t in trades]
            avg_trade_duration    = float(np.mean(dur_bars))
            avg_duration_minutes  = avg_trade_duration * self.bar_minutes
            min_duration_minutes  = int(min(dur_bars)) * self.bar_minutes
            max_duration_minutes  = int(max(dur_bars)) * self.bar_minutes
        else:
            avg_trade_duration   = 0.0
            avg_duration_minutes = 0.0
            min_duration_minutes = 0
            max_duration_minutes = 0

        # ── Max win / loss dollars ────────────────────────────
        max_win_dollars  = float(max((t.pnl_dollars for t in wins),   default=0.0))
        max_loss_dollars = float(min((t.pnl_dollars for t in losses), default=0.0))

        # ── RTH vs ETH split ──────────────────────────────────
        rth_trades_n = rth_wins_n = eth_trades_n = eth_wins_n = 0
        if self._session_bars is not None and trades:
            try:
                rth_s = pd.Timestamp(f"2000-01-01 {self.rth_start}").time()
                rth_e = pd.Timestamp(f"2000-01-01 {self.rth_end}").time()
                for t in trades:
                    idx = min(t.entry_bar_idx, len(self._session_bars) - 1)
                    row = self._session_bars.iloc[idx]
                    # After random_start reset_index(drop=False) the DatetimeIndex
                    # becomes integers and the timestamp moves to a "datetime" column.
                    if hasattr(self._session_bars.index, "hour"):
                        bar_ts = self._session_bars.index[idx]
                    elif "datetime" in self._session_bars.columns:
                        bar_ts = pd.Timestamp(row["datetime"])
                    else:
                        bar_ts = None
                    if bar_ts is None:
                        rth_trades_n += 1
                        rth_wins_n   += int(t.is_win)
                        continue
                    bar_time = bar_ts.time()
                    if rth_s <= bar_time <= rth_e:
                        rth_trades_n += 1
                        rth_wins_n   += int(t.is_win)
                    else:
                        eth_trades_n += 1
                        eth_wins_n   += int(t.is_win)
            except Exception:
                rth_trades_n = n_trades  # fallback: all RTH

        return {
            "date":                    self._current_day,
            "total_reward":            sum(self._episode_rewards),
            # P&L
            "total_pnl_r":             total_r,
            "total_pnl_dollars":       total_dollars,
            # Counts
            "n_trades":                n_trades,
            "n_wins":                  n_wins,
            "n_losses":                n_losses,
            "win_rate":                win_rate,
            # Per-trade averages
            "avg_win_r":               avg_win_r,
            "avg_loss_r":              avg_loss_r,
            "avg_win_dollars":         avg_win_dollars,
            "avg_loss_dollars":        avg_loss_dollars,
            "max_win_dollars":         max_win_dollars,
            "max_loss_dollars":        max_loss_dollars,
            # Ratios
            "profit_factor":           profit_factor,
            "avg_rr":                  avg_rr,
            "win_loss_ratio":          win_loss_ratio,
            "expected_return":         expected_return,
            "sharpe_ratio":            sharpe_ratio,
            # Drawdown
            "max_drawdown_r":          max_drawdown_r,
            "max_drawdown_pct":        max_drawdown_pct,
            # Duration
            "avg_trade_duration":      avg_trade_duration,
            "avg_duration_minutes":    avg_duration_minutes,
            "min_duration_minutes":    min_duration_minutes,
            "max_duration_minutes":    max_duration_minutes,
            # Session split
            "rth_trades":              rth_trades_n,
            "rth_wins":                rth_wins_n,
            "eth_trades":              eth_trades_n,
            "eth_wins":                eth_wins_n,
            # Individual trade records (plain dicts — safe across SubprocVecEnv)
            "trades_list": [
                {
                    "date":           self._current_day,
                    "direction":      t.direction.name,
                    "entry_price":    round(t.entry_price,    2),
                    "stop_price":     round(t.stop_price,     2),
                    "initial_target": round(t.initial_target, 2),
                    "exit_price":     round(t.exit_price,     2),
                    "pnl_r":          round(t.pnl_r,          4),
                    "pnl_dollars":    round(t.pnl_dollars,    2),
                    "pnl_points":     round(t.pnl_points,     2),
                    "n_contracts":    t.n_contracts,
                    "entry_bar_idx":  t.entry_bar_idx,
                    "exit_bar_idx":   t.exit_bar_idx,
                    "duration_bars":  t.duration_bars,
                    "duration_min":   t.duration_bars * self.bar_minutes,
                    "exit_reason":    t.exit_reason.value,
                    "is_win":         t.is_win,
                    "mae_r":          round(t.max_adverse_excursion, 4),
                }
                for t in trades
            ],
        }