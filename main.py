"""
main.py
========
Entry point for the RL Trading Agent.

Usage:
    # Train from scratch
    python main.py --mode train --config config/ --data data

    # Continue training from checkpoint
    python main.py --mode train --config config/ --data data/ --checkpoint logs/checkpoints/best_model.zip

    # Evaluate (backtest) a trained model
    python main.py --mode evaluate --config config/ --data data/ \\
                   --checkpoint logs/checkpoints/best_model.zip

    # train on custom date range (e.g. for walk-forward fold)
    python main.py --mode walk_forward --config config/ --data data/\\
        --train-start 2021-01-02 --train-end 2025-12-31

    # Evaluate ALL saved checkpoints/hotsaves against a specific date range
    python main.py --mode test_fold --config config/ --data data/ \\
        --models-dir logs/walk_forward/fold_00/models \\
        --test-start 2026-03-01 --test-end 2026-04-09

    # Print journal analysis for a completed backtest
    python main.py --mode analyse --journal logs/journal/

    # Walk-forward analysis (rolling 12-month train / 5-week val folds)
    python main.py --mode walk_forward --config config/ --data data/

All parameters are loaded from YAML config files — no command-line
parameter overrides for model hyperparameters (edit the YAML instead).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from utils.logger import configure_logging, get_logger, tee_stdout
from utils.metrics_printer import init_console
from utils.validators import validate_all_configs, assert_instrument_allowed

log = get_logger(__name__)


# Zone detector defaults — shared by build_components() and run_walk_forward()
_ZONE_DETECTOR_DEFAULTS: dict = {
    "consolidation_min_bars": 2, "consolidation_max_bars": 8,
    "consolidation_range_atr_pct": 0.20, "impulse_min_body_atr_pct": 0.15,
    "max_zone_age_bars": 200, "max_zone_touches": 3, "zone_buffer_atr_pct": 0.02,
}


# ── Config loader ─────────────────────────────────────────────────────────────

def load_configs(config_dir: str) -> dict:
    """
    Load all YAML config files from config_dir.

    Parameters
    ----------
    config_dir : str
        Path to directory containing: agent_config.yaml,
        environment_config.yaml, features_config.yaml,
        risk_config.yaml, reward_config.yaml, logging_config.yaml.

    Returns
    -------
    dict with keys: agent, environment, features, risk, reward, logging.
    """
    config_dir = Path(config_dir)
    file_map = {
        "agent": "agent_config.yaml",
        "environment": "environment_config.yaml",
        "features": "features_config.yaml",
        "risk": "risk_config.yaml",
        "reward": "reward_config.yaml",
        "logging": "logging_config.yaml",
    }

    configs = {}
    for key, filename in file_map.items():
        path = config_dir / filename
        if path.exists():
            with open(path, "r") as f:
                configs[key] = yaml.safe_load(f)
        else:
            log.warning(f"Config file not found: {path} — using defaults.")
            configs[key] = {}

    return configs


# ── Factory functions (wire components together) ──────────────────────────────

def build_components(
    configs: dict,
    data_dir: str,
    train_start: str | None = None,
    train_end: str | None = None,
    val_weeks: int = 5,
):
    """
    Instantiate all components from config.

    Returns a namespace with all constructed objects.
    """
    import numpy as np
    from data.data_augmentor import OHLCVAugmentor
    from data.data_loader import DataLoader
    from data.data_splitter import DataSplitter
    from environment.action_space import ActionMasker
    from environment.position_manager import PositionManager
    from environment.reward_calculator import RewardCalculator
    from environment.trading_env import TradingEnv
    from features.atr_calculator import ATRCalculator
    from features.observation_builder import ObservationBuilder
    from features.order_zone_engine import OrderZoneEngine
    from features.zone_detector import ZoneDetector
    from training.curriculum import CurriculumScheduler

    env_cfg = configs["environment"]
    feat_cfg = configs["features"]
    risk_cfg = configs["risk"]
    reward_cfg = configs["reward"]
    agent_cfg = configs["agent"]

    instrument = env_cfg.get("instruments", {}).get("default", "NQ")
    allowed = env_cfg.get("instruments", {}).get("allowed", ["NQ", "ES", "MNQ", "MES"])
    assert_instrument_allowed(instrument, allowed)

    # ── Data ─────────────────────────────────────────────────
    session_cfg = env_cfg.get("session", {})
    data_loader = DataLoader(
        data_dir=data_dir,
        instrument=instrument,
        intraday_tf=f"{session_cfg.get('bar_timeframe_minutes', 5)}min",
        daily_tf=session_cfg.get("daily_timeframe", "1D"),
        tz=session_cfg.get("timezone", "America/New_York"),
    )
    data_loader.load()

    all_days = data_loader.get_trading_days()

    # ── ATR ──────────────────────────────────────────────────
    atr_cfg = feat_cfg.get("atr", {})
    atr_calculator = ATRCalculator(
        atr_period=atr_cfg.get("period", 14),
        exhaustion_threshold=atr_cfg.get("exhaustion_threshold", 0.95),
    )
    atr_calculator.fit(data_loader.daily)

    # Filter to valid weekday sessions that have ATR history
    import pandas as _pd
    trading_days = [
        d for d in all_days
        if _pd.Timestamp(d).weekday() < 5                  # Mon–Fri only
        and atr_calculator.get_atr_for_date(d) is not None  # ATR warmup complete
    ]

    # ── Train/val/test split ──────────────────────────────────
    # Date-based split: when --train-start/--train-end are provided the user
    # defines the exact training window.  Val is the next val_weeks*5 trading
    # days after train_end; test is everything that follows.
    # Fallback: fixed-count split (252 train, 26 val) using the full day pool.
    n_val_days = val_weeks * 5

    if train_end:
        # Clip pool start when requested
        pool = [d for d in trading_days if (not train_start or d >= train_start)]
        train_days_list = [d for d in pool if d <= train_end]
        post_train      = [d for d in pool if d >  train_end]
        val_days_list   = post_train[:n_val_days]
        test_days_list  = post_train[n_val_days:]

        if not train_days_list:
            raise ValueError(
                f"No trading days found in train window "
                f"[{train_start or 'start'} → {train_end}]. "
                f"Check --train-start / --train-end and your CSV date range."
            )
        if not val_days_list:
            raise ValueError(
                f"No trading days available for validation after {train_end}. "
                f"Extend --train-end or reduce --val-weeks."
            )
        if not test_days_list:
            log.warning(
                "No test days after validation window — test set is empty.",
                val_end=val_days_list[-1],
            )

        from data.data_splitter import DataSplit
        split = DataSplit(
            train=train_days_list,
            validation=val_days_list,
            test=test_days_list,
        )
    else:
        # Legacy fixed-count split: first 252 days train, next 26 val, rest test
        if train_start:
            trading_days = [d for d in trading_days if d >= train_start]
        split = DataSplitter.split_by_counts(trading_days, n_train=252, n_val=n_val_days)

    log.info(
        "Data split",
        train_days=len(split.train),
        val_days=len(split.validation),
        test_days=len(split.test),
        train_range=f"{split.train[0]} → {split.train[-1]}",
        val_range=f"{split.validation[0]} → {split.validation[-1]}",
        test_range=f"{split.test[0]} → {split.test[-1]}" if split.test else "none",
    )

    # ── Zone Detector ─────────────────────────────────────────
    zones_cfg = feat_cfg.get("zones", {})
    zone_detector = ZoneDetector(
        consolidation_min_bars=zones_cfg.get("consolidation_min_bars", 2),
        consolidation_max_bars=zones_cfg.get("consolidation_max_bars", 8),
        consolidation_range_atr_pct=zones_cfg.get("consolidation_range_atr_pct", 0.20),
        impulse_min_body_atr_pct=zones_cfg.get("impulse_min_body_atr_pct", 0.15),
        max_zone_age_bars=zones_cfg.get("max_zone_age_bars", 200),
        max_zone_touches=zones_cfg.get("max_zone_touches", 3),
        zone_buffer_atr_pct=zones_cfg.get("zone_buffer_atr_pct", 0.02),
    )

    # ── Order Zone Engine ─────────────────────────────────────
    oz_cfg = feat_cfg.get("order_zone", {})
    order_zone_engine = OrderZoneEngine(
        weights=oz_cfg.get("weights"),
        min_confluence_score=oz_cfg.get("min_confluence_score", 0.60),
        min_rr_ratio=risk_cfg.get("take_profit", {}).get("min_rr_ratio", 1.5),
    )

    # ── Observation Builder ───────────────────────────────────
    obs_cfg = env_cfg.get("observation", {})
    observation_builder = ObservationBuilder(
        clip_value=obs_cfg.get("clip_observations", 10.0),
        normalize_observations=obs_cfg.get("normalize_observations", True),
        lookback_bars=obs_cfg.get("lookback_bars", 20),
        max_zone_age_bars=zones_cfg.get("max_zone_age_bars", 300),
    )

    # ── Action Masker ─────────────────────────────────────────
    atr_gate_cfg = risk_cfg.get("atr_gate", {})
    daily_lim_cfg = risk_cfg.get("daily_limits", {})
    session_risk_cfg = risk_cfg.get("session", {})
    action_masker = ActionMasker(
        atr_exhaustion_threshold=atr_gate_cfg.get("block_entries_above_pct", 0.95),
        trail_min_r=risk_cfg.get("trailing", {}).get("activate_at_r", 2.0),
        max_trades_per_day=daily_lim_cfg.get("max_trades_per_day", 5),
        no_entry_last_n_bars=session_risk_cfg.get("no_entry_last_n_bars", 3),
    )

    # ── Position Manager ──────────────────────────────────────
    account_cfg = env_cfg.get("account", {})
    sizing_cfg = risk_cfg.get("sizing", {})
    trail_cfg = risk_cfg.get("trailing", {})
    contracts_cfg = env_cfg.get("contracts", {}).get(instrument, {})
    real_capital = float(account_cfg.get("initial_balance", 2500))
    point_value = float(contracts_cfg.get("micro_point_value", 2.0))

    def make_position_manager():
        return PositionManager(
            real_capital=real_capital,
            risk_per_trade_pct=sizing_cfg.get("risk_per_trade_pct", 0.01),
            min_contracts=sizing_cfg.get("min_contracts", 0.5),
            max_contracts=sizing_cfg.get("max_contracts", 2.5),
            point_value=point_value,
            max_trades_per_day=daily_lim_cfg.get("max_trades_per_day", 5),
            trail_activate_r=trail_cfg.get("activate_at_r", 2.0),
            trail_aggressive_r=trail_cfg.get("trail_aggressively_at_r", 4.0),
            trail_lock_in_r=trail_cfg.get("lock_in_r_at_trail", 2.0),
            max_daily_loss_r=daily_lim_cfg.get("max_daily_loss_r", 3.0),
            max_daily_loss_dollars=daily_lim_cfg.get("max_daily_loss_dollars", 1000.0),
            max_drawdown_r=risk_cfg.get("position", {}).get("max_drawdown_r", 5.0),
            pause_bars_after_loss_streak=daily_lim_cfg.get("pause_bars_after_loss_streak", 6),
            loss_streak_threshold=daily_lim_cfg.get("max_consecutive_losses_before_pause", 3),
            zone_buffer_atr_pct=risk_cfg.get("stop_loss", {}).get("zone_buffer_atr_pct", 0.03),
            contract_tiers=sizing_cfg.get("contract_tiers"),
            confluence_tier_thresholds=sizing_cfg.get("confluence_tier_thresholds"),
        )

    # ── Reward Calculator ─────────────────────────────────────
    reward_calculator = RewardCalculator.from_config(reward_cfg)

    # ── Curriculum ────────────────────────────────────────────
    curriculum_cfg = env_cfg.get("curriculum", {})
    curriculum_scheduler = None
    if curriculum_cfg.get("enabled", True):
        curriculum_scheduler = CurriculumScheduler.from_config(
            curriculum_cfg.get("stages", [])
        )
        log.info(curriculum_scheduler.stage_summary())

    # ── Build environments ────────────────────────────────────
    session_start   = session_cfg.get("rth_start_utc", "08:30")
    session_end     = session_cfg.get("rth_end_utc",   "15:00")
    session_type    = session_cfg.get("session_type", "RTH").upper()

    # ── OHLCV Augmentor (training only) ──────────────────────
    # Jitter is discrete: OHLC {-0.5,-0.25,0,+0.25,+0.5}, Volume {-10,-5,0,+5,+10}
    train_augmentor = OHLCVAugmentor(rng=np.random.default_rng(agent_cfg.get("seed", 42)))

    bar_minutes = int(session_cfg.get("bar_timeframe_minutes", 5))

    def make_env(day_list, is_eval=False, worker_seed_offset=0):
        return TradingEnv(
            data_loader=data_loader,
            trading_days=day_list,
            position_manager=make_position_manager(),
            reward_calculator=reward_calculator,
            observation_builder=observation_builder,
            atr_calculator=atr_calculator,
            zone_detector=ZoneDetector(**{
                k: zones_cfg.get(k, v)
                for k, v in _ZONE_DETECTOR_DEFAULTS.items()
            }),
            order_zone_engine=order_zone_engine,
            action_masker=action_masker,
            rth_start=session_start,
            rth_end=session_end,
            no_entry_last_n_bars=session_risk_cfg.get("no_entry_last_n_bars", 3),
            early_terminate_on_max_dd=env_cfg.get("episode", {}).get("early_termination_on_max_drawdown", True),
            point_value=point_value,
            bar_minutes=bar_minutes,
            curriculum_filter_fn=None,
            augmentor=None if is_eval else train_augmentor,
            session_type=session_type,
            random_start=not is_eval,
            seed=agent_cfg.get("seed", 42) + worker_seed_offset + (100 if is_eval else 0),
            zone_lookback_bars=feat_cfg.get("zone_lookback_bars", 500),
        )

    # ── Vectorised training env (multiprocessing) ─────────────
    mp_cfg  = agent_cfg.get("multiprocessing", {})
    n_envs  = int(mp_cfg.get("n_envs", 4))
    use_subproc = mp_cfg.get("use_subprocess", True)

    def _make_train_env_fn(offset: int):
        """Return a zero-argument callable for VecEnv factories."""
        def _fn():
            return make_env(split.train, is_eval=False, worker_seed_offset=offset)
        return _fn

    train_env_fns = [_make_train_env_fn(i) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        try:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            train_vec_env = SubprocVecEnv(train_env_fns, start_method="spawn")
            log.info("Using SubprocVecEnv for training", n_envs=n_envs)
        except Exception as exc:
            log.warning("SubprocVecEnv failed, falling back to DummyVecEnv", error=str(exc))
            from stable_baselines3.common.vec_env import DummyVecEnv
            train_vec_env = DummyVecEnv(train_env_fns)
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        train_vec_env = DummyVecEnv(train_env_fns)
        log.info("Using DummyVecEnv for training", n_envs=n_envs)

    # ── VecNormalize: normalise observations, NOT rewards ────────────────────
    # norm_obs=True  → running mean/std normalisation of observations
    # norm_reward=False → keep raw R-multiple rewards (LSTM value estimates
    #                     are sensitive to reward scale changes mid-training)
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv as _DummyVecEnv
    train_vec_env = VecNormalize(
        train_vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=True,
    )

    # Eval/test share the same normalisation stats (frozen — no update during eval)
    eval_vec_env = VecNormalize(
        _DummyVecEnv([lambda: make_env(split.validation, is_eval=True)]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,   # don't update running stats during eval
    )
    test_vec_env = VecNormalize(
        _DummyVecEnv([lambda: make_env(split.test, is_eval=True)]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    # Keep train VecNormalize reference so trainer can save its stats
    train_env = train_vec_env
    eval_env  = eval_vec_env
    test_env  = test_vec_env

    # ── Namespace ─────────────────────────────────────────────
    class Components:
        pass

    c = Components()
    c.data_loader = data_loader
    c.split = split
    c.atr_calculator = atr_calculator
    c.train_env      = train_env
    c.eval_env       = eval_env
    c.test_env       = test_env
    c.vec_normalize  = train_vec_env   # reference for saving normalisation stats
    c.curriculum_scheduler = curriculum_scheduler
    c.reward_calculator = reward_calculator
    c.real_capital = real_capital

    return c


# ── Clean-slate helper ────────────────────────────────────────────────────────

def clean_run_dirs(log_dir: Path) -> None:
    """
    Delete all generated output under log_dir so each run starts fresh.
    Skipped automatically when --no-clean is passed or --checkpoint is set.
    Uses an onerror handler to force-remove read-only files on Windows.
    """
    import shutil
    import stat

    def _force_remove(func, path, *_):
        """On Windows, read-only files raise PermissionError — chmod then retry."""
        try:
            import os
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass  # best-effort; skip files that are truly locked

    subdirs = [
        log_dir / "models",
        log_dir / "checkpoints",
        log_dir / "tensorboard",
        log_dir / "journal",
        log_dir / "walk_forward",
        log_dir / "hotsaves",
    ]
    log_file = log_dir / "metrics.log"

    print("\n========================================")
    print("  CLEAN SLATE — removing previous outputs")
    print("========================================")

    removed = []
    for d in subdirs:
        if d.exists():
            print(f"  Removing folder : {d}")
            shutil.rmtree(d, onerror=_force_remove)
            removed.append(str(d))
    if log_file.exists():
        print(f"  Removing file   : {log_file}")
        log_file.unlink(missing_ok=True)
        removed.append(str(log_file))

    if removed:
        print(f"  Done — {len(removed)} item(s) removed.")
    else:
        print("  Nothing to remove — already clean.")
    print("========================================\n")


# ── Mode handlers ─────────────────────────────────────────────────────────────

def run_train(args: argparse.Namespace, configs: dict) -> None:
    from agent.ppo_agent import PPOAgent
    from training.checkpoint_manager import CheckpointManager
    from training.trainer import Trainer

    log.info("Mode: TRAIN")

    c = build_components(
        configs, args.data,
        train_start=args.train_start,
        train_end=args.train_end,
        val_weeks=args.val_weeks if args.val_weeks is not None else 5,
    )

    agent_cfg = configs["agent"]
    ppo_cfg = agent_cfg.get("ppo", {})
    net_cfg = agent_cfg.get("network", {})
    exp_cfg = agent_cfg.get("exploration", {})

    from datetime import datetime
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)

    # ── Folder layout ─────────────────────────────────────────
    #   logs/
    #     models/          ← best model + periodic checkpoints saved here
    #     tensorboard/     ← TensorBoard event files
    #     metrics.log      ← console table mirror
    models_dir      = log_dir / "models"
    checkpoints_dir = models_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "tensorboard").mkdir(parents=True, exist_ok=True)

    log.info("Run folders created", models=str(models_dir), run_id=run_id)

    agent = PPOAgent(
        env=c.train_env,
        algorithm=agent_cfg.get("algorithm", "RecurrentPPO"),
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        learning_rate_schedule=ppo_cfg.get("learning_rate_schedule", "linear"),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 256),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        hidden_dims=net_cfg.get("net_arch", {}).get("pi", [256, 128]),
        use_layer_norm=net_cfg.get("use_layer_norm", True),
        use_lstm=net_cfg.get("use_lstm", True),
        lstm_hidden_size=net_cfg.get("lstm_hidden_size", 256),
        n_lstm_layers=net_cfg.get("n_lstm_layers", 1),
        activation_fn_name=net_cfg.get("activation_fn", "ReLU"),
        ortho_init=net_cfg.get("ortho_init", True),
        device=agent_cfg.get("device", "auto"),
        seed=agent_cfg.get("seed", 42),
        tensorboard_log=str(log_dir / "tensorboard"),
    )

    if args.checkpoint:
        log.info("Loading checkpoint for resume", path=args.checkpoint)
        agent.load(args.checkpoint, env=c.train_env)
        # Restore VecNormalize stats so running mean/std continues from where
        # training left off rather than starting fresh.
        ckpt_path = Path(args.checkpoint)
        for candidate in ["vecnormalize.pkl", "vec_normalize.pkl"]:
            vn_resume_path = ckpt_path.parent / candidate
            if not vn_resume_path.exists():
                # Also check models dir (two levels up from checkpoints/)
                vn_resume_path = ckpt_path.parent.parent / candidate
            if vn_resume_path.exists():
                from stable_baselines3.common.vec_env import VecNormalize
                c.train_env = VecNormalize.load(str(vn_resume_path), c.train_env)
                c.train_env.training    = True
                c.train_env.norm_reward = False
                c.vec_normalize         = c.train_env
                log.info("VecNormalize stats restored for resume", path=str(vn_resume_path))
                break
        else:
            log.warning("No VecNormalize stats found for resume — running stats will restart from scratch")

    ckpt_cfg = agent_cfg.get("checkpointing", {})
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoints_dir,
        save_freq=ckpt_cfg.get("save_freq", 100_000),
        keep_n_checkpoints=ckpt_cfg.get("keep_n_checkpoints", 5),
    )

    eval_cfg    = agent_cfg.get("evaluation", {})
    hotsave_cfg = agent_cfg.get("hotsave", {})
    n_training_days   = len(c.split.train)
    min_trades_per_wk = hotsave_cfg.get("min_trades_per_week", 1)
    hotsave_min_trades = max(10, n_training_days * min_trades_per_wk // 5)

    trainer = Trainer(
        agent=agent,
        train_env=c.train_env,
        eval_env=c.eval_env,
        checkpoint_manager=checkpoint_manager,
        curriculum_scheduler=c.curriculum_scheduler,
        n_training_days=n_training_days,
        total_timesteps=ppo_cfg.get("total_timesteps", 2_000_000),
        eval_freq=eval_cfg.get("eval_freq",       50_000),
        n_eval_episodes=eval_cfg.get("n_eval_episodes", 20),
        warmup_steps=eval_cfg.get("warmup_steps",   400_000),
        patience_steps=eval_cfg.get("patience_steps", 550_000),
        w_sharpe=eval_cfg.get("w_sharpe", 0.30),
        w_pnl=eval_cfg.get("w_pnl",    0.25),
        w_wl=eval_cfg.get("w_wl",     0.25),
        w_dd=eval_cfg.get("w_dd",     0.20),
        eval_save_enabled=eval_cfg.get("save_enabled", True),
        ent_coef_start=exp_cfg.get("ent_coef_start",        0.05),
        ent_coef_end=exp_cfg.get("ent_coef_end",          0.005),
        ent_coef_decay_steps=exp_cfg.get("ent_coef_decay_steps", 1_000_000),
        log_dir=str(log_dir),
        models_dir=str(models_dir),
        train_date_range=f"{c.split.train[0]}→{c.split.train[-1]}",
        vec_normalize=c.vec_normalize,
        resume=bool(args.checkpoint),
        initial_capital=real_capital,
        hotsave_pf=hotsave_cfg.get("pf_threshold",          1.60),
        hotsave_wr=hotsave_cfg.get("wr_threshold",          0.40),
        hotsave_min_trades=hotsave_min_trades,
        hotsave_min_envs=hotsave_cfg.get("min_envs_passing",      2),
        hotsave_cooldown=hotsave_cfg.get("cooldown_steps",        50_000),
        hotsave_sharpe=hotsave_cfg.get("sharpe_threshold",      1.2),
        hotsave_sharpe_pf=hotsave_cfg.get("sharpe_pf_threshold",   1.85),
        hotsave_sharpe_cooldown=hotsave_cfg.get("sharpe_cooldown_steps", 50_000),
        hotsave_wr70_cooldown=hotsave_cfg.get("wr70_cooldown_steps",        50_000),
        hotsave_elite_pnl_multiplier=hotsave_cfg.get("elite_pnl_multiplier",   1.5),
        hotsave_elite_wr_pf_threshold=hotsave_cfg.get("elite_wr_pf_threshold", 1.5),
        hotsave_elite_sharpe=hotsave_cfg.get("elite_sharpe_threshold",         3.0),
        hotsave_elite_cooldown=hotsave_cfg.get("elite_cooldown_steps",         50_000),
    )

    trainer.run()
    log.info("Training run finished. Models saved to: %s", models_dir)
    # VecNormalize stats are saved by Trainer.run() as logs/models/vecnormalize.pkl
    # and by TradingEvalCallback as logs/models/best_model_vecnormalize.pkl —
    # no additional save needed here.


def run_evaluate(args: argparse.Namespace, configs: dict) -> None:
    from agent.ppo_agent import PPOAgent
    from evaluation.backtester import Backtester
    from evaluation.metrics_calculator import MetricsCalculator
    from evaluation.trade_journal import TradeJournal

    if not args.checkpoint:
        log.error("--checkpoint is required for evaluate mode.")
        sys.exit(1)

    log.info("Mode: EVALUATE", checkpoint=args.checkpoint)
    c = build_components(
        configs, args.data,
        train_start=args.train_start,
        train_end=args.train_end,
        val_weeks=args.val_weeks if args.val_weeks is not None else 5,
    )

    # Load VecNormalize stats so the agent receives the same scaled observations
    # it was trained with.  The filename depends on which checkpoint is loaded:
    #   best_model.zip        → best_model_vecnormalize.pkl  (stats at best eval)
    #   final_model.zip       → vecnormalize.pkl             (stats at end of training)
    #   anything else         → vec_normalize.pkl            (legacy fallback)
    from stable_baselines3.common.vec_env import VecNormalize
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.stem == "best_model":
        vec_candidates = ["best_model_vecnormalize.pkl", "vecnormalize.pkl", "vec_normalize.pkl"]
    elif ckpt_path.stem == "final_model":
        vec_candidates = ["vecnormalize.pkl", "vec_normalize.pkl"]
    else:
        vec_candidates = ["vec_normalize.pkl", "vecnormalize.pkl"]

    vec_path = None
    for candidate in vec_candidates:
        candidate_path = ckpt_path.parent / candidate
        if candidate_path.exists():
            vec_path = candidate_path
            break

    if vec_path is not None:
        c.test_env = VecNormalize.load(str(vec_path), c.test_env)
        c.test_env.training    = False
        c.test_env.norm_reward = False
        log.info("VecNormalize stats loaded", path=str(vec_path))
    else:
        log.warning(
            "No VecNormalize stats file found — observations will not be correctly scaled!",
            searched=[str(ckpt_path.parent / f) for f in vec_candidates],
        )

    agent = PPOAgent.from_checkpoint(
        args.checkpoint,
        env=c.test_env,
        algorithm=configs["agent"].get("algorithm", "RecurrentPPO"),
    )

    journal = TradeJournal(
        journal_dir=Path(args.log_dir) / "journal",
        agent_run_id=Path(args.checkpoint).stem,
        min_rr_ratio=configs["risk"].get("take_profit", {}).get("min_rr_ratio", 4.0),
    )

    metrics_calculator = MetricsCalculator()

    backtester = Backtester(
        env=c.test_env,
        agent=agent,
        journal=journal,
        metrics_calculator=metrics_calculator,
        deterministic=True,
    )

    results = backtester.run()
    journal.print_summary()
    journal.export_csv()

    log.info("Backtest results", **{
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in results["metrics"].items()
    })


def run_walk_forward(args: argparse.Namespace, configs: dict) -> None:
    """
    Walk-forward analysis.

    For each fold:
      1. Build train/val environments from the fold's date ranges.
      2. Train a fresh agent on the training window.
      3. Save fold journal (Excel + HTML) to <output_dir>/fold_<N>/.

    Config keys (agent_config.yaml → walk_forward):
      n_folds      : int  — folds to run (-1 = all possible, default 1)
      train_months : int  — training window in calendar months (default 12)
      val_weeks    : int  — validation window in weeks (default 5)
      output_dir   : str  — root output dir (default logs/walk_forward)
    """
    import numpy as np
    import pandas as _pd
    from datetime import datetime

    from agent.ppo_agent import PPOAgent
    from data.data_augmentor import OHLCVAugmentor
    from data.data_loader import DataLoader
    from data.data_splitter import DataSplitter
    from environment.action_space import ActionMasker
    from environment.position_manager import PositionManager
    from environment.reward_calculator import RewardCalculator
    from environment.trading_env import TradingEnv
    from features.atr_calculator import ATRCalculator
    from features.observation_builder import ObservationBuilder
    from features.order_zone_engine import OrderZoneEngine
    from features.zone_detector import ZoneDetector
    from training.checkpoint_manager import CheckpointManager

    from training.fold_journal_callback import FoldJournalCallback
    from training.trainer import Trainer

    log.info("Mode: WALK_FORWARD")

    agent_cfg   = configs["agent"]
    env_cfg     = configs["environment"]
    feat_cfg    = configs["features"]
    risk_cfg    = configs["risk"]
    reward_cfg  = configs["reward"]
    ppo_cfg     = agent_cfg.get("ppo", {})
    net_cfg     = agent_cfg.get("network", {})
    exp_cfg     = agent_cfg.get("exploration", {})
    mp_cfg      = agent_cfg.get("multiprocessing", {})
    wf_cfg      = agent_cfg.get("walk_forward", {})
    eval_cfg    = agent_cfg.get("evaluation", {})
    ckpt_cfg    = agent_cfg.get("checkpointing", {})
    hotsave_cfg = agent_cfg.get("hotsave", {})

    # ── Walk-forward parameters ───────────────────────────────
    n_folds_cfg    = int(wf_cfg.get("n_folds", 1))
    train_months   = int(wf_cfg.get("train_months", 12))
    val_weeks      = int(wf_cfg.get("val_weeks", 5))
    n_train_days   = train_months * 21      # approx trading days per month
    # CLI --val-weeks overrides config when explicitly passed; otherwise config wins.
    _cli_val_weeks = getattr(args, "val_weeks", None)
    n_val_days = (_cli_val_weeks if _cli_val_weeks is not None else val_weeks) * 5
    output_root    = Path(wf_cfg.get("output_dir", "logs/walk_forward"))
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Common infrastructure (shared across folds) ────────────
    instrument   = env_cfg.get("instruments", {}).get("default", "NQ")
    session_cfg  = env_cfg.get("session", {})
    atr_cfg      = feat_cfg.get("atr", {})
    zones_cfg    = feat_cfg.get("zones", {})
    oz_cfg       = feat_cfg.get("order_zone", {})
    obs_cfg      = env_cfg.get("observation", {})
    atr_gate_cfg = risk_cfg.get("atr_gate", {})
    daily_lim_cfg= risk_cfg.get("daily_limits", {})
    session_risk = risk_cfg.get("session", {})
    account_cfg  = env_cfg.get("account", {})
    sizing_cfg   = risk_cfg.get("sizing", {})
    trail_cfg    = risk_cfg.get("trailing", {})
    contracts_cfg= env_cfg.get("contracts", {}).get(instrument, {})

    data_loader = DataLoader(
        data_dir=args.data,
        instrument=instrument,
        intraday_tf=f"{session_cfg.get('bar_timeframe_minutes', 5)}min",
        daily_tf=session_cfg.get("daily_timeframe", "1D"),
        tz=session_cfg.get("timezone", "America/New_York"),
    )
    data_loader.load()

    atr_calculator = ATRCalculator(
        atr_period=atr_cfg.get("period", 14),
        exhaustion_threshold=atr_cfg.get("exhaustion_threshold", 0.95),
    )
    atr_calculator.fit(data_loader.daily)

    all_days = data_loader.get_trading_days()
    trading_days = [
        d for d in all_days
        if _pd.Timestamp(d).weekday() < 5
        and atr_calculator.get_atr_for_date(d) is not None
    ]

    # Optional date-range clip (--train-start / --train-end)
    if getattr(args, "train_start", None):
        trading_days = [d for d in trading_days if d >= args.train_start]
    if getattr(args, "train_end", None):
        trading_days = [d for d in trading_days if d <= args.train_end]

    log.info(
        "Valid trading days for walk-forward pool",
        total=len(trading_days),
        date_range=f"{trading_days[0]} → {trading_days[-1]}" if trading_days else "empty",
    )

    # ── Build walk-forward folds ──────────────────────────────
    # When --train-end is supplied the user wants to train on the FULL date range
    # [train_start, train_end] as a single fold, with val immediately after.
    # Rolling walk-forward is only used when no explicit end date is given.
    train_end_arg = getattr(args, "train_end", None)
    if train_end_arg:
        all_days_pool = data_loader.get_trading_days()
        all_trading = [
            d for d in all_days_pool
            if _pd.Timestamp(d).weekday() < 5
            and atr_calculator.get_atr_for_date(d) is not None
        ]
        train_start_arg = getattr(args, "train_start", None)
        train_fold = [d for d in all_trading
                      if (not train_start_arg or d >= train_start_arg)
                      and d <= train_end_arg]
        post_train  = [d for d in all_trading if d > train_end_arg]
        val_fold    = post_train[:n_val_days]
        folds = [(train_fold, val_fold)]
        log.info(
            "Single fold from explicit date range",
            train_days=len(train_fold),
            val_days=len(val_fold),
            train_range=f"{train_fold[0]} → {train_fold[-1]}" if train_fold else "empty",
            val_range=f"{val_fold[0]} → {val_fold[-1]}" if val_fold else "empty",
        )
    else:
        folds = DataSplitter.walk_forward_splits(
            trading_days,
            n_train_days=n_train_days,
            n_val_days=n_val_days,
            n_folds=n_folds_cfg,
        )
        log.info(
            "Walk-forward folds",
            n_folds=len(folds),
            n_train_days=n_train_days,
            n_val_days=n_val_days,
        )

    # ── Shared component factories ────────────────────────────
    real_capital = float(account_cfg.get("initial_balance", 2500))
    point_value  = float(contracts_cfg.get("micro_point_value", 2.0))
    session_start= session_cfg.get("rth_start_utc", "08:30")
    session_end  = session_cfg.get("rth_end_utc",   "15:00")
    session_type = session_cfg.get("session_type", "RTH").upper()
    bar_minutes  = int(session_cfg.get("bar_timeframe_minutes", 5))
    n_envs       = int(mp_cfg.get("n_envs", 4))
    use_subproc  = mp_cfg.get("use_subprocess", True)

    order_zone_engine = OrderZoneEngine(
        weights=oz_cfg.get("weights"),
        min_confluence_score=oz_cfg.get("min_confluence_score", 0.60),
        min_rr_ratio=risk_cfg.get("take_profit", {}).get("min_rr_ratio", 1.5),
    )
    observation_builder = ObservationBuilder(
        clip_value=obs_cfg.get("clip_observations", 10.0),
        normalize_observations=obs_cfg.get("normalize_observations", True),
        lookback_bars=obs_cfg.get("lookback_bars", 20),
        max_zone_age_bars=zones_cfg.get("max_zone_age_bars", 300),
    )
    action_masker = ActionMasker(
        atr_exhaustion_threshold=atr_gate_cfg.get("block_entries_above_pct", 0.95),
        trail_min_r=trail_cfg.get("activate_at_r", 2.0),
        max_trades_per_day=daily_lim_cfg.get("max_trades_per_day", 5),
        no_entry_last_n_bars=session_risk.get("no_entry_last_n_bars", 3),
    )
    reward_calculator = RewardCalculator.from_config(reward_cfg)
    train_augmentor   = OHLCVAugmentor(rng=np.random.default_rng(agent_cfg.get("seed", 42)))

    def make_position_manager():
        return PositionManager(
            real_capital=real_capital,
            risk_per_trade_pct=sizing_cfg.get("risk_per_trade_pct", 0.01),
            min_contracts=sizing_cfg.get("min_contracts", 0.5),
            max_contracts=sizing_cfg.get("max_contracts", 2.5),
            point_value=point_value,
            max_trades_per_day=daily_lim_cfg.get("max_trades_per_day", 5),
            trail_activate_r=trail_cfg.get("activate_at_r", 2.0),
            trail_aggressive_r=trail_cfg.get("trail_aggressively_at_r", 4.0),
            trail_lock_in_r=trail_cfg.get("lock_in_r_at_trail", 2.0),
            max_daily_loss_r=daily_lim_cfg.get("max_daily_loss_r", 3.0),
            max_daily_loss_dollars=daily_lim_cfg.get("max_daily_loss_dollars", 1000.0),
            max_drawdown_r=risk_cfg.get("position", {}).get("max_drawdown_r", 5.0),
            pause_bars_after_loss_streak=daily_lim_cfg.get("pause_bars_after_loss_streak", 6),
            loss_streak_threshold=daily_lim_cfg.get("max_consecutive_losses_before_pause", 3),
            zone_buffer_atr_pct=risk_cfg.get("stop_loss", {}).get("zone_buffer_atr_pct", 0.03),
            contract_tiers=sizing_cfg.get("contract_tiers"),
            confluence_tier_thresholds=sizing_cfg.get("confluence_tier_thresholds"),
        )

    def make_env(day_list, is_eval=False, worker_seed_offset=0):
        return TradingEnv(
            data_loader=data_loader,
            trading_days=day_list,
            position_manager=make_position_manager(),
            reward_calculator=reward_calculator,
            observation_builder=observation_builder,
            atr_calculator=atr_calculator,
            zone_detector=ZoneDetector(**{
                k: zones_cfg.get(k, v) for k, v in _ZONE_DETECTOR_DEFAULTS.items()
            }),
            order_zone_engine=order_zone_engine,
            action_masker=action_masker,
            rth_start=session_start,
            rth_end=session_end,
            no_entry_last_n_bars=session_risk.get("no_entry_last_n_bars", 3),
            early_terminate_on_max_dd=env_cfg.get("episode", {}).get("early_termination_on_max_drawdown", True),
            point_value=point_value,
            bar_minutes=bar_minutes,
            curriculum_filter_fn=None,
            augmentor=None if is_eval else train_augmentor,
            session_type=session_type,
            random_start=not is_eval,
            seed=agent_cfg.get("seed", 42) + worker_seed_offset + (100 if is_eval else 0),
            zone_lookback_bars=feat_cfg.get("zone_lookback_bars", 500),
        )

    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    fold_results = []

    # ── Per-fold training loop ─────────────────────────────────
    for fold_id, (train_days, val_days) in enumerate(folds):
        fold_dir = output_root / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        models_dir      = fold_dir / "models"
        checkpoints_dir = models_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "tensorboard").mkdir(parents=True, exist_ok=True)

        log.info(
            "Fold starting",
            fold=fold_id,
            train=f"{train_days[0]} → {train_days[-1]}",
            val=f"{val_days[0]} → {val_days[-1]}",
            train_days=len(train_days),
            val_days=len(val_days),
        )

        # ── Build envs ─────────────────────────────────────────
        def _make_train_fn(offset: int, td=train_days):
            def _fn():
                return make_env(td, is_eval=False, worker_seed_offset=offset)
            return _fn

        train_env_fns = [_make_train_fn(i) for i in range(n_envs)]

        if use_subproc and n_envs > 1:
            try:
                from stable_baselines3.common.vec_env import SubprocVecEnv
                log.info("Spawning training environments — first-time PyTorch CUDA init may take several minutes", n_envs=n_envs)
                raw_train_env = SubprocVecEnv(train_env_fns, start_method="spawn")
                log.info("Training environments ready")
            except Exception:
                raw_train_env = DummyVecEnv(train_env_fns)
        else:
            raw_train_env = DummyVecEnv(train_env_fns)

        train_vec_env = VecNormalize(
            raw_train_env,
            norm_obs=True, norm_reward=False, clip_obs=10.0, training=True,
        )
        eval_vec_env = VecNormalize(
            DummyVecEnv([lambda vd=val_days: make_env(vd, is_eval=True)]),
            norm_obs=True, norm_reward=False, clip_obs=10.0, training=False,
        )

        # ── Fresh agent per fold ───────────────────────────────
        agent = PPOAgent(
            env=train_vec_env,
            algorithm=agent_cfg.get("algorithm", "RecurrentPPO"),
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            learning_rate_schedule=ppo_cfg.get("learning_rate_schedule", "linear"),
            n_steps=ppo_cfg.get("n_steps", 2048),
            batch_size=ppo_cfg.get("batch_size", 256),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.01),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            hidden_dims=net_cfg.get("net_arch", {}).get("pi", [256, 128]),
            use_layer_norm=net_cfg.get("use_layer_norm", True),
            use_lstm=net_cfg.get("use_lstm", True),
            lstm_hidden_size=net_cfg.get("lstm_hidden_size", 256),
            n_lstm_layers=net_cfg.get("n_lstm_layers", 1),
            activation_fn_name=net_cfg.get("activation_fn", "ReLU"),
            ortho_init=net_cfg.get("ortho_init", True),
            device=agent_cfg.get("device", "auto"),
            seed=agent_cfg.get("seed", 42) + fold_id,
            tensorboard_log=str(fold_dir / "tensorboard"),
        )

        # ── Fold journal callback ──────────────────────────────
        fold_journal_cb = FoldJournalCallback(n_envs=n_envs, verbose=1)

        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoints_dir,
            save_freq=ckpt_cfg.get("save_freq", 100_000),
            keep_n_checkpoints=ckpt_cfg.get("keep_n_checkpoints", 5),
        )

        min_trades_per_wk  = hotsave_cfg.get("min_trades_per_week", 1)
        fold_n_training_days = len(train_days)
        fold_min_trades    = max(10, fold_n_training_days * min_trades_per_wk // 5)

        trainer = Trainer(
            agent=agent,
            train_env=train_vec_env,
            eval_env=eval_vec_env,
            n_training_days=fold_n_training_days,
            checkpoint_manager=checkpoint_manager,
            curriculum_scheduler=None,
            total_timesteps=ppo_cfg.get("total_timesteps", 2_000_000),
            eval_freq=eval_cfg.get("eval_freq", 50_000),
            n_eval_episodes=eval_cfg.get("n_eval_episodes", 20),
            warmup_steps=eval_cfg.get("warmup_steps", 400_000),
            patience_steps=eval_cfg.get("patience_steps", 550_000),
            w_sharpe=eval_cfg.get("w_sharpe", 0.30),
            w_pnl=eval_cfg.get("w_pnl",    0.25),
            w_wl=eval_cfg.get("w_wl",     0.25),
            w_dd=eval_cfg.get("w_dd",     0.20),
            eval_save_enabled=eval_cfg.get("save_enabled", True),
            ent_coef_start=exp_cfg.get("ent_coef_start",        0.05),
            ent_coef_end=exp_cfg.get("ent_coef_end",          0.005),
            ent_coef_decay_steps=exp_cfg.get("ent_coef_decay_steps", 1_000_000),
            log_dir=str(fold_dir),
            models_dir=str(models_dir),
            train_date_range=f"{train_days[0]}→{train_days[-1]}",
            vec_normalize=train_vec_env,
            resume=False,
            initial_capital=real_capital,
            hotsave_pf=hotsave_cfg.get("pf_threshold",          1.60),
            hotsave_wr=hotsave_cfg.get("wr_threshold",          0.40),
            hotsave_min_trades=fold_min_trades,
            hotsave_min_envs=hotsave_cfg.get("min_envs_passing",      2),
            hotsave_cooldown=hotsave_cfg.get("cooldown_steps",        50_000),
            hotsave_sharpe=hotsave_cfg.get("sharpe_threshold",      1.2),
            hotsave_sharpe_pf=hotsave_cfg.get("sharpe_pf_threshold",   1.85),
            hotsave_sharpe_cooldown=hotsave_cfg.get("sharpe_cooldown_steps", 50_000),
            hotsave_wr70_cooldown=hotsave_cfg.get("wr70_cooldown_steps",   50_000),
            hotsave_elite_pnl_multiplier=hotsave_cfg.get("elite_pnl_multiplier",  1.5),
            hotsave_elite_wr_pf_threshold=hotsave_cfg.get("elite_wr_pf_threshold", 1.5),
            hotsave_elite_sharpe=hotsave_cfg.get("elite_sharpe_threshold",    3.0),
            hotsave_elite_cooldown=hotsave_cfg.get("elite_cooldown_steps",    50_000),
        )

        # Inject fold_journal_cb into Trainer callbacks
        from stable_baselines3.common.callbacks import CallbackList
        base_callbacks, eval_cb = trainer._build_callbacks()
        all_callbacks = CallbackList([base_callbacks, fold_journal_cb])

        log.info("Training fold", fold=fold_id, timesteps=ppo_cfg.get("total_timesteps", 2_000_000))
        agent.train(
            total_timesteps=ppo_cfg.get("total_timesteps", 2_000_000),
            callback=all_callbacks,
            progress_bar=True,
            reset_num_timesteps=True,
        )

        # Always write a FINAL_STEP checkpoint as safety net
        eval_cb.save_final_checkpoint()

        # ── Save fold journal ──────────────────────────────────
        fold_journal_cb.save(fold_id=fold_id, fold_dir=fold_dir)

        # ── Save final model ───────────────────────────────────
        final_path = models_dir / "final_model"
        agent.save(final_path)
        vn_path = str(models_dir / "vecnormalize.pkl")
        train_vec_env.save(vn_path)
        log.info("Fold complete", fold=fold_id, model=str(final_path))

        fold_results.append({
            "fold": fold_id,
            "train": f"{train_days[0]} → {train_days[-1]}",
            "val":   f"{val_days[0]} → {val_days[-1]}",
            "train_days": len(train_days),
            "val_days": len(val_days),
        })

        # Cleanup envs before next fold
        try:
            train_vec_env.close()
            eval_vec_env.close()
        except Exception:
            pass

    # ── Print fold summary ────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  WALK-FORWARD COMPLETE — {len(fold_results)} fold(s)")
    print("=" * 70)
    for r in fold_results:
        print(
            f"  Fold {r['fold']:02d} | "
            f"Train: {r['train']} ({r['train_days']}d) | "
            f"Val: {r['val']} ({r['val_days']}d)"
        )
    print(f"\n  Results saved to: {output_root}")
    print("=" * 70)


def run_analyse(args: argparse.Namespace) -> None:
    from evaluation.trade_journal import TradeJournal

    journal_dir = Path(args.journal or (Path(args.log_dir) / "journal"))
    csv_files = list(journal_dir.glob("*.csv"))

    if not csv_files:
        log.error("No journal CSV files found.", directory=str(journal_dir))
        sys.exit(1)

    import pandas as pd
    from evaluation.metrics_calculator import MetricsCalculator

    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)

    mc = MetricsCalculator()
    metrics = mc.compute_from_dataframe(combined)

    print(f"\nJournal Analysis — {len(combined)} total trades")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<40} {v:.4f}")
        else:
            print(f"  {k:<40} {v}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL Trading Agent — Order Zone Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "analyse", "walk_forward", "test_fold"],
        required=True,
        help="Operation mode.",
    )
    parser.add_argument(
        "--config",
        default="config/",
        help="Path to config directory containing YAML files. Default: config/",
    )
    parser.add_argument(
        "--data",
        default="data/",
        help="Path to raw OHLCV data directory. Default: data/",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a saved model checkpoint (.zip). Required for evaluate mode.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/",
        help="Directory for logs, checkpoints, journal. Default: logs/",
    )
    parser.add_argument(
        "--journal",
        default=None,
        help="Path to journal directory for analyse mode.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        dest="no_clean",
        help="Skip wiping output dirs at run start (use when resuming from --checkpoint).",
    )
    parser.add_argument(
        "--train-start",
        default=None,
        dest="train_start",
        metavar="YYYY-MM-DD",
        help="Earliest date to include in the day pool (inclusive). "
             "Filters all modes — train, evaluate, walk_forward.",
    )
    parser.add_argument(
        "--train-end",
        default=None,
        dest="train_end",
        metavar="YYYY-MM-DD",
        help="Last date of the training window (inclusive). When provided together "
             "with --train-start, the splitter uses exact date boundaries instead of "
             "fixed day counts. Val is the next --val-weeks trading days; test follows.",
    )
    parser.add_argument(
        "--val-weeks",
        default=None,
        type=int,
        dest="val_weeks",
        metavar="N",
        help="Number of weeks (× 5 trading days) allocated to validation. Default: from agent_config.yaml walk_forward.val_weeks.",
    )
    # ── test_fold mode args ───────────────────────────────────────────────────
    parser.add_argument(
        "--models-dir",
        default=None,
        dest="models_dir",
        help="(test_fold) Folder containing checkpoint/hotsave .zip files.",
    )
    parser.add_argument(
        "--test-start",
        default=None,
        dest="test_start",
        metavar="YYYY-MM-DD",
        help="(test_fold) Start of the evaluation date range (inclusive).",
    )
    parser.add_argument(
        "--test-end",
        default=None,
        dest="test_end",
        metavar="YYYY-MM-DD",
        help="(test_fold) End of the evaluation date range (inclusive).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        dest="out_dir",
        help="(test_fold) Output directory for journals and leaderboard.",
    )
    parser.add_argument(
        "--n-episodes",
        default=0,
        type=int,
        dest="n_episodes",
        help="(test_fold) Episodes per checkpoint. 0 = run every test day once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load all configs
    configs = load_configs(args.config)

    # ── Clean BEFORE opening any log file handles ─────────────────────────────
    # Must run here so metrics.log is not yet open when we delete it.
    if args.mode in ("train", "walk_forward"):
        if not getattr(args, "no_clean", False) and not getattr(args, "checkpoint", None):
            clean_run_dirs(Path(args.log_dir))

    # Configure logging from config
    log_cfg = configs.get("logging", {})
    file_cfg = log_cfg.get("outputs", {}).get("file", {})
    metrics_log_path = file_cfg.get("path") if file_cfg.get("enabled", False) else None

    configure_logging(
        level=log_cfg.get("level", "INFO"),
        log_format=log_cfg.get("format", "console"),
        log_file=metrics_log_path,
        rich_formatting=log_cfg.get("outputs", {}).get("console", {}).get("rich_formatting", True),
    )

    # Mirror all stdout (print + log) to a timestamped console transcript
    from datetime import datetime as _dt
    _run_ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    _console_log_dir = Path(metrics_log_path).parent if metrics_log_path else Path("logs")
    tee_stdout(_console_log_dir / f"console_{args.mode}_{_run_ts}.log")

    # Initialise the shared Rich console (dual terminal + file output for tables)
    init_console(log_path=metrics_log_path)

    log.info("RL Trading Agent starting", mode=args.mode, config=args.config)

    # Validate all configs
    try:
        validate_all_configs(configs)
    except ValueError as e:
        log.error("Config validation failed", error=str(e))
        sys.exit(1)

    # Dispatch to mode
    if args.mode == "train":
        run_train(args, configs)
    elif args.mode == "evaluate":
        run_evaluate(args, configs)
    elif args.mode == "analyse":
        run_analyse(args)
    elif args.mode == "walk_forward":
        run_walk_forward(args, configs)
    elif args.mode == "test_fold":
        _run_test_fold(args)


def _run_test_fold(args) -> None:
    """Delegate to evaluation/test_fold.py main() with the CLI args translated."""
    from evaluation.test_fold import main as _tf_main
    argv = [
        "--models-dir", args.models_dir if hasattr(args, "models_dir") and args.models_dir
                        else str(Path(args.log_dir) / "models"),
        "--config",     args.config,
        "--data",       args.data,
    ]
    if getattr(args, "test_start", None):
        argv += ["--test-start", args.test_start]
    if getattr(args, "test_end", None):
        argv += ["--test-end", args.test_end]
    if getattr(args, "n_episodes", None):
        argv += ["--n-episodes", str(args.n_episodes)]
    if getattr(args, "out_dir", None):
        argv += ["--out-dir", args.out_dir]
    _tf_main(argv)


if __name__ == "__main__":
    main()