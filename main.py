"""
main.py
========
Entry point for the RL Trading Agent.

Usage:
    # Train from scratch
    python main.py --mode train --config config/ --data data/raw/

    # Continue training from checkpoint
    python main.py --mode train --config config/ --data data/raw/ \\
                   --checkpoint logs/checkpoints/best_model.zip

    # Evaluate (backtest) a trained model
    python main.py --mode evaluate --config config/ --data data/raw/ \\
                   --checkpoint logs/checkpoints/best_model.zip

    # Print journal analysis for a completed backtest
    python main.py --mode analyse --journal logs/journal/

All parameters are loaded from YAML config files — no command-line
parameter overrides for model hyperparameters (edit the YAML instead).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from utils.logger import configure_logging, get_logger
from utils.metrics_printer import init_console
from utils.validators import validate_all_configs, assert_instrument_allowed

log = get_logger(__name__)


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

def build_components(configs: dict, data_dir: str):
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
    from features.liquidity_detector import LiquidityDetector
    from features.observation_builder import ObservationBuilder
    from features.order_zone_engine import OrderZoneEngine
    from features.trend_classifier import TrendClassifier
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
        warning_threshold=atr_cfg.get("danger_threshold", 0.85),
    )
    atr_calculator.fit(data_loader.daily)

    # Filter to valid weekday sessions that have ATR history
    import pandas as _pd
    trading_days = [
        d for d in all_days
        if _pd.Timestamp(d).weekday() < 5                  # Mon–Fri only
        and atr_calculator.get_atr_for_date(d) is not None  # ATR warmup complete
    ]
    log.info("Valid trading days after filtering", total=len(all_days), valid=len(trading_days))

    # Fixed-count chronological split: 252 trading days (~12 months) train,
    # 26 days (~5 weeks) validation, remainder held out as test.
    # No lookahead — val/test data is strictly after all training dates.
    split = DataSplitter.split_by_counts(trading_days, n_train=252, n_val=26)
    log.info(
        "Data split",
        train_days=len(split.train),
        val_days=len(split.validation),
        test_days=len(split.test),
        train_range=f"{split.train[0]} → {split.train[-1]}",
        val_range=f"{split.validation[0]} → {split.validation[-1]}",
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

    # ── Liquidity Detector ────────────────────────────────────
    liq_cfg = feat_cfg.get("liquidity", {})
    swing_cfg = feat_cfg.get("swing", {})
    liquidity_detector = LiquidityDetector(
        swing_lookback=swing_cfg.get("lookback_bars", 5),
        proximity_atr_pct=liq_cfg.get("proximity_atr_pct", 0.05),
        sweep_wick_min_atr_pct=liq_cfg.get("sweep_wick_min_atr_pct", 0.03),
        sweep_lookback_bars=liq_cfg.get("sweep_lookback_bars", 5),
    )

    # ── Trend Classifier ──────────────────────────────────────
    trend_cfg = feat_cfg.get("trend", {})
    trend_classifier = TrendClassifier(
        swing_lookback=swing_cfg.get("lookback_bars", 5),
        min_hh_hl_for_uptrend=trend_cfg.get("min_hh_hl_for_uptrend", 2),
        min_ll_lh_for_downtrend=trend_cfg.get("min_ll_lh_for_downtrend", 2),
        reversal_requires_breaks=trend_cfg.get("reversal_requires_breaks", 2),
        strength_lookback_bars=trend_cfg.get("strength_lookback_bars", 40),
    )

    # ── Order Zone Engine ─────────────────────────────────────
    oz_cfg = feat_cfg.get("order_zone", {})
    order_zone_engine = OrderZoneEngine(
        weights=oz_cfg.get("weights"),
        min_confluence_score=oz_cfg.get("min_confluence_score", 0.60),
        min_rr_ratio=risk_cfg.get("take_profit", {}).get("min_rr_ratio", 4.0),
        pin_bar_wick_ratio=oz_cfg.get("rejection", {}).get("pin_bar_wick_ratio", 2.0),
        engulfing_body_ratio=oz_cfg.get("rejection", {}).get("engulfing_body_ratio", 1.1),
    )

    # ── Observation Builder ───────────────────────────────────
    obs_cfg = env_cfg.get("observation", {})
    observation_builder = ObservationBuilder(
        clip_value=obs_cfg.get("clip_observations", 10.0),
        normalize_observations=obs_cfg.get("normalize_observations", True),
        lookback_bars=obs_cfg.get("lookback_bars", 20),
    )

    # ── Action Masker ─────────────────────────────────────────
    atr_gate_cfg = risk_cfg.get("atr_gate", {})
    daily_lim_cfg = risk_cfg.get("daily_limits", {})
    session_risk_cfg = risk_cfg.get("session", {})
    action_masker = ActionMasker(
        min_rr_ratio=risk_cfg.get("take_profit", {}).get("min_rr_ratio", 4.0),
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
            min_contracts=sizing_cfg.get("min_contracts", 1),
            max_contracts=sizing_cfg.get("max_contracts", 3),
            point_value=point_value,
            max_trades_per_day=daily_lim_cfg.get("max_trades_per_day", 5),
            trail_activate_r=trail_cfg.get("activate_at_r", 2.0),
            trail_aggressive_r=trail_cfg.get("trail_aggressively_at_r", 4.0),
            trail_lock_in_r=trail_cfg.get("lock_in_r_at_trail", 2.0),
            max_daily_loss_r=daily_lim_cfg.get("max_daily_loss_r", 3.0),
            max_drawdown_r=5.0,
            pause_bars_after_loss_streak=daily_lim_cfg.get("pause_bars_after_loss_streak", 6),
            loss_streak_threshold=daily_lim_cfg.get("max_consecutive_losses_before_pause", 3),
            zone_buffer_atr_pct=risk_cfg.get("stop_loss", {}).get("zone_buffer_atr_pct", 0.03),
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

    zone_detector_defaults = {
        "consolidation_min_bars": 2, "consolidation_max_bars": 8,
        "consolidation_range_atr_pct": 0.20, "impulse_min_body_atr_pct": 0.15,
        "max_zone_age_bars": 200, "max_zone_touches": 3, "zone_buffer_atr_pct": 0.02,
    }

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
                for k, v in zone_detector_defaults.items()
            }),
            liquidity_detector=liquidity_detector,
            trend_classifier=trend_classifier,
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

    eval_env = make_env(split.validation, is_eval=True)
    test_env = make_env(split.test, is_eval=True)

    # Keep a single-env handle for the trainer (eval/curriculum logic)
    train_env = train_vec_env

    # ── Namespace ─────────────────────────────────────────────
    class Components:
        pass

    c = Components()
    c.data_loader = data_loader
    c.split = split
    c.atr_calculator = atr_calculator
    c.train_env = train_env
    c.eval_env = eval_env
    c.test_env = test_env
    c.curriculum_scheduler = curriculum_scheduler
    c.reward_calculator = reward_calculator
    c.real_capital = real_capital

    return c


# ── Mode handlers ─────────────────────────────────────────────────────────────

def run_train(args: argparse.Namespace, configs: dict) -> None:
    from agent.ppo_agent import PPOAgent
    from training.checkpoint_manager import CheckpointManager
    from training.trainer import Trainer

    log.info("Mode: TRAIN")
    c = build_components(configs, args.data)

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
        log.info("Loading checkpoint", path=args.checkpoint)
        agent.load(args.checkpoint, env=c.train_env)

    ckpt_cfg = agent_cfg.get("checkpointing", {})
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoints_dir,
        save_freq=ckpt_cfg.get("save_freq", 100_000),
        keep_n_checkpoints=ckpt_cfg.get("keep_n_checkpoints", 5),
    )

    eval_cfg = agent_cfg.get("evaluation", {})
    trainer = Trainer(
        agent=agent,
        train_env=c.train_env,
        eval_env=c.eval_env,
        checkpoint_manager=checkpoint_manager,
        curriculum_scheduler=c.curriculum_scheduler,
        total_timesteps=ppo_cfg.get("total_timesteps", 2_000_000),
        eval_freq=eval_cfg.get("eval_freq",       50_000),
        n_eval_episodes=eval_cfg.get("n_eval_episodes", 20),
        warmup_steps=eval_cfg.get("warmup_steps",   400_000),
        patience_steps=eval_cfg.get("patience_steps", 550_000),
        w_sharpe=eval_cfg.get("w_sharpe", 0.30),
        w_pnl=eval_cfg.get("w_pnl",    0.25),
        w_wl=eval_cfg.get("w_wl",     0.25),
        w_dd=eval_cfg.get("w_dd",     0.20),
        ent_coef_start=exp_cfg.get("ent_coef_start",        0.05),
        ent_coef_end=exp_cfg.get("ent_coef_end",          0.005),
        ent_coef_decay_steps=exp_cfg.get("ent_coef_decay_steps", 1_000_000),
        log_dir=str(log_dir),
        models_dir=str(models_dir),
        train_date_range=f"{c.split.train[0]}→{c.split.train[-1]}",
    )

    trainer.run()
    log.info("Training run finished. Models saved to: %s", models_dir)


def run_evaluate(args: argparse.Namespace, configs: dict) -> None:
    from agent.ppo_agent import PPOAgent
    from evaluation.backtester import Backtester
    from evaluation.metrics_calculator import MetricsCalculator
    from evaluation.trade_journal import TradeJournal

    if not args.checkpoint:
        log.error("--checkpoint is required for evaluate mode.")
        sys.exit(1)

    log.info("Mode: EVALUATE", checkpoint=args.checkpoint)
    c = build_components(configs, args.data)

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


def run_analyse(args: argparse.Namespace) -> None:
    from evaluation.trade_journal import TradeJournal
    import glob

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
        choices=["train", "evaluate", "analyse"],
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load all configs
    configs = load_configs(args.config)

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


if __name__ == "__main__":
    main()