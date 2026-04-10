"""
training/trainer.py
====================
Main training orchestration loop.

Handles:
  - Multi-stage curriculum (trending → mixed → full market)
  - Evaluation callbacks
  - Checkpoint saving
  - TensorBoard metric logging
  - Early stopping on validation performance
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from training.trading_eval_callback import TradingEvalCallback
from training.metrics_logger_callback import MetricsPrinterCallback
from training.shaping_decay_callback import ShapingDecayCallback
from training.training_journal_callback import TrainingJournalCallback
from training.training_hotsave_callback import TrainingHotSaveCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agent.ppo_agent import PPOAgent
from training.checkpoint_manager import CheckpointManager
from training.curriculum import CurriculumScheduler
from utils.logger import get_logger

log = get_logger(__name__)


# ── Custom Callbacks ──────────────────────────────────────────────────────────

class TradingMetricsCallback(BaseCallback):
    """
    Logs trading-specific metrics to TensorBoard after each episode.

    Tracks:
      - Win rate
      - Average R:R achieved
      - Trades per episode
      - Profit factor
      - % trades in Order Zone
      - % trades trend-aligned
      - Max drawdown per episode
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_metrics: List[dict] = []

    def _on_step(self) -> bool:
        # SB3 passes episode info in self.locals["infos"]
        infos = self.locals.get("infos", [])
        for info in infos:
            if "total_pnl_r" in info:  # Episode summary info
                self._episode_metrics.append(info)
                self._log_episode_metrics(info)
        return True

    def _log_episode_metrics(self, info: dict) -> None:
        step = self.num_timesteps
        self.logger.record("trading/win_rate", info.get("win_rate", 0.0))
        self.logger.record("trading/total_pnl_r", info.get("total_pnl_r", 0.0))
        self.logger.record("trading/n_trades", info.get("n_trades", 0))
        self.logger.record("trading/profit_factor", info.get("profit_factor", 0.0))
        self.logger.record("trading/avg_win_r", info.get("avg_win_r", 0.0))
        self.logger.record("trading/avg_loss_r", info.get("avg_loss_r", 0.0))


class CurriculumCallback(BaseCallback):
    """
    Updates the curriculum stage based on training progress.
    Injects curriculum filter functions into all envs when stage advances.
    """

    def __init__(self, scheduler: CurriculumScheduler, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._scheduler = scheduler
        self._last_stage: Optional[str] = None

    def _on_step(self) -> bool:
        stage = self._scheduler.current_stage(self.num_timesteps)
        if stage and stage.name != self._last_stage:
            log.info(
                "Curriculum stage advanced",
                stage=stage.name,
                description=stage.description,
                at_step=self.num_timesteps,
            )
            self._last_stage = stage.name
            # Update filter function in all envs
            filter_fn = self._scheduler.build_filter_fn(stage)
            self._update_env_filter(filter_fn)
        return True

    def _update_env_filter(self, filter_fn: Any) -> None:
        """Push curriculum filter to all vectorised envs."""
        try:
            for env in self.training_env.envs:  # DummyVecEnv
                if hasattr(env, "curriculum_filter_fn"):
                    env.curriculum_filter_fn = filter_fn
                elif hasattr(env, "env"):
                    env.env.curriculum_filter_fn = filter_fn
        except AttributeError:
            pass  # SubprocVecEnv — more complex; skip for now


class EntropyAnnealingCallback(BaseCallback):
    """
    Linearly anneals the entropy coefficient during training.
    Encourages more exploration early, exploitation later.
    """

    def __init__(
        self,
        ent_coef_start: float = 0.05,
        ent_coef_end: float = 0.005,
        decay_steps: int = 1_000_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.decay_steps, 1.0)
        new_ent_coef = self.ent_coef_start + progress * (self.ent_coef_end - self.ent_coef_start)
        self.model.ent_coef = new_ent_coef
        if self.num_timesteps % 10_000 == 0:
            self.logger.record("train/ent_coef", new_ent_coef)
        return True


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Orchestrates the full training pipeline.

    Parameters
    ----------
    agent : PPOAgent
    train_env : gymnasium.Env or VecEnv
        Training environment (possibly vectorised).
    eval_env : gymnasium.Env
        Single evaluation environment (non-vectorised).
    checkpoint_manager : CheckpointManager
    curriculum_scheduler : Optional[CurriculumScheduler]
    total_timesteps : int
    eval_freq : int
        How often (in timesteps) to run evaluation.
    n_eval_episodes : int
        Number of episodes per evaluation run.
    ent_coef_start : float
    ent_coef_end : float
    ent_coef_decay_steps : int
    log_dir : str | Path
    """

    def __init__(
        self,
        agent: PPOAgent,
        train_env: Any,
        eval_env: Any,
        checkpoint_manager: CheckpointManager,
        curriculum_scheduler: Optional[CurriculumScheduler] = None,
        total_timesteps: int = 2_000_000,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 20,
        warmup_steps: int = 400_000,
        patience_steps: int = 550_000,
        print_train_episodes: bool = True,
        rl_diag_every: int = 1,
        # Composite score weights
        w_sharpe: float = 0.30,
        w_pnl:    float = 0.25,
        w_wl:     float = 0.25,
        w_dd:     float = 0.20,
        ent_coef_start: float = 0.05,
        ent_coef_end: float = 0.005,
        ent_coef_decay_steps: int = 1_000_000,
        log_dir: str = "logs",
        models_dir: str = "logs/models",
        train_date_range: str = "",
        vec_normalize=None,   # VecNormalize wrapper — stats saved alongside model
        resume: bool = False, # True when continuing from a checkpoint
        eval_save_enabled: bool = True,  # False = run eval metrics but save no models
        # Training hot-save — Gate 1 (PF/WR)
        hotsave_pf: float = 1.60,
        hotsave_wr: float = 0.40,
        hotsave_min_trades: int = 20,
        hotsave_min_envs: int = 2,
        hotsave_cooldown: int = 50_000,
        # Training hot-save — Gate 2 (Sharpe quality)
        hotsave_sharpe: float = 1.2,
        hotsave_sharpe_pf: float = 1.85,
        hotsave_sharpe_cooldown: int = 50_000,
    ) -> None:
        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.vec_normalize = vec_normalize
        self.checkpoint_manager = checkpoint_manager
        self.curriculum_scheduler = curriculum_scheduler
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.warmup_steps = warmup_steps
        self.patience_steps = patience_steps
        self.w_sharpe = w_sharpe
        self.w_pnl    = w_pnl
        self.w_wl     = w_wl
        self.w_dd     = w_dd
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.ent_coef_decay_steps = ent_coef_decay_steps
        self.print_train_episodes = print_train_episodes
        self.rl_diag_every = rl_diag_every
        self.log_dir        = Path(log_dir)
        self.models_dir     = Path(models_dir)
        self.train_date_range = train_date_range
        self.resume               = resume
        self.eval_save_enabled    = eval_save_enabled
        self.hotsave_pf           = hotsave_pf
        self.hotsave_wr           = hotsave_wr
        self.hotsave_min_trades   = hotsave_min_trades
        self.hotsave_min_envs     = hotsave_min_envs
        self.hotsave_cooldown     = hotsave_cooldown
        self.hotsave_sharpe       = hotsave_sharpe
        self.hotsave_sharpe_pf    = hotsave_sharpe_pf
        self.hotsave_sharpe_cooldown = hotsave_sharpe_cooldown

    def run(self) -> PPOAgent:
        """
        Execute the full training run.

        Returns
        -------
        PPOAgent
            The trained agent.
        """
        callbacks, eval_cb = self._build_callbacks()

        log.info(
            "Training run started",
            total_timesteps=self.total_timesteps,
            eval_freq=self.eval_freq,
            log_dir=str(self.log_dir),
        )

        start_time = time.time()

        self.agent.train(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not self.resume,
        )

        elapsed = time.time() - start_time
        log.info(
            "Training complete",
            elapsed_seconds=round(elapsed, 1),
            steps_per_second=round(self.total_timesteps / elapsed, 0),
        )

        # Always write a FINAL_STEP checkpoint — guarantees at least one
        # testable model exists per fold even if no phase gate was ever cleared.
        eval_cb.save_final_checkpoint()

        # Save final model + VecNormalize stats into the dedicated models folder
        self.models_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.models_dir / "final_model"
        self.agent.save(final_path)
        if self.vec_normalize is not None:
            vn_path = str(self.models_dir / "vecnormalize.pkl")
            self.vec_normalize.save(vn_path)
            log.info("VecNormalize stats saved", path=vn_path)
        log.info("Final model saved", path=str(final_path))

        return self.agent

    # ── Callback assembly ─────────────────────────────────────

    def _build_callbacks(self) -> tuple[CallbackList, TradingEvalCallback]:
        cbs = []

        # 1. Pretty-printed tables: training episodes + RL diagnostics
        cbs.append(
            MetricsPrinterCallback(
                print_train_episodes=self.print_train_episodes,
                rl_diag_every=self.rl_diag_every,
                train_date_range=self.train_date_range,
                verbose=0,
            )
        )

        # 2. TensorBoard trading metrics logger (raw scalars, no tables)
        cbs.append(TradingMetricsCallback(verbose=0))

        # 2. Trading-metrics evaluation + composite best-model saving + early stop
        eval_cb = TradingEvalCallback(
            eval_env=self.eval_env,
            save_path=self.models_dir,
            eval_freq=self.eval_freq,
            n_eval_episodes=self.n_eval_episodes,
            warmup_steps=self.warmup_steps,
            patience_steps=self.patience_steps,
            w_sharpe=self.w_sharpe,
            w_pnl=self.w_pnl,
            w_wl=self.w_wl,
            w_dd=self.w_dd,
            save_enabled=self.eval_save_enabled,
            verbose=1,
        )
        cbs.append(eval_cb)

        # 3. Periodic checkpoint saving
        checkpoint_cb = CheckpointCallback(
            save_freq=self.checkpoint_manager.save_freq,
            save_path=str(self.log_dir / "checkpoints"),
            name_prefix="rl_trading",
            verbose=0,
        )
        cbs.append(checkpoint_cb)

        # 4. Entropy annealing
        cbs.append(
            EntropyAnnealingCallback(
                ent_coef_start=self.ent_coef_start,
                ent_coef_end=self.ent_coef_end,
                decay_steps=self.ent_coef_decay_steps,
            )
        )

        # 5. Reward shaping decay (Stage 1 → 2 → 3)
        cbs.append(
            ShapingDecayCallback(
                stage1_end=600_000,
                stage2_end=1_200_000,
                verbose=0,
            )
        )

        # 6. Training journal — Excel + Plotly HTML saved every 50k steps
        cbs.append(
            TrainingJournalCallback(
                journal_dir=self.log_dir / "journal",
                save_every_steps=50_000,
                verbose=1,
            )
        )

        # 7. Training hot-saves — two gates in one callback
        cbs.append(
            TrainingHotSaveCallback(
                models_dir=self.models_dir / "hotsaves",
                # Gate 1 — PF/WR
                pf_threshold=self.hotsave_pf,
                wr_threshold=self.hotsave_wr,
                min_trades=self.hotsave_min_trades,
                min_envs_passing=self.hotsave_min_envs,
                cooldown_steps=self.hotsave_cooldown,
                # Gate 2 — Sharpe quality
                sharpe_threshold=self.hotsave_sharpe,
                sharpe_pf_threshold=self.hotsave_sharpe_pf,
                sharpe_cooldown_steps=self.hotsave_sharpe_cooldown,
                vec_normalize=self.vec_normalize,
                verbose=1,
            )
        )

        # 8. Curriculum (optional)
        if self.curriculum_scheduler is not None:
            cbs.append(CurriculumCallback(self.curriculum_scheduler))

        return CallbackList(cbs), eval_cb