"""
training/trading_eval_callback.py
===================================
Custom SB3 callback that replaces the generic EvalCallback with full
trading-specific metric evaluation.

Metrics computed after every evaluation run
-------------------------------------------
  total_pnl_r        : Sum of all trade P&L in R-multiples
  sharpe_ratio        : Mean episode P&L / Std episode P&L  (episode-level)
  win_loss_ratio      : n_wins / max(n_losses, 1)
  max_drawdown_r      : Worst peak-to-trough cumulative R across all episodes
  win_rate            : n_wins / n_trades
  n_trades            : Total trades executed
  n_wins / n_losses   : Win and loss counts
  avg_rr              : avg_win_r / avg_loss_r
  expected_return     : win_rate * avg_win_r − (1−win_rate) * avg_loss_r
  avg_trade_duration  : Mean trade duration in bars
  max_win_dollars     : Largest single winning trade in $
  max_loss_dollars    : Largest single losing trade in $ (most negative)

Tiered save schedule
--------------------
  As training progresses the bar for saving escalates.  A checkpoint is
  written only when BOTH conditions are met:
    (a) composite_score >= phase minimum threshold
    (b) n_trades >= MIN_TRADES_FOR_SAVE (avoids saving on lucky zero-trade runs)

  SAVE_SCHEDULE  (step_threshold, min_composite_score):
    Phase 0 (0–400k)      : 0.05  — any positive signal
    Phase 1 (400k–750k)   : 0.15  — post-warmup; need real signal
    Phase 2 (750k–1.1M)   : 0.25  — consistency required
    Phase 3 (1.1M–1.5M)   : 0.35  — quality grade
    Phase 4 (1.5M+)        : 0.45  — elite; only clear best saved

  Every checkpoint that passes the phase gate is saved as a numbered file:
    checkpoint_s{N:02d}_step{step}_c{score:.2f}.zip
  alongside its VecNormalize stats:
    checkpoint_s{N:02d}_step{step}_c{score:.2f}_vecnormalize.pkl

  The all-time best is additionally written as best_model.zip /
  best_model_vecnormalize.pkl for backwards-compatible loading.

Composite save score
--------------------
  score = w_sharpe * sharpe_norm
        + w_pnl    * pnl_norm
        + w_wl     * wl_norm
        + w_dd     * (1 − dd_norm)          ← lower DD is better

Early stopping
--------------
  - No early stopping before warmup_steps  (default 400 000)
  - Stop if no composite improvement for patience_steps (default 550 000)
  - Training budget: total_timesteps (default 2 000 000)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from utils.logger import get_logger

log = get_logger(__name__)


# ── Tiered save schedule ──────────────────────────────────────────────────────
# (step_threshold, minimum_composite_score_to_save)
SAVE_SCHEDULE: List[Tuple[int, float]] = [
    (0,         0.05),   # Phase 0: Warmup — any positive composite
    (400_000,   0.15),   # Phase 1: Post-warmup — need real signal
    (750_000,   0.25),   # Phase 2: Consistency required
    (1_100_000, 0.35),   # Phase 3: Quality grade
    (1_500_000, 0.45),   # Phase 4: Elite — only clear best
]

# Minimum trades in an eval run before we consider saving.
# Prevents saving a checkpoint that got lucky on 1-2 trades with no statistical validity.
MIN_TRADES_FOR_SAVE = 8


# ── Metric container ──────────────────────────────────────────────────────────

@dataclass
class ValMetrics:
    """All trading metrics from one validation run."""
    # Core metrics
    total_pnl_r:        float = 0.0
    sharpe_ratio:       float = 0.0
    win_loss_ratio:     float = 0.0
    max_drawdown_r:     float = 0.0
    # Detail
    win_rate:           float = 0.0
    n_trades:           int   = 0
    n_wins:             int   = 0
    n_losses:           int   = 0
    avg_rr:             float = 0.0   # avg_win_r / avg_loss_r
    expected_return:    float = 0.0   # (win_rate * avg_win) − ((1−wr) * avg_loss)
    avg_trade_duration: float = 0.0   # mean bars per trade
    max_win_dollars:    float = 0.0
    max_loss_dollars:   float = 0.0
    # Composite
    composite_score:    float = -999.0

    def log_str(self) -> str:
        return (
            f"P&L={self.total_pnl_r:+.2f}R  Sharpe={self.sharpe_ratio:+.3f}  "
            f"WL={self.win_loss_ratio:.2f}  MaxDD={self.max_drawdown_r:.2f}R  "
            f"WR={self.win_rate*100:.1f}%  Trades={self.n_trades}  "
            f"RR={self.avg_rr:.2f}  ExpR={self.expected_return:+.3f}  "
            f"AvgDur={self.avg_trade_duration:.1f}bars  "
            f"MaxWin=${self.max_win_dollars:.0f}  MaxLoss=${self.max_loss_dollars:.0f}  "
            f"[composite={self.composite_score:.4f}]"
        )


# ── Callback ──────────────────────────────────────────────────────────────────

class TradingEvalCallback(BaseCallback):
    """
    Evaluates the agent on a validation environment using trading metrics,
    saves numbered checkpoints on every phase-passing eval, and maintains
    a best_model alias pointing to the all-time best checkpoint.

    Parameters
    ----------
    eval_env : VecEnv | gymnasium.Env
    save_path : str | Path
        Directory for all checkpoint files.
    eval_freq : int
    n_eval_episodes : int
    warmup_steps : int
    patience_steps : int
    w_sharpe, w_pnl, w_wl, w_dd : float
        Composite score weights.
    """

    def __init__(
        self,
        eval_env: Any,
        save_path: str | Path,
        eval_freq: int           = 50_000,
        n_eval_episodes: int     = 20,
        warmup_steps: int        = 400_000,
        patience_steps: int      = 550_000,
        # Composite weights
        w_sharpe: float          = 0.30,
        w_pnl:    float          = 0.25,
        w_wl:     float          = 0.25,
        w_dd:     float          = 0.20,
        # Normalization anchors
        sharpe_ref: float        = 1.5,   # annualised Sharpe (*sqrt(252)); 1.5 = excellent
        pnl_ref:    float        = 20.0,
        wl_ref:     float        = 3.0,
        dd_ref:     float        = 8.0,
        save_enabled: bool       = True,  # set False to run eval without saving any models
        verbose: int             = 1,
    ) -> None:
        super().__init__(verbose)

        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
        self.eval_env = eval_env

        self.save_path       = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.warmup_steps    = warmup_steps
        self.patience_steps  = patience_steps

        total_w = w_sharpe + w_pnl + w_wl + w_dd
        self.w_sharpe = w_sharpe / total_w
        self.w_pnl    = w_pnl    / total_w
        self.w_wl     = w_wl     / total_w
        self.w_dd     = w_dd     / total_w

        self.sharpe_ref   = max(sharpe_ref, 1e-6)
        self.pnl_ref      = max(pnl_ref,    1e-6)
        self.wl_ref       = max(wl_ref,     1e-6)
        self.dd_ref       = max(dd_ref,     1e-6)
        self.save_enabled = save_enabled

        # State
        self._best_score:      float = float("-inf")
        self._best_score_step: int   = 0
        self._last_eval_step:  int   = 0
        self._eval_history:    List[ValMetrics] = []
        self._early_stopped:   bool  = False
        self._save_n:          int   = 0   # counter for numbered checkpoints
        self._cur_phase:       int   = 0

    # ── Phase helpers ─────────────────────────────────────────────────────────

    def _phase_min_composite(self) -> float:
        """Return the minimum composite score required to save at current step."""
        threshold = SAVE_SCHEDULE[0][1]
        for step_thr, min_score in SAVE_SCHEDULE:
            if self.num_timesteps >= step_thr:
                threshold = min_score
        return threshold

    def _phase_idx(self) -> int:
        idx = 0
        for i, (step_thr, _) in enumerate(SAVE_SCHEDULE):
            if self.num_timesteps >= step_thr:
                idx = i
        return idx

    # ── SB3 hook ─────────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps

        # Phase advancement — reset patience when phase advances
        phase_now = self._phase_idx()
        if phase_now > self._cur_phase:
            self._cur_phase = phase_now
            log.info(
                "Eval phase advanced",
                phase=phase_now,
                min_composite=self._phase_min_composite(),
            )

        self._sync_vec_normalize()
        metrics = self._run_eval()

        if metrics is None:
            return True

        self._eval_history.append(metrics)
        self._log_metrics(metrics)

        # ── Tiered save logic ─────────────────────────────────
        if not self.save_enabled:
            log.info("Eval save disabled (save_enabled=False) — metrics logged only")
        else:
            min_composite = self._phase_min_composite()
            passes_phase  = metrics.composite_score >= min_composite
            passes_trades = metrics.n_trades >= MIN_TRADES_FOR_SAVE

            if passes_phase and passes_trades:
                self._save_checkpoint(metrics)
            else:
                reasons = []
                if not passes_phase:
                    reasons.append(f"composite={metrics.composite_score:.3f} < phase_min={min_composite:.2f}")
                if not passes_trades:
                    reasons.append(f"n_trades={metrics.n_trades} < min={MIN_TRADES_FOR_SAVE}")
                log.info("Checkpoint not saved", reasons=", ".join(reasons))

        # ── Early stopping ────────────────────────────────────
        if self.num_timesteps >= self.warmup_steps:
            steps_no_improve = self.num_timesteps - self._best_score_step
            if steps_no_improve >= self.patience_steps:
                log.info(
                    "Early stopping triggered",
                    reason="no improvement",
                    steps_without_improvement=steps_no_improve,
                    patience=self.patience_steps,
                    best_step=self._best_score_step,
                    best_score=round(self._best_score, 4),
                )
                self._early_stopped = True
                return False

        return True

    # ── Save helpers ──────────────────────────────────────────────────────────

    def _save_checkpoint(self, metrics: ValMetrics) -> None:
        """Save a numbered checkpoint + VecNorm. Also update best_model alias."""
        self._save_n += 1
        is_new_best = metrics.composite_score > self._best_score

        # Numbered checkpoint filename — explicit .zip so SB3 doesn't mangle
        # the score suffix (e.g. c0.57 would be treated as the file extension)
        stem = (
            f"checkpoint_s{self._save_n:02d}"
            f"_step{self.num_timesteps}"
            f"_c{metrics.composite_score:.2f}"
        )
        ckpt_path   = self.save_path / f"{stem}.zip"
        vn_path     = self.save_path / f"{stem}_vecnormalize.pkl"

        self.model.save(str(ckpt_path))
        self._save_vec_normalize(vn_path)

        if is_new_best:
            self._best_score      = metrics.composite_score
            self._best_score_step = self.num_timesteps
            best_path = self.save_path / "best_model"
            best_vn   = self.save_path / "best_model_vecnormalize.pkl"
            self.model.save(str(best_path))
            self._save_vec_normalize(best_vn)

        # ── Visible save banner ───────────────────────────────
        tag  = "★  NEW BEST" if is_new_best else "✓  SAVED"
        line = "=" * 70
        print(f"\n{line}")
        print(f"  MODEL {tag}  #{self._save_n}  |  step {self.num_timesteps:,}  |  phase {self._cur_phase}")
        print(f"  Score : {metrics.composite_score:.4f}   Trades : {metrics.n_trades}   "
              f"WR : {metrics.win_rate*100:.1f}%   PF : {metrics.win_loss_ratio:.2f}   "
              f"Sharpe : {metrics.sharpe_ratio:.2f}")
        print(f"  File  : {ckpt_path}.zip")
        if is_new_best:
            print(f"  Best  : {best_path}.zip  (best_model alias updated)")
        print(f"{line}\n")

    def save_final_checkpoint(self) -> None:
        """
        Save a FINAL_STEP checkpoint regardless of composite score.
        Called by Trainer.run() at end of training as a safety-net so at
        least one testable checkpoint always exists per fold.
        Skipped when save_enabled=False.
        """
        if not self.save_enabled:
            log.info("FINAL_STEP save skipped (save_enabled=False)")
            return
        stem      = f"checkpoint_FINAL_STEP{self.num_timesteps}"
        ckpt_path = self.save_path / f"{stem}.zip"
        vn_path   = self.save_path / f"{stem}_vecnormalize.pkl"
        self.model.save(str(ckpt_path))
        self._save_vec_normalize(vn_path)
        line = "=" * 70
        print(f"\n{line}")
        print(f"  MODEL FINAL_STEP  |  step {self.num_timesteps:,}  (end-of-training safety net)")
        print(f"  File  : {ckpt_path}.zip")
        print(f"{line}\n")

    # ── VecNormalize helpers ──────────────────────────────────────────────────

    def _sync_vec_normalize(self) -> None:
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            train_vn = self.training_env
            eval_vn  = self.eval_env
            if isinstance(train_vn, VecNormalize) and isinstance(eval_vn, VecNormalize):
                eval_vn.obs_rms  = train_vn.obs_rms
                eval_vn.ret_rms  = train_vn.ret_rms
                eval_vn.clip_obs = train_vn.clip_obs
        except Exception:
            pass

    def _save_vec_normalize(self, path: Path) -> None:
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(str(path))
        except Exception:
            pass

    # ── Evaluation runner ─────────────────────────────────────────────────────

    def _run_eval(self) -> Optional[ValMetrics]:
        episode_pnl_r:    List[float] = []
        all_wins_r:       List[float] = []
        all_losses_r:     List[float] = []
        all_durations:    List[float] = []
        all_win_dollars:  List[float] = []
        all_loss_dollars: List[float] = []
        n_wins = n_losses = 0
        cumulative_r: List[float] = [0.0]

        obs            = self.eval_env.reset()
        lstm_states    = None
        episode_starts = np.ones(self.eval_env.num_envs, dtype=bool)
        episodes_done  = 0

        while episodes_done < self.n_eval_episodes:
            action, lstm_states = self.model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, _rewards, dones, infos = self.eval_env.step(action)
            episode_starts = dones

            for env_idx, (done, info) in enumerate(zip(dones, infos)):
                if not done:
                    continue
                episodes_done += 1

                ep_pnl = float(info.get("total_pnl_r", 0.0))
                episode_pnl_r.append(ep_pnl)
                cumulative_r.append(cumulative_r[-1] + ep_pnl)

                n_ep_wins   = int(info.get("n_wins",   0))
                n_ep_losses = int(info.get("n_losses", 0))
                n_wins   += n_ep_wins
                n_losses += n_ep_losses

                avg_win  = float(info.get("avg_win_r",  0.0))
                avg_loss = float(info.get("avg_loss_r", 0.0))
                dur      = float(info.get("avg_trade_duration", 0.0))
                mw       = float(info.get("max_win_dollars",  0.0))
                ml       = float(info.get("max_loss_dollars", 0.0))

                if n_ep_wins  > 0: all_wins_r.append(avg_win)
                if n_ep_losses > 0: all_losses_r.append(avg_loss)
                if dur > 0:         all_durations.append(dur)
                if mw != 0:         all_win_dollars.append(mw)
                if ml != 0:         all_loss_dollars.append(ml)

                if episodes_done >= self.n_eval_episodes:
                    break

        n_trades = n_wins + n_losses
        if n_trades == 0:
            log.warning("No trades in eval — skipping metric update.")
            return None

        total_pnl_r    = float(sum(episode_pnl_r))
        win_rate       = n_wins / n_trades
        win_loss_ratio = n_wins / max(n_losses, 1)

        avg_win_r  = float(np.mean(all_wins_r))   if all_wins_r   else 0.0
        avg_loss_r = float(np.mean(all_losses_r)) if all_losses_r else 0.0
        avg_rr     = avg_win_r / max(avg_loss_r, 1e-6)
        exp_return = win_rate * avg_win_r - (1.0 - win_rate) * avg_loss_r
        avg_dur    = float(np.mean(all_durations)) if all_durations else 0.0

        max_win_d  = float(max(all_win_dollars,  default=0.0))
        max_loss_d = float(min(all_loss_dollars, default=0.0))

        if len(episode_pnl_r) >= 2:
            mu     = float(np.mean(episode_pnl_r))
            std    = float(np.std(episode_pnl_r, ddof=1))
            sharpe = (mu / max(std, 1e-6)) * np.sqrt(252)   # annualised
        else:
            sharpe = float(np.mean(episode_pnl_r)) if episode_pnl_r else 0.0

        max_drawdown_r = self._max_drawdown(cumulative_r)
        composite      = self._composite_score(sharpe, total_pnl_r, win_loss_ratio, max_drawdown_r)

        return ValMetrics(
            total_pnl_r=total_pnl_r,
            sharpe_ratio=sharpe,
            win_loss_ratio=win_loss_ratio,
            max_drawdown_r=max_drawdown_r,
            win_rate=win_rate,
            n_trades=n_trades,
            n_wins=n_wins,
            n_losses=n_losses,
            avg_rr=avg_rr,
            expected_return=exp_return,
            avg_trade_duration=avg_dur,
            max_win_dollars=max_win_d,
            max_loss_dollars=max_loss_d,
            composite_score=composite,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _composite_score(self, sharpe, pnl_r, wl_ratio, max_dd_r) -> float:
        sharpe_norm = float(np.clip((sharpe + 1.0) / (self.sharpe_ref + 1.0), 0.0, 1.0))
        pnl_norm    = float(np.clip(pnl_r    / self.pnl_ref, 0.0, 1.0))
        wl_norm     = float(np.clip(wl_ratio / self.wl_ref,  0.0, 1.0))
        dd_penalty  = float(np.clip(max_dd_r / self.dd_ref,  0.0, 1.0))
        return (
            self.w_sharpe * sharpe_norm
            + self.w_pnl  * pnl_norm
            + self.w_wl   * wl_norm
            + self.w_dd   * (1.0 - dd_penalty)
        )

    @staticmethod
    def _max_drawdown(cumulative: List[float]) -> float:
        arr  = np.array(cumulative, dtype=np.float64)
        peak = arr[0]
        max_dd = 0.0
        for v in arr:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd
        return float(max_dd)

    def _log_metrics(self, m: ValMetrics) -> None:
        step = self.num_timesteps
        phase_min = self._phase_min_composite()
        if self.verbose >= 1:
            log.info(
                "Eval",
                step=step,
                metrics=m.log_str(),
                phase=self._cur_phase,
                phase_min_composite=round(phase_min, 2),
                warmup_passed=(step >= self.warmup_steps),
                steps_since_best=(step - self._best_score_step),
                patience=self.patience_steps,
            )
        self.logger.record("eval/total_pnl_r",         m.total_pnl_r)
        self.logger.record("eval/sharpe_ratio",         m.sharpe_ratio)
        self.logger.record("eval/win_loss_ratio",       m.win_loss_ratio)
        self.logger.record("eval/max_drawdown_r",       m.max_drawdown_r)
        self.logger.record("eval/win_rate",             m.win_rate)
        self.logger.record("eval/n_trades",             m.n_trades)
        self.logger.record("eval/n_wins",               m.n_wins)
        self.logger.record("eval/n_losses",             m.n_losses)
        self.logger.record("eval/avg_rr",               m.avg_rr)
        self.logger.record("eval/expected_return",      m.expected_return)
        self.logger.record("eval/avg_trade_duration",   m.avg_trade_duration)
        self.logger.record("eval/max_win_dollars",      m.max_win_dollars)
        self.logger.record("eval/max_loss_dollars",     m.max_loss_dollars)
        self.logger.record("eval/composite_score",      m.composite_score)
        self.logger.record("eval/phase_min_composite",  phase_min)
        self.logger.record("eval/best_composite",       self._best_score)
        self.logger.record("eval/steps_since_best",     step - self._best_score_step)
        self.logger.dump(step)
