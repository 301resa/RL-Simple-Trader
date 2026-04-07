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

Composite save score
--------------------
  score = w_sharpe * sharpe_norm
        + w_pnl    * pnl_norm
        + w_wl     * wl_norm
        + w_dd     * (1 − dd_norm)          ← lower DD is better

  Normalization reference points (configurable):
    sharpe_norm  = clip((sharpe + 1) / 4,  0, 1)   −1→0, 0→0.25, 3→1
    pnl_norm     = clip(total_pnl_r / 20,  0, 1)    20R  per eval run → 1
    wl_norm      = clip(wl_ratio / 3,       0, 1)   3:1 WL ratio     → 1
    dd_norm      = clip(max_dd_r / 8,       0, 1)   8R drawdown      → 1

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
from typing import Any, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from utils.logger import get_logger

log = get_logger(__name__)


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
    saves the best model, and early-stops when no improvement is seen.

    Parameters
    ----------
    eval_env : VecEnv | gymnasium.Env
        Validation environment (single, non-augmented).
    save_path : str | Path
        Directory to write ``best_model.zip``.
    eval_freq : int
        Evaluate every N training steps (total across all envs).
    n_eval_episodes : int
        Number of val episodes per evaluation run.
    warmup_steps : int
        Minimum training steps before early stopping can trigger.
    patience_steps : int
        Stop training if no improvement for this many steps.

    Composite score weights (must sum to 1.0):
    ------------------------------------------
    w_sharpe, w_pnl, w_wl, w_dd

    Normalization anchors (1-unit = score of 1.0 on that axis):
    -----------------------------------------------------------
    sharpe_ref  :  Sharpe of +3 maps to norm = 1.0
    pnl_ref     :  total_pnl_r of +20R maps to norm = 1.0
    wl_ref      :  win/loss ratio of 3.0 maps to norm = 1.0
    dd_ref      :  max_drawdown_r of 8R maps to norm = 1.0 (worst)
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
        sharpe_ref: float        = 3.0,
        pnl_ref:    float        = 20.0,
        wl_ref:     float        = 3.0,
        dd_ref:     float        = 8.0,
        verbose: int             = 1,
    ) -> None:
        super().__init__(verbose)

        # Wrap bare env in DummyVecEnv if needed
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
        self.eval_env = eval_env

        self.save_path       = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.warmup_steps    = warmup_steps
        self.patience_steps  = patience_steps

        # Weights
        total_w = w_sharpe + w_pnl + w_wl + w_dd
        self.w_sharpe = w_sharpe / total_w
        self.w_pnl    = w_pnl    / total_w
        self.w_wl     = w_wl     / total_w
        self.w_dd     = w_dd     / total_w

        # Anchors
        self.sharpe_ref = max(sharpe_ref, 1e-6)
        self.pnl_ref    = max(pnl_ref,    1e-6)
        self.wl_ref     = max(wl_ref,     1e-6)
        self.dd_ref     = max(dd_ref,     1e-6)

        # State
        self._best_score:      float = float("-inf")
        self._best_score_step: int   = 0
        self._last_eval_step:  int   = 0
        self._eval_history:    List[ValMetrics] = []
        self._early_stopped:   bool  = False

    # ── SB3 hook ─────────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        """Called after every environment step during training."""
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True   # not time yet

        self._last_eval_step = self.num_timesteps

        # Sync VecNormalize obs stats from train → eval env so eval uses
        # the same normalisation the model was trained with.
        self._sync_vec_normalize()

        metrics = self._run_eval()

        if metrics is None:
            return True   # nothing collected (no trades in warm-up episodes)

        self._eval_history.append(metrics)
        self._log_metrics(metrics)

        # ── Save best model ───────────────────────────────────
        if metrics.composite_score > self._best_score:
            self._best_score      = metrics.composite_score
            self._best_score_step = self.num_timesteps
            best_path = self.save_path / "best_model"
            self.model.save(str(best_path))
            # Also save VecNormalize stats alongside best model
            self._save_vec_normalize(self.save_path / "best_model_vecnormalize.pkl")
            log.info(
                "New best model saved",
                step=self.num_timesteps,
                composite=round(metrics.composite_score, 4),
                path=str(best_path),
            )
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
                return False  # signals SB3 to stop training

        return True

    # ── VecNormalize helpers ──────────────────────────────────────────────────

    def _sync_vec_normalize(self) -> None:
        """Copy obs running stats from training env → eval env."""
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            train_vn = self.training_env
            eval_vn  = self.eval_env
            if isinstance(train_vn, VecNormalize) and isinstance(eval_vn, VecNormalize):
                eval_vn.obs_rms   = train_vn.obs_rms
                eval_vn.ret_rms   = train_vn.ret_rms
                eval_vn.clip_obs  = train_vn.clip_obs
        except Exception:
            pass

    def _save_vec_normalize(self, path) -> None:
        """Save VecNormalize stats if the training env is wrapped."""
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(str(path))
        except Exception:
            pass

    # ── Evaluation runner ─────────────────────────────────────────────────────

    def _run_eval(self) -> Optional[ValMetrics]:
        """
        Roll out n_eval_episodes on the val env and collect episode summaries.

        Returns ValMetrics or None if no trades were executed.
        """
        # Episode-level P&L list (for Sharpe across episodes)
        episode_pnl_r: List[float] = []

        # Trade-level aggregates
        all_wins_r:      List[float] = []
        all_losses_r:    List[float] = []
        all_durations:   List[float] = []
        all_win_dollars: List[float] = []
        all_loss_dollars: List[float] = []
        n_wins = n_losses = 0

        # Running cumulative R for drawdown calculation
        cumulative_r:   List[float] = [0.0]

        # Recurrent PPO requires carrying LSTM state between steps
        obs      = self.eval_env.reset()
        lstm_states = None
        episode_starts = np.ones(self.eval_env.num_envs, dtype=bool)

        episodes_done = 0

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

                # Trade-level data from episode summary
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

        # ── Compute aggregate metrics ─────────────────────────
        n_trades = n_wins + n_losses
        if n_trades == 0:
            log.warning("No trades in eval — skipping metric update.")
            return None

        total_pnl_r  = float(sum(episode_pnl_r))
        win_rate     = n_wins / n_trades
        win_loss_ratio = n_wins / max(n_losses, 1)

        avg_win_r  = float(np.mean(all_wins_r))  if all_wins_r  else 0.0
        avg_loss_r = float(np.mean(all_losses_r)) if all_losses_r else 0.0
        avg_rr     = avg_win_r / max(avg_loss_r, 1e-6)
        exp_return = win_rate * avg_win_r - (1.0 - win_rate) * avg_loss_r
        avg_dur    = float(np.mean(all_durations)) if all_durations else 0.0

        max_win_d  = float(max(all_win_dollars,  default=0.0))
        max_loss_d = float(min(all_loss_dollars, default=0.0))

        # Episode-level Sharpe
        if len(episode_pnl_r) >= 2:
            mu  = float(np.mean(episode_pnl_r))
            std = float(np.std(episode_pnl_r, ddof=1))
            sharpe = mu / max(std, 1e-6)
        else:
            sharpe = float(np.mean(episode_pnl_r)) if episode_pnl_r else 0.0

        # Max drawdown across all episodes (on cumulative R curve)
        max_drawdown_r = self._max_drawdown(cumulative_r)

        composite = self._composite_score(sharpe, total_pnl_r, win_loss_ratio, max_drawdown_r)

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

    def _composite_score(
        self,
        sharpe:    float,
        pnl_r:    float,
        wl_ratio: float,
        max_dd_r: float,
    ) -> float:
        """
        Weighted normalised composite score.

        All components normalised to [0, 1] before weighting.
        Drawdown is inverted (lower DD = higher norm score).
        """
        # Sharpe: −1 → 0,  0 → 0.25,  3 → 1.0
        sharpe_norm = float(np.clip((sharpe + 1.0) / (self.sharpe_ref + 1.0), 0.0, 1.0))
        # P&L R: 0 → 0,  pnl_ref → 1.0
        pnl_norm    = float(np.clip(pnl_r    / self.pnl_ref, 0.0, 1.0))
        # Win/Loss: 0 → 0,  wl_ref → 1.0
        wl_norm     = float(np.clip(wl_ratio / self.wl_ref,  0.0, 1.0))
        # Max DD: 0 → 1.0,  dd_ref → 0.0  (inverted)
        dd_penalty  = float(np.clip(max_dd_r / self.dd_ref,  0.0, 1.0))

        return (
            self.w_sharpe * sharpe_norm
            + self.w_pnl  * pnl_norm
            + self.w_wl   * wl_norm
            + self.w_dd   * (1.0 - dd_penalty)
        )

    @staticmethod
    def _max_drawdown(cumulative: List[float]) -> float:
        """Peak-to-trough maximum drawdown of a cumulative R series."""
        arr = np.array(cumulative, dtype=np.float64)
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
        if self.verbose >= 1:
            log.info(
                "Eval",
                step=step,
                metrics=m.log_str(),
                warmup_passed=(step >= self.warmup_steps),
                steps_since_best=(step - self._best_score_step),
                patience=self.patience_steps,
            )

        # TensorBoard
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
        self.logger.record("eval/best_composite",       self._best_score)
        self.logger.record("eval/steps_since_best",     step - self._best_score_step)
        self.logger.dump(step)
