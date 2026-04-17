"""
training/training_hotsave_callback.py
=======================================
Saves the model mid-training when per-env rolling metrics cross one of
three quality gates.

Gate 1 — Standard PF/WR gate:
  At least min_envs_passing training envs simultaneously satisfy:
    • PF  > pf_threshold   (default 1.60)
    • WR  >= wr_threshold  (default 0.40)
    • trades >= min_trades (default 20)

Gate 2 — Sharpe quality gate (high-quality saves):
  At least min_envs_passing training envs simultaneously satisfy:
    • Sharpe > sharpe_threshold  (default 1.2)
    • PF     > sharpe_pf_threshold (default 1.85)
    • trades >= min_trades        (default 20)
    • win_loss_ratio > 1.0        (avg winner > avg loser)
    • total_pnl_r   > 0.0         (episode net positive)

Gate 3 — Win-rate 70 gate (saves any env reaching clear positive quality):
  Any single training env satisfies ALL of:
    • WR  >= 0.70             (70%+ win rate)
    • total_pnl_dollars > 0   (dollar PnL positive)
    • trades >= 20            (statistically meaningful)
  Saved with prefix "hotsave_wr70_".

Each gate has its own cooldown. Gate-2 saves use the prefix "hotsave_sh_",
Gate-3 saves use "hotsave_wr70_".

Saves: <models_dir>/hotsave[_sh|_wr70]_<step>.zip  +  ..._vecnormalize.pkl
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from utils.logger import get_logger

log = get_logger(__name__)


class TrainingHotSaveCallback(BaseCallback):
    """
    Saves the model during training when either quality gate is cleared.

    Parameters
    ----------
    models_dir : str | Path
        Directory to write hot-save files into.
    pf_threshold : float
        Gate 1 — each qualifying env must have PF > this value.
    wr_threshold : float
        Gate 1 — each qualifying env must have win rate >= this (0–1).
    min_trades : int
        Both gates — minimum trades an env must have to be counted.
    min_envs_passing : int
        Both gates — how many envs must simultaneously pass all criteria.
    cooldown_steps : int
        Minimum steps between consecutive Gate-1 hot-saves.
    sharpe_threshold : float
        Gate 2 — each qualifying env must have annualised Sharpe > this.
    sharpe_pf_threshold : float
        Gate 2 — each qualifying env must have PF > this value.
    sharpe_cooldown_steps : int
        Minimum steps between consecutive Gate-2 hot-saves.
    check_every_steps : int
        How often (in steps) to check both gates (~one rollout = 4096).
    vec_normalize : VecNormalize | None
        If provided, saves normalisation stats alongside each checkpoint.
    """

    # Window size: aggregate this many recent episodes per env for gate checks.
    # With ~5 trades/session, a window of 5 gives ~25 trades — enough to satisfy
    # min_trades=20 while remaining responsive to recent policy quality.
    EPISODE_WINDOW: int = 5

    def __init__(
        self,
        models_dir: str | Path,
        pf_threshold: float = 1.60,
        wr_threshold: float = 0.40,
        min_trades: int = 20,
        min_envs_passing: int = 2,
        cooldown_steps: int = 50_000,
        sharpe_threshold: float = 1.2,
        sharpe_pf_threshold: float = 1.85,
        sharpe_cooldown_steps: int = 50_000,
        # Gate 3 — WR70 quality gate
        wr70_min_trades: int = 20,
        wr70_cooldown_steps: int = 50_000,
        # Minimum dollar PnL for WR70 gate = 0.5% of initial capital
        initial_capital: float = 2500.0,
        check_every_steps: int = 4_096,
        vec_normalize=None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.models_dir           = Path(models_dir)
        self.pf_threshold         = pf_threshold
        self.wr_threshold         = wr_threshold
        self.min_trades           = min_trades
        self.min_envs_passing     = min_envs_passing
        self.cooldown_steps       = cooldown_steps
        self.sharpe_threshold     = sharpe_threshold
        self.sharpe_pf_threshold  = sharpe_pf_threshold
        self.sharpe_cooldown_steps= sharpe_cooldown_steps
        self.wr70_min_trades      = wr70_min_trades
        self.wr70_cooldown_steps  = wr70_cooldown_steps
        self.wr70_min_pnl_dollars = 0.005 * initial_capital  # 0.5% of capital
        self.check_every_steps    = check_every_steps
        self.vec_normalize        = vec_normalize

        # Rolling window of last EPISODE_WINDOW episodes per env.
        # Gates aggregate across the window so a single short session
        # (2-3 trades) doesn't prevent saving when recent quality is high.
        self._env_windows: Dict[int, deque] = {}
        self._last_save_step:        int = -cooldown_steps
        self._last_sharpe_save_step: int = -sharpe_cooldown_steps
        self._last_wr70_save_step:   int = -wr70_cooldown_steps

    # ── Episode capture ───────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, (info, done) in enumerate(zip(infos, dones)):
            if done and "profit_factor" in info:
                if i not in self._env_windows:
                    self._env_windows[i] = deque(maxlen=self.EPISODE_WINDOW)
                self._env_windows[i].append(dict(info))

        # Check all gates every N steps
        if self.num_timesteps % self.check_every_steps < self.n_envs:
            self._check_pf_gate()
            self._check_sharpe_gate()
            self._check_wr70_gate()

        return True

    # ── Window aggregation helper ─────────────────────────────────────────────

    def _aggregate(self, window: deque) -> dict:
        """Aggregate a deque of per-episode info dicts into one combined dict.

        Trades, wins, losses, and dollar/R P&L are summed; ratio metrics
        (WR, PF, Sharpe, win_loss_ratio) are recomputed from the sums so
        they reflect the full window — not a simple average of episode values.
        """
        eps = list(window)
        if not eps:
            return {}

        n_trades = sum(e.get("n_trades", 0)  for e in eps)
        n_wins   = sum(e.get("n_wins",   0)  for e in eps)
        n_losses = sum(e.get("n_losses", 0)  for e in eps)
        total_pnl_r       = sum(e.get("total_pnl_r",       0.0) for e in eps)
        total_pnl_dollars = sum(e.get("total_pnl_dollars",  0.0) for e in eps)

        win_rate = n_wins / n_trades if n_trades > 0 else 0.0

        # Reconstruct gross win/loss R from per-episode averages × counts
        gross_win_r  = sum(e.get("avg_win_r",  0.0) * e.get("n_wins",   0) for e in eps)
        gross_loss_r = sum(e.get("avg_loss_r", 0.0) * e.get("n_losses", 0) for e in eps)
        profit_factor = min(gross_win_r / max(gross_loss_r, 1e-6), 99.99) if n_trades else 0.0

        avg_win_r  = gross_win_r  / n_wins   if n_wins   else 0.0
        avg_loss_r = gross_loss_r / n_losses if n_losses else 1.0
        win_loss_ratio = avg_win_r / max(avg_loss_r, 1e-6)

        # Sharpe from window-level per-trade returns (collect all individual pnl_r)
        all_r = []
        for e in eps:
            aw, al, nw, nl = (e.get("avg_win_r",0.), e.get("avg_loss_r",0.),
                              e.get("n_wins",0),     e.get("n_losses",0))
            all_r.extend([aw] * nw + [-al] * nl)
        if len(all_r) >= 5:
            arr = np.array(all_r, dtype=np.float32)
            std = float(np.std(arr))
            sharpe_ratio = float(np.clip(np.mean(arr) / std * np.sqrt(252), -9.99, 9.99)) if std > 0.01 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            "n_trades":          n_trades,
            "n_wins":            n_wins,
            "n_losses":          n_losses,
            "win_rate":          win_rate,
            "total_pnl_r":       total_pnl_r,
            "total_pnl_dollars": total_pnl_dollars,
            "profit_factor":     profit_factor,
            "win_loss_ratio":    win_loss_ratio,
            "sharpe_ratio":      sharpe_ratio,
        }

    # ── Gate 1 — PF / WR ─────────────────────────────────────────────────────

    def _check_pf_gate(self) -> None:
        step = self.num_timesteps
        if step - self._last_save_step < self.cooldown_steps:
            return
        if not self._env_windows:
            return

        agg_list = [self._aggregate(w) for w in self._env_windows.values() if w]
        passing  = [
            d for d in agg_list
            if (
                d.get("profit_factor",   0.0) > self.pf_threshold
                and d.get("win_rate",    0.0) >= self.wr_threshold
                and d.get("n_trades",    0)   >= self.min_trades
                and d.get("total_pnl_r", 0.0) > 0.0   # never save on net loss
            )
        ]
        if len(passing) < self.min_envs_passing:
            return

        avg_pf = float(np.mean([d["profit_factor"] for d in passing]))
        avg_wr = float(np.mean([d["win_rate"]       for d in passing]))
        self._save(
            step=step,
            prefix="hotsave",
            tag="PF gate",
            metrics={"avg_PF": avg_pf, "avg_WR": avg_wr, "envs": len(passing)},
        )
        self._last_save_step = step

    # ── Gate 2 — Sharpe quality ───────────────────────────────────────────────

    def _check_sharpe_gate(self) -> None:
        step = self.num_timesteps
        if step - self._last_sharpe_save_step < self.sharpe_cooldown_steps:
            return
        if not self._env_windows:
            return

        agg_list = [self._aggregate(w) for w in self._env_windows.values() if w]
        passing  = [
            d for d in agg_list
            if (
                d.get("sharpe_ratio",    0.0) > self.sharpe_threshold
                and d.get("profit_factor", 0.0) > self.sharpe_pf_threshold
                and d.get("n_trades",      0)   >= self.min_trades
                and d.get("win_loss_ratio", 0.0) > 1.0
                and d.get("total_pnl_r",   0.0) > 0.0
            )
        ]
        if len(passing) < self.min_envs_passing:
            return

        avg_sh = float(np.mean([d["sharpe_ratio"]   for d in passing]))
        avg_pf = float(np.mean([d["profit_factor"]  for d in passing]))
        avg_rr = float(np.mean([d["win_loss_ratio"] for d in passing]))
        self._save(
            step=step,
            prefix="hotsave_sh",
            tag="Sharpe gate",
            metrics={"avg_SH": avg_sh, "avg_PF": avg_pf, "avg_RR": avg_rr, "envs": len(passing)},
        )
        self._last_sharpe_save_step = step

    # ── Gate 3 — WR70 quality gate ────────────────────────────────────────────

    def _check_wr70_gate(self) -> None:
        """Save when any env's rolling window achieves WR ≥ 70%, PnL > 0, trades ≥ 20."""
        step = self.num_timesteps
        if step - self._last_wr70_save_step < self.wr70_cooldown_steps:
            return
        if not self._env_windows:
            return

        agg_list   = [self._aggregate(w) for w in self._env_windows.values() if w]
        qualifying = [
            d for d in agg_list
            if (
                d.get("win_rate",            0.0) >= 0.70
                and d.get("total_pnl_dollars", d.get("total_pnl_r", 0.0)) >= self.wr70_min_pnl_dollars
                and d.get("n_trades",          0)   >= self.wr70_min_trades
                and d.get("total_pnl_r",       0.0) > 0.0   # R-based sanity guard
            )
        ]
        if not qualifying:
            return

        best = max(qualifying, key=lambda d: d.get("win_rate", 0.0))
        self._save(
            step=step,
            prefix="hotsave_wr70",
            tag="WR70 gate",
            metrics={
                "WR":     best.get("win_rate", 0.0),
                "PnL_$":  best.get("total_pnl_dollars", 0.0),
                "trades": best.get("n_trades", 0),
                "envs":   len(qualifying),
            },
        )
        self._last_wr70_save_step = step

    # ── Shared save helper ────────────────────────────────────────────────────

    def _save(self, step: int, prefix: str, tag: str, metrics: dict) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        name       = f"{prefix}_{step:010d}"
        model_path = self.models_dir / name

        self.model.save(str(model_path))

        if self.vec_normalize is not None:
            vn_path = self.models_dir / f"{name}_vecnormalize.pkl"
            self.vec_normalize.save(str(vn_path))

        if self.verbose >= 1:
            vn_note  = " + VecNormalize" if self.vec_normalize is not None else ""
            met_str  = "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in metrics.items())
            line = "=" * 70
            print(f"\n{line}")
            print(f"  HOTSAVE [{tag}]  |  step {step:,}")
            print(f"  {met_str}")
            print(f"  File : {model_path}.zip{vn_note}")
            print(line)

        log.info(
            "Training hot-save written",
            gate=tag,
            step=step,
            path=str(model_path),
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()},
        )

    # ── n_envs helper ─────────────────────────────────────────────────────────

    @property
    def n_envs(self) -> int:
        try:
            return self.training_env.num_envs
        except AttributeError:
            return 1
