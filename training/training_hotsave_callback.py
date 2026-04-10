"""
training/training_hotsave_callback.py
=======================================
Saves the model mid-training when per-env rolling metrics cross one of
two quality gates.

Gate 1 — Standard PF/WR gate (as before):
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

Each gate has its own cooldown. Gate-2 saves use the prefix "hotsave_sh_"
to distinguish them from Gate-1 saves ("hotsave_").

Saves: <models_dir>/hotsave[_sh]_<step>.zip  +  ..._vecnormalize.pkl
"""

from __future__ import annotations

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
        self.check_every_steps    = check_every_steps
        self.vec_normalize        = vec_normalize

        self._env_latest: Dict[int, dict] = {}
        self._last_save_step:        int = -cooldown_steps
        self._last_sharpe_save_step: int = -sharpe_cooldown_steps

    # ── Episode capture ───────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, (info, done) in enumerate(zip(infos, dones)):
            if done and "profit_factor" in info:
                self._env_latest[i] = dict(info)

        # Check both gates every N steps
        if self.num_timesteps % self.check_every_steps < self.n_envs:
            self._check_pf_gate()
            self._check_sharpe_gate()

        return True

    # ── Gate 1 — PF / WR ─────────────────────────────────────────────────────

    def _check_pf_gate(self) -> None:
        step = self.num_timesteps
        if step - self._last_save_step < self.cooldown_steps:
            return

        filled = [v for v in self._env_latest.values() if v]
        if not filled:
            return

        passing = [
            d for d in filled
            if (
                d.get("profit_factor", 0.0) > self.pf_threshold
                and d.get("win_rate",      0.0) >= self.wr_threshold
                and d.get("n_trades",      0)   >= self.min_trades
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

        filled = [v for v in self._env_latest.values() if v]
        if not filled:
            return

        passing = [
            d for d in filled
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
