"""
training/training_hotsave_callback.py
=======================================
V7-style training hot-save: saves the model mid-training when the
rolling PF/WR metrics across live training envs cross quality gates.

Two-tier gate (both must pass):
  Tier 1 (aggregate): mean PF across ALL envs > pf_avg_threshold
  Tier 2 (individual): at least min_envs_passing envs individually satisfy
                       PF > pf_ind_threshold AND WR >= wr_threshold
                       AND trades >= min_trades

A cooldown of cooldown_steps between hot-saves prevents flooding the disk
when the agent briefly crosses the gate.

Saves: <models_dir>/hotsave_<step>.zip  +  <models_dir>/hotsave_<step>_vecnormalize.pkl
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from utils.logger import get_logger

log = get_logger(__name__)


class TrainingHotSaveCallback(BaseCallback):
    """
    Saves the model during training whenever rolling per-env metrics
    cross a two-tier quality gate.

    Parameters
    ----------
    models_dir : str | Path
        Directory to write hot-save files into.
    pf_avg_threshold : float
        Tier-1 gate — mean PF across all envs must exceed this.
    pf_ind_threshold : float
        Tier-2 gate — individual env PF must exceed this.
    wr_threshold : float
        Tier-2 gate — individual env win rate must be >= this (0–1).
    min_trades : int
        Minimum trades an env must have before it counts toward Tier-2.
    min_envs_passing : int
        How many individual envs must pass Tier-2 simultaneously.
    cooldown_steps : int
        Minimum steps between consecutive hot-saves.
    check_every_steps : int
        How often (in steps) to evaluate the gate (default 4096 — one rollout).
    vec_normalize : VecNormalize | None
        If provided, saves normalisation stats alongside each checkpoint.
    """

    def __init__(
        self,
        models_dir: str | Path,
        pf_avg_threshold: float = 1.30,
        pf_ind_threshold: float = 1.80,
        wr_threshold: float = 0.40,
        min_trades: int = 20,
        min_envs_passing: int = 2,
        cooldown_steps: int = 50_000,
        check_every_steps: int = 4_096,
        vec_normalize=None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.models_dir        = Path(models_dir)
        self.pf_avg_threshold  = pf_avg_threshold
        self.pf_ind_threshold  = pf_ind_threshold
        self.wr_threshold      = wr_threshold
        self.min_trades        = min_trades
        self.min_envs_passing  = min_envs_passing
        self.cooldown_steps    = cooldown_steps
        self.check_every_steps = check_every_steps
        self.vec_normalize     = vec_normalize

        self._env_latest: Dict[int, dict] = {}
        self._last_save_step: int = -cooldown_steps  # allow save at step 0 if gate passes

    # ── Episode capture ───────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, (info, done) in enumerate(zip(infos, dones)):
            if done and "profit_factor" in info:
                self._env_latest[i] = dict(info)

        # Gate check every N steps
        if self.num_timesteps % self.check_every_steps < self.n_envs:
            self._maybe_save()

        return True

    # ── Gate logic ────────────────────────────────────────────────────────────

    def _maybe_save(self) -> None:
        step = self.num_timesteps

        # Cooldown guard
        if step - self._last_save_step < self.cooldown_steps:
            return

        filled = [v for v in self._env_latest.values() if v]
        if not filled:
            return

        # Tier 1 — aggregate PF
        pf_values = [d.get("profit_factor", 0.0) for d in filled]
        avg_pf = float(np.mean(pf_values))
        if avg_pf <= self.pf_avg_threshold:
            return

        # Tier 2 — count individual envs passing all three criteria
        n_passing = sum(
            1 for d in filled
            if (
                d.get("profit_factor", 0.0) > self.pf_ind_threshold
                and d.get("win_rate", 0.0) >= self.wr_threshold
                and d.get("n_trades", 0) >= self.min_trades
            )
        )
        if n_passing < self.min_envs_passing:
            return

        # Both tiers cleared — save
        self._save(step, avg_pf, n_passing)

    def _save(self, step: int, avg_pf: float, n_passing: int) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        name = f"hotsave_{step:010d}"
        model_path = self.models_dir / name

        self.model.save(str(model_path))
        self._last_save_step = step

        if self.vec_normalize is not None:
            vn_path = self.models_dir / f"{name}_vecnormalize.pkl"
            self.vec_normalize.save(str(vn_path))

        if self.verbose >= 1:
            vn_note = " + VecNormalize" if self.vec_normalize is not None else ""
            print(
                f"\n[HotSave] step={step:,}  avg_PF={avg_pf:.2f}  "
                f"envs_passing={n_passing}  → {model_path}.zip{vn_note}"
            )
        log.info(
            "Training hot-save written",
            step=step,
            avg_pf=round(avg_pf, 3),
            envs_passing=n_passing,
            path=str(model_path),
        )

    # ── n_envs helper ─────────────────────────────────────────────────────────

    @property
    def n_envs(self) -> int:
        try:
            return self.training_env.num_envs
        except AttributeError:
            return 1
