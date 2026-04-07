"""
training/shaping_decay_callback.py
=====================================
Decays reward shaping bonuses/penalties over 3 training stages:

  Stage 1 — Mechanics  (0 → stage1_end):
      shaping_scale = 1.0  (full shaping — teach agent what good setups look like)

  Stage 2 — Transition (stage1_end → stage2_end):
      shaping_scale linearly decays 1.0 → 0.0

  Stage 3 — Pure P&L   (stage2_end → end):
      shaping_scale = 0.0  (only realised R matters — agent discovers own alpha)

The callback pushes the updated scale to ALL environments in the VecEnv by
calling env.set_attr() — works with both DummyVecEnv and SubprocVecEnv.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from utils.logger import get_logger

log = get_logger(__name__)


class ShapingDecayCallback(BaseCallback):
    """
    Linearly decays reward shaping weight from 1.0 → 0.0 between two
    timestep boundaries, then holds at 0.0 for the rest of training.

    Parameters
    ----------
    stage1_end : int
        Timestep at which shaping starts decaying (default 600k).
    stage2_end : int
        Timestep at which shaping reaches 0.0 (default 1.2M).
    verbose : int
    """

    def __init__(
        self,
        stage1_end: int = 600_000,
        stage2_end: int = 1_200_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.stage1_end = stage1_end
        self.stage2_end = stage2_end
        self._last_scale: float = 1.0
        self._last_stage: int = 0

    def _on_step(self) -> bool:
        t = self.num_timesteps
        scale = self._compute_scale(t)

        # Only push to envs when scale actually changes (avoid spam)
        if abs(scale - self._last_scale) > 0.001:
            self._set_shaping_scale(scale)
            self._last_scale = scale

        # Log stage transitions
        stage = self._current_stage(t)
        if stage != self._last_stage:
            stage_names = {1: "MECHANICS", 2: "TRANSITION", 3: "PURE P&L"}
            log.info(
                "Reward shaping stage transition",
                stage=stage_names.get(stage, stage),
                step=t,
                shaping_scale=round(scale, 3),
            )
            self._last_stage = stage

        return True

    def _compute_scale(self, t: int) -> float:
        if t <= self.stage1_end:
            return 1.0
        if t >= self.stage2_end:
            return 0.0
        # Linear decay between stage1_end and stage2_end
        progress = (t - self.stage1_end) / (self.stage2_end - self.stage1_end)
        return round(1.0 - progress, 4)

    def _current_stage(self, t: int) -> int:
        if t <= self.stage1_end:
            return 1
        if t <= self.stage2_end:
            return 2
        return 3

    def _set_shaping_scale(self, scale: float) -> None:
        """Push shaping_scale into reward_calculator of every env."""
        try:
            self.training_env.env_method("_set_shaping_scale", scale)
        except Exception:
            # Fallback for DummyVecEnv or single env
            try:
                for env in self.training_env.envs:
                    _push_scale(env, scale)
            except Exception:
                pass


def _push_scale(env, scale: float) -> None:
    """Recursively unwrap env to find reward_calculator and set scale."""
    target = env
    while target is not None:
        if hasattr(target, "reward_calculator"):
            target.reward_calculator.shaping_scale = scale
            return
        target = getattr(target, "env", None)
