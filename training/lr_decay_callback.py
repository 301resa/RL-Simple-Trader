"""
training/lr_decay_callback.py
=============================
Hot-swaps learning rate schedule mid-training without restarting.

Use case: After model convergence, reduce LR to fine-tune without instability.

Two modes:

  1. trigger_step (recommended):
     At step N, switch from current schedule to a new one.
     Useful when you want to let training converge, THEN decay.

  2. Two-phase static:
     Set stage1_lr, stage1_end, stage2_lr.
     Linear decay from stage1_lr to stage2_lr at stage1_end.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from stable_baselines3.common.callbacks import BaseCallback

from utils.logger import get_logger

log = get_logger(__name__)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linearly decay a learning rate from initial_value to 0."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def cosine_schedule(initial_value: float, min_value: float = 1e-6) -> Callable[[float], float]:
    """Cosine annealing schedule."""
    import math
    def schedule(progress_remaining: float) -> float:
        cos_val = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return min_value + (initial_value - min_value) * cos_val
    return schedule


class LRDecayCallback(BaseCallback):
    """
    Swaps learning rate schedule at a trigger step or decays across two stages.

    Mode 1: trigger_step (preferred for convergence-aware decay)
    ────────────────────────────────────────────────────────────
    At `trigger_step`, switch to a new schedule.
    Useful when you checkpoint test_fold, see good tracking, then decide to decay.

    Args:
        trigger_step: int — switch schedule at this step
        new_initial_lr: float — new base LR for the new schedule
        new_schedule: str — "linear", "cosine", or "constant"

    Mode 2: Two-phase static (legacy, for pre-planned decay)
    ──────────────────────────────────────────────────────────
    stage1_lr → stage2_lr, decaying over [stage1_end, stage2_end].

    Args:
        stage1_lr: float
        stage1_end: int
        stage2_end: int
        stage2_lr: float (final LR)
    """

    def __init__(
        self,
        # Mode 1: trigger-based
        trigger_step: Optional[int] = None,
        new_initial_lr: Optional[float] = None,
        new_schedule: Optional[str] = None,
        # Mode 2: two-phase static
        stage1_lr: Optional[float] = None,
        stage1_end: Optional[int] = None,
        stage2_end: Optional[int] = None,
        stage2_lr: Optional[float] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)

        # Validate: either trigger_step XOR stage-based, not both
        has_trigger = trigger_step is not None
        has_stages = stage1_end is not None
        if has_trigger and has_stages:
            raise ValueError("Specify either trigger_step OR stage1_end, not both")
        if not has_trigger and not has_stages:
            raise ValueError("Specify either trigger_step + new_initial_lr + new_schedule, OR stage1_end + stage2_end + stage1_lr + stage2_lr")

        self.mode = "trigger" if has_trigger else "stages"

        # Mode 1: trigger-based
        self.trigger_step = trigger_step
        self.new_initial_lr = new_initial_lr
        self.new_schedule = new_schedule
        self._trigger_fired = False

        # Mode 2: two-phase static
        self.stage1_lr = stage1_lr or 3e-4
        self.stage1_end = stage1_end or 600_000
        self.stage2_end = stage2_end or 1_200_000
        self.stage2_lr = stage2_lr or 1e-4
        self._last_stage = 0
        self._last_lr = None

    def _on_step(self) -> bool:
        t = self.num_timesteps

        if self.mode == "trigger":
            self._on_step_trigger(t)
        else:
            self._on_step_stages(t)

        return True

    def _on_step_trigger(self, t: int) -> None:
        """Fire once at trigger_step to swap schedule."""
        if t >= self.trigger_step and not self._trigger_fired:
            if self.new_schedule == "constant":
                self.model.learning_rate = self.new_initial_lr
            else:
                schedule_func = {
                    "linear": linear_schedule,
                    "cosine": cosine_schedule,
                }.get(self.new_schedule)
                if schedule_func is None:
                    raise ValueError(f"Unknown schedule: {self.new_schedule}")
                self.model.learning_rate = schedule_func(self.new_initial_lr)

            log.info(
                "LR decay trigger fired",
                step=t,
                new_schedule=self.new_schedule,
                new_initial_lr=self.new_initial_lr,
            )
            self._trigger_fired = True

    def _on_step_stages(self, t: int) -> None:
        """Linearly decay LR across two stages."""
        lr = self._compute_lr_stages(t)

        # Only log when LR significantly changes (avoid spam)
        if self._last_lr is None or abs(lr - self._last_lr) > 1e-6:
            stage = self._current_stage(t)
            if stage != self._last_stage:
                stage_names = {1: "STAGE1 (constant)", 2: "DECAY", 3: "STAGE2 (final)"}
                log.info(
                    "LR schedule stage transition",
                    stage=stage_names.get(stage, stage),
                    step=t,
                    learning_rate=f"{lr:.2e}",
                )
                self._last_stage = stage
            self._last_lr = lr

        # Update the model's learning rate
        self.model.learning_rate = lr

    def _compute_lr_stages(self, t: int) -> float:
        """Compute LR for two-stage mode."""
        if t <= self.stage1_end:
            return self.stage1_lr
        if t >= self.stage2_end:
            return self.stage2_lr
        # Linear decay between stage1_end and stage2_end
        progress = (t - self.stage1_end) / (self.stage2_end - self.stage1_end)
        return self.stage1_lr + (self.stage2_lr - self.stage1_lr) * progress

    def _current_stage(self, t: int) -> int:
        if t <= self.stage1_end:
            return 1
        if t <= self.stage2_end:
            return 2
        return 3
