"""
training/curriculum.py
=======================
Curriculum Learning Scheduler.

Implements staged training as described in the roadmap:
  Stage 1 — trending_only   : Strong trend days only (easy)
  Stage 2 — mixed           : Trending + ranging markets
  Stage 3 — full            : All regimes including high-volatility

Each stage runs for a configured number of timesteps, then
automatically advances. The curriculum filter function is injected
into the environment to restrict episode sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CurriculumStage:
    """
    Configuration for a single curriculum stage.

    Attributes
    ----------
    name : str
        Stage identifier (e.g. "trending_only").
    description : str
    timesteps : int
        Number of training timesteps this stage runs for.
    trend_strength_min : float
        Minimum trend strength (0.0–1.0) for a day to be included.
        0.0 = include all days.
    include_high_vix : bool
        Whether to include high-volatility (VIX spike) days.
    start_step : int
        Timestep at which this stage begins (computed automatically).
    end_step : int
        Timestep at which this stage ends (computed automatically).
    """
    name: str
    description: str
    timesteps: int
    trend_strength_min: float = 0.0
    include_high_vix: bool = True
    start_step: int = field(default=0, init=False)
    end_step: int = field(default=0, init=False)


class CurriculumScheduler:
    """
    Manages curriculum stage progression based on timestep count.

    Parameters
    ----------
    stages : List[CurriculumStage]
        Ordered list of curriculum stages.
    trend_strength_col : str
        Column name in daily_bar used to measure trend strength.
        If not present, trend strength is approximated from OHLCV.
    """

    def __init__(
        self,
        stages: List[CurriculumStage],
        trend_strength_col: str = "trend_strength",
    ) -> None:
        if not stages:
            raise ValueError("At least one curriculum stage required.")

        self.stages = stages
        self.trend_strength_col = trend_strength_col
        self._assign_step_ranges()

        log.info(
            "Curriculum initialized",
            stages=[s.name for s in stages],
            total_steps=sum(s.timesteps for s in stages),
        )

    def current_stage(self, num_timesteps: int) -> Optional[CurriculumStage]:
        """
        Return the active curriculum stage for the given timestep count.

        Parameters
        ----------
        num_timesteps : int
            Current training step count.

        Returns
        -------
        CurriculumStage or None if all stages complete.
        """
        for stage in self.stages:
            if stage.start_step <= num_timesteps < stage.end_step:
                return stage
        # Past all stages — use final stage indefinitely
        return self.stages[-1]

    def build_filter_fn(
        self, stage: CurriculumStage
    ) -> Optional[Callable[[str, pd.Series], bool]]:
        """
        Build a filter function for the given stage.

        The filter function is injected into TradingEnv and called as:
            filter_fn(date: str, daily_bar: pd.Series) → bool

        Returns True if the day should be included in the episode pool.

        Parameters
        ----------
        stage : CurriculumStage

        Returns
        -------
        Callable or None (None = include all days)
        """
        if stage.trend_strength_min <= 0.0 and stage.include_high_vix:
            return None  # No filtering — include all days

        min_strength = stage.trend_strength_min
        include_high_vix = stage.include_high_vix
        col = self.trend_strength_col

        def filter_fn(date: str, daily_bar: pd.Series) -> bool:
            # Trend strength filter
            if min_strength > 0.0:
                if col in daily_bar.index:
                    strength = float(daily_bar[col])
                else:
                    # Approximate: use (|close - open|) / (high - low)
                    h = float(daily_bar.get("high", 1.0))
                    l = float(daily_bar.get("low", 0.0))
                    o = float(daily_bar.get("open", 0.0))
                    c = float(daily_bar.get("close", 0.0))
                    rng = max(h - l, 1e-6)
                    strength = abs(c - o) / rng

                if strength < min_strength:
                    return False

            # High-VIX filter (if column present)
            if not include_high_vix and "vix" in daily_bar.index:
                if float(daily_bar["vix"]) > 30.0:
                    return False

            return True

        return filter_fn

    def stage_summary(self) -> str:
        """Return a human-readable summary of all stages."""
        lines = ["Curriculum Stages:"]
        for s in self.stages:
            lines.append(
                f"  [{s.name}] steps {s.start_step:,}–{s.end_step:,} "
                f"| min_trend_strength={s.trend_strength_min} "
                f"| {s.description}"
            )
        return "\n".join(lines)

    # ── Private ───────────────────────────────────────────────

    def _assign_step_ranges(self) -> None:
        """Compute start_step and end_step for each stage."""
        cumulative = 0
        for stage in self.stages:
            stage.start_step = cumulative
            stage.end_step = cumulative + stage.timesteps
            cumulative += stage.timesteps

    @classmethod
    def from_config(cls, cfg: List[dict]) -> "CurriculumScheduler":
        """
        Build from the curriculum section of environment_config.yaml.

        Parameters
        ----------
        cfg : List[dict]
            List of stage dicts from config["curriculum"]["stages"].
        """
        stages = []
        for s in cfg:
            stages.append(
                CurriculumStage(
                    name=s["name"],
                    description=s.get("description", ""),
                    timesteps=s["timesteps"],
                    trend_strength_min=s.get("trend_strength_min", 0.0),
                    include_high_vix=s.get("include_high_vix", True),
                )
            )
        return cls(stages=stages)