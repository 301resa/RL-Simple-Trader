"""
utils/normalizer.py
====================
Online feature normalisation for the observation vector.

Uses Welford's online algorithm to compute running mean and
variance without storing all past observations — O(1) memory.

This is applied AFTER the per-feature normalisation already
done inside ObservationBuilder (ATR-relative scaling, z-scores, etc.)
to further standardise the full observation vector for the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class RunningNormalizer:
    """
    Online mean/variance normalisation using Welford's algorithm.

    Normalises an observation vector by subtracting the running mean
    and dividing by the running standard deviation, estimated online
    across all timesteps seen so far.

    Thread-safe for single-threaded use. For parallel envs, use
    stable-baselines3's VecNormalize wrapper instead.

    Parameters
    ----------
    clip_value : float
        Clip normalised observations to ± this value after normalisation.
    epsilon : float
        Small constant added to variance to prevent division by zero.
    """

    def __init__(
        self,
        clip_value: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        self.clip_value = clip_value
        self.epsilon = epsilon

        self._count: int = 0
        self._mean: Optional[np.ndarray] = None
        self._m2: Optional[np.ndarray] = None       # Sum of squared deviations

    def normalise(self, obs: np.ndarray) -> np.ndarray:
        """
        Update running statistics and return normalised observation.

        Parameters
        ----------
        obs : np.ndarray
            Raw observation vector (float32).

        Returns
        -------
        np.ndarray
            Normalised observation vector (float32).
        """
        self._update(obs)
        std = np.sqrt(self._m2 / max(self._count - 1, 1) + self.epsilon)
        normalised = (obs - self._mean) / std
        return np.clip(normalised, -self.clip_value, self.clip_value).astype(np.float32)

    def reset(self) -> None:
        """Reset all running statistics."""
        self._count = 0
        self._mean = None
        self._m2 = None

    def save(self, path: str | Path) -> None:
        """Persist normaliser statistics to disk (for evaluation reuse)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            count=np.array([self._count]),
            mean=self._mean if self._mean is not None else np.array([]),
            m2=self._m2 if self._m2 is not None else np.array([]),
        )

    def load(self, path: str | Path) -> None:
        """Load normaliser statistics from disk."""
        path = Path(path)
        if not path.exists() and not Path(str(path) + ".npz").exists():
            raise FileNotFoundError(f"Normaliser file not found: {path}")
        data = np.load(str(path) + ".npz" if not str(path).endswith(".npz") else str(path))
        self._count = int(data["count"][0])
        self._mean = data["mean"] if data["mean"].size > 0 else None
        self._m2 = data["m2"] if data["m2"].size > 0 else None

    @property
    def n_samples(self) -> int:
        return self._count

    @property
    def running_mean(self) -> Optional[np.ndarray]:
        return self._mean.copy() if self._mean is not None else None

    @property
    def running_std(self) -> Optional[np.ndarray]:
        if self._m2 is None or self._count < 2:
            return None
        return np.sqrt(self._m2 / (self._count - 1) + self.epsilon)

    # ── Welford update ────────────────────────────────────────

    def _update(self, obs: np.ndarray) -> None:
        """
        Update running mean and M2 (sum of squared deviations)
        using Welford's online algorithm.

        This is numerically stable for large sample counts.
        """
        if self._mean is None:
            self._mean = np.zeros_like(obs, dtype=np.float64)
            self._m2 = np.zeros_like(obs, dtype=np.float64)

        self._count += 1
        delta = obs.astype(np.float64) - self._mean
        self._mean += delta / self._count
        delta2 = obs.astype(np.float64) - self._mean
        self._m2 += delta * delta2