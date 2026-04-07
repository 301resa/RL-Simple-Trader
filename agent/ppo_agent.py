"""
agent/ppo_agent.py
==================
PPO Agent supporting two algorithm modes:

  RecurrentPPO  (default) — sb3-contrib RecurrentPPO with MlpLstmPolicy.
      The LSTM hidden state carries memory across every candle in an episode
      so the agent doesn't need a multi-bar observation window.
      Action validity is enforced at the environment level (step() overrides
      invalid actions to HOLD) rather than through hard masking.

  MaskablePPO   (legacy)  — sb3-contrib MaskablePPO with MlpPolicy.
      Full hard action masking; no LSTM memory.

Select via agent_config.yaml:  algorithm: RecurrentPPO | MaskablePPO
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn

from agent.network import build_policy_kwargs
from utils.logger import get_logger

log = get_logger(__name__)

# ── Algorithm imports ─────────────────────────────────────────────────────────
try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_PPO_AVAILABLE = True
except ImportError:
    RECURRENT_PPO_AVAILABLE = False
    RecurrentPPO = None  # type: ignore[assignment]

try:
    from sb3_contrib import MaskablePPO
    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    MaskablePPO = None  # type: ignore[assignment]

if not RECURRENT_PPO_AVAILABLE and not MASKABLE_PPO_AVAILABLE:
    from stable_baselines3 import PPO as _FallbackPPO
    log.warning(
        "sb3-contrib not installed — falling back to standard PPO. "
        "Install with: pip install sb3-contrib"
    )
else:
    _FallbackPPO = None  # type: ignore[assignment]

# Whether ANY sb3-contrib variant is available
SB3_CONTRIB_AVAILABLE = RECURRENT_PPO_AVAILABLE or MASKABLE_PPO_AVAILABLE


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linearly decay a learning rate from initial_value to 0.

    Parameters
    ----------
    initial_value : float
        Starting learning rate.

    Returns
    -------
    Callable
        Function that takes progress_remaining (1.0 → 0.0) and
        returns the current learning rate.
    """
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def cosine_schedule(initial_value: float, min_value: float = 1e-6) -> Callable[[float], float]:
    """Cosine annealing schedule from initial_value to min_value."""
    import math
    def schedule(progress_remaining: float) -> float:
        # progress_remaining: 1.0 (start) → 0.0 (end)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return min_value + (initial_value - min_value) * cos_val
    return schedule


class PPOAgent:
    """
    Wrapper around RecurrentPPO (default) or MaskablePPO.

    Parameters
    ----------
    env : gymnasium.Env | VecEnv
    algorithm : str
        "RecurrentPPO"  — LSTM policy, session memory, no hard masking
        "MaskablePPO"   — hard action masking, no LSTM
    learning_rate : float
    learning_rate_schedule : str  "linear" | "cosine" | "constant"
    n_steps : int       Steps per rollout per env.
    batch_size : int
    n_epochs : int
    gamma : float
    gae_lambda : float
    clip_range : float
    ent_coef : float
    vf_coef : float
    max_grad_norm : float
    hidden_dims : List[int]
    use_layer_norm : bool
    use_lstm : bool     Must be True for RecurrentPPO.
    lstm_hidden_size : int
    n_lstm_layers : int
    activation_fn_name : str  "ReLU" | "Tanh" | "ELU"
    ortho_init : bool
    device : str        "auto" | "cpu" | "cuda" | "mps"
    seed : int
    tensorboard_log : Optional[str]
    """

    def __init__(
        self,
        env: Any,
        algorithm: str = "RecurrentPPO",
        learning_rate: float = 3e-4,
        learning_rate_schedule: str = "linear",
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_dims: Optional[List[int]] = None,
        use_layer_norm: bool = True,
        use_lstm: bool = True,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        activation_fn_name: str = "ReLU",
        ortho_init: bool = True,
        device: str = "auto",
        seed: int = 42,
        tensorboard_log: Optional[str] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.learning_rate_schedule = learning_rate_schedule

        lr: Union[float, Callable] = self._build_lr_schedule(learning_rate, learning_rate_schedule)

        activation_map: Dict[str, Type[nn.Module]] = {
            "ReLU": nn.ReLU, "Tanh": nn.Tanh, "ELU": nn.ELU,
        }
        activation_fn = activation_map.get(activation_fn_name, nn.ReLU)

        # ── Select algorithm ──────────────────────────────────
        algo_name = algorithm.strip()
        use_recurrent = algo_name == "RecurrentPPO"

        if use_recurrent:
            if not RECURRENT_PPO_AVAILABLE:
                raise ImportError(
                    "RecurrentPPO requires sb3-contrib. "
                    "Install with: pip install sb3-contrib"
                )
            AlgoClass = RecurrentPPO
            policy_name = "MlpLstmPolicy"
            # RecurrentPPO policy_kwargs: lstm_hidden_size and n_lstm_layers
            # go directly into policy_kwargs, not net_arch
            policy_kwargs = {
                "lstm_hidden_size": lstm_hidden_size,
                "n_lstm_layers": n_lstm_layers,
                "net_arch": hidden_dims or [256, 128],
                "activation_fn": activation_fn,
                "ortho_init": ortho_init,
                "enable_critic_lstm": True,
            }
        else:
            if not MASKABLE_PPO_AVAILABLE:
                raise ImportError(
                    "MaskablePPO requires sb3-contrib. "
                    "Install with: pip install sb3-contrib"
                )
            AlgoClass = MaskablePPO
            policy_name = "MlpPolicy"
            policy_kwargs = build_policy_kwargs(
                hidden_dims=hidden_dims or [512, 256, 128],
                use_layer_norm=use_layer_norm,
                use_lstm=False,
                lstm_hidden_size=lstm_hidden_size,
                activation_fn=activation_fn,
                ortho_init=ortho_init,
            )

        log.info(
            "Creating PPO agent",
            algorithm=algo_name,
            policy=policy_name,
            device=device,
            lr=learning_rate,
            lstm=use_recurrent,
            lstm_hidden=lstm_hidden_size if use_recurrent else None,
        )

        self.model = AlgoClass(
            policy=policy_name,
            env=env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
            seed=seed,
            verbose=0,
            tensorboard_log=tensorboard_log,
        )
        self._algorithm = algo_name

    # ── Training API ──────────────────────────────────────────

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        progress_bar: bool = True,
        reset_num_timesteps: bool = True,
    ) -> None:
        """
        Run the PPO training loop.

        Parameters
        ----------
        total_timesteps : int
        callback : Optional[stable_baselines3 callback]
        progress_bar : bool
        reset_num_timesteps : bool
            False = continue training from current step count.
        """
        log.info("Training started", total_timesteps=total_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
        )
        log.info("Training complete")

    def predict(
        self,
        observation: np.ndarray,
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple:
        """
        Get action from the policy.

        Parameters
        ----------
        observation : np.ndarray
        action_masks : Optional[np.ndarray]
            Binary mask for valid actions. Required for MaskablePPO.
        deterministic : bool
            True = take the greedy action (for evaluation).

        Returns
        -------
        (action, state) where state is None for non-recurrent policies.
        """
        if SB3_CONTRIB_AVAILABLE and action_masks is not None:
            return self.model.predict(
                observation,
                action_masks=action_masks,
                deterministic=deterministic,
            )
        return self.model.predict(observation, deterministic=deterministic)

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Save model weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        log.info("Model saved", path=str(path))

    def load(self, path: Union[str, Path], env: Optional[Any] = None) -> None:
        """Load model weights from disk (preserves current algorithm type)."""
        path = Path(path)
        if not path.exists() and not Path(str(path) + ".zip").exists():
            raise FileNotFoundError(f"No model file found at {path}")
        AlgoClass = RecurrentPPO if self._algorithm == "RecurrentPPO" else MaskablePPO
        self.model = AlgoClass.load(str(path), env=env)
        log.info("Model loaded", path=str(path))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        env: Any,
        algorithm: str = "RecurrentPPO",
    ) -> "PPOAgent":
        """Load a previously saved agent from a checkpoint."""
        agent = cls.__new__(cls)
        agent._algorithm = algorithm
        AlgoClass = RecurrentPPO if algorithm == "RecurrentPPO" else MaskablePPO
        agent.model = AlgoClass.load(str(checkpoint_path), env=env)
        log.info("Agent loaded from checkpoint", path=str(checkpoint_path), algorithm=algorithm)
        return agent

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _build_lr_schedule(
        lr: float, schedule: str
    ) -> Union[float, Callable]:
        if schedule == "linear":
            return linear_schedule(lr)
        elif schedule == "cosine":
            return cosine_schedule(lr)
        elif schedule == "constant":
            return lr
        else:
            raise ValueError(f"Unknown lr_schedule: {schedule}. Use 'linear', 'cosine', or 'constant'.")

    @property
    def num_timesteps(self) -> int:
        return self.model.num_timesteps

    @property
    def device(self) -> torch.device:
        return self.model.device