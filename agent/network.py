
"""
agent/network.py
=================
Policy and Value neural network architectures.

Uses a shared-trunk MLP with separate policy and value heads,
plus optional LSTM wrapper for sequential memory.

Design:
  - Input: flat observation vector (~65 features)
  - Shared layers: 512 → 256 → 128 (ReLU + LayerNorm)
  - Policy head: 128 → n_actions (softmax)
  - Value head:  128 → 1 (linear)

Orthogonal initialisation is used throughout for training stability.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class LayerNormMLP(nn.Module):
    """
    Multi-Layer Perceptron with optional LayerNorm between hidden layers.

    Parameters
    ----------
    input_dim : int
    hidden_dims : List[int]
        Sizes of hidden layers.
    output_dim : int
        Output dimension (0 = no output layer, just hidden stack).
    activation : nn.Module class
        Activation function class (e.g. nn.ReLU).
    use_layer_norm : bool
    dropout_rate : float
        0.0 = no dropout.
    ortho_init : bool
        Use orthogonal weight initialisation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 0,
        activation: Type[nn.Module] = nn.ReLU,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0,
        ortho_init: bool = True,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            linear = nn.Linear(in_dim, h_dim)
            if ortho_init:
                nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
                nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(activation())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output_dim = in_dim  # Track for downstream use

        if output_dim > 0:
            out_layer = nn.Linear(in_dim, output_dim)
            if ortho_init:
                nn.init.orthogonal_(out_layer.weight, gain=0.01)
                nn.init.constant_(out_layer.bias, 0.0)
            self.output_layer: Optional[nn.Module] = out_layer
        else:
            self.output_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x


class OrderZoneFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for stable-baselines3.

    Processes the flat observation vector through shared hidden layers
    before splitting into policy and value heads.

    Parameters
    ----------
    observation_space : spaces.Box
    hidden_dims : List[int]
        Hidden layer sizes for the shared trunk.
    use_layer_norm : bool
    activation_fn : Type[nn.Module]
    ortho_init : bool
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        hidden_dims: Optional[List[int]] = None,
        use_layer_norm: bool = True,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
    ) -> None:
        hidden_dims = hidden_dims or [512, 256, 128]
        # features_dim is the output of the shared trunk
        super().__init__(observation_space, features_dim=hidden_dims[-1])

        input_dim = int(np.prod(observation_space.shape))
        self.mlp = LayerNormMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=0,          # No explicit output layer — trunk only
            activation=activation_fn,
            use_layer_norm=use_layer_norm,
            ortho_init=ortho_init,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


class OrderZoneLSTMExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for sequential memory.
    Wraps the MLP trunk with an LSTM to provide temporal context.

    Use when the agent needs to "remember" recent bars beyond the
    lookback window already in the observation.

    Parameters
    ----------
    observation_space : spaces.Box
    mlp_hidden_dims : List[int]
    lstm_hidden_size : int
    n_lstm_layers : int
    use_layer_norm : bool
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        mlp_hidden_dims: Optional[List[int]] = None,
        lstm_hidden_size: int = 128,
        n_lstm_layers: int = 1,
        use_layer_norm: bool = True,
    ) -> None:
        mlp_hidden_dims = mlp_hidden_dims or [256, 128]
        super().__init__(observation_space, features_dim=lstm_hidden_size)

        input_dim = int(np.prod(observation_space.shape))

        # MLP pre-processes each timestep's features
        self.mlp = LayerNormMLP(
            input_dim=input_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=0,
            use_layer_norm=use_layer_norm,
        )

        self.lstm = nn.LSTM(
            input_size=mlp_hidden_dims[-1],
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
        )
        self.lstm_hidden_size = lstm_hidden_size
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_hidden(self, batch_size: int, device: torch.device) -> None:
        """Reset LSTM hidden state (call at episode start)."""
        self._hidden = (
            torch.zeros(1, batch_size, self.lstm_hidden_size, device=device),
            torch.zeros(1, batch_size, self.lstm_hidden_size, device=device),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, obs_dim) → treat as sequence of length 1
        x = self.mlp(observations)           # (batch, mlp_out)
        x = x.unsqueeze(1)                   # (batch, 1, mlp_out) — single timestep
        out, self._hidden = self.lstm(x, self._hidden)
        return out.squeeze(1)                # (batch, lstm_hidden)


def build_policy_kwargs(
    hidden_dims: Optional[List[int]] = None,
    use_layer_norm: bool = True,
    use_lstm: bool = False,
    lstm_hidden_size: int = 128,
    n_lstm_layers: int = 1,
    activation_fn: Type[nn.Module] = nn.ReLU,
    ortho_init: bool = True,
) -> dict:
    """
    Build the policy_kwargs dict for stable-baselines3 PPO.

    Parameters
    ----------
    hidden_dims : List[int]
        Shared trunk hidden layer sizes.
    use_layer_norm : bool
    use_lstm : bool
        If True, uses LSTM extractor instead of MLP.
    lstm_hidden_size : int
    n_lstm_layers : int
    activation_fn : type
    ortho_init : bool

    Returns
    -------
    dict
        Keyword arguments to pass to PPO(..., policy_kwargs=...).
    """
    hidden_dims = hidden_dims or [512, 256, 128]

    if use_lstm:
        extractor_cls = OrderZoneLSTMExtractor
        extractor_kwargs: dict = {
            "mlp_hidden_dims": hidden_dims,
            "lstm_hidden_size": lstm_hidden_size,
            "n_lstm_layers": n_lstm_layers,
            "use_layer_norm": use_layer_norm,
        }
    else:
        extractor_cls = OrderZoneFeaturesExtractor
        extractor_kwargs = {
            "hidden_dims": hidden_dims,
            "use_layer_norm": use_layer_norm,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        }

    return {
        "features_extractor_class": extractor_cls,
        "features_extractor_kwargs": extractor_kwargs,
        "activation_fn": activation_fn,
        "ortho_init": ortho_init,
        # Empty net_arch since the feature extractor handles depth
        "net_arch": dict(pi=[], vf=[]),
    }