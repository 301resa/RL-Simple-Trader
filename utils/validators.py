"""
utils/validators.py
====================
Runtime validation for configuration and data.

Provides:
  - Config schema validators (YAML dicts → validated objects)
  - OHLCV dataframe validator (used by DataLoader)
  - Hard guardrail assertions (called at environment start)

These prevent silent failures from misconfiguration or bad data
from corrupting a training run.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


# ── OHLCV Validation ─────────────────────────────────────────────────────────

def validate_ohlcv_dataframe(df: pd.DataFrame, context: str = "") -> None:
    """
    Validate an OHLCV DataFrame for minimum integrity requirements.

    Raises ValueError on critical failures.
    Logs warnings for minor issues.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume.
    context : str
        Label for error messages.
    """
    from data.data_validator import DataValidator
    validator = DataValidator()
    validator.validate(df, context=context)


# ── Config Validation ─────────────────────────────────────────────────────────

def validate_agent_config(cfg: dict) -> None:
    """Validate agent_config.yaml structure."""
    _require_keys(cfg, ["algorithm", "ppo", "network"], section="agent_config")
    ppo = cfg["ppo"]
    _require_keys(ppo, ["learning_rate", "n_steps", "batch_size", "n_epochs", "gamma"], section="agent_config.ppo")

    _assert_range(ppo["learning_rate"], 1e-6, 1e-2, "learning_rate")
    _assert_range(ppo["gamma"], 0.9, 1.0, "gamma")
    _assert_range(ppo["gae_lambda"], 0.8, 1.0, "gae_lambda")
    _assert_range(ppo["clip_range"], 0.05, 0.5, "clip_range")
    _assert_positive_int(ppo["n_steps"], "n_steps")
    _assert_positive_int(ppo["batch_size"], "batch_size")
    _assert_positive_int(ppo["n_epochs"], "n_epochs")

    if ppo["batch_size"] > ppo["n_steps"]:
        raise ValueError(
            f"batch_size ({ppo['batch_size']}) must be <= n_steps ({ppo['n_steps']}). "
            "Each mini-batch is sampled from the rollout buffer."
        )

    log.debug("Agent config validated")


def validate_environment_config(cfg: dict) -> None:
    """Validate environment_config.yaml structure."""
    _require_keys(cfg, ["instruments", "session", "contracts", "account"], section="environment_config")

    allowed = cfg["instruments"].get("allowed", [])
    if not allowed:
        raise ValueError("environment_config: instruments.allowed must be a non-empty list.")

    default = cfg["instruments"].get("default", "")
    if default not in allowed:
        raise ValueError(
            f"environment_config: default instrument '{default}' "
            f"not in allowed list {allowed}."
        )

    account = cfg["account"]
    _require_keys(account, ["initial_balance", "max_drawdown_limit"], section="environment_config.account")

    if account["initial_balance"] <= 0:
        raise ValueError("account.initial_balance must be > 0.")
    if account["max_drawdown_limit"] <= 0:
        raise ValueError("account.max_drawdown_limit must be > 0.")

    log.debug("Environment config validated")


def validate_risk_config(cfg: dict) -> None:
    """Validate risk_config.yaml structure."""
    _require_keys(cfg, ["sizing", "stop_loss", "take_profit", "trailing", "daily_limits"], section="risk_config")

    sizing = cfg["sizing"]
    _assert_range(sizing["risk_per_trade_pct"], 0.001, 0.05, "risk_per_trade_pct")
    _assert_positive_int(sizing["min_contracts"], "min_contracts")
    _assert_positive_int(sizing["max_contracts"], "max_contracts")

    if sizing["min_contracts"] > sizing["max_contracts"]:
        raise ValueError("min_contracts must be <= max_contracts.")

    tp = cfg["take_profit"]
    if tp.get("min_rr_ratio", 0) < 1.0:
        raise ValueError("take_profit.min_rr_ratio must be >= 1.0. Recommended: >= 4.0.")

    if cfg["stop_loss"].get("allow_stop_widening", True):
        log.warning(
            "RISK WARNING: stop_loss.allow_stop_widening is True. "
            "This is against the strategy rules. Set to false."
        )

    log.debug("Risk config validated")


def validate_features_config(cfg: dict) -> None:
    """Validate features_config.yaml structure."""
    _require_keys(cfg, ["atr", "swing", "zones", "liquidity", "order_zone", "trend"], section="features_config")

    atr = cfg["atr"]
    _assert_range(atr["exhaustion_threshold"], 0.7, 1.5, "atr.exhaustion_threshold")
    _assert_positive_int(atr["period"], "atr.period")

    zones = cfg["zones"]
    if zones["consolidation_min_bars"] >= zones["consolidation_max_bars"]:
        raise ValueError(
            "zones.consolidation_min_bars must be < consolidation_max_bars."
        )

    order_zone = cfg["order_zone"]
    weights = order_zone.get("weights", {})
    weight_sum = sum(weights.values())
    if not (0.95 <= weight_sum <= 1.05):
        raise ValueError(
            f"order_zone.weights must sum to ~1.0, got {weight_sum:.3f}. "
            f"Weights: {weights}"
        )

    log.debug("Features config validated")


def validate_reward_config(cfg: dict) -> None:
    """Validate reward_config.yaml structure."""
    _require_keys(cfg, ["core", "step", "entry_bonuses", "entry_penalties"], section="reward_config")
    log.debug("Reward config validated")


def validate_all_configs(configs: dict) -> None:
    """
    Run all config validators in sequence.

    Parameters
    ----------
    configs : dict
        Keys: "agent", "environment", "risk", "features", "reward".
        Values: parsed YAML dicts.
    """
    validators = {
        "agent": validate_agent_config,
        "environment": validate_environment_config,
        "risk": validate_risk_config,
        "features": validate_features_config,
        "reward": validate_reward_config,
    }

    errors: List[str] = []
    for name, validator_fn in validators.items():
        if name in configs:
            try:
                validator_fn(configs[name])
            except (ValueError, KeyError) as e:
                errors.append(f"[{name}_config] {e}")
        else:
            log.warning(f"Config section '{name}' not provided — skipping validation.")

    if errors:
        error_msg = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(
            f"Configuration validation failed with {len(errors)} error(s):\n{error_msg}"
        )

    log.info("All configs validated successfully")


# ── Guardrail Assertions ──────────────────────────────────────────────────────

def assert_no_lookahead(feature_bar_idx: int, data_length: int) -> None:
    """Assert that feature computation does not use future data."""
    if feature_bar_idx >= data_length:
        raise ValueError(
            f"Lookahead bias detected: feature_bar_idx={feature_bar_idx} "
            f">= data_length={data_length}. Features must only use past data."
        )


def assert_instrument_allowed(instrument: str, allowed: List[str]) -> None:
    """Assert that the instrument is on the approved list."""
    if instrument.upper() not in [a.upper() for a in allowed]:
        raise ValueError(
            f"Instrument '{instrument}' is not in the allowed list: {allowed}. "
            "Only regulated futures (NQ, ES, MNQ, MES) are permitted."
        )


# ── Private helpers ───────────────────────────────────────────────────────────

def _require_keys(cfg: dict, keys: List[str], section: str) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(
            f"{section}: Missing required keys: {missing}. "
            f"Present keys: {list(cfg.keys())}"
        )


def _assert_range(value: float, lo: float, hi: float, name: str) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} = {value} is outside expected range [{lo}, {hi}].")


def _assert_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")