"""
utils/instrument.py
====================
Per-instrument profile: ALL price-based thresholds expressed in TICKS.

One place to switch instruments (ES ↔ NQ ↔ MES ↔ MNQ). Every downstream module
reads geometry from this profile — nothing else should hardcode point values,
stop buffers, zone widths, or jitter magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class InstrumentProfile:
    symbol: str
    point_value: float           # $ per point (ES=$50, MES=$5, NQ=$20, MNQ=$2)
    tick_size: float             # price increment (e.g. 0.25)
    tick_value: float            # $ per tick per contract

    # Geometry (all in POINTS, derived from tick counts × tick_size)
    stop_buffer_pts: float       # distance beyond far zone edge for stop
    min_target_pts: float        # minimum TP distance from entry
    jitter_pts: float            # data-augmentor jitter magnitude
    min_zone_pts: float          # reject zones narrower than this
    max_zone_pts: float          # reject zones wider than this
    fallback_stop_pts: float     # emergency stop when zone geometry unavailable

    # Sizing
    contract_tiers: List[float]             # graduated contract sizes
    confluence_tier_thresholds: List[float] # N-1 thresholds → N tiers


def load_instrument_profile(env_cfg: Dict[str, Any]) -> InstrumentProfile:
    """Read `default` instrument and assemble its profile from tick counts."""
    default = env_cfg["instruments"]["default"]
    contracts = env_cfg["contracts"]
    if default not in contracts:
        raise ValueError(f"Instrument '{default}' not in contracts config")
    c = contracts[default]

    required = [
        "micro_point_value", "tick_size", "tick_value",
        "stop_buffer_ticks", "min_target_ticks", "jitter_ticks",
        "min_zone_ticks", "max_zone_ticks", "fallback_stop_ticks",
        "contract_tiers", "confluence_tier_thresholds",
    ]
    missing = [k for k in required if k not in c]
    if missing:
        raise ValueError(
            f"Instrument '{default}' missing fields: {missing}. "
            f"Add them under contracts.{default} in environment_config.yaml."
        )

    tick = float(c["tick_size"])
    return InstrumentProfile(
        symbol=default,
        point_value=float(c["micro_point_value"]),
        tick_size=tick,
        tick_value=float(c["tick_value"]),
        stop_buffer_pts=float(c["stop_buffer_ticks"]) * tick,
        min_target_pts=float(c["min_target_ticks"]) * tick,
        jitter_pts=float(c["jitter_ticks"]) * tick,
        min_zone_pts=float(c["min_zone_ticks"]) * tick,
        max_zone_pts=float(c["max_zone_ticks"]) * tick,
        fallback_stop_pts=float(c["fallback_stop_ticks"]) * tick,
        contract_tiers=[float(x) for x in c["contract_tiers"]],
        confluence_tier_thresholds=[float(x) for x in c["confluence_tier_thresholds"]],
    )
