"""
utils/feature_exporter.py
==========================
Exports all computed features for every bar of every trading day to Excel.

Usage (standalone):
    python -m utils.feature_exporter --config config/ --data data/ --out features.xlsx

Usage (from code):
    from utils.feature_exporter import export_features
    export_features(data_loader, atr_calculator, zone_detector,
                    order_zone_engine, trading_days, out_path="features.xlsx")

Columns exported per bar:
  datetime, date, open, high, low, close, volume,
  atr, atr_pct, atr_short_exhausted, atr_long_exhausted,
  n_supply_zones, n_demand_zones, nearest_supply, nearest_demand,
  in_bullish_oz, in_bearish_oz, confluence_score, rr_ratio,
  zone_score_bearish, zone_score_bullish, atr_room_bearish, atr_room_bullish
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def export_features(
    data_loader,
    atr_calculator,
    zone_detector,
    order_zone_engine,
    trading_days: List[str],
    out_path: str = "features.xlsx",
    max_days: Optional[int] = None,
) -> str:
    """
    Compute and export all features for every bar across trading_days.

    Parameters
    ----------
    data_loader       : DataLoader
    atr_calculator    : ATRCalculator
    zone_detector     : ZoneDetector
    order_zone_engine : OrderZoneEngine
    trading_days      : list of YYYY-MM-DD strings (train or all days)
    out_path          : output .xlsx path
    max_days          : limit export to first N days (None = all)

    Returns
    -------
    Absolute path to the written file.
    """
    days = trading_days[:max_days] if max_days else trading_days
    rows = []

    print(f"[FeatureExporter] Processing {len(days)} trading days…")

    for day_idx, date in enumerate(days):
        bars = data_loader.get_day_bars(date)
        if bars.empty:
            continue

        daily_atr = atr_calculator.get_atr_for_date(date)
        if daily_atr is None:
            continue

        atr_series = pd.Series([daily_atr] * len(bars), index=bars.index)

        for bar_idx in range(len(bars)):
            bar = bars.iloc[bar_idx]

            # ── ATR ───────────────────────────────────────────
            atr_state = atr_calculator.compute_session_state(
                date=date,
                session_bars=bars,
                current_bar_idx=bar_idx,
            )
            if atr_state is None:
                continue

            # ── Zone ──────────────────────────────────────────
            zone_state = zone_detector.scan_and_update(
                bars=bars,
                atr_series=atr_series,
                current_bar_idx=bar_idx,
            )

            # ── Order zone ────────────────────────────────────
            oz_state = order_zone_engine.compute(
                bars=bars,
                current_bar_idx=bar_idx,
                atr_state=atr_state,
                zone_state=zone_state,
            )

            rows.append({
                # Identity
                "datetime":          bar.name,
                "date":              date,
                "bar_idx":           bar_idx,
                # OHLCV
                "open":              float(bar["open"]),
                "high":              float(bar["high"]),
                "low":               float(bar["low"]),
                "close":             float(bar["close"]),
                "volume":            int(bar.get("volume", 0)),
                # ATR
                "atr":               round(atr_state.atr_daily, 4),
                "atr_remaining_pts": round(atr_state.atr_remaining_pts, 4),
                "atr_pct_used":      round(atr_state.atr_pct_used, 4),
                "atr_short_exhausted": int(atr_state.atr_short_exhausted),
                "atr_long_exhausted":  int(atr_state.atr_long_exhausted),
                # Zones
                "has_supply_zone":   int(zone_state.nearest_supply is not None and zone_state.nearest_supply.is_valid) if zone_state else 0,
                "has_demand_zone":   int(zone_state.nearest_demand is not None and zone_state.nearest_demand.is_valid) if zone_state else 0,
                "nearest_supply":    round(zone_state.nearest_supply.midpoint, 4) if (zone_state and zone_state.nearest_supply) else 0.0,
                "nearest_demand":    round(zone_state.nearest_demand.midpoint, 4) if (zone_state and zone_state.nearest_demand) else 0.0,
                # Order zone
                "in_bullish_oz":     int(oz_state.in_bullish_order_zone),
                "in_bearish_oz":     int(oz_state.in_bearish_order_zone),
                "confluence_score":  round(oz_state.confluence_score, 4),
                "rr_ratio":          round(oz_state.rr_ratio, 4),
                # Component scores
                "zone_score_bearish":    round(getattr(oz_state, "component_scores", {}).get("zone_bearish", 0.0), 4),
                "zone_score_bullish":    round(getattr(oz_state, "component_scores", {}).get("zone_bullish", 0.0), 4),
                "atr_room_bearish":      round(getattr(oz_state, "component_scores", {}).get("atr_room_bearish", 0.0), 4),
                "atr_room_bullish":      round(getattr(oz_state, "component_scores", {}).get("atr_room_bullish", 0.0), 4),
            })

        if (day_idx + 1) % 50 == 0:
            print(f"  … {day_idx + 1}/{len(days)} days processed")

    if not rows:
        print("[FeatureExporter] No data rows — nothing written.")
        return out_path

    df = pd.DataFrame(rows)

    # Excel cannot handle tz-aware datetimes — strip timezone
    if "datetime" in df.columns and hasattr(df["datetime"].dtype, "tz"):
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Features", index=False)

        # ── Auto-fit column widths ─────────────────────────────
        ws = writer.sheets["Features"]
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 30)

    abs_path = str(out_path.resolve())
    print(f"[FeatureExporter] Saved {len(df):,} rows → {abs_path}")
    return abs_path


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Export all RL agent features to Excel")
    parser.add_argument("--config", default="config/",  help="Config directory")
    parser.add_argument("--data",   default="data/",    help="Data directory")
    parser.add_argument("--out",    default="logs/features.xlsx", help="Output .xlsx path")
    parser.add_argument("--days",   type=int, default=None, help="Limit to first N days")
    args = parser.parse_args()

    import yaml
    from pathlib import Path as _Path

    def _load(cfg_dir, fname):
        p = _Path(cfg_dir) / fname
        return yaml.safe_load(p.read_text()) if p.exists() else {}

    env_cfg  = _load(args.config, "environment_config.yaml")
    feat_cfg = _load(args.config, "features_config.yaml")
    risk_cfg = _load(args.config, "risk_config.yaml")

    from data.data_loader import DataLoader
    from features.atr_calculator import ATRCalculator
    from features.order_zone_engine import OrderZoneEngine
    from features.zone_detector import ZoneDetector

    sess  = env_cfg.get("session", {})
    instr = env_cfg.get("instruments", {}).get("default", "ES")

    data_cfg = env_cfg.get("data", {})
    _tf = data_cfg.get("timeframe", f"{sess.get('bar_timeframe_minutes', 5)}min")
    dl = DataLoader(
        data_dir=args.data,
        instrument=data_cfg.get("instrument", instr),
        timeframe=_tf,
        tz=sess.get("timezone", "America/New_York"),
    )
    dl.load()

    atr_cfg = feat_cfg.get("atr", {})
    atr = ATRCalculator(
        atr_period=atr_cfg.get("period", 14),
        exhaustion_threshold=atr_cfg.get("exhaustion_threshold", 0.95),
    )
    atr.fit(dl.daily)

    zones_cfg = feat_cfg.get("zones", {})
    zd = ZoneDetector(
        consolidation_min_bars=zones_cfg.get("consolidation_min_bars", 2),
        consolidation_max_bars=zones_cfg.get("consolidation_max_bars", 8),
        consolidation_range_atr_pct=zones_cfg.get("consolidation_range_atr_pct", 0.20),
        impulse_min_body_atr_pct=zones_cfg.get("impulse_min_body_atr_pct", 0.15),
        max_zone_age_bars=zones_cfg.get("max_zone_age_bars", 200),
        max_zone_touches=zones_cfg.get("max_zone_touches", 3),
        zone_buffer_atr_pct=zones_cfg.get("zone_buffer_atr_pct", 0.02),
    )

    oz_cfg = feat_cfg.get("order_zone", {})
    oz = OrderZoneEngine(
        weights=oz_cfg.get("weights"),
        min_confluence_score=oz_cfg.get("min_confluence_score", 0.35),
        min_rr_ratio=risk_cfg.get("take_profit", {}).get("min_rr_ratio", 2.5),
    )

    import pandas as pd
    trading_days = [
        d for d in dl.get_trading_days()
        if pd.Timestamp(d).weekday() < 5 and atr.get_atr_for_date(d) is not None
    ]

    export_features(dl, atr, zd, oz, trading_days, out_path=args.out, max_days=args.days)


if __name__ == "__main__":
    _cli()
