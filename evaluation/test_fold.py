"""
evaluation/test_fold.py
========================
Standalone test-fold runner.

Loads every checkpoint saved during a training fold, runs deterministic
episodes on the test date range, prints a ranked summary table, and
generates a per-checkpoint Plotly HTML with:
  • Candlestick chart — entry ▲/▼ markers, coloured by win/loss
  • SL and TP horizontal lines per trade
  • Equity-curve subplot

Usage
-----
    python -m evaluation.test_fold \\
        --models-dir  logs/walk_forward/fold_00/models \\
        --config      config/ \\
        --data        data/ \\
        [--test-start 2024-01-01] \\
        [--test-end   2024-06-30] \\
        [--out-dir    logs/walk_forward/fold_00/test_results]

If --test-start / --test-end are omitted the script uses the DataSplitter
default test split (the held-out block after val).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Plotly (optional but expected) ───────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_base(p: Path) -> str:
    """Return the base name of a checkpoint without the .zip extension."""
    name = p.name
    return name[:-4] if name.endswith(".zip") else name


def _find_checkpoints(models_dir: Path) -> List[Path]:
    """
    Return all checkpoint files sorted by composite score then step.

    Handles both correctly-saved files (checkpoint_*.zip) and legacy saves
    where SB3 stripped the numeric score suffix and saved without .zip.
    """
    zips = sorted(models_dir.glob("checkpoint_*.zip"))
    if zips:
        return zips
    # Legacy: SB3 treated the score decimal (e.g. .57) as the file extension
    # and saved without .zip — match any checkpoint_* that is not a .pkl or dir
    candidates = sorted(
        f for f in models_dir.glob("checkpoint_*")
        if not f.name.endswith(".pkl") and not f.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files found in {models_dir}"
        )
    return candidates


def _vecnorm_path(ckpt: Path) -> Optional[Path]:
    """Derive paired VecNormalize .pkl path from checkpoint path."""
    base = _checkpoint_base(ckpt)   # strip .zip if present
    candidate = ckpt.parent / f"{base}_vecnormalize.pkl"
    if candidate.exists():
        return candidate
    # Fallback: best_model_vecnormalize.pkl or vecnormalize.pkl in same dir
    for name in ("best_model_vecnormalize.pkl", "vecnormalize.pkl"):
        fb = ckpt.parent / name
        if fb.exists():
            return fb
    return None


def _composite_from_name(p: Path) -> float:
    """Parse composite score from checkpoint filename, or 0 if FINAL_STEP."""
    base = _checkpoint_base(p)
    if "_c" in base:
        try:
            return float(base.split("_c")[-1])
        except ValueError:
            pass
    return 0.0


def _step_from_name(p: Path) -> int:
    """Parse training step from checkpoint filename."""
    base = _checkpoint_base(p)
    for token in base.split("_"):
        if token.startswith("step"):
            try:
                return int(token[4:])
            except ValueError:
                pass
    if "FINAL" in base:
        part = base.split("FINAL_STEP")[-1]
        try:
            return int(part)
        except ValueError:
            pass
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────

def _run_checkpoint(
    ckpt_path: Path,
    vn_path: Optional[Path],
    env_factory,
    n_episodes: int = 20,
) -> Tuple[List[dict], List[dict]]:
    """
    Load checkpoint + VecNormalize, run n_episodes deterministically.

    Returns
    -------
    trades   : list of trade dicts (raw, from _episode_summary)
    episodes : list of episode-level summary dicts
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sb3_contrib import RecurrentPPO

    # Build a fresh single-env VecNormalize
    vec_env = DummyVecEnv([env_factory])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                           clip_obs=10.0, training=False)

    if vn_path is not None:
        # Load running stats (mean/var) into the wrapper
        vec_env = VecNormalize.load(str(vn_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = RecurrentPPO.load(str(ckpt_path), env=vec_env)

    all_trades: List[dict] = []
    all_episodes: List[dict] = []

    for _ in range(n_episodes):
        obs = vec_env.reset()
        done = False
        lstm_states = None
        episode_start = np.ones((vec_env.num_envs,), dtype=bool)

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            obs, _reward, dones, infos = vec_env.step(action)
            episode_start = dones
            done = bool(dones[0])
            if done and infos[0]:
                info = infos[0]
                all_episodes.append(info)
                all_trades.extend(info.get("trades_list", []))

    vec_env.close()
    return all_trades, all_episodes


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(trades: List[dict], episodes: List[dict]) -> dict:
    """Compute summary metrics from raw trade list."""
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0.0, "total_pnl_r": 0.0,
            "avg_win_r": 0.0, "avg_loss_r": 0.0, "profit_factor": 0.0,
            "sharpe": 0.0, "max_dd_r": 0.0, "avg_duration_min": 0.0,
        }

    pnl = [t["pnl_r"] for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    n = len(pnl)

    win_rate = len(wins) / n
    total_pnl = sum(pnl)
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(abs(np.mean(losses))) if losses else 0.0
    gp = sum(wins) if wins else 0.0
    gl = abs(sum(losses)) if losses else 1e-9
    pf = gp / gl

    # Sharpe: episode-level daily PnL std
    ep_pnls = [ep.get("total_pnl_r", 0.0) for ep in episodes if ep]
    sharpe = 0.0
    if len(ep_pnls) >= 2:
        mu = float(np.mean(ep_pnls))
        sd = float(np.std(ep_pnls))
        sharpe = (mu / sd) * np.sqrt(252) if sd > 1e-9 else 0.0  # annualised

    # Max drawdown
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max()) if len(equity) > 0 else 0.0

    avg_dur = float(np.mean([t.get("duration_min", 0) for t in trades]))

    return {
        "n_trades":       n,
        "win_rate":       round(win_rate, 4),
        "total_pnl_r":    round(total_pnl, 4),
        "avg_win_r":      round(avg_win,   4),
        "avg_loss_r":     round(avg_loss,  4),
        "profit_factor":  round(pf,        4),
        "sharpe":         round(sharpe,    4),
        "max_dd_r":       round(max_dd,    4),
        "avg_duration_min": round(avg_dur, 1),
    }


def _composite(m: dict,
               w_sharpe: float = 0.30,
               w_pnl: float = 0.25,
               w_wl: float = 0.25,
               w_dd: float = 0.20) -> float:
    """Replicate training composite score for ranking."""
    sharpe  = max(m["sharpe"],       0.0)
    pnl     = max(m["total_pnl_r"],  0.0)
    wl      = m["win_rate"] * (m["avg_win_r"] / max(m["avg_loss_r"], 1e-6))
    dd      = m["max_dd_r"]

    ref_sharpe = 1.5   # annualised Sharpe (*sqrt(252))
    ref_pnl    = 20.0
    ref_wl     = 3.0
    ref_dd     = 5.0

    sn = min(sharpe / ref_sharpe, 1.0)
    pn = min(pnl    / ref_pnl,    1.0)
    wn = min(wl     / ref_wl,     1.0)
    dn = min(dd     / ref_dd,     1.0)

    total_w = w_sharpe + w_pnl + w_wl + w_dd
    return (w_sharpe * sn + w_pnl * pn + w_wl * wn + w_dd * (1.0 - dn)) / total_w


# ─────────────────────────────────────────────────────────────────────────────
# Console tables
# ─────────────────────────────────────────────────────────────────────────────

_COL_W = {
    "ckpt":      28,
    "trades":     6,
    "wr":         7,
    "pnl":        8,
    "pf":         6,
    "sharpe":     8,
    "dd":         7,
    "dur":        7,
    "comp":       7,
}

def _header_row() -> str:
    return (
        f"{'Checkpoint':<{_COL_W['ckpt']}}"
        f"{'Trades':>{_COL_W['trades']}}"
        f"{'WR%':>{_COL_W['wr']}}"
        f"{'PnL(R)':>{_COL_W['pnl']}}"
        f"{'PF':>{_COL_W['pf']}}"
        f"{'Sharpe':>{_COL_W['sharpe']}}"
        f"{'MaxDD':>{_COL_W['dd']}}"
        f"{'AvgMin':>{_COL_W['dur']}}"
        f"{'Score':>{_COL_W['comp']}}"
    )


def _data_row(name: str, m: dict, score: float) -> str:
    short = name[:_COL_W["ckpt"] - 1] if len(name) > _COL_W["ckpt"] - 1 else name
    return (
        f"{short:<{_COL_W['ckpt']}}"
        f"{m['n_trades']:>{_COL_W['trades']}}"
        f"{m['win_rate']*100:>{_COL_W['wr']}.1f}"
        f"{m['total_pnl_r']:>{_COL_W['pnl']}.2f}"
        f"{m['profit_factor']:>{_COL_W['pf']}.2f}"
        f"{m['sharpe']:>{_COL_W['sharpe']}.3f}"
        f"{m['max_dd_r']:>{_COL_W['dd']}.2f}"
        f"{m['avg_duration_min']:>{_COL_W['dur']}.1f}"
        f"{score:>{_COL_W['comp']}.3f}"
    )


def _separator() -> str:
    return "-" * sum(_COL_W.values())


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart per checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _build_chart(
    ckpt_name: str,
    trades: List[dict],
    data_loader,
    out_path: Path,
) -> None:
    """
    Build a Plotly HTML with:
      Upper panel: candlestick + trade entry/exit markers + SL/TP lines
      Lower panel: equity curve
    """
    if not PLOTLY_OK:
        return
    if not trades:
        return

    # ── Gather OHLCV bars for the date range ─────────────────────────────────
    dates = sorted({t["date"] for t in trades})
    if not dates:
        return

    try:
        bars_list = []
        for d in dates:
            day_bars = data_loader.get_bars_for_day(d)
            if day_bars is not None and not day_bars.empty:
                bars_list.append(day_bars)
        if not bars_list:
            return
        bars = pd.concat(bars_list).sort_index()
    except Exception:
        return

    # ── Reset index for Plotly ────────────────────────────────────────────────
    bars = bars.reset_index()
    time_col = bars.columns[0]   # typically "datetime" or "timestamp"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.04,
        subplot_titles=[f"Trades — {ckpt_name}", "Equity Curve (R)"],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=bars[time_col],
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ── Trade markers + SL/TP lines ──────────────────────────────────────────
    equity = 0.0
    eq_times: List[Any] = []
    eq_vals:  List[float] = []

    for t in trades:
        is_long  = t["direction"].upper() == "LONG"
        is_win   = t["is_win"]
        entry_px = t["entry_price"]
        stop_px  = t["stop_price"]
        tgt_px   = t["initial_target"]
        exit_px  = t["exit_price"]
        pnl_r    = t["pnl_r"]

        # Entry marker
        marker_sym  = "triangle-up" if is_long else "triangle-down"
        marker_col  = "#2196F3"   # blue = entry
        entry_size  = 12

        # Find approx time for entry_bar_idx
        entry_idx   = t.get("entry_bar_idx", 0)
        exit_idx    = t.get("exit_bar_idx",  0)

        entry_time  = bars[time_col].iloc[min(entry_idx, len(bars) - 1)]
        exit_time   = bars[time_col].iloc[min(exit_idx,  len(bars) - 1)]

        exit_col = "#26a69a" if is_win else "#ef5350"

        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[entry_time], y=[entry_px],
                mode="markers",
                marker=dict(symbol=marker_sym, color=marker_col,
                            size=entry_size, line=dict(width=1, color="white")),
                name="Entry",
                showlegend=False,
                hovertemplate=(
                    f"<b>{'LONG' if is_long else 'SHORT'} Entry</b><br>"
                    f"Price: {entry_px}<br>"
                    f"SL: {stop_px}<br>"
                    f"TP: {tgt_px}<br>"
                    f"PnL: {pnl_r:+.2f}R<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

        # Exit marker
        fig.add_trace(
            go.Scatter(
                x=[exit_time], y=[exit_px],
                mode="markers",
                marker=dict(symbol="x", color=exit_col,
                            size=10, line=dict(width=2, color=exit_col)),
                name="Exit",
                showlegend=False,
                hovertemplate=(
                    f"<b>Exit ({t.get('exit_reason','')})</b><br>"
                    f"Price: {exit_px}<br>"
                    f"PnL: {pnl_r:+.2f}R<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

        # SL line (dashed red)
        fig.add_shape(
            type="line",
            x0=entry_time, x1=exit_time,
            y0=stop_px,    y1=stop_px,
            line=dict(color="rgba(239,83,80,0.6)", width=1, dash="dot"),
            row=1, col=1,
        )
        # TP line (dashed green)
        fig.add_shape(
            type="line",
            x0=entry_time, x1=exit_time,
            y0=tgt_px,     y1=tgt_px,
            line=dict(color="rgba(38,166,154,0.6)", width=1, dash="dot"),
            row=1, col=1,
        )

        # Equity
        equity += pnl_r
        eq_times.append(exit_time)
        eq_vals.append(round(equity, 4))

    # ── Equity subplot ────────────────────────────────────────────────────────
    eq_color = "#26a69a" if equity >= 0 else "#ef5350"
    fig.add_trace(
        go.Scatter(
            x=eq_times, y=eq_vals,
            mode="lines+markers",
            line=dict(color=eq_color, width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor=(
                "rgba(38,166,154,0.15)" if equity >= 0
                else "rgba(239,83,80,0.15)"
            ),
            name="Equity (R)",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Zero line on equity panel
    fig.add_hline(y=0, line=dict(color="white", width=1, dash="dash"),
                  row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"Test Fold — {ckpt_name}", font=dict(size=14)),
        height=800,
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", y=1.02),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity (R)", row=2, col=1)

    fig.write_html(str(out_path))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run test-fold evaluation across all saved checkpoints."
    )
    parser.add_argument(
        "--models-dir", required=True,
        help="Path to folder containing checkpoint_*.zip files",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to config/ directory (YAML files)",
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to data/ directory",
    )
    parser.add_argument(
        "--test-start", default=None,
        help="Override test period start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--test-end", default=None,
        help="Override test period end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=20,
        help="Episodes per checkpoint (default 20)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to write HTML charts and results txt (default: models-dir/test_results)",
    )
    args = parser.parse_args(argv)

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir) if args.out_dir else models_dir.parent / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build environment components ─────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    # Import here to avoid circular imports at module level
    import yaml
    from utils.logger import configure_logging, get_logger as _get_logger
    configure_logging()
    _log = _get_logger(__name__)

    def _load_configs(config_dir: str) -> dict:
        config_dir = Path(config_dir)
        file_map = {
            "agent":       "agent_config.yaml",
            "environment": "environment_config.yaml",
            "features":    "features_config.yaml",
            "risk":        "risk_config.yaml",
            "reward":      "reward_config.yaml",
            "logging":     "logging_config.yaml",
        }
        cfgs = {}
        for key, fname in file_map.items():
            p = config_dir / fname
            cfgs[key] = yaml.safe_load(p.read_text()) if p.exists() else {}
        return cfgs

    configs = _load_configs(args.config)
    env_cfg     = configs["environment"]
    feat_cfg    = configs["features"]
    risk_cfg    = configs["risk"]
    reward_cfg  = configs["reward"]
    agent_cfg   = configs["agent"]

    from data.data_loader     import DataLoader
    from data.data_splitter   import DataSplitter
    from environment.action_space       import ActionMasker
    from environment.position_manager   import PositionManager
    from environment.reward_calculator  import RewardCalculator
    from environment.trading_env        import TradingEnv
    from features.atr_calculator        import ATRCalculator
    from features.observation_builder   import ObservationBuilder
    from features.order_zone_engine     import OrderZoneEngine
    from features.trend_classifier      import TrendClassifier
    from features.zone_detector         import ZoneDetector

    instrument  = env_cfg.get("instruments", {}).get("default", "NQ")
    session_cfg = env_cfg.get("session", {})
    bar_minutes = int(session_cfg.get("bar_timeframe_minutes", 5))

    data_loader = DataLoader(
        data_dir=args.data,
        instrument=instrument,
        intraday_tf=f"{bar_minutes}min",
        daily_tf=session_cfg.get("daily_timeframe", "1D"),
        tz=session_cfg.get("timezone", "America/New_York"),
    )
    data_loader.load()

    all_days = data_loader.get_trading_days()
    atr_cfg  = feat_cfg.get("atr", {})
    atr_calc = ATRCalculator(
        atr_period=atr_cfg.get("period", 14),
        exhaustion_threshold=atr_cfg.get("exhaustion_threshold", 0.95),
        warning_threshold=atr_cfg.get("danger_threshold", 0.85),
    )
    atr_calc.fit(data_loader.daily)

    import pandas as _pd
    trading_days = [
        d for d in all_days
        if _pd.Timestamp(d).weekday() < 5
        and atr_calc.get_atr_for_date(d) is not None
    ]

    # Determine test days
    if args.test_start or args.test_end:
        ts = args.test_start or "1970-01-01"
        te = args.test_end   or "2099-12-31"
        test_days = [d for d in trading_days if ts <= str(d) <= te]
    else:
        split = DataSplitter.split_by_counts(trading_days, n_train=252, n_val=26)
        test_days = split.test

    if not test_days:
        print("ERROR: No test days found — check --test-start / --test-end or data range.")
        sys.exit(1)

    _log.info("Test period", days=len(test_days),
              start=str(test_days[0]), end=str(test_days[-1]))

    # ── Build component factories ─────────────────────────────────────────────
    zones_cfg   = feat_cfg.get("zones", {})
    swing_cfg   = feat_cfg.get("swing", {})
    trend_cfg   = feat_cfg.get("trend", {})
    oz_cfg      = feat_cfg.get("order_zone", {})
    obs_cfg     = env_cfg.get("observation", {})
    atr_gate    = risk_cfg.get("atr_gate", {})
    daily_lim   = risk_cfg.get("daily_limits", {})
    session_risk = risk_cfg.get("session", {})
    account_cfg = env_cfg.get("account", {})
    sizing_cfg  = risk_cfg.get("sizing", {})
    trail_cfg   = risk_cfg.get("trailing", {})
    contracts_cfg = env_cfg.get("contracts", {}).get(instrument, {})

    real_capital = float(account_cfg.get("initial_balance", 2500))
    point_value  = float(contracts_cfg.get("micro_point_value", 2.0))

    trend_classifier = TrendClassifier(
        swing_lookback=swing_cfg.get("lookback_bars", 5),
        min_hh_hl_for_uptrend=trend_cfg.get("min_hh_hl_for_uptrend", 2),
        min_ll_lh_for_downtrend=trend_cfg.get("min_ll_lh_for_downtrend", 2),
        reversal_requires_breaks=trend_cfg.get("reversal_requires_breaks", 2),
        strength_lookback_bars=trend_cfg.get("strength_lookback_bars", 40),
    )
    observation_builder = ObservationBuilder(
        clip_value=obs_cfg.get("clip_observations", 10.0),
        normalize_observations=obs_cfg.get("normalize_observations", True),
        lookback_bars=obs_cfg.get("lookback_bars", 20),
    )
    order_zone_engine = OrderZoneEngine(
        weights=oz_cfg.get("weights"),
        min_confluence_score=oz_cfg.get("min_confluence_score", 0.60),
        min_rr_ratio=risk_cfg.get("take_profit", {}).get("min_rr_ratio", 4.0),
        pin_bar_wick_ratio=oz_cfg.get("rejection", {}).get("pin_bar_wick_ratio", 2.0),
        engulfing_body_ratio=oz_cfg.get("rejection", {}).get("engulfing_body_ratio", 1.1),
    )
    action_masker = ActionMasker(
        min_rr_ratio=risk_cfg.get("take_profit", {}).get("min_rr_ratio", 4.0),
        atr_exhaustion_threshold=atr_gate.get("block_entries_above_pct", 0.95),
        trail_min_r=trail_cfg.get("activate_at_r", 2.0),
        max_trades_per_day=daily_lim.get("max_trades_per_day", 5),
        no_entry_last_n_bars=session_risk.get("no_entry_last_n_bars", 3),
    )
    reward_calculator = RewardCalculator.from_config(reward_cfg)

    zone_detector_defaults = {
        "consolidation_min_bars":       2,
        "consolidation_max_bars":       8,
        "consolidation_range_atr_pct":  0.20,
        "impulse_min_body_atr_pct":     0.15,
        "max_zone_age_bars":            200,
        "max_zone_touches":             3,
        "zone_buffer_atr_pct":          0.02,
    }

    def make_position_manager():
        return PositionManager(
            real_capital=real_capital,
            risk_per_trade_pct=sizing_cfg.get("risk_per_trade_pct", 0.01),
            min_contracts=sizing_cfg.get("min_contracts", 1),
            max_contracts=sizing_cfg.get("max_contracts", 3),
            point_value=point_value,
            max_trades_per_day=daily_lim.get("max_trades_per_day", 5),
            trail_activate_r=trail_cfg.get("activate_at_r", 2.0),
            trail_aggressive_r=trail_cfg.get("trail_aggressively_at_r", 4.0),
            trail_lock_in_r=trail_cfg.get("lock_in_r_at_trail", 2.0),
            max_daily_loss_r=daily_lim.get("max_daily_loss_r", 3.0),
            max_drawdown_r=5.0,
            pause_bars_after_loss_streak=daily_lim.get("pause_bars_after_loss_streak", 6),
            loss_streak_threshold=daily_lim.get("max_consecutive_losses_before_pause", 3),
            zone_buffer_atr_pct=risk_cfg.get("stop_loss", {}).get("zone_buffer_atr_pct", 0.03),
        )

    session_start = session_cfg.get("rth_start_utc", "08:30")
    session_end   = session_cfg.get("rth_end_utc",   "15:00")
    session_type  = session_cfg.get("session_type", "RTH").upper()

    def env_factory():
        return TradingEnv(
            data_loader=data_loader,
            trading_days=test_days,
            position_manager=make_position_manager(),
            reward_calculator=reward_calculator,
            observation_builder=observation_builder,
            atr_calculator=atr_calc,
            zone_detector=ZoneDetector(**{
                k: zones_cfg.get(k, v)
                for k, v in zone_detector_defaults.items()
            }),
            trend_classifier=trend_classifier,
            order_zone_engine=order_zone_engine,
            action_masker=action_masker,
            rth_start=session_start,
            rth_end=session_end,
            no_entry_last_n_bars=session_risk.get("no_entry_last_n_bars", 3),
            early_terminate_on_max_dd=env_cfg.get("episode", {}).get(
                "early_termination_on_max_drawdown", True),
            point_value=point_value,
            bar_minutes=bar_minutes,
            curriculum_filter_fn=None,
            augmentor=None,
            session_type=session_type,
            random_start=False,
            seed=agent_cfg.get("seed", 42) + 999,
            zone_lookback_bars=feat_cfg.get("zone_lookback_bars", 500),
        )

    # ── Find and evaluate checkpoints ────────────────────────────────────────
    checkpoints = _find_checkpoints(models_dir)
    print(f"\nFound {len(checkpoints)} checkpoint(s) in {models_dir}\n")

    results: List[Dict] = []
    sep = _separator()

    print(sep)
    print(_header_row())
    print(sep)

    for ckpt in checkpoints:
        name = ckpt.stem
        vn   = _vecnorm_path(ckpt)

        print(f"  → Evaluating {name} ...", end="", flush=True)
        try:
            trades, episodes = _run_checkpoint(
                ckpt, vn, env_factory, n_episodes=args.n_episodes
            )
        except Exception as exc:
            print(f" FAILED: {exc}")
            continue

        m     = _compute_metrics(trades, episodes)
        score = _composite(m)

        print(f"\r{_data_row(name, m, score)}")

        results.append({
            "name":      name,
            "ckpt_path": ckpt,
            "metrics":   m,
            "score":     score,
            "trades":    trades,
            "episodes":  episodes,
        })

        # Per-checkpoint HTML chart
        if PLOTLY_OK and trades:
            chart_path = out_dir / f"{name}_test_chart.html"
            try:
                _build_chart(name, trades, data_loader, chart_path)
            except Exception as exc:
                _log.warning("Chart generation failed", ckpt=name, error=str(exc))

    print(sep)

    if not results:
        print("No results to rank.")
        return

    # ── Ranked summary (ascending score — best last) ──────────────────────────
    results.sort(key=lambda r: r["score"])

    print("\n── Ranked Summary (best last) ──────────────────────────────────────────\n")
    print(sep)
    print(_header_row())
    print(sep)
    for r in results:
        print(_data_row(r["name"], r["metrics"], r["score"]))
    print(sep)

    best = results[-1]
    print(f"\n  Best checkpoint: {best['name']}")
    print(f"  Score={best['score']:.3f}  "
          f"Trades={best['metrics']['n_trades']}  "
          f"WR={best['metrics']['win_rate']*100:.1f}%  "
          f"PnL={best['metrics']['total_pnl_r']:+.2f}R  "
          f"PF={best['metrics']['profit_factor']:.2f}")

    # ── Write text results ────────────────────────────────────────────────────
    txt_path = out_dir / "test_fold_results.txt"
    lines: List[str] = [
        f"Test Fold Results",
        f"Models dir : {models_dir}",
        f"Test period: {test_days[0]} → {test_days[-1]}  ({len(test_days)} days)",
        f"Episodes   : {args.n_episodes} per checkpoint",
        "",
        sep,
        _header_row(),
        sep,
    ]
    for r in results:
        lines.append(_data_row(r["name"], r["metrics"], r["score"]))
    lines += [sep, "", f"Best: {best['name']}  score={best['score']:.4f}"]
    txt_path.write_text("\n".join(lines))
    print(f"\n  Results saved → {txt_path}")

    if PLOTLY_OK:
        charts = list(out_dir.glob("*_test_chart.html"))
        print(f"  Charts saved  → {out_dir} ({len(charts)} file(s))")
    else:
        print("  (Plotly not installed — no charts generated)")


if __name__ == "__main__":
    main()
