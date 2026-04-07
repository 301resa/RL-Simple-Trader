"""
training/metrics_logger_callback.py
=====================================
SB3 callback that prints per-environment training tables and RL diagnostics
to the console (and mirrored to logs/metrics.log).

Output format (after every rollout):

  ######################################################################
  # STEP: 65,536 | ALL 16 ENVS | TRAIN: 2024-10-16→2025-10-03
  ######################################################################
  +------+------+-------+---------+-------+-------+------+ ... +
  | ENV  | Tr   | WR%   |   PnL   |  PF   |  Sh   | DD%  | ... |
  +------+------+-------+---------+-------+-------+------+ ... +
  | 00   |   14 | 64.3% |  +4,478 |  5.34 |  7.98 | 1.5% | ... |
  ...
  | AVG  |   .. | ..%   |  +....  |  ..   |  ..   | ..%  | ... |
  +------+------+-------+---------+-------+-------+------+ ... +
  ┌─ TRAINING ASSESSMENT ────────────────────────────────────────────
  │  Entropy          : 0.0391  [EXPLORING — high randomness, early phase]
  │  Explained Var.   : 0.2327  [LOW — critic not yet explaining returns]
  └──────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# ── Column definitions ────────────────────────────────────────────────────────

_COLS = [
    ("ENV",   4,  "l"),
    ("Tr",    5,  "r"),
    ("WR%",   6,  "r"),
    ("PnL",   8,  "r"),
    ("PF",    6,  "r"),
    ("Sh",    6,  "r"),
    ("DD%",   5,  "r"),
    ("R-Tr",  5,  "r"),
    ("R-WR%", 6,  "r"),
    ("E-Tr",  5,  "r"),
    ("E-WR%", 6,  "r"),
    ("MaxW",  8,  "r"),
    ("MaxL",  8,  "r"),
    ("AvgW",  8,  "r"),
    ("AvgL",  8,  "r"),
    ("RR",    5,  "r"),
    ("Dur",   5,  "r"),
    ("Min",   5,  "r"),
    ("Max",   5,  "r"),
]

_SEP  = "+"
_PIPE = "|"


def _divider() -> str:
    parts = [f"{_SEP}{'-' * (w + 2)}" for _, w, _ in _COLS]
    return "".join(parts) + _SEP


def _header() -> str:
    parts = []
    for name, width, align in _COLS:
        if align == "r":
            parts.append(f"{_PIPE} {name:>{width}} ")
        else:
            parts.append(f"{_PIPE} {name:<{width}} ")
    return "".join(parts) + _PIPE


def _fmt_pnl(v: float) -> str:
    if v >= 0:
        return f"+{v:,.0f}"
    return f"{v:,.0f}"


def _fmt_dollars(v: float) -> str:
    if v == 0:
        return "    +0"
    if v >= 0:
        return f"+{v:,.0f}"
    return f"{v:,.0f}"


def _fmt_pf(pf: float) -> str:
    """Cap profit factor display at 99.99 to prevent column overflow."""
    if pf >= 100.0:
        return "99.99"
    return f"{pf:.2f}"


def _row(env_idx: int | str, info: dict) -> str:
    n_trades   = int(round(info.get("n_trades", 0)))
    win_rate   = info.get("win_rate", 0.0)
    pnl_d      = info.get("total_pnl_dollars", 0.0)
    pf         = info.get("profit_factor", 0.0)
    sh         = info.get("sharpe_ratio", 0.0)
    dd_pct     = info.get("max_drawdown_pct", 0.0)
    rth_tr     = int(round(info.get("rth_trades", 0)))
    rth_wr     = info.get("rth_wins", 0) / max(rth_tr, 1)
    eth_tr     = int(round(info.get("eth_trades", 0)))
    eth_wr     = info.get("eth_wins", 0) / max(eth_tr, 1)
    max_w      = info.get("max_win_dollars", 0.0)
    max_l      = info.get("max_loss_dollars", 0.0)
    avg_w      = info.get("avg_win_dollars", 0.0)
    avg_l      = info.get("avg_loss_dollars", 0.0)
    rr         = info.get("avg_rr", 0.0)
    dur        = info.get("avg_duration_minutes", 0.0)
    dur_min    = int(round(info.get("min_duration_minutes", 0)))
    dur_max    = int(round(info.get("max_duration_minutes", 0)))

    env_str  = f"{env_idx:>4}" if isinstance(env_idx, int) else f"{env_idx:>4}"

    cells = [
        (env_str,              4,  "l"),
        (f"{n_trades}",        5,  "r"),
        (f"{win_rate*100:.1f}%",6,  "r"),
        (_fmt_pnl(pnl_d),      8,  "r"),
        (_fmt_pf(pf),          6,  "r"),
        (f"{sh:.2f}",          6,  "r"),
        (f"{dd_pct:.1f}%",     5,  "r"),
        (f"{rth_tr}",          5,  "r"),
        (f"{rth_wr*100:.1f}%", 6,  "r"),
        (f"{eth_tr}",          5,  "r"),
        (f"{eth_wr*100:.1f}%", 6,  "r"),
        (_fmt_dollars(max_w),  8,  "r"),
        (_fmt_dollars(max_l),  8,  "r"),
        (_fmt_dollars(avg_w),  8,  "r"),
        (_fmt_dollars(avg_l),  8,  "r"),
        (f"{rr:.2f}",          5,  "r"),
        (f"{dur:.0f}m",        5,  "r"),
        (f"{dur_min}m",        5,  "r"),
        (f"{dur_max}m",        5,  "r"),
    ]

    parts = []
    for val, width, align in cells:
        if align == "r":
            parts.append(f"{_PIPE} {val:>{width}} ")
        else:
            parts.append(f"{_PIPE} {val:<{width}} ")
    return "".join(parts) + _PIPE


def _avg_row(infos_list: List[dict]) -> str:
    """Compute aggregate averages across all envs that have data."""
    if not infos_list:
        return _row("AVG", {})

    def _mean(key):
        vals = [d.get(key, 0.0) for d in infos_list]
        return float(np.mean(vals)) if vals else 0.0

    def _sum(key):
        return sum(d.get(key, 0) for d in infos_list)

    n_envs = len(infos_list)
    agg = {
        "n_trades":             _mean("n_trades"),
        "win_rate":             _mean("win_rate"),
        "total_pnl_dollars":    _mean("total_pnl_dollars"),
        "profit_factor":        _mean("profit_factor"),
        "sharpe_ratio":         _mean("sharpe_ratio"),
        "max_drawdown_pct":     _mean("max_drawdown_pct"),
        "rth_trades":           _mean("rth_trades"),
        "rth_wins":             _mean("rth_wins"),
        "eth_trades":           _mean("eth_trades"),
        "eth_wins":             _mean("eth_wins"),
        "max_win_dollars":      _mean("max_win_dollars"),
        "max_loss_dollars":     _mean("max_loss_dollars"),
        "avg_win_dollars":      _mean("avg_win_dollars"),
        "avg_loss_dollars":     _mean("avg_loss_dollars"),
        "avg_rr":               _mean("avg_rr"),
        "avg_duration_minutes": _mean("avg_duration_minutes"),
        "min_duration_minutes": _mean("min_duration_minutes"),
        "max_duration_minutes": _mean("max_duration_minutes"),
    }
    return _row("AVG", agg)


def _entropy_label(entropy: float) -> str:
    if entropy >= 1.2:
        return "EXPLORING  — high randomness, early phase"
    if entropy >= 0.5:
        return "LEARNING SELECTIVITY — narrowing action choices"
    if entropy >= 0.1:
        return "CONVERGING — exploitation increasing"
    return "MAY BE STUCK — very low exploration, check rewards"


def _ev_label(ev: float) -> str:
    if ev >= 0.8:
        return "EXCELLENT — critic explains variance well"
    if ev >= 0.5:
        return "GOOD — critic learning well"
    if ev >= 0.0:
        return "DEVELOPING — normal in early training"
    return "NEGATIVE — critic worse than baseline, check lr/arch"


def _print_lines(lines: List[str]) -> None:
    """Write lines to stdout and optionally to a log file."""
    try:
        from utils.metrics_printer import _get_log_file
        fh = _get_log_file()
    except Exception:
        fh = None

    for line in lines:
        print(line)
        if fh:
            try:
                fh.write(line + "\n")
                fh.flush()
            except Exception:
                pass


# ── Callback ──────────────────────────────────────────────────────────────────

class MetricsPrinterCallback(BaseCallback):
    """
    Tracks the most recent completed episode per training environment and
    prints a per-env summary table after every N rollouts.

    Parameters
    ----------
    rl_diag_every : int
        Print after every N rollouts (default 1 = every rollout).
    print_train_episodes : bool
        If False, skip individual episode prints (table still shows).
    train_date_range : str
        Shown in the header (e.g. "2024-10-16→2025-10-03").
    """

    def __init__(
        self,
        print_train_episodes: bool = True,
        rl_diag_every: int = 1,
        train_date_range: str = "",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.print_train_episodes = print_train_episodes
        self.rl_diag_every = max(1, rl_diag_every)
        self.train_date_range = train_date_range
        self._rollout_count  = 0

        # latest episode info per env index
        self._env_latest: Dict[int, dict] = {}

    # ── Per step: capture latest episode per env ──────────────────────────────

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, (info, done) in enumerate(zip(infos, dones)):
            if done and "n_trades" in info:
                self._env_latest[i] = dict(info)

        return True

    # ── After rollout: print the table ───────────────────────────────────────

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.rl_diag_every != 0:
            return

        n_envs = getattr(self.training_env, "num_envs", 1)
        step   = self.num_timesteps

        entropy, ev = self._read_rl_diagnostics()

        lines: List[str] = []

        # ── Banner ────────────────────────────────────────────
        banner_mid = (
            f" STEP: {step:>10,} | ALL {n_envs} ENVS"
            + (f" | TRAIN: {self.train_date_range}" if self.train_date_range else "")
        )
        lines.append("")
        lines.append("#" * 70)
        lines.append(f"#{banner_mid}")
        lines.append("#" * 70)

        # ── Entropy line (inline) ─────────────────────────────
        if entropy is not None:
            ent_pct = entropy / 1.6094 * 100  # 1.6094 = ln(5)
            lines.append(
                f"  ENT: {entropy:.4f}  ({ent_pct:.0f}% of max 1.61 nats)"
            )

        # ── Per-env table ─────────────────────────────────────
        div = _divider()
        lines.append("  " + div)
        lines.append("  " + _header())
        lines.append("  " + div)

        env_infos: List[dict] = []
        for i in range(n_envs):
            info = self._env_latest.get(i, {})
            env_infos.append(info)
            lines.append(f"  {_row(i, info)}")

        lines.append("  " + div)
        lines.append(f"  {_row('AVG', _avg_info(env_infos))}")
        lines.append("  " + div)

        # ── RL Assessment ─────────────────────────────────────
        lines.append("  ┌─ TRAINING ASSESSMENT " + "─" * 46)
        if entropy is not None:
            bar = _progress_bar(entropy / 1.6094)
            lines.append(f"  │  Entropy          : {entropy:.4f}  [{_entropy_label(entropy)}]")
            lines.append(f"  │                    {bar}")
        else:
            lines.append("  │  Entropy          : (awaiting first update)")
        if ev is not None:
            lines.append(f"  │  Explained Var.   : {ev:.4f}  [{_ev_label(ev)}]")
        else:
            lines.append("  │  Explained Var.   : (awaiting first update)")
        lines.append("  └" + "─" * 68)

        _print_lines(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _read_rl_diagnostics(self):
        try:
            ntv = self.logger.name_to_value
            el  = ntv.get("train/entropy_loss")
            ev  = ntv.get("train/explained_variance")
            if el is None or ev is None:
                return None, None
            return -float(el), float(ev)
        except Exception:
            return None, None


_INT_KEYS = {
    "n_trades", "n_wins", "n_losses",
    "rth_trades", "rth_wins", "eth_trades", "eth_wins",
    "min_duration_minutes", "max_duration_minutes",
}


def _avg_info(infos: List[dict]) -> dict:
    """Mean of all numeric keys across non-empty env infos."""
    filled = [d for d in infos if d]
    if not filled:
        return {}

    result = {}
    all_keys = set().union(*[d.keys() for d in filled])
    for k in all_keys:
        vals = [d[k] for d in filled if k in d and isinstance(d[k], (int, float))]
        if vals:
            mean_val = float(np.mean(vals))
            result[k] = int(round(mean_val)) if k in _INT_KEYS else mean_val
    return result


def _progress_bar(fraction: float, width: int = 20) -> str:
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"
