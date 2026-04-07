"""
utils/metrics_printer.py
=========================
Pretty-prints trading metrics and RL diagnostics as Rich tables.

Two table types:
  1. Trading table  — P&L, win rate, Sharpe, drawdown, etc.
                      used for training episodes, validation runs, and test.
  2. RL diagnostics — Entropy and Explained Variance with plain-English
                      guidance on what each value means.

All output goes to the console AND to the session log file simultaneously
via a shared Console object created once at startup with `init_console()`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

# ── Rich imports ──────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ── Module-level shared console ───────────────────────────────────────────────
_console: Optional["Console"] = None
_log_file = None   # open file handle for plain-text fallback


def init_console(log_path: Optional[str] = None) -> None:
    """
    Initialise the shared console.  Call once at training startup.

    Parameters
    ----------
    log_path : str, optional
        If given, all table output is also written to this file.
    """
    global _console, _log_file

    if not RICH_AVAILABLE:
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            _log_file = open(log_path, "a", encoding="utf-8")
        return

    Path(log_path).parent.mkdir(parents=True, exist_ok=True) if log_path else None
    if log_path:
        # Dual output: terminal + file
        _log_file = open(log_path, "a", encoding="utf-8")
        _console = Console(
            record=True,
            highlight=False,
        )
    else:
        _console = Console(highlight=False)


def _get_console() -> "Console":
    global _console
    if _console is None:
        if RICH_AVAILABLE:
            _console = Console(highlight=False)
    return _console


def _flush_to_file(text: str) -> None:
    """Write plain text to log file."""
    if _log_file and not _log_file.closed:
        _log_file.write(text + "\n")
        _log_file.flush()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"

def _r(v: float, decimals: int = 2) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}R"

def _dollars(v: float) -> str:
    if v >= 0:
        return f"+${v:,.0f}"
    return f"-${abs(v):,.0f}"

def _colour(value: float, good_above: Optional[float] = None,
            bad_below: Optional[float] = None) -> str:
    """Return a coloured Rich markup string."""
    if not RICH_AVAILABLE:
        return str(value)
    s = f"{value:.4f}"
    if good_above is not None and value >= good_above:
        return f"[green]{s}[/green]"
    if bad_below is not None and value <= bad_below:
        return f"[red]{s}[/red]"
    return f"[yellow]{s}[/yellow]"


# ── 1. Trading metrics table ──────────────────────────────────────────────────

def print_trading_metrics(
    metrics: dict,
    phase: str,           # "TRAINING", "VALIDATION", or "TEST"
    step: Optional[int] = None,
    is_best: bool = False,
    composite_score: Optional[float] = None,
) -> None:
    """
    Print a two-column trading metrics table to console and log file.

    Parameters
    ----------
    metrics : dict
        Keys: total_pnl_r, sharpe_ratio, win_loss_ratio, max_drawdown_r,
              win_rate, n_trades, n_wins, n_losses, avg_rr, expected_return,
              avg_trade_duration, max_win_dollars, max_loss_dollars,
              profit_factor, avg_win_r, avg_loss_r.
              (Missing keys are shown as "—".)
    phase : str   "TRAINING" | "VALIDATION" | "TEST"
    step : int    Current training step.
    is_best : bool  Highlight as new best model.
    composite_score : float   Composite save score.
    """
    def _get(key, fmt=None):
        v = metrics.get(key)
        if v is None:
            return "—"
        return fmt(v) if fmt else str(v)

    step_str  = f"  step {step:,}" if step else ""
    best_str  = "  ★ NEW BEST" if is_best else ""
    title = f" {phase}{step_str}{best_str} "

    if RICH_AVAILABLE:
        c = _get_console()
        t = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            title_style="bold white on blue" if phase == "VALIDATION"
                        else "bold white on dark_green" if phase == "TEST"
                        else "bold white on grey35",
            min_width=52,
        )
        t.add_column("Metric",  style="dim",   min_width=26)
        t.add_column("Value",   justify="right", min_width=18)

        pnl  = metrics.get("total_pnl_r", 0.0)
        dd   = metrics.get("max_drawdown_r", 0.0)
        shr  = metrics.get("sharpe_ratio", 0.0)
        wl   = metrics.get("win_loss_ratio", 0.0)

        pnl_style = "green" if pnl >= 0 else "red"
        dd_style  = "green" if dd < 3 else "yellow" if dd < 6 else "red"
        shr_style = "green" if shr >= 1 else "yellow" if shr >= 0 else "red"
        wl_style  = "green" if wl >= 2 else "yellow" if wl >= 1 else "red"

        rows = [
            ("Total P&L (R)",          f"[{pnl_style}]{_r(pnl)}[/{pnl_style}]"),
            ("Sharpe Ratio",            f"[{shr_style}]{shr:.3f}[/{shr_style}]"),
            ("Win / Loss Ratio",        f"[{wl_style}]{wl:.2f}[/{wl_style}]"),
            ("Max Drawdown",            f"[{dd_style}]{dd:.2f}R[/{dd_style}]"),
            ("Win Rate",                _pct(metrics.get("win_rate", 0.0))),
            ("Trades",                  str(metrics.get("n_trades", 0))),
            ("Wins",                    str(metrics.get("n_wins",   0))),
            ("Losses",                  str(metrics.get("n_losses", 0))),
            ("Avg R:R",                 f"{metrics.get('avg_rr', 0.0):.2f}"),
            ("Expected Return",         _r(metrics.get("expected_return", 0.0), 3)),
            ("Avg Win (R)",             f"{metrics.get('avg_win_r',  0.0):.2f}R"),
            ("Avg Loss (R)",            f"-{abs(metrics.get('avg_loss_r', 0.0)):.2f}R"),
            ("Profit Factor",           f"{metrics.get('profit_factor', 0.0):.2f}"),
            ("Avg Trade Duration",      f"{metrics.get('avg_trade_duration', 0.0):.1f} bars"),
            ("Max Win  $",              _dollars(metrics.get("max_win_dollars",  0.0))),
            ("Max Loss $",              _dollars(metrics.get("max_loss_dollars", 0.0))),
        ]
        if composite_score is not None:
            cs_style = "green" if composite_score >= 0.5 else "yellow" if composite_score >= 0.25 else "red"
            rows.append(("─" * 26, "─" * 18))
            rows.append(("Composite Score",
                         f"[bold {cs_style}]{composite_score:.4f}[/bold {cs_style}]"))

        for k, v in rows:
            if k.startswith("─"):
                t.add_section()
            else:
                t.add_row(k, v)

        c.print(t)
        if _log_file:
            _log_file.write(c.export_text())
            _log_file.flush()

    else:
        # Plain-text fallback
        sep = "─" * 54
        lines = [sep, f"  {title}", sep]
        lines += [
            f"  {'Total P&L (R)':<28} {_r(metrics.get('total_pnl_r', 0.0))}",
            f"  {'Sharpe Ratio':<28} {metrics.get('sharpe_ratio', 0.0):.3f}",
            f"  {'Win / Loss Ratio':<28} {metrics.get('win_loss_ratio', 0.0):.2f}",
            f"  {'Max Drawdown':<28} {metrics.get('max_drawdown_r', 0.0):.2f}R",
            f"  {'Win Rate':<28} {_pct(metrics.get('win_rate', 0.0))}",
            f"  {'Trades':<28} {metrics.get('n_trades', 0)}",
            f"  {'Wins / Losses':<28} {metrics.get('n_wins', 0)} / {metrics.get('n_losses', 0)}",
            f"  {'Avg R:R':<28} {metrics.get('avg_rr', 0.0):.2f}",
            f"  {'Expected Return':<28} {_r(metrics.get('expected_return', 0.0), 3)}",
            f"  {'Avg Trade Duration':<28} {metrics.get('avg_trade_duration', 0.0):.1f} bars",
            f"  {'Max Win $':<28} {_dollars(metrics.get('max_win_dollars', 0.0))}",
            f"  {'Max Loss $':<28} {_dollars(metrics.get('max_loss_dollars', 0.0))}",
        ]
        if composite_score is not None:
            lines.append(f"  {'Composite Score':<28} {composite_score:.4f}")
        lines.append(sep)
        out = "\n".join(lines)
        print(out)
        _flush_to_file(out)


# ── 2. RL diagnostics table (Entropy + Explained Variance) ───────────────────

# With 5 discrete actions, max entropy = ln(5) ≈ 1.609 nats
_MAX_ENTROPY = math.log(5)   # update if Action.n_actions() changes

def print_rl_diagnostics(
    entropy: float,
    explained_var: float,
    step: Optional[int] = None,
) -> None:
    """
    Print a three-column table explaining Entropy and Explained Variance.

    Parameters
    ----------
    entropy : float
        Policy entropy in nats (SB3 reports entropy_loss = -entropy,
        so pass -entropy_loss).
    explained_var : float
        Explained variance of the value function (SB3: train/explained_variance).
    step : int
        Current training step for the table header.
    """
    step_str = f"  step {step:,}" if step else ""

    def _entropy_label(e: float) -> tuple:
        pct = e / _MAX_ENTROPY * 100
        if e >= 1.2:
            return "Exploring (high randomness)", "cyan", pct
        if e >= 0.5:
            return "Learning selectivity", "green", pct
        if e >= 0.1:
            return "Converging / exploiting", "yellow", pct
        return "May be stuck — check learning", "red", pct

    def _ev_label(ev: float) -> tuple:
        if ev >= 0.8:
            return "Excellent — value fn accurate", "green"
        if ev >= 0.5:
            return "Good — value fn learning", "yellow"
        if ev >= 0.0:
            return "Developing — early training OK", "yellow"
        return "Negative — value fn inaccurate", "red"

    ent_label, ent_col, ent_pct = _entropy_label(entropy)
    ev_label,  ev_col            = _ev_label(explained_var)

    if RICH_AVAILABLE:
        c = _get_console()
        t = Table(
            title=f" RL DIAGNOSTICS{step_str} ",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_style="bold white on dark_violet",
            min_width=72,
        )
        t.add_column("Metric",          style="dim",  min_width=20)
        t.add_column("Value",           justify="right", min_width=10)
        t.add_column("Status / Guidance",              min_width=36)

        # Entropy row
        ent_bar = "█" * int(ent_pct / 5) + "░" * (20 - int(ent_pct / 5))
        t.add_row(
            "Entropy",
            f"[{ent_col}]{entropy:.4f}[/{ent_col}]",
            f"[{ent_col}]{ent_label}[/{ent_col}]\n"
            f"[dim]{ent_pct:.0f}% of max ({_MAX_ENTROPY:.2f}nats)  {ent_bar}[/dim]",
        )
        t.add_row(
            "[dim]range[/dim]",
            "[dim]0 – 1.61[/dim]",
            "[dim]0=fully deterministic · 1.61=uniform random (5 actions)[/dim]",
        )

        t.add_section()

        # Explained Variance row
        t.add_row(
            "Explained Variance",
            f"[{ev_col}]{explained_var:.4f}[/{ev_col}]",
            f"[{ev_col}]{ev_label}[/{ev_col}]",
        )
        t.add_row(
            "[dim]range[/dim]",
            "[dim](−∞, 1.0][/dim]",
            "[dim]1.0=perfect · 0=mean baseline · <0=value fn wrong[/dim]",
        )

        c.print(t)
        if _log_file:
            _log_file.write(c.export_text())
            _log_file.flush()

    else:
        sep = "─" * 72
        lines = [
            sep,
            f"  RL DIAGNOSTICS{step_str}",
            sep,
            f"  {'Metric':<22} {'Value':>10}  {'Status / Guidance'}",
            sep,
            f"  {'Entropy':<22} {entropy:>10.4f}  {ent_label}",
            f"  {'  range 0 – 1.61':<22} {'':>10}  0=deterministic · 1.61=fully random (5 actions)",
            sep,
            f"  {'Explained Variance':<22} {explained_var:>10.4f}  {ev_label}",
            f"  {'  range (−∞, 1.0]':<22} {'':>10}  1=perfect · 0=mean baseline · <0=value fn wrong",
            sep,
        ]
        out = "\n".join(lines)
        print(out)
        _flush_to_file(out)
