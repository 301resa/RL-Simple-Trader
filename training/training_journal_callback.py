"""
training/training_journal_callback.py
=======================================
SB3 callback that accumulates every trade from all training envs and
periodically writes:

  logs/journal/training_journal.xlsx   — Excel workbook (3 sheets)
  logs/journal/training_journal.html   — Interactive Plotly chart

Both files are overwritten on each save so you always have the latest
snapshot without accumulating multiple files.

Excel sheets
------------
  Trades   : one row per trade — entry/SL/TP/close, PnL, duration, etc.
  Daily    : per-date aggregates (trades, win rate, total PnL)
  Summary  : overall training stats (win rate, Sharpe, profit factor …)

Plotly chart
------------
  Row 1: Cumulative equity curve (R)
  Row 2: Per-trade PnL bar chart (green=win, red=loss)
  Row 3: Rolling 20-trade win rate
  Row 4: Trade duration histogram
  Bottom: summary stats table
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# ── Colour constants (dark theme) ────────────────────────────────────────────
_GREEN = "#26a69a"
_RED   = "#ef5350"
_BLUE  = "#42a5f5"
_GOLD  = "#ffd700"
_BG    = "#131722"
_PAPER = "#1e2230"
_GRID  = "#2a2e39"
_TEXT  = "#d1d4dc"


class TrainingJournalCallback(BaseCallback):
    """
    Collects trades from all envs during training and saves Excel + Plotly.

    Parameters
    ----------
    journal_dir : str | Path
        Output directory (created if missing).
    save_every_steps : int
        Write files after every N timesteps (default 50 000).
    verbose : int
    """

    def __init__(
        self,
        journal_dir: str | Path = "logs/journal",
        save_every_steps: int = 50_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.journal_dir      = Path(journal_dir)
        self.save_every_steps = save_every_steps
        self._trades: List[dict] = []
        self._last_save_step  = 0
        self._save_n          = 0   # snapshot counter — incremented on every save

    # ── SB3 hooks ─────────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                for t in info.get("trades_list", []):
                    t["global_step"] = self.num_timesteps
                    self._trades.append(t)

        if (self.num_timesteps - self._last_save_step) >= self.save_every_steps:
            self._save()
            self._last_save_step = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        if self._trades:
            self._save()

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save(self) -> None:
        if not self._trades:
            return
        self._save_n += 1
        snap_dir = self.journal_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._write_excel(snap_dir)
        except Exception as exc:
            if self.verbose:
                print(f"[TrainingJournal] Excel write failed: {exc}")
        try:
            self._write_plotly(snap_dir)
        except Exception as exc:
            if self.verbose:
                print(f"[TrainingJournal] Plotly write failed: {exc}")

    # ── Excel ─────────────────────────────────────────────────────────────────

    def _write_excel(self, snap_dir: Path) -> None:
        import pandas as pd
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        df = pd.DataFrame(self._trades)
        stem = f"journal_s{self._save_n:04d}_step{self.num_timesteps:010d}"
        path = snap_dir / f"{stem}.xlsx"

        with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
            # ── Sheet 1: Trades ───────────────────────────────
            cols = [
                "global_step", "date", "direction",
                "entry_price", "stop_price", "initial_target", "exit_price",
                "pnl_r", "pnl_dollars", "pnl_points",
                "n_contracts", "duration_min", "exit_reason",
                "is_win", "mae_r",
            ]
            trades_df = df[[c for c in cols if c in df.columns]].copy()
            trades_df.to_excel(writer, sheet_name="Trades", index=False)
            ws = writer.sheets["Trades"]
            _style_trades_sheet(ws, trades_df)

            # ── Sheet 2: Daily ────────────────────────────────
            if "date" in df.columns:
                daily = (
                    df.groupby("date")
                    .agg(
                        trades=("pnl_r", "count"),
                        wins=("is_win", "sum"),
                        total_pnl_r=("pnl_r", "sum"),
                        total_pnl_dollars=("pnl_dollars", "sum"),
                        avg_pnl_r=("pnl_r", "mean"),
                        avg_duration_min=("duration_min", "mean"),
                    )
                    .reset_index()
                )
                daily["win_rate"] = (daily["wins"] / daily["trades"].clip(lower=1)).round(3)
                daily.to_excel(writer, sheet_name="Daily", index=False)
                _style_daily_sheet(writer.sheets["Daily"], daily)

            # ── Sheet 3: Summary ──────────────────────────────
            summary = _compute_summary(df, self.num_timesteps)
            summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            _style_summary_sheet(writer.sheets["Summary"])

        # Also overwrite a fixed-name "latest" copy for quick access
        import shutil
        latest = self.journal_dir / "training_journal.xlsx"
        shutil.copy2(str(path), str(latest))

        if self.verbose:
            print(f"[TrainingJournal] Excel saved → {path}  ({len(df)} trades)")

    # ── Plotly ────────────────────────────────────────────────────────────────

    def _write_plotly(self, snap_dir: Path) -> None:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        df_raw = self._trades
        if not df_raw:
            return

        pnl_r  = [t["pnl_r"]       for t in df_raw]
        is_win = [t["is_win"]       for t in df_raw]
        durs   = [t["duration_min"] for t in df_raw]
        dates  = [t["date"]         for t in df_raw]
        steps  = [t["global_step"]  for t in df_raw]
        dirs   = [t["direction"]    for t in df_raw]

        n = len(pnl_r)
        trade_nums = list(range(1, n + 1))
        equity     = list(np.cumsum(pnl_r))

        # Rolling 20-trade win rate
        win_arr    = np.array([1.0 if w else 0.0 for w in is_win])
        roll_wr    = [
            float(np.mean(win_arr[max(0, i - 19): i + 1]))
            for i in range(n)
        ]

        total_r   = sum(pnl_r)
        n_wins    = sum(is_win)
        wr_pct    = 100 * n_wins / n if n else 0
        color_r   = _GREEN if total_r >= 0 else _RED

        fig = make_subplots(
            rows=5, cols=1,
            row_heights=[0.28, 0.20, 0.16, 0.16, 0.20],
            vertical_spacing=0.04,
            specs=[
                [{"type": "scatter"}],
                [{"type": "bar"}],
                [{"type": "scatter"}],
                [{"type": "histogram"}],
                [{"type": "table"}],
            ],
            subplot_titles=[
                "Cumulative Equity (R)",
                "Per-Trade PnL (R)",
                "Rolling 20-Trade Win Rate",
                "Trade Duration Distribution (min)",
                "",
            ],
        )

        # ── Row 1: Equity curve ───────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=trade_nums, y=equity,
                mode="lines",
                line=dict(color=_BLUE, width=2),
                fill="tozeroy",
                fillcolor=f"rgba(66,165,245,0.1)",
                hovertemplate="Trade %{x}<br>Equity: %{y:.2f}R<extra></extra>",
                name="Equity",
            ),
            row=1, col=1,
        )
        fig.add_hline(y=0, line=dict(color=_GRID, width=1), row=1, col=1)

        # ── Row 2: Per-trade PnL bars ─────────────────────────
        bar_colors = [_GREEN if w else _RED for w in is_win]
        fig.add_trace(
            go.Bar(
                x=trade_nums,
                y=pnl_r,
                marker_color=bar_colors,
                hovertemplate=(
                    "Trade %{x}<br>"
                    "PnL: %{y:.2f}R<br>"
                    "<extra></extra>"
                ),
                name="PnL",
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line=dict(color=_GRID, width=1), row=2, col=1)

        # ── Row 3: Rolling win rate ───────────────────────────
        fig.add_trace(
            go.Scatter(
                x=trade_nums, y=[r * 100 for r in roll_wr],
                mode="lines",
                line=dict(color=_GOLD, width=1.5),
                hovertemplate="Trade %{x}<br>Win Rate: %{y:.1f}%<extra></extra>",
                name="Win Rate",
            ),
            row=3, col=1,
        )
        fig.add_hline(y=50, line=dict(color=_GRID, width=1, dash="dot"), row=3, col=1)

        # ── Row 4: Duration histogram ─────────────────────────
        fig.add_trace(
            go.Histogram(
                x=durs,
                marker_color=_BLUE,
                opacity=0.7,
                hovertemplate="Duration: %{x}m<br>Count: %{y}<extra></extra>",
                name="Duration",
            ),
            row=4, col=1,
        )

        # ── Row 5: Summary table ──────────────────────────────
        summary = _compute_summary(
            __import__("pandas").DataFrame(df_raw), self.num_timesteps
        )
        s_keys = list(summary.keys())
        s_vals = [str(v) for v in summary.values()]
        half   = len(s_keys) // 2 + len(s_keys) % 2

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>",
                            "<b>Metric</b>", "<b>Value</b>"],
                    fill_color=_PAPER,
                    font=dict(color=_TEXT, size=11),
                    align="center",
                    line_color=_GRID,
                    height=26,
                ),
                cells=dict(
                    values=[
                        s_keys[:half],  s_vals[:half],
                        s_keys[half:],  s_vals[half:],
                    ],
                    fill_color=_BG,
                    font=dict(color=_TEXT, size=10),
                    align=["left", "right", "left", "right"],
                    line_color=_GRID,
                    height=22,
                ),
            ),
            row=5, col=1,
        )

        # ── Layout ────────────────────────────────────────────
        title = (
            f"Training Journal  —  Step {self.num_timesteps:,}  |  "
            f"{n} trades  |  WR {wr_pct:.0f}%  |  "
            f"<span style='color:{color_r}'>{total_r:+.2f}R</span>"
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=14, color=_TEXT), x=0.5),
            paper_bgcolor=_PAPER,
            plot_bgcolor=_BG,
            font=dict(color=_TEXT, family="monospace"),
            showlegend=False,
            margin=dict(l=60, r=30, t=60, b=20),
            height=1100,
        )
        for row in range(1, 5):
            fig.update_xaxes(gridcolor=_GRID, zeroline=False, row=row, col=1)
            fig.update_yaxes(gridcolor=_GRID, zeroline=False, row=row, col=1)

        stem = f"journal_s{self._save_n:04d}_step{self.num_timesteps:010d}"
        path = snap_dir / f"{stem}.html"
        fig.write_html(str(path), include_plotlyjs="cdn")

        # Also overwrite latest copy
        import shutil
        latest = self.journal_dir / "training_journal.html"
        shutil.copy2(str(path), str(latest))

        if self.verbose:
            print(f"[TrainingJournal] Chart  saved → {path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_summary(df: Any, step: int) -> dict:
    """Compute summary stats from a trades DataFrame."""
    import pandas as pd
    if df.empty:
        return {"error": "no trades"}

    n      = len(df)
    wins   = df[df["is_win"]]
    losses = df[~df["is_win"]]
    wr     = len(wins) / n if n else 0.0
    total_r  = float(df["pnl_r"].sum())
    total_usd = float(df["pnl_dollars"].sum()) if "pnl_dollars" in df else 0.0
    avg_win  = float(wins["pnl_r"].mean())  if len(wins)   else 0.0
    avg_loss = float(losses["pnl_r"].mean()) if len(losses) else 0.0
    gross_w  = float(wins["pnl_r"].sum())   if len(wins)   else 0.0
    gross_l  = abs(float(losses["pnl_r"].sum())) if len(losses) else 1e-6
    pf       = min(gross_w / max(gross_l, 1e-6), 99.99)
    avg_dur  = float(df["duration_min"].mean()) if "duration_min" in df else 0.0

    tr = df["pnl_r"].values
    if len(tr) >= 5 and tr.std() > 0.01:
        sharpe = float(np.clip(tr.mean() / tr.std(), -9.99, 9.99))
    else:
        sharpe = 0.0

    return {
        "Training Step":    f"{step:,}",
        "Total Trades":     n,
        "Win Rate":         f"{wr*100:.1f}%",
        "Total PnL (R)":    f"{total_r:+.2f}",
        "Total PnL ($)":    f"${total_usd:+,.0f}",
        "Profit Factor":    f"{pf:.2f}",
        "Sharpe Ratio":     f"{sharpe:.2f}",
        "Avg Win (R)":      f"{avg_win:+.3f}",
        "Avg Loss (R)":     f"{avg_loss:+.3f}",
        "Avg Duration":     f"{avg_dur:.0f} min",
        "Total Wins":       len(wins),
        "Total Losses":     len(losses),
    }


def _style_trades_sheet(ws: Any, df: Any) -> None:
    """Apply green/red row fills and auto column widths to Trades sheet."""
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter

    green_fill = PatternFill("solid", fgColor="C8E6C9")
    red_fill   = PatternFill("solid", fgColor="FFCDD2")
    header_font = Font(bold=True)

    for cell in ws[1]:
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    is_win_col = None
    for i, cell in enumerate(ws[1], 1):
        if cell.value == "is_win":
            is_win_col = i
            break

    for row in ws.iter_rows(min_row=2):
        if is_win_col:
            win_val = row[is_win_col - 1].value
            fill = green_fill if win_val else red_fill
            for cell in row:
                cell.fill = fill

    for i, col in enumerate(ws.columns, 1):
        max_len = max((len(str(cell.value or "")) for cell in col), default=8)
        ws.column_dimensions[get_column_letter(i)].width = min(max_len + 2, 20)


def _style_daily_sheet(ws: Any, df: Any) -> None:
    from openpyxl.styles import PatternFill, Font
    from openpyxl.utils import get_column_letter

    green_fill = PatternFill("solid", fgColor="C8E6C9")
    red_fill   = PatternFill("solid", fgColor="FFCDD2")

    for cell in ws[1]:
        cell.font = Font(bold=True)

    pnl_col = None
    for i, cell in enumerate(ws[1], 1):
        if cell.value == "total_pnl_r":
            pnl_col = i
            break

    for row in ws.iter_rows(min_row=2):
        if pnl_col:
            val = row[pnl_col - 1].value
            fill = green_fill if (val or 0) >= 0 else red_fill
            for cell in row:
                cell.fill = fill

    for i, col in enumerate(ws.columns, 1):
        max_len = max((len(str(c.value or "")) for c in col), default=8)
        ws.column_dimensions[get_column_letter(i)].width = min(max_len + 2, 18)


def _style_summary_sheet(ws: Any) -> None:
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        row[0].font = Font(bold=True)

    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 16
