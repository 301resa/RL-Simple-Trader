"""
evaluation/journal_viewer.py
==============================
Interactive trade journal viewer — candlestick chart + trade table.

For every trading day that has recorded trades the viewer generates an
interactive HTML file containing:
  - Candlestick chart with entry/exit markers, SL and TP level lines
  - A colour-coded trade table (green = win, red = loss)
  - Summary statistics banner

Usage
-----
    python evaluation/journal_viewer.py \\
        --journal logs/journal/best_model_trades.csv \\
        --data    data/ \\
        --out     logs/journal/charts/

    # Optional: filter to a single date
    python evaluation/journal_viewer.py \\
        --journal logs/journal/best_model_trades.csv \\
        --data data/ --date 2025-03-14

All output files are self-contained HTML — open in any browser.
An index page (index.html) is also written to --out listing every day.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ── Plotly imports ────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Plotly is required:  pip install plotly")
    sys.exit(1)


# ── Colour palette ────────────────────────────────────────────────────────────
_GREEN      = "#26a69a"
_RED        = "#ef5350"
_GOLD       = "#ffd700"
_BLUE       = "#42a5f5"
_GREY       = "#607d8b"
_BG         = "#131722"
_PAPER      = "#1e2230"
_GRID       = "#2a2e39"
_TEXT       = "#d1d4dc"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_ohlcv(data_dir: str, instrument: str, date: str) -> pd.DataFrame:
    """
    Load 5-min bars for a single trading date directly from the CSV
    (no DataLoader dependency so this script runs standalone).
    """
    data_path = Path(data_dir)
    candidates = list(data_path.glob(f"*{instrument}*.csv")) + list(data_path.glob("*.csv"))
    if not candidates:
        return pd.DataFrame()

    raw = pd.read_csv(candidates[0], dtype=str)
    raw.columns = raw.columns.str.strip()

    col_map = {
        "Date": "date", "Time": "time",
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Volume": "volume",
    }
    raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

    dt = pd.to_datetime(raw["date"].str.strip() + " " + raw["time"].str.strip(), dayfirst=True)
    raw.index = dt
    for col in ("open", "high", "low", "close"):
        raw[col] = raw[col].astype(float)

    target = pd.Timestamp(date).date()
    day_bars = raw[raw.index.date == target][["open", "high", "low", "close", "volume"]].copy()
    day_bars.sort_index(inplace=True)
    return day_bars


# ── Chart builder ─────────────────────────────────────────────────────────────

def _build_day_chart(
    date: str,
    bars: pd.DataFrame,
    trades: pd.DataFrame,
) -> go.Figure:
    """
    Build a Plotly figure for one trading day.

    Layout
    ------
    Row 1 (70%): candlestick + trade annotations
    Row 2 (30%): trade table
    """

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.04,
        specs=[[{"type": "candlestick"}], [{"type": "table"}]],
    )

    # ── Candlestick ───────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            name="Price",
            increasing_line_color=_GREEN,
            decreasing_line_color=_RED,
            increasing_fillcolor=_GREEN,
            decreasing_fillcolor=_RED,
        ),
        row=1, col=1,
    )

    # ── Trade overlays ────────────────────────────────────────
    for _, t in trades.iterrows():
        _add_trade(fig, bars, t)

    # ── Summary annotation (top-left) ─────────────────────────
    n_trades  = len(trades)
    n_wins    = int(trades["is_win"].sum())
    total_r   = float(trades["pnl_r"].sum())
    total_usd = float(trades["pnl_dollars"].sum())
    wr_pct    = 100 * n_wins / n_trades if n_trades else 0.0

    color_r = _GREEN if total_r >= 0 else _RED
    summary = (
        f"<b>{date}</b>   "
        f"Trades: {n_trades}   "
        f"Win Rate: {wr_pct:.0f}%   "
        f"<span style='color:{color_r}'>PnL: {total_r:+.2f}R  ${total_usd:+,.0f}</span>"
    )
    fig.add_annotation(
        text=summary,
        xref="paper", yref="paper",
        x=0.0, y=1.045,
        showarrow=False,
        font=dict(size=13, color=_TEXT),
        align="left",
    )

    # ── Trade table ───────────────────────────────────────────
    _add_table(fig, trades)

    # ── Layout ────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=_PAPER,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="monospace"),
        showlegend=False,
        margin=dict(l=60, r=30, t=60, b=20),
        xaxis_rangeslider_visible=False,
        height=950,
        title=dict(
            text=f"Trade Journal — {date}",
            font=dict(size=16, color=_TEXT),
            x=0.5,
        ),
    )
    fig.update_xaxes(
        gridcolor=_GRID, showgrid=True,
        zeroline=False,
        tickfont=dict(color=_TEXT),
    )
    fig.update_yaxes(
        gridcolor=_GRID, showgrid=True,
        zeroline=False,
        tickfont=dict(color=_TEXT),
    )

    return fig


def _add_trade(
    fig: go.Figure,
    bars: pd.DataFrame,
    trade: pd.Series,
) -> None:
    """Overlay one trade onto the candlestick chart."""
    n = len(bars)
    entry_idx = min(int(trade["entry_bar_idx"]), n - 1)
    exit_idx  = min(int(trade["exit_bar_idx"]),  n - 1)

    if entry_idx >= n or exit_idx >= n:
        return

    entry_time = bars.index[entry_idx]
    exit_time  = bars.index[exit_idx]

    is_long  = str(trade["direction"]).upper() == "LONG"
    is_win   = bool(trade["is_win"])
    color    = _GREEN if is_win else _RED
    entry_px = float(trade["entry_price"])
    exit_px  = float(trade["exit_price"])
    sl_px    = float(trade["stop_price"])
    tp_px    = float(trade["initial_target"])

    # ── Entry marker ──────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=[entry_time],
            y=[entry_px],
            mode="markers+text",
            marker=dict(
                symbol="triangle-up" if is_long else "triangle-down",
                size=14,
                color=_BLUE,
                line=dict(color="#ffffff", width=1),
            ),
            text=["E"],
            textposition="top center" if is_long else "bottom center",
            textfont=dict(size=8, color=_BLUE),
            hovertemplate=(
                f"<b>ENTRY {'LONG' if is_long else 'SHORT'}</b><br>"
                f"Price: {entry_px:.2f}<br>"
                f"SL:    {sl_px:.2f}<br>"
                f"TP:    {tp_px:.2f}<br>"
                f"<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ── Exit marker ───────────────────────────────────────────
    exit_reason = str(trade.get("exit_reason", ""))
    fig.add_trace(
        go.Scatter(
            x=[exit_time],
            y=[exit_px],
            mode="markers",
            marker=dict(
                symbol="x",
                size=12,
                color=color,
                line=dict(color="#ffffff", width=1),
            ),
            hovertemplate=(
                f"<b>EXIT — {exit_reason}</b><br>"
                f"Price:  {exit_px:.2f}<br>"
                f"PnL:    {trade['pnl_r']:+.2f}R  ${trade['pnl_dollars']:+.0f}<br>"
                f"Bars:   {trade['duration_bars']}<br>"
                f"<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ── SL line (red dashed) ──────────────────────────────────
    fig.add_shape(
        type="line",
        x0=entry_time, x1=exit_time,
        y0=sl_px,      y1=sl_px,
        line=dict(color=_RED, width=1.5, dash="dot"),
        row=1, col=1,
    )

    # ── TP line (green dashed) ────────────────────────────────
    fig.add_shape(
        type="line",
        x0=entry_time, x1=exit_time,
        y0=tp_px,      y1=tp_px,
        line=dict(color=_GREEN, width=1.5, dash="dot"),
        row=1, col=1,
    )

    # ── Trade region shading ──────────────────────────────────
    fig.add_vrect(
        x0=entry_time, x1=exit_time,
        fillcolor=color,
        opacity=0.06,
        layer="below",
        line_width=0,
        row=1, col=1,
    )

    # ── Entry price horizontal line (grey) ────────────────────
    fig.add_shape(
        type="line",
        x0=entry_time, x1=exit_time,
        y0=entry_px,   y1=entry_px,
        line=dict(color=_BLUE, width=1, dash="dash"),
        row=1, col=1,
    )

    # ── Labels at right edge ──────────────────────────────────
    for label, price, lcolor in [("SL", sl_px, _RED), ("TP", tp_px, _GREEN)]:
        fig.add_annotation(
            x=exit_time, y=price,
            text=f"  {label} {price:.1f}",
            showarrow=False,
            font=dict(size=9, color=lcolor),
            xanchor="left",
            row=1, col=1,
        )


def _add_table(fig: go.Figure, trades: pd.DataFrame) -> None:
    """Add a colour-coded trade table in row 2."""
    cols_show = [
        "trade_number", "direction", "entry_price", "stop_price",
        "initial_target", "exit_price", "pnl_r", "pnl_dollars",
        "duration_bars", "exit_reason", "is_win",
    ]
    col_labels = [
        "#", "Dir", "Entry", "SL", "TP", "Close",
        "PnL (R)", "PnL ($)", "Bars", "Exit", "W/L",
    ]

    df = trades[cols_show].copy()
    df["pnl_r"]       = df["pnl_r"].map(lambda x: f"{x:+.2f}")
    df["pnl_dollars"]  = df["pnl_dollars"].map(lambda x: f"${x:+,.0f}")
    df["entry_price"]  = df["entry_price"].map(lambda x: f"{x:.2f}")
    df["stop_price"]   = df["stop_price"].map(lambda x: f"{x:.2f}")
    df["initial_target"] = df["initial_target"].map(lambda x: f"{x:.2f}")
    df["exit_price"]   = df["exit_price"].map(lambda x: f"{x:.2f}")
    df["is_win"]       = df["is_win"].map(lambda x: "WIN" if x else "LOSS")

    fill_colors = []
    for col in cols_show:
        col_fill = []
        for _, row in trades.iterrows():
            col_fill.append(_GREEN + "33" if row["is_win"] else _RED + "33")
        fill_colors.append(col_fill)

    fig.add_trace(
        go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in col_labels],
                fill_color=_PAPER,
                font=dict(color=_TEXT, size=11),
                align="center",
                line_color=_GRID,
                height=28,
            ),
            cells=dict(
                values=[df[c].tolist() for c in cols_show],
                fill_color=fill_colors,
                font=dict(color=_TEXT, size=10),
                align=["center"] * len(cols_show),
                line_color=_GRID,
                height=24,
            ),
        ),
        row=2, col=1,
    )


# ── Index page ────────────────────────────────────────────────────────────────

def _build_index(all_trades: pd.DataFrame, day_files: Dict[str, str]) -> str:
    """Return HTML for the index / summary page."""

    total_r    = float(all_trades["pnl_r"].sum())
    total_usd  = float(all_trades["pnl_dollars"].sum())
    n_trades   = len(all_trades)
    n_wins     = int(all_trades["is_win"].sum())
    wr         = 100 * n_wins / n_trades if n_trades else 0.0
    avg_win_r  = float(all_trades[all_trades["is_win"]]["pnl_r"].mean()) if n_wins else 0.0
    avg_loss_r = float(all_trades[~all_trades["is_win"]]["pnl_r"].mean()) if (n_trades - n_wins) else 0.0

    color = "#26a69a" if total_r >= 0 else "#ef5350"
    rows = ""
    for date, fname in sorted(day_files.items()):
        day_trades = all_trades[all_trades["episode_date"] == date]
        day_r   = float(day_trades["pnl_r"].sum())
        day_usd = float(day_trades["pnl_dollars"].sum())
        day_wr  = 100 * int(day_trades["is_win"].sum()) / len(day_trades)
        row_color = "#26a69a22" if day_r >= 0 else "#ef535022"
        rows += (
            f"<tr style='background:{row_color}'>"
            f"<td><a href='{fname}' style='color:#42a5f5'>{date}</a></td>"
            f"<td>{len(day_trades)}</td>"
            f"<td>{day_wr:.0f}%</td>"
            f"<td style='color:{'#26a69a' if day_r>=0 else '#ef5350'}'>{day_r:+.2f}</td>"
            f"<td style='color:{'#26a69a' if day_usd>=0 else '#ef5350'}'>${day_usd:+,.0f}</td>"
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Trade Journal — Summary</title>
<style>
  body   {{ background:#131722; color:#d1d4dc; font-family:monospace; padding:30px; }}
  h1     {{ color:#d1d4dc; border-bottom:1px solid #2a2e39; padding-bottom:8px; }}
  .stat  {{ display:inline-block; background:#1e2230; border:1px solid #2a2e39;
             border-radius:6px; padding:14px 22px; margin:6px; min-width:130px;
             text-align:center; }}
  .sv    {{ font-size:22px; font-weight:bold; }}
  .sl    {{ font-size:11px; color:#607d8b; margin-top:4px; }}
  table  {{ border-collapse:collapse; width:100%; margin-top:20px; }}
  th,td  {{ border:1px solid #2a2e39; padding:8px 14px; text-align:center; }}
  th     {{ background:#1e2230; }}
</style>
</head><body>
<h1>📒 Trade Journal — Summary</h1>
<div>
  <div class='stat'><div class='sv'>{n_trades}</div><div class='sl'>Trades</div></div>
  <div class='stat'><div class='sv'>{wr:.0f}%</div><div class='sl'>Win Rate</div></div>
  <div class='stat'><div class='sv' style='color:{color}'>{total_r:+.2f}R</div><div class='sl'>Total PnL (R)</div></div>
  <div class='stat'><div class='sv' style='color:{color}'>${total_usd:+,.0f}</div><div class='sl'>Total PnL ($)</div></div>
  <div class='stat'><div class='sv' style='color:#26a69a'>{avg_win_r:+.2f}R</div><div class='sl'>Avg Win</div></div>
  <div class='stat'><div class='sv' style='color:#ef5350'>{avg_loss_r:+.2f}R</div><div class='sl'>Avg Loss</div></div>
</div>
<table>
<tr><th>Date</th><th>Trades</th><th>Win%</th><th>PnL (R)</th><th>PnL ($)</th></tr>
{rows}
</table>
</body></html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate interactive trade journal charts from a journal CSV."
    )
    parser.add_argument("--journal",     required=True, help="Path to trades CSV file")
    parser.add_argument("--data",        required=True, help="Path to raw OHLCV data directory")
    parser.add_argument("--instrument",  default="ES",  help="Instrument ticker (default: ES)")
    parser.add_argument("--out",         default=None,  help="Output directory (default: same as --journal)")
    parser.add_argument("--date",        default=None,  help="Only chart this date (YYYY-MM-DD)")
    parser.add_argument("--no-browser",  action="store_true", help="Don't open browser after generation")
    args = parser.parse_args()

    journal_path = Path(args.journal)
    if not journal_path.exists():
        print(f"Journal file not found: {journal_path}")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else journal_path.parent / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_trades = pd.read_csv(journal_path)
    if all_trades.empty:
        print("No trades in journal.")
        sys.exit(0)

    if args.date:
        all_trades = all_trades[all_trades["episode_date"] == args.date]
        if all_trades.empty:
            print(f"No trades found for date {args.date}")
            sys.exit(0)

    dates = sorted(all_trades["episode_date"].unique())
    print(f"Found {len(all_trades)} trades across {len(dates)} day(s)")

    day_files: Dict[str, str] = {}

    for date in dates:
        day_trades = all_trades[all_trades["episode_date"] == date].copy()
        bars = _load_ohlcv(args.data, args.instrument, date)

        if bars.empty:
            print(f"  [{date}] WARNING: no OHLCV data found — skipping chart")
            continue

        print(f"  [{date}] {len(day_trades)} trades — building chart...", end=" ")
        fig  = _build_day_chart(date, bars, day_trades)
        fname = f"journal_{date}.html"
        out_path = out_dir / fname
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        day_files[date] = fname
        print(f"saved → {out_path.name}")

    # ── Index page ────────────────────────────────────────────
    index_html = _build_index(all_trades, day_files)
    index_path = out_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    print(f"\nIndex page → {index_path}")

    if not args.no_browser and day_files:
        import webbrowser
        webbrowser.open(index_path.as_uri())


if __name__ == "__main__":
    main()
