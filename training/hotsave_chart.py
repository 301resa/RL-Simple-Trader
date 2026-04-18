"""
training/hotsave_chart.py
==========================
Generates a self-contained per-env interactive Plotly HTML when a model
is hot-saved.  No OHLC bar data required — built entirely from trade dicts.

Layout (4 rows):
  Row 1 (50%): Price view  — entry marker, exit marker, SL/TP horizontal
               lines, entry→exit connector, coloured by direction / outcome
  Row 2 (16%): Cumulative PnL ($)
  Row 3 (16%): Cumulative Drawdown ($)
  Row 4      : Full trade table with all fields
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

_GREEN = "#26a69a"
_RED   = "#ef5350"
_BG    = "#131722"
_PAPER = "#1e2230"
_GRID  = "#2a2e39"
_TEXT  = "#d1d4dc"


def write_env_hotsave_chart(
    trades:       List[dict],
    output_path:  str | Path,
    env_id:       int,
    step:         int,
    gate_tag:     str = "",
) -> None:
    """
    Write one per-env trade chart HTML to output_path.

    Parameters
    ----------
    trades      : trade dicts for ONE env (must already be filtered by env_id)
    output_path : full path for the HTML file
    env_id      : environment index (shown in title)
    step        : training step at save time (shown in title)
    gate_tag    : e.g. "PF gate", "WR70 gate", "Elite gate"
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not trades:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_trades = sorted(
        trades,
        key=lambda t: (t.get("entry_time") or t.get("date", ""), t.get("entry_price", 0.0)),
    )
    n = len(sorted_trades)
    if n == 0:
        return

    # X-axis: trade sequence 1..N
    xs       = list(range(1, n + 1))
    x_entry  = [i - 0.25 for i in xs]
    x_exit   = [i + 0.25 for i in xs]

    # Equity / drawdown
    pnl_d   = np.array([t.get("pnl_dollars", t.get("pnl_r", 0.0)) for t in sorted_trades], dtype=float)
    cum_pnl = np.cumsum(pnl_d).tolist()
    peak    = np.maximum.accumulate(np.array(cum_pnl, dtype=float)).tolist()
    dd      = [cp - pk for cp, pk in zip(cum_pnl, peak)]

    n_wins    = sum(1 for t in sorted_trades if t.get("is_win"))
    wr_pct    = n_wins / n * 100 if n else 0.0
    total_pnl = cum_pnl[-1] if cum_pnl else 0.0
    max_dd    = -min(dd) if dd else 0.0
    clr_title = _GREEN if total_pnl >= 0 else _RED

    # ── Figure ────────────────────────────────────────────────────────────────
    table_row_px = max(300, min(n * 22 + 70, 900))
    total_height = 550 + 200 + 200 + table_row_px + 120

    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.50, 0.16, 0.16, 0.18],
        shared_xaxes=False,
        vertical_spacing=0.03,
        specs=[
            [{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "table"}],
        ],
        subplot_titles=[
            "Trade Prices  —  Entry / Exit / Stop / Target",
            "Cumulative PnL ($)",
            "Cumulative Drawdown ($)",
            "",
        ],
    )

    # ── Row 1: Price view ─────────────────────────────────────────────────────

    # Stop lines (red dashed, one trace with None separators)
    sl_x: list = []
    sl_y: list = []
    for i, t in enumerate(sorted_trades, 1):
        sl = t.get("stop_price")
        if sl is not None:
            sl_x.extend([i - 0.48, i + 0.48, None])
            sl_y.extend([sl, sl, None])
    if sl_x:
        fig.add_trace(
            go.Scatter(
                x=sl_x, y=sl_y, mode="lines",
                line=dict(color="rgba(239,83,80,0.55)", width=1.2, dash="dash"),
                name="Stop", showlegend=True, hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # Target lines (green dashed)
    tp_x: list = []
    tp_y: list = []
    for i, t in enumerate(sorted_trades, 1):
        tp = t.get("initial_target")
        if tp is not None:
            tp_x.extend([i - 0.48, i + 0.48, None])
            tp_y.extend([tp, tp, None])
    if tp_x:
        fig.add_trace(
            go.Scatter(
                x=tp_x, y=tp_y, mode="lines",
                line=dict(color="rgba(38,166,154,0.55)", width=1.2, dash="dash"),
                name="Target", showlegend=True, hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # Entry→Exit connector lines (dotted, split by outcome)
    for is_win in (True, False):
        cx: list = []
        cy: list = []
        for i, t in enumerate(sorted_trades, 1):
            if bool(t.get("is_win")) != is_win:
                continue
            cx.extend([i - 0.25, i + 0.25, None])
            cy.extend([t.get("entry_price", 0), t.get("exit_price", 0), None])
        if cx:
            fig.add_trace(
                go.Scatter(
                    x=cx, y=cy, mode="lines",
                    line=dict(color=_GREEN if is_win else _RED, width=1.5, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1,
            )

    # Entry markers (triangle-up = LONG, triangle-down = SHORT)
    fig.add_trace(
        go.Scatter(
            x=x_entry,
            y=[t.get("entry_price", 0) for t in sorted_trades],
            mode="markers",
            marker=dict(
                symbol=["triangle-up" if t.get("direction") == "LONG" else "triangle-down"
                        for t in sorted_trades],
                color=[_GREEN if t.get("direction") == "LONG" else _RED
                       for t in sorted_trades],
                size=11,
                line=dict(width=1, color=_TEXT),
            ),
            name="Entry",
            customdata=[
                [
                    t.get("direction", "?"),
                    t.get("n_contracts", 1),
                    t.get("stop_price", 0),
                    t.get("initial_target", 0),
                    "RTH" if t.get("is_rth", True) else "ETH",
                    t.get("date", ""),
                ]
                for t in sorted_trades
            ],
            hovertemplate=(
                "<b>ENTRY  %{customdata[0]}</b><br>"
                "Date  : %{customdata[5]}<br>"
                "Price : %{y:.2f}<br>"
                "Lots  : %{customdata[1]}<br>"
                "Stop  : %{customdata[2]:.2f}<br>"
                "Target: %{customdata[3]:.2f}<br>"
                "Sess  : %{customdata[4]}<extra></extra>"
            ),
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Exit markers (X, green = win, red = loss)
    fig.add_trace(
        go.Scatter(
            x=x_exit,
            y=[t.get("exit_price", 0) for t in sorted_trades],
            mode="markers",
            marker=dict(
                symbol="x",
                color=[_GREEN if t.get("is_win") else _RED for t in sorted_trades],
                size=10,
                line=dict(width=2),
            ),
            name="Exit",
            customdata=[
                [
                    t.get("pnl_dollars", t.get("pnl_r", 0.0)),
                    t.get("pnl_r", 0.0),
                    t.get("exit_reason", "?"),
                    "Win" if t.get("is_win") else "Loss",
                    t.get("duration_min", 0),
                ]
                for t in sorted_trades
            ],
            hovertemplate=(
                "<b>EXIT  %{customdata[3]}</b><br>"
                "Price  : %{y:.2f}<br>"
                "PnL $  : $%{customdata[0]:+,.0f}<br>"
                "PnL R  : %{customdata[1]:+.3f}<br>"
                "Reason : %{customdata[2]}<br>"
                "Dur    : %{customdata[4]:.0f}m<extra></extra>"
            ),
            showlegend=True,
        ),
        row=1, col=1,
    )

    # ── Row 2: Cumulative PnL ─────────────────────────────────────────────────
    pnl_color = _GREEN if total_pnl >= 0 else _RED
    pnl_fill  = "rgba(38,166,154,0.12)" if pnl_color == _GREEN else "rgba(239,83,80,0.12)"
    fig.add_trace(
        go.Scatter(
            x=xs, y=cum_pnl, mode="lines",
            line=dict(color=pnl_color, width=2),
            fill="tozeroy", fillcolor=pnl_fill,
            hovertemplate="Trade %{x}<br>Cum PnL: $%{y:+,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line=dict(color=_GRID, width=1), row=2, col=1)

    # ── Row 3: Drawdown ───────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=xs, y=dd, mode="lines",
            line=dict(color=_RED, width=1.5),
            fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
            hovertemplate="Trade %{x}<br>Drawdown: $%{y:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line=dict(color=_GRID, width=1, dash="dot"), row=3, col=1)

    # ── Row 4: Trade table ────────────────────────────────────────────────────
    tbl_hdrs = [
        "#", "Date", "Dir", "Lots",
        "Entry", "Stop", "Target", "Exit",
        "PnL($)", "PnL(R)", "W/L", "Sess", "Dur(m)", "Reason",
    ]
    tbl_data = [
        [i + 1                                                  for i in range(n)],
        [t.get("date", "")                                      for t in sorted_trades],
        [t.get("direction", "")                                 for t in sorted_trades],
        [t.get("n_contracts", 1)                                for t in sorted_trades],
        [f"{t.get('entry_price', 0):.2f}"                      for t in sorted_trades],
        [f"{t.get('stop_price', 0):.2f}"                       for t in sorted_trades],
        [f"{t.get('initial_target', 0):.2f}"                   for t in sorted_trades],
        [f"{t.get('exit_price', 0):.2f}"                       for t in sorted_trades],
        [f"${t.get('pnl_dollars', t.get('pnl_r', 0)):+,.0f}"  for t in sorted_trades],
        [f"{t.get('pnl_r', 0):+.3f}R"                         for t in sorted_trades],
        ["Win" if t.get("is_win") else "Loss"                   for t in sorted_trades],
        ["RTH" if t.get("is_rth", True) else "ETH"             for t in sorted_trades],
        [f"{t.get('duration_min', 0):.0f}"                     for t in sorted_trades],
        [t.get("exit_reason", "")                               for t in sorted_trades],
    ]
    row_fills = [
        "rgba(38,166,154,0.10)" if t.get("is_win") else "rgba(239,83,80,0.10)"
        for t in sorted_trades
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in tbl_hdrs],
                fill_color=_PAPER,
                font=dict(color=_TEXT, size=11),
                align="center",
                line_color=_GRID,
                height=26,
            ),
            cells=dict(
                values=tbl_data,
                fill_color=[row_fills] * len(tbl_hdrs),
                font=dict(color=_TEXT, size=10),
                align=["center"] * len(tbl_hdrs),
                line_color=_GRID,
                height=22,
            ),
        ),
        row=4, col=1,
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    gate_label = f" [{gate_tag}]" if gate_tag else ""
    title_str = (
        f"HotSave{gate_label}  |  Env {env_id:02d}  |  Step {step:,}  |  "
        f"{n} trades  WR {wr_pct:.0f}%  "
        f"<span style='color:{clr_title}'>PnL ${total_pnl:+,.0f}</span>  "
        f"MaxDD ${max_dd:,.0f}<br>"
        f"<span style='font-size:11px;color:#888'>"
        f"Prices are augmented (bar jitter + session scaling) — "
        f"absolute levels differ from live data.  PnL in R is scale-invariant."
        f"</span>"
    )

    tick_step = max(1, n // 30)
    tick_vals = xs[::tick_step]

    fig.update_layout(
        title=dict(text=title_str, font=dict(size=13, color=_TEXT), x=0.5),
        paper_bgcolor=_PAPER,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="monospace"),
        showlegend=True,
        legend=dict(
            orientation="h", x=0.0, y=1.02,
            font=dict(size=10, color=_TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
        dragmode="pan",
        margin=dict(l=70, r=20, t=90, b=20),
        height=total_height,
        xaxis=dict(
            gridcolor=_GRID, zeroline=False,
            title="Trade #",
            tickvals=tick_vals,
            ticktext=[str(v) for v in tick_vals],
        ),
        xaxis2=dict(gridcolor=_GRID, zeroline=False, title="Trade #"),
        xaxis3=dict(gridcolor=_GRID, zeroline=False, title="Trade #"),
        yaxis=dict(gridcolor=_GRID, zeroline=False, fixedrange=False, title="Price"),
        yaxis2=dict(gridcolor=_GRID, zeroline=False, tickprefix="$", title="PnL"),
        yaxis3=dict(gridcolor=_GRID, zeroline=False, tickprefix="$", title="Drawdown"),
    )

    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["pan2d", "zoom2d"],
        },
    )
