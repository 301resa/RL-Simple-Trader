"""
training/trade_chart.py
========================
Generates a comprehensive trade verification HTML report.

Layout (4 rows, shared datetime x-axis on rows 1–3):
  Row 1: OHLC candlestick + trade annotations
            entry marker  — triangle-up (LONG) / triangle-down (SHORT)
            exit marker   — ✕ green=win / red=loss
            SL line       — red dashed horizontal (entry → exit bar)
            TP line       — green dashed horizontal (entry → exit bar)
            trade line    — dotted connector entry_price → exit_price
  Row 2: Cumulative PnL ($)  — linked x-axis
  Row 3: Cumulative Drawdown ($) — linked x-axis
  Row 4: Full trade table (all fields)

Usage
-----
    from training.trade_chart import write_trade_chart
    write_trade_chart(trades, data_dir="data/", output_path="logs/trades.html",
                      instrument="ES", bar_minutes=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


# ── Colour palette (dark theme) ──────────────────────────────────────────────
_GREEN  = "#26a69a"
_RED    = "#ef5350"
_BG     = "#131722"
_PAPER  = "#1e2230"
_GRID   = "#2a2e39"
_TEXT   = "#d1d4dc"


def write_trade_chart(
    trades:          List[dict],
    data_dir:        str | Path,
    output_path:     str | Path,
    instrument:      str   = "ES",
    bar_minutes:     int   = 5,
    initial_capital: float = 100_000.0,
    title_prefix:    str   = "Trade Verification",
) -> None:
    """
    Build an OHLC + trade annotation Plotly HTML.

    Parameters
    ----------
    trades          : trade dicts — must include entry_time / exit_time
    data_dir        : directory containing the instrument CSV / pickle cache
    output_path     : full path for the HTML output file
    instrument      : "ES", "NQ", etc.
    bar_minutes     : bar timeframe in minutes (must match the data)
    initial_capital : used in the subtitle only
    title_prefix    : prepended to the HTML title
    """
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from data.data_loader import DataLoader

    if not trades:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load bar data ─────────────────────────────────────────────────────────
    dl = DataLoader(
        data_dir=str(data_dir),
        instrument=instrument,
        timeframe=f"{bar_minutes}min",
    )
    dl.load()

    dates      = sorted(set(t["date"] for t in trades if "date" in t))
    bar_frames = [dl.get_day_bars(d) for d in dates]
    bar_frames = [f for f in bar_frames if not f.empty]
    if not bar_frames:
        return

    bars = pd.concat(bar_frames).sort_index()

    # ── Sort trades chronologically ───────────────────────────────────────────
    sorted_trades = sorted(
        trades,
        key=lambda t: t.get("entry_time", t.get("date", "")),
    )
    n_trades = len(sorted_trades)

    def _ts(s: Optional[str]) -> Optional[pd.Timestamp]:
        try:
            return pd.Timestamp(s) if s else None
        except Exception:
            return None

    # ── PnL / drawdown arrays ─────────────────────────────────────────────────
    pnl_d   = np.array([t.get("pnl_dollars", t.get("pnl_r", 0.0)) for t in sorted_trades],
                       dtype=float)
    cum_pnl = np.cumsum(pnl_d)
    peak    = np.maximum.accumulate(cum_pnl)
    dd      = cum_pnl - peak                       # always ≤ 0

    entry_times = [_ts(t.get("entry_time")) for t in sorted_trades]
    x_times     = [et if et is not None else t.get("date") for et, t in zip(entry_times, sorted_trades)]
    x_seq       = list(range(1, n_trades + 1))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.50, 0.14, 0.14, 0.22],
        shared_xaxes=False,      # OHLC=datetime, PnL/DD=trade seq, table=none
        vertical_spacing=0.025,
        specs=[
            [{}],
            [{}],
            [{}],
            [{"type": "table"}],
        ],
        subplot_titles=[
            "OHLC + Trade Annotations",
            "Cumulative PnL ($)",
            "Cumulative Drawdown ($)",
            "",
        ],
    )

    # ── Row 1: Candlestick ────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=bars["open"].values,
            high=bars["high"].values,
            low=bars["low"].values,
            close=bars["close"].values,
            increasing_line_color=_GREEN,
            decreasing_line_color=_RED,
            name="OHLC",
            showlegend=False,
            hoverinfo="x+y",
            whiskerwidth=0,
        ),
        row=1, col=1,
    )

    # Trade connector lines (entry_price → exit_price): green=win, red=loss
    for is_win in (True, False):
        xs: list = []
        ys: list = []
        for t in sorted_trades:
            if bool(t.get("is_win")) != is_win:
                continue
            et = _ts(t.get("entry_time"))
            xt = _ts(t.get("exit_time"))
            if et is None or xt is None:
                continue
            xs.extend([et, xt, None])
            ys.extend([t["entry_price"], t["exit_price"], None])
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    line=dict(
                        color=_GREEN if is_win else _RED,
                        width=1.5, dash="dot",
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )

    # SL / TP horizontal shapes for every trade
    for t in sorted_trades:
        et = _ts(t.get("entry_time"))
        xt = _ts(t.get("exit_time"))
        if et is None or xt is None:
            continue
        sl = t.get("stop_price")
        tp = t.get("initial_target")
        if sl is not None:
            fig.add_shape(
                type="line", x0=et, x1=xt, y0=sl, y1=sl,
                line=dict(color="rgba(239,83,80,0.65)", width=1, dash="dash"),
                row=1, col=1,
            )
        if tp is not None:
            fig.add_shape(
                type="line", x0=et, x1=xt, y0=tp, y1=tp,
                line=dict(color="rgba(38,166,154,0.65)", width=1, dash="dash"),
                row=1, col=1,
            )

    # Entry markers
    fig.add_trace(
        go.Scatter(
            x=entry_times,
            y=[t["entry_price"] for t in sorted_trades],
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
                    t.get("n_contracts", 0),
                    t.get("stop_price", 0),
                    t.get("initial_target", 0),
                    "RTH" if t.get("is_rth", True) else "ETH",
                ]
                for t in sorted_trades
            ],
            hovertemplate=(
                "<b>ENTRY %{customdata[0]}</b><br>"
                "Price : %{y:.2f}<br>"
                "Lots  : %{customdata[1]}<br>"
                "SL    : %{customdata[2]:.2f}<br>"
                "TP    : %{customdata[3]:.2f}<br>"
                "Sess  : %{customdata[4]}<extra></extra>"
            ),
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Exit markers
    fig.add_trace(
        go.Scatter(
            x=[_ts(t.get("exit_time")) for t in sorted_trades],
            y=[t["exit_price"] for t in sorted_trades],
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
                ]
                for t in sorted_trades
            ],
            hovertemplate=(
                "<b>EXIT %{customdata[3]}</b><br>"
                "Price  : %{y:.2f}<br>"
                "PnL $  : $%{customdata[0]:+,.0f}<br>"
                "PnL R  : %{customdata[1]:+.3f}<br>"
                "Reason : %{customdata[2]}<extra></extra>"
            ),
            showlegend=True,
        ),
        row=1, col=1,
    )

    # ── Row 2: Cumulative PnL ─────────────────────────────────────────────────
    pnl_color   = _GREEN if (float(cum_pnl[-1]) if n_trades else 0) >= 0 else _RED
    pnl_fill    = "rgba(38,166,154,0.12)" if pnl_color == _GREEN else "rgba(239,83,80,0.12)"
    fig.add_trace(
        go.Scatter(
            x=x_seq, y=cum_pnl.tolist(),
            mode="lines",
            line=dict(color=pnl_color, width=2),
            fill="tozeroy",
            fillcolor=pnl_fill,
            hovertemplate="Trade %{x}<br>Cum PnL: $%{y:+,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line=dict(color=_GRID, width=1), row=2, col=1)

    # ── Row 3: Drawdown ───────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x_seq, y=dd.tolist(),
            mode="lines",
            line=dict(color=_RED, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.15)",
            hovertemplate="Trade %{x}<br>Drawdown: $%{y:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line=dict(color=_GRID, width=1, dash="dot"), row=3, col=1)

    # ── Row 4: Trade table ────────────────────────────────────────────────────
    tbl_hdrs = [
        "#", "Date", "Dir", "Entry", "SL", "TP", "Exit",
        "Lots", "PnL($)", "PnL(R)", "W/L", "Sess", "Dur(m)", "Reason",
    ]
    tbl_cols_data = [
        [i + 1                                                  for i, _ in enumerate(sorted_trades)],
        [t.get("date", "")                                      for t in sorted_trades],
        [t.get("direction", "")                                 for t in sorted_trades],
        [f"{t.get('entry_price', 0):.2f}"                      for t in sorted_trades],
        [f"{t.get('stop_price', 0):.2f}"                       for t in sorted_trades],
        [f"{t.get('initial_target', 0):.2f}"                   for t in sorted_trades],
        [f"{t.get('exit_price', 0):.2f}"                       for t in sorted_trades],
        [t.get("n_contracts", 0)                                for t in sorted_trades],
        [f"${t.get('pnl_dollars', t.get('pnl_r', 0)):+,.0f}"  for t in sorted_trades],
        [f"{t.get('pnl_r', 0):+.3f}"                           for t in sorted_trades],
        ["Win" if t.get("is_win") else "Loss"                   for t in sorted_trades],
        ["RTH" if t.get("is_rth", True) else "ETH"             for t in sorted_trades],
        [f"{t.get('duration_min', 0):.0f}"                     for t in sorted_trades],
        [t.get("exit_reason", "")                               for t in sorted_trades],
    ]

    row_fills = [
        "rgba(38,166,154,0.08)" if t.get("is_win") else "rgba(239,83,80,0.08)"
        for t in sorted_trades
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in tbl_hdrs],
                fill_color=_PAPER,
                font=dict(color=_TEXT, size=10),
                align="center",
                line_color=_GRID,
                height=26,
            ),
            cells=dict(
                values=tbl_cols_data,
                fill_color=[row_fills] * len(tbl_hdrs),
                font=dict(color=_TEXT, size=10),
                align=["center"] * len(tbl_hdrs),
                line_color=_GRID,
                height=21,
            ),
        ),
        row=4, col=1,
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    total_pnl = float(cum_pnl[-1]) if n_trades else 0.0
    max_dd    = float(-dd.min())   if n_trades else 0.0
    n_wins    = sum(1 for t in sorted_trades if t.get("is_win"))
    wr_pct    = n_wins / n_trades * 100 if n_trades else 0.0
    clr       = _GREEN if total_pnl >= 0 else _RED

    title_str = (
        f"{title_prefix}  —  {n_trades} trades  |  WR {wr_pct:.0f}%  |  "
        f"<span style='color:{clr}'>PnL ${total_pnl:+,.0f}</span>  |  "
        f"Max DD ${max_dd:,.0f}"
    )

    fig.update_layout(
        title=dict(text=title_str, font=dict(size=14, color=_TEXT), x=0.5),
        paper_bgcolor=_PAPER,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="monospace"),
        showlegend=True,
        legend=dict(
            orientation="h", x=0.0, y=1.01,
            font=dict(size=10, color=_TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
        dragmode="pan",
        margin=dict(l=70, r=20, t=80, b=20),
        height=1650,
        # OHLC x-axis (xaxis)
        xaxis=dict(
            gridcolor=_GRID, zeroline=False,
            rangeslider=dict(visible=True, thickness=0.035, bgcolor=_PAPER),
            rangebreaks=[dict(bounds=["sat", "mon"])],   # hide weekends
        ),
        xaxis2=dict(gridcolor=_GRID, zeroline=False, title="Trade #"),
        xaxis3=dict(gridcolor=_GRID, zeroline=False, title="Trade #"),
        yaxis=dict(gridcolor=_GRID, zeroline=False, fixedrange=False),
        yaxis2=dict(gridcolor=_GRID, zeroline=False, tickprefix="$"),
        yaxis3=dict(gridcolor=_GRID, zeroline=False, tickprefix="$"),
    )

    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["pan2d", "zoom2d", "drawline", "drawopenpath"],
        },
    )
