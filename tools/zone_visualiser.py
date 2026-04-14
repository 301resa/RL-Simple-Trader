"""
tools/zone_visualiser.py
=========================
Visual zone quality audit tool.

Runs the live ZoneDetector (exactly as used during training — no lookahead)
over a sample of trading days and produces a single Plotly HTML report with
one candlestick chart per day.  Each chart shows:

  - Green rectangles  : demand zones (valid when formed)
  - Red rectangles    : supply zones (valid when formed)
  - Faded rectangles  : zones that were invalidated before EOD
  - Orange dashed line: zone midpoint (reference only — entry is now at edge)
  - Zone edge markers : the actual limit-order entry price (bottom / top)

Usage
-----
    conda run -n base python tools/zone_visualiser.py
    conda run -n base python tools/zone_visualiser.py --days 20 --out logs/zones.html
    conda run -n base python tools/zone_visualiser.py --date 2024-11-15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.data_loader import DataLoader
from features.atr_calculator import ATRCalculator
from features.zone_detector import ZoneDetector, ZoneType

# ── Config mirrors features_config.yaml ──────────────────────────────────────
_ZONE_CFG = dict(
    consolidation_min_bars=2,
    consolidation_max_bars=10,
    consolidation_range_atr_pct=0.08,
    impulse_min_body_atr_pct=0.08,
    max_zone_touches=3,
    max_zone_age_bars=300,
    zone_buffer_atr_pct=0.03,
)
_MAX_ZONE_WIDTH = 10.0   # zones wider than this are skipped at entry
_ZONE_LOOKBACK  = 500    # prior-session bars fed to detector on reset


def _build_atr_series(bars: pd.DataFrame, atr_calc: ATRCalculator) -> pd.Series:
    """Return a bar-aligned ATR series for the given bars DataFrame."""
    date_strs = bars.index.strftime("%Y-%m-%d")
    return pd.Series(
        [atr_calc.get_atr_for_date(d) for d in date_strs],
        index=bars.index,
    )


def _run_zone_detection(
    combined_bars: pd.DataFrame,
    session_start_idx: int,
    atr_series: pd.Series,
) -> list[dict]:
    """
    Run ZoneDetector bar-by-bar over the session portion of combined_bars.
    Returns a list of zone event dicts for plotting.

    Each event:
        top, bottom, zone_type, formed_bar, is_valid_eod, width
    """
    detector = ZoneDetector(**_ZONE_CFG)

    # Scan all bars up to the start of the session window to warm up
    # the detector with prior-session history — same as TradingEnv does.
    for i in range(session_start_idx):
        detector.scan_and_update(combined_bars, atr_series, i)

    # Track every zone snapshot at the bar it was first detected
    seen_zones: dict[int, dict] = {}  # id(zone) → dict

    for i in range(session_start_idx, len(combined_bars)):
        state = detector.scan_and_update(combined_bars, atr_series, i)

        # Capture supply zones
        for z in detector._supply_zones:
            zid = id(z)
            if zid not in seen_zones:
                seen_zones[zid] = {
                    "top": z.top,
                    "bottom": z.bottom,
                    "zone_type": z.zone_type,
                    "formed_bar": z.bar_formed_idx,
                    "is_valid_eod": z.is_valid,
                    "width": z.top - z.bottom,
                    "_zone_ref": z,
                }

        # Capture demand zones
        for z in detector._demand_zones:
            zid = id(z)
            if zid not in seen_zones:
                seen_zones[zid] = {
                    "top": z.top,
                    "bottom": z.bottom,
                    "zone_type": z.zone_type,
                    "formed_bar": z.bar_formed_idx,
                    "is_valid_eod": z.is_valid,
                    "width": z.top - z.bottom,
                    "_zone_ref": z,
                }

    # Update final validity from the live zone references
    for ev in seen_zones.values():
        ev["is_valid_eod"] = ev["_zone_ref"].is_valid
        del ev["_zone_ref"]

    return list(seen_zones.values())


def _zone_count_series(
    combined_bars: pd.DataFrame,
    session_start_idx: int,
    atr_series: pd.Series,
) -> tuple[list[int], list[int]]:
    """
    Run detector bar-by-bar and return (supply_counts, demand_counts)
    for each session bar — used for the zone count sub-trace.
    """
    detector = ZoneDetector(**_ZONE_CFG)
    for i in range(session_start_idx):
        detector.scan_and_update(combined_bars, atr_series, i)

    supply_counts, demand_counts = [], []
    for i in range(session_start_idx, len(combined_bars)):
        detector.scan_and_update(combined_bars, atr_series, i)
        supply_counts.append(sum(1 for z in detector._supply_zones if z.is_valid))
        demand_counts.append(sum(1 for z in detector._demand_zones if z.is_valid))

    return supply_counts, demand_counts


def _build_chart(
    date: str,
    session_bars: pd.DataFrame,
    combined_bars: pd.DataFrame,
    session_start_idx: int,
    atr_series: pd.Series,
    daily_atr: float,
) -> go.Figure:
    """Build a single-day zone chart."""
    zones = _run_zone_detection(combined_bars, session_start_idx, atr_series)
    supply_cnts, demand_cnts = _zone_count_series(combined_bars, session_start_idx, atr_series)

    session_times = session_bars.index
    t_start = session_times[0]
    t_end   = session_times[-1]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.80, 0.20],
        vertical_spacing=0.02,
    )

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=session_times,
            open=session_bars["open"],
            high=session_bars["high"],
            low=session_bars["low"],
            close=session_bars["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            name="Price",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ── Zone rectangles ───────────────────────────────────────────────────────
    supply_zones_valid = 0
    demand_zones_valid = 0
    wide_zones = 0

    for z in zones:
        is_supply  = z["zone_type"] == ZoneType.SUPPLY
        is_valid   = z["is_valid_eod"]
        is_wide    = z["width"] > _MAX_ZONE_WIDTH

        if is_wide:
            wide_zones += 1

        # Colour coding:
        #   valid supply  → semi-transparent red
        #   valid demand  → semi-transparent green
        #   invalidated   → grey (faded, dashed border)
        #   too wide      → orange (agent would skip)
        if is_wide:
            fill  = "rgba(255, 165, 0, 0.10)"
            line  = "rgba(255, 165, 0, 0.50)"
        elif is_supply and is_valid:
            fill  = "rgba(239, 83, 80, 0.15)"
            line  = "rgba(239, 83, 80, 0.60)"
            supply_zones_valid += 1
        elif not is_supply and is_valid:
            fill  = "rgba(38, 166, 154, 0.15)"
            line  = "rgba(38, 166, 154, 0.60)"
            demand_zones_valid += 1
        else:
            fill  = "rgba(150, 150, 150, 0.06)"
            line  = "rgba(150, 150, 150, 0.25)"

        # Rectangle spanning full session width at zone price levels
        fig.add_shape(
            type="rect",
            x0=t_start, x1=t_end,
            y0=z["bottom"], y1=z["top"],
            fillcolor=fill,
            line=dict(color=line, width=1),
            row=1, col=1,
        )

        # Zone label — placed at the right edge of the chart, vertically centred in the zone
        label_text = (
            f"{'SUPPLY' if is_supply else 'DEMAND'}  {z['bottom']:.2f}–{z['top']:.2f}"
            + ("  [WIDE]" if is_wide else "")
            + ("  [INVALID]" if not is_valid and not is_wide else "")
        )
        label_color = (
            "rgba(255,165,0,0.85)"   if is_wide  else
            "rgba(239,83,80,0.90)"   if is_supply and is_valid else
            "rgba(38,166,154,0.90)"  if not is_supply and is_valid else
            "rgba(160,160,160,0.60)"
        )
        fig.add_annotation(
            x=t_end,
            y=(z["top"] + z["bottom"]) / 2,
            text=label_text,
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font=dict(size=10, color=label_color),
            bgcolor="rgba(22,33,62,0.70)",
            borderpad=2,
            row=1, col=1,
        )

        # Entry line + label — the exact limit-order price the agent would use
        if not is_wide and is_valid:
            entry_y = z["top"] if is_supply else z["bottom"]
            entry_color = "rgba(239,83,80,1.0)" if is_supply else "rgba(38,166,154,1.0)"

            # Solid bright entry line (thicker than zone borders)
            fig.add_shape(
                type="line",
                x0=t_start, x1=t_end,
                y0=entry_y, y1=entry_y,
                line=dict(color=entry_color, width=2.0, dash="dashdot"),
                row=1, col=1,
            )

            # Arrow + label pinned to the LEFT edge of the session
            entry_label = f"ENTRY @ {entry_y:.2f}"
            fig.add_annotation(
                x=t_start,
                y=entry_y,
                text=entry_label,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=entry_color,
                ax=40,          # arrow tip offset right (pixels) → label sits left of tip
                ay=0,
                xanchor="right",
                yanchor="middle",
                font=dict(size=10, color="#ffffff", family="monospace"),
                bgcolor=entry_color.replace("1.0", "0.80"),
                borderpad=3,
                row=1, col=1,
            )

    # ── Zone count sub-chart ──────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=session_times,
            y=supply_cnts,
            name="Supply zones",
            line=dict(color="#ef5350", width=1.5),
            mode="lines",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=session_times,
            y=demand_cnts,
            name="Demand zones",
            line=dict(color="#26a69a", width=1.5),
            mode="lines",
        ),
        row=2, col=1,
    )

    n_total  = len(zones)
    n_skipped = wide_zones
    title_txt = (
        f"{date}  |  ATR={daily_atr:.1f} pts  |  "
        f"Zones detected: {n_total} "
        f"(supply={supply_zones_valid} valid, demand={demand_zones_valid} valid, "
        f"wide/skipped={n_skipped})"
    )

    fig.update_layout(
        title=dict(text=title_txt, font=dict(size=13)),
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        height=700,
        margin=dict(l=60, r=20, t=60, b=40),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    fig.update_yaxes(gridcolor="#2a2a4a", row=1, col=1)
    fig.update_yaxes(gridcolor="#2a2a4a", title_text="# Zones", row=2, col=1)
    fig.update_xaxes(gridcolor="#2a2a4a")

    return fig


def main(n_days: int = 15, out_path: str = "logs/zone_audit.html", target_date: str = None) -> None:
    """Run zone audit and write HTML report."""
    print("Loading data...")
    loader = DataLoader(str(ROOT / "data"), instrument="ES")
    loader.load()

    atr_calc = ATRCalculator(atr_period=14, exhaustion_threshold=0.85)
    atr_calc.fit(loader.daily)

    all_days = loader.get_trading_days()
    print(f"  {len(all_days)} trading days available ({all_days[0]} → {all_days[-1]})")

    if target_date:
        if target_date not in all_days:
            print(f"ERROR: date {target_date!r} not found. Available range: {all_days[0]}–{all_days[-1]}")
            sys.exit(1)
        sample_days = [target_date]
    else:
        # Sample evenly across the dataset so we audit early, mid, and recent zones
        step = max(1, len(all_days) // n_days)
        sample_days = all_days[::step][:n_days]
        print(f"  Sampling {len(sample_days)} days (every {step} days)")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Collect all figures
    figs: list[tuple[str, go.Figure]] = []
    for i, date in enumerate(sample_days):
        print(f"  [{i+1:02d}/{len(sample_days)}] {date} ...", end=" ", flush=True)

        session_bars = loader.get_day_bars(date)
        if session_bars.empty:
            print("no bars — skipped")
            continue

        prior_bars = loader.get_bars_before(date, _ZONE_LOOKBACK)
        combined   = pd.concat([prior_bars, session_bars]).sort_index()
        session_start_idx = len(prior_bars)

        atr_ser = _build_atr_series(combined, atr_calc)
        daily_atr = atr_calc.get_atr_for_date(date)

        fig = _build_chart(
            date=date,
            session_bars=session_bars,
            combined_bars=combined,
            session_start_idx=session_start_idx,
            atr_series=atr_ser,
            daily_atr=daily_atr,
        )
        figs.append((date, fig))
        print("done")

    if not figs:
        print("No charts generated — nothing to write.")
        sys.exit(1)

    # ── Write single-file HTML with a simple tab selector ────────────────────
    print(f"\nWriting {len(figs)} charts to {out} ...")

    html_parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Zone Quality Audit — ES 5-min</title>
<style>
  body { margin:0; background:#1a1a2e; color:#e0e0e0; font-family:sans-serif; }
  #tab-bar { display:flex; flex-wrap:wrap; padding:8px 12px; gap:6px; background:#0f3460; }
  .tab-btn {
    padding:5px 12px; border:none; border-radius:4px; cursor:pointer;
    background:#16213e; color:#e0e0e0; font-size:12px;
  }
  .tab-btn.active { background:#e94560; color:#fff; }
  .tab-content { display:none; }
  .tab-content.active { display:block; }
  h3 { padding:4px 12px; margin:0; font-size:13px; color:#aaa; }
</style>
</head>
<body>
<div id="tab-bar">
"""]

    for idx, (date, _) in enumerate(figs):
        active = " active" if idx == 0 else ""
        html_parts.append(
            f'  <button class="tab-btn{active}" onclick="showTab({idx})">{date}</button>\n'
        )

    html_parts.append("</div>\n")

    for idx, (date, fig) in enumerate(figs):
        active = " active" if idx == 0 else ""
        inner = fig.to_html(full_html=False, include_plotlyjs=("cdn" if idx == 0 else False))
        html_parts.append(f'<div class="tab-content{active}" id="tab-{idx}">\n{inner}\n</div>\n')

    html_parts.append("""<script>
function showTab(i) {
  document.querySelectorAll('.tab-content').forEach((el,j) => {
    el.classList.toggle('active', j === i);
  });
  document.querySelectorAll('.tab-btn').forEach((el,j) => {
    el.classList.toggle('active', j === i);
  });
}
</script>
</body></html>""")

    out.write_text("".join(html_parts), encoding="utf-8")
    print(f"Done. Open in browser:\n  {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zone quality visualiser")
    parser.add_argument("--days",  type=int, default=15,
                        help="Number of days to sample across the dataset (default 15)")
    parser.add_argument("--out",   type=str, default="logs/zone_audit.html",
                        help="Output HTML path (default logs/zone_audit.html)")
    parser.add_argument("--date",  type=str, default=None,
                        help="Audit a single specific date (YYYY-MM-DD)")
    args = parser.parse_args()
    main(n_days=args.days, out_path=args.out, target_date=args.date)
