"""
evaluation/trade_journal.py
============================
Trade Journal — records every trade and provides analysis tools.

This implements Mistake #7 from the video:
  "Without screenshots, you'll repeat the mistakes indefinitely."

The journal logs every completed trade with full metadata so you can:
  - Identify which rule violations caused the most losses
  - Compute win rates broken down by confluence factor
  - Track improvement over training iterations
  - Export to CSV for external analysis

All data is persisted to disk after each episode so nothing is lost
on crash or early termination.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from environment.position_manager import Trade
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class JournalEntry:
    """
    Full journal record for one completed trade.

    Captures both execution data and the confluence context
    at the time of entry — enabling post-hoc analysis of
    which conditions predicted success vs failure.
    """
    # Identity
    episode_date: str
    trade_number: int                   # Sequential trade number (global)
    episode_trade_number: int           # Trade number within the episode

    # Execution
    direction: str                      # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    stop_price: float
    initial_target: float
    n_contracts: int
    entry_bar_idx: int
    exit_bar_idx: int
    exit_reason: str
    duration_bars: int

    # Outcome
    pnl_r: float                        # R-multiples
    pnl_points: float
    pnl_dollars: float
    is_win: bool

    # Excursion data
    max_adverse_excursion_r: float      # Worst drawdown during trade (in R)
    max_favourable_excursion_r: float   # Best unrealised gain during trade (in R)

    # Entry context (confluence state at entry)
    trend_state: str
    in_supply_demand_zone: bool
    in_order_zone: bool
    confluence_score: float
    liquidity_sweep_present: bool
    rejection_candle_present: bool
    atr_pct_used_at_entry: float
    rr_ratio_at_entry: float

    # Rule compliance flags
    trend_aligned: bool
    atr_gate_respected: bool            # ATR not exhausted at entry
    min_rr_respected: bool              # R:R >= 4:1 at entry
    order_zone_used: bool               # All 3 confluence factors present

    # Timestamp
    recorded_at: str = ""

    def __post_init__(self) -> None:
        if not self.recorded_at:
            self.recorded_at = datetime.utcnow().isoformat()


class TradeJournal:
    """
    Persistent trade journal with analysis capabilities.

    Parameters
    ----------
    journal_dir : str | Path
        Directory for journal files.
    agent_run_id : str
        Unique identifier for this training run.
    min_rr_ratio : float
        Minimum R:R threshold (for compliance tagging).
    atr_exhaustion_threshold : float
        ATR % threshold for gate compliance.
    """

    def __init__(
        self,
        journal_dir: str | Path,
        agent_run_id: str = "default",
        min_rr_ratio: float = 4.0,
        atr_exhaustion_threshold: float = 0.95,
    ) -> None:
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.agent_run_id = agent_run_id
        self.min_rr_ratio = min_rr_ratio
        self.atr_exhaustion_threshold = atr_exhaustion_threshold

        self._entries: List[JournalEntry] = []
        self._global_trade_counter: int = 0
        self._csv_path = self.journal_dir / f"{agent_run_id}_trades.csv"
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file = None
        self._csv_headers_written: bool = False

    # ── Entry recording ───────────────────────────────────────

    def record(
        self,
        trade: Trade,
        episode_date: str,
        episode_trade_number: int,
        trend_state: str,
        in_supply_demand_zone: bool,
        in_order_zone: bool,
        confluence_score: float,
        liquidity_sweep_present: bool,
        rejection_candle_present: bool,
        atr_pct_used_at_entry: float,
        rr_ratio_at_entry: float,
        peak_unrealised_r: float,
    ) -> JournalEntry:
        """
        Record a completed trade in the journal.

        Parameters
        ----------
        trade : Trade
            The completed trade from PositionManager.
        episode_date : str
        episode_trade_number : int
        trend_state : str
        in_supply_demand_zone : bool
        in_order_zone : bool
        confluence_score : float
        liquidity_sweep_present : bool
        rejection_candle_present : bool
        atr_pct_used_at_entry : float
        rr_ratio_at_entry : float
        peak_unrealised_r : float
            Maximum unrealised R during trade (MFE).

        Returns
        -------
        JournalEntry
        """
        self._global_trade_counter += 1

        entry = JournalEntry(
            episode_date=episode_date,
            trade_number=self._global_trade_counter,
            episode_trade_number=episode_trade_number,
            direction=trade.direction.name,
            entry_price=round(trade.entry_price, 4),
            exit_price=round(trade.exit_price, 4),
            stop_price=round(trade.stop_price, 4),
            initial_target=round(trade.initial_target, 4),
            n_contracts=trade.n_contracts,
            entry_bar_idx=trade.entry_bar_idx,
            exit_bar_idx=trade.exit_bar_idx,
            exit_reason=trade.exit_reason.value,
            duration_bars=trade.duration_bars,
            pnl_r=round(trade.pnl_r, 4),
            pnl_points=round(trade.pnl_points, 4),
            pnl_dollars=round(trade.pnl_dollars, 2),
            is_win=trade.is_win,
            max_adverse_excursion_r=round(trade.max_adverse_excursion, 4),
            max_favourable_excursion_r=round(peak_unrealised_r, 4),
            trend_state=trend_state,
            in_supply_demand_zone=in_supply_demand_zone,
            in_order_zone=in_order_zone,
            confluence_score=round(confluence_score, 4),
            liquidity_sweep_present=liquidity_sweep_present,
            rejection_candle_present=rejection_candle_present,
            atr_pct_used_at_entry=round(atr_pct_used_at_entry, 4),
            rr_ratio_at_entry=round(rr_ratio_at_entry, 4),
            trend_aligned=self._is_trend_aligned(trade.direction.name, trend_state),
            atr_gate_respected=atr_pct_used_at_entry < self.atr_exhaustion_threshold,
            min_rr_respected=rr_ratio_at_entry >= self.min_rr_ratio,
            order_zone_used=in_order_zone,
        )

        self._entries.append(entry)
        self._write_csv_row(entry)

        return entry

    # ── Analysis ──────────────────────────────────────────────

    def analyse(self) -> dict:
        """
        Compute comprehensive performance analysis.

        Returns a dict suitable for logging or printing.
        Breaks down win rate and average R by each confluence factor.
        """
        if not self._entries:
            return {"error": "No trades recorded yet."}

        df = self.to_dataframe()
        return {
            "summary": self._summary_stats(df),
            "by_trend_aligned": self._breakdown_by(df, "trend_aligned"),
            "by_order_zone": self._breakdown_by(df, "order_zone_used"),
            "by_atr_gate": self._breakdown_by(df, "atr_gate_respected"),
            "by_min_rr": self._breakdown_by(df, "min_rr_respected"),
            "by_zone_present": self._breakdown_by(df, "in_supply_demand_zone"),
            "by_liquidity_sweep": self._breakdown_by(df, "liquidity_sweep_present"),
            "exit_reason_breakdown": df.groupby("exit_reason")["pnl_r"].agg(
                ["count", "mean", "sum"]
            ).to_dict(),
            "top_loss_categories": self._top_loss_categories(df),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return all journal entries as a Pandas DataFrame."""
        if not self._entries:
            return pd.DataFrame()
        return pd.DataFrame([asdict(e) for e in self._entries])

    def export_csv(self, path: Optional[str] = None) -> Path:
        """Export full journal to a CSV file."""
        out_path = Path(path) if path else self._csv_path
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(out_path, index=False)
            log.info("Journal exported", path=str(out_path), n_trades=len(df))
        return out_path

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        analysis = self.analyse()
        summary = analysis.get("summary", {})
        print("\n" + "=" * 60)
        print(f"  TRADE JOURNAL SUMMARY  ({summary.get('n_trades', 0)} trades)")
        print("=" * 60)
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k:<35} {v:.4f}")
            else:
                print(f"  {k:<35} {v}")
        print("\n  WIN RATE BREAKDOWN BY FACTOR:")
        for factor in ["by_trend_aligned", "by_order_zone", "by_atr_gate", "by_min_rr"]:
            bd = analysis.get(factor, {})
            label = factor.replace("by_", "").replace("_", " ").title()
            print(f"\n  [{label}]")
            for group, stats in bd.items():
                wr = stats.get("win_rate", 0)
                ar = stats.get("avg_r", 0)
                n = stats.get("count", 0)
                print(f"    {str(group):<8} n={n:<5} win_rate={wr:.1%}  avg_r={ar:.3f}")
        print("\n  TOP LOSS CATEGORIES:")
        for cat, loss in analysis.get("top_loss_categories", {}).items():
            print(f"    {cat:<40} {loss:.4f} R")
        print("=" * 60 + "\n")

    # ── Private helpers ───────────────────────────────────────

    def _summary_stats(self, df: pd.DataFrame) -> dict:
        wins = df[df["is_win"]]
        losses = df[~df["is_win"]]
        gross_profit = wins["pnl_r"].sum()
        gross_loss = abs(losses["pnl_r"].sum())

        return {
            "n_trades": len(df),
            "win_rate": len(wins) / len(df) if len(df) > 0 else 0.0,
            "total_pnl_r": float(df["pnl_r"].sum()),
            "avg_win_r": float(wins["pnl_r"].mean()) if len(wins) > 0 else 0.0,
            "avg_loss_r": float(losses["pnl_r"].mean()) if len(losses) > 0 else 0.0,
            "profit_factor": gross_profit / max(gross_loss, 1e-6),
            "max_consecutive_losses": self._max_consecutive_losses(df),
            "avg_duration_bars": float(df["duration_bars"].mean()),
            "avg_mfe_r": float(df["max_favourable_excursion_r"].mean()),
            "avg_mae_r": float(df["max_adverse_excursion_r"].mean()),
            "pct_trades_in_order_zone": float(df["order_zone_used"].mean()),
            "pct_trades_trend_aligned": float(df["trend_aligned"].mean()),
            "pct_atr_gate_respected": float(df["atr_gate_respected"].mean()),
        }

    def _breakdown_by(self, df: pd.DataFrame, col: str) -> dict:
        result = {}
        for val, group in df.groupby(col):
            wins = group[group["is_win"]]
            result[val] = {
                "count": len(group),
                "win_rate": len(wins) / len(group),
                "avg_r": float(group["pnl_r"].mean()),
                "total_r": float(group["pnl_r"].sum()),
            }
        return result

    def _top_loss_categories(self, df: pd.DataFrame) -> dict:
        """
        Identify which rule violations caused the most total R-loss.
        This is the key output for iterative strategy improvement.
        """
        losses = df[~df["is_win"]]
        if losses.empty:
            return {}

        categories = {
            "counter_trend_entries": losses[~losses["trend_aligned"]]["pnl_r"].sum(),
            "no_order_zone": losses[~losses["order_zone_used"]]["pnl_r"].sum(),
            "atr_gate_violated": losses[~losses["atr_gate_respected"]]["pnl_r"].sum(),
            "rr_below_min": losses[~losses["min_rr_respected"]]["pnl_r"].sum(),
            "no_liquidity_sweep": losses[~losses["liquidity_sweep_present"]]["pnl_r"].sum(),
        }
        # Sort by worst (most negative) loss category
        return dict(sorted(categories.items(), key=lambda x: x[1]))

    def _max_consecutive_losses(self, df: pd.DataFrame) -> int:
        max_streak = 0
        current = 0
        for is_win in df["is_win"]:
            if not is_win:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @staticmethod
    def _is_trend_aligned(direction: str, trend_state: str) -> bool:
        bearish_trends = {"DOWNTREND", "W_DOWNTREND"}
        bullish_trends = {"UPTREND", "M_UPTREND"}
        if direction == "SHORT":
            return trend_state in bearish_trends
        elif direction == "LONG":
            return trend_state in bullish_trends
        return False

    def _write_csv_row(self, entry: JournalEntry) -> None:
        """Append a row to the persistent CSV file."""
        row = asdict(entry)
        if not self._csv_headers_written:
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
            self._csv_headers_written = True

        if self._csv_writer:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

    def __del__(self) -> None:
        if self._csv_file:
            try:
                self._csv_file.close()
            except Exception:
                pass