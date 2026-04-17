"""
evaluation/metrics_calculator.py
==================================
Computes performance metrics from a completed trade journal DataFrame.

Metrics produced:
  - win_rate, total_trades, total_pnl_r
  - avg_win_r, avg_loss_r, profit_factor
  - sharpe_ratio (daily R-multiple returns)
  - max_consecutive_losses, max_drawdown_r
  - overfitting_score (train vs test comparison)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class MetricsCalculator:
    """
    Computes trading performance metrics from a journal DataFrame.

    The DataFrame is expected to have at minimum these columns
    (as written by TradeJournal.to_dataframe()):
      pnl_r, direction, exit_reason, entry_timestamp
    """

    def compute_from_dataframe(self, df: pd.DataFrame) -> Dict:
        """
        Compute all metrics from a trade journal DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            One row per completed trade.  Must contain column 'pnl_r'.

        Returns
        -------
        dict of metric name → value.
        """
        if df.empty or "pnl_r" not in df.columns:
            return self._empty_metrics()

        pnl = df["pnl_r"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]

        total_trades = len(pnl)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        total_pnl_r = float(pnl.sum())
        avg_win_r = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss_r = float(abs(losses.mean())) if len(losses) > 0 else 0.0
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1e-6
        profit_factor = gross_profit / gross_loss

        sharpe = self._sharpe_ratio(df)
        max_consec_losses = self._max_consecutive_losses(pnl.tolist())
        max_dd_r = self._max_drawdown_r(pnl.tolist())

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "total_pnl_r": round(total_pnl_r, 4),
            "avg_win_r": round(avg_win_r, 4),
            "avg_loss_r": round(avg_loss_r, 4),
            "profit_factor": round(profit_factor, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_consecutive_losses": max_consec_losses,
            "max_drawdown_r": round(max_dd_r, 4),
        }

    def overfitting_score(
        self, train_metrics: dict, test_metrics: dict
    ) -> dict:
        """
        Compare training vs test performance to assess overfitting.

        Returns degradation ratios and an overfitting_flag.

        A win-rate drop > 20pp or profit_factor dropping below 1.0
        while train profit_factor > 2.0 are red flags (roadmap Phase 10).
        """
        train_wr = train_metrics.get("win_rate", 0.0)
        test_wr = test_metrics.get("win_rate", 0.0)
        train_pf = train_metrics.get("profit_factor", 1.0)
        test_pf = test_metrics.get("profit_factor", 1.0)

        wr_drop = train_wr - test_wr
        pf_ratio = test_pf / max(train_pf, 1e-6)

        overfitting_flag = (
            (wr_drop > 0.20 and test_wr < 0.35)
            or (train_pf > 2.0 and test_pf < 1.0)
        )

        return {
            "train_win_rate": train_wr,
            "test_win_rate": test_wr,
            "win_rate_drop": round(wr_drop, 4),
            "train_profit_factor": train_pf,
            "test_profit_factor": test_pf,
            "profit_factor_ratio": round(pf_ratio, 4),
            "overfitting_flag": overfitting_flag,
        }

    # ── Private helpers ───────────────────────────────────────

    def _sharpe_ratio(self, df: pd.DataFrame) -> float:
        """Annualised Sharpe ratio of daily R-multiple returns (×√252)."""
        if "entry_timestamp" not in df.columns or "pnl_r" not in df.columns:
            return 0.0
        try:
            df2 = df.copy()
            df2["date"] = pd.to_datetime(df2["entry_timestamp"]).dt.date
            daily = df2.groupby("date")["pnl_r"].sum()
            if len(daily) < 2:
                return 0.0
            mean_r = daily.mean()
            std_r = daily.std()
            return float(mean_r / std_r * np.sqrt(252)) if std_r > 1e-9 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _max_consecutive_losses(pnl_list: List[float]) -> int:
        """Count the longest consecutive losing streak."""
        max_streak = current = 0
        for r in pnl_list:
            if r <= 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @staticmethod
    def _max_drawdown_r(pnl_list: List[float]) -> float:
        """Peak-to-trough drawdown in R-multiples (positive number)."""
        if not pnl_list:
            return 0.0
        equity = np.cumsum(pnl_list)
        peak = np.maximum.accumulate(equity)
        return float((peak - equity).max())

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_consecutive_losses": 0,
            "max_drawdown_r": 0.0,
        }
