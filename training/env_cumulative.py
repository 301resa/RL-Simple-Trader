"""
training/env_cumulative.py
===========================
Per-environment cumulative statistics accumulator.

Aggregates every completed episode for a single training env since
training started.  All ratio metrics (WR, PF, Sharpe, RR) are
recomputed from running sums so they always reflect the full history —
never a rolling slice.

Used by:
  - MetricsPrinterCallback  (display table)
  - TrainingHotSaveCallback (gate checks)
"""

from __future__ import annotations

import numpy as np


class EnvCumulative:
    """Running totals for one env across the full training run."""

    __slots__ = (
        "n_trades", "n_wins", "n_losses",
        "n_episodes",                           # completed episodes — used for Tr/wk
        "total_pnl_r", "total_pnl_dollars",
        "gross_win_r", "gross_loss_r",
        "gross_win_dollars", "gross_loss_dollars",
        "rth_trades", "rth_wins", "eth_trades", "eth_wins",
        # RTH/ETH breakdown — for PF, RR and Dur per session type
        "rth_gross_win_r", "rth_gross_loss_r", "rth_n_losses",
        "eth_gross_win_r", "eth_gross_loss_r", "eth_n_losses",
        "rth_total_duration", "rth_n_dur",
        "eth_total_duration", "eth_n_dur",
        "max_win_dollars", "max_loss_dollars",
        "total_duration_minutes",
        "min_duration_minutes", "max_duration_minutes",
        # Welford-style running moments for Sharpe (O(1) per episode)
        "sum_r", "sum_r_sq", "n_r",
        # Running equity curve for drawdown
        "running_pnl_dollars", "equity_peak_dollars", "max_drawdown_dollars",
    )

    def __init__(self) -> None:
        self.n_trades = self.n_wins = self.n_losses = 0
        self.n_episodes = 0
        self.total_pnl_r = self.total_pnl_dollars = 0.0
        self.gross_win_r = self.gross_loss_r = 0.0
        self.gross_win_dollars = self.gross_loss_dollars = 0.0
        self.rth_trades = self.rth_wins = 0
        self.eth_trades = self.eth_wins = 0
        self.rth_gross_win_r  = self.rth_gross_loss_r = 0.0
        self.rth_n_losses     = 0
        self.eth_gross_win_r  = self.eth_gross_loss_r = 0.0
        self.eth_n_losses     = 0
        self.rth_total_duration = self.rth_n_dur = 0.0
        self.eth_total_duration = self.eth_n_dur = 0.0
        self.max_win_dollars  = 0.0
        self.max_loss_dollars = 0.0
        self.total_duration_minutes = 0.0
        self.min_duration_minutes   = float("inf")
        self.max_duration_minutes   = 0.0
        self.sum_r    = self.sum_r_sq = 0.0
        self.n_r      = 0
        self.running_pnl_dollars   = 0.0
        self.equity_peak_dollars   = 0.0
        self.max_drawdown_dollars  = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, ep: dict) -> None:
        """Accumulate one completed episode's info dict."""
        self.n_episodes += 1          # count every session, even zero-trade ones
        n  = ep.get("n_trades",  0)
        nw = ep.get("n_wins",    0)
        nl = ep.get("n_losses",  0)
        if n == 0:
            return

        self.n_trades  += n
        self.n_wins    += nw
        self.n_losses  += nl

        pnl_r = ep.get("total_pnl_r",       0.0)
        pnl_d = ep.get("total_pnl_dollars",  0.0)
        self.total_pnl_r       += pnl_r
        self.total_pnl_dollars += pnl_d

        aw   = ep.get("avg_win_r",       0.0)
        al   = ep.get("avg_loss_r",      0.0)
        aw_d = ep.get("avg_win_dollars",  0.0)
        al_d = ep.get("avg_loss_dollars", 0.0)
        self.gross_win_r        += aw   * nw
        self.gross_loss_r       += al   * nl
        self.gross_win_dollars  += aw_d * nw
        self.gross_loss_dollars += al_d * nl

        self.rth_trades += ep.get("rth_trades", 0)
        self.rth_wins   += ep.get("rth_wins",   0)
        self.eth_trades += ep.get("eth_trades", 0)
        self.eth_wins   += ep.get("eth_wins",   0)

        # RTH/ETH PF, RR, Dur from per-trade list (O(trades_per_episode))
        for t in ep.get("trades_list", []):
            t_pnl_r = float(t.get("pnl_r", 0.0))
            t_dur   = float(t.get("duration_min", 0.0))
            t_win   = bool(t.get("is_win", False))
            if t.get("is_rth", True):
                if t_win:
                    self.rth_gross_win_r += t_pnl_r
                else:
                    self.rth_gross_loss_r += abs(t_pnl_r)
                    self.rth_n_losses     += 1
                self.rth_total_duration += t_dur
                self.rth_n_dur          += 1
            else:
                if t_win:
                    self.eth_gross_win_r += t_pnl_r
                else:
                    self.eth_gross_loss_r += abs(t_pnl_r)
                    self.eth_n_losses     += 1
                self.eth_total_duration += t_dur
                self.eth_n_dur          += 1

        mw = ep.get("max_win_dollars",  0.0)
        ml = ep.get("max_loss_dollars", 0.0)
        if mw > self.max_win_dollars:
            self.max_win_dollars = mw
        if ml < self.max_loss_dollars:
            self.max_loss_dollars = ml

        dur = ep.get("avg_duration_minutes", 0.0)
        mn  = ep.get("min_duration_minutes", dur)
        mx  = ep.get("max_duration_minutes", dur)
        self.total_duration_minutes += dur * n
        if mn < self.min_duration_minutes:
            self.min_duration_minutes = mn
        if mx > self.max_duration_minutes:
            self.max_duration_minutes = mx

        # Running Sharpe: accumulate episode-level PnL in O(1).
        # One observation per episode avoids inflating Sharpe by treating all
        # trades within an episode as independent identical-return observations.
        self.sum_r    += pnl_r
        self.sum_r_sq += pnl_r * pnl_r
        self.n_r      += 1

        # Running drawdown (equity curve from episode-level PnL)
        self.running_pnl_dollars += pnl_d
        if self.running_pnl_dollars > self.equity_peak_dollars:
            self.equity_peak_dollars = self.running_pnl_dollars
        dd = self.equity_peak_dollars - self.running_pnl_dollars
        if dd > self.max_drawdown_dollars:
            self.max_drawdown_dollars = dd

    def to_info_dict(
        self,
        initial_capital: float = 2500.0,
        n_training_days: int   = 0,
    ) -> dict:
        """
        Return a dict with the same keys as a trading episode info dict,
        computed from all accumulated history.  Returns empty dict if no trades.

        n_training_days : total trading days in the configured date range.
            When provided, Tr/wk = n_trades / (n_training_days / 5).
            Falls back to elapsed episodes when 0 (avoids import dependency).
        """
        n  = self.n_trades
        nw = self.n_wins
        nl = self.n_losses
        if n == 0:
            return {}

        win_rate  = nw / n
        pf        = min(self.gross_win_r / max(self.gross_loss_r, 1e-6), 99.99)
        avg_win_r = self.gross_win_r  / nw if nw else 0.0
        avg_los_r = self.gross_loss_r / nl if nl else 1.0
        win_loss_ratio = avg_win_r / max(avg_los_r, 1e-6)

        avg_win_d = self.gross_win_dollars  / nw if nw else 0.0
        avg_los_d = self.gross_loss_dollars / nl if nl else 0.0
        # avg_loss_dollars is negative — use abs() so RR is always positive
        avg_rr    = abs(avg_win_d) / max(abs(avg_los_d), 1e-6) if avg_los_d else 0.0
        avg_dur   = self.total_duration_minutes / n
        min_dur   = self.min_duration_minutes if self.min_duration_minutes != float("inf") else 0.0

        if self.n_r >= 5:
            mean_r = self.sum_r / self.n_r
            var_r  = max(self.sum_r_sq / self.n_r - mean_r ** 2, 0.0)
            std_r  = float(np.sqrt(var_r))
            sharpe = float(np.clip(mean_r / std_r * np.sqrt(252), -9.99, 9.99)) if std_r > 0.01 else 0.0
        else:
            sharpe = 0.0

        max_dd_pct = (
            self.max_drawdown_dollars / initial_capital * 100
            if initial_capital > 0 else 0.0
        )

        # RTH breakdown
        rth_nl = self.rth_n_losses
        rth_nw = self.rth_wins
        rth_pf = min(self.rth_gross_win_r / max(self.rth_gross_loss_r, 1e-6), 99.99)
        rth_aw = self.rth_gross_win_r  / max(rth_nw, 1)
        rth_al = self.rth_gross_loss_r / max(rth_nl, 1)
        rth_rr = rth_aw / max(rth_al, 1e-6) if rth_nl > 0 else 0.0
        rth_dur = self.rth_total_duration / max(self.rth_n_dur, 1)

        # ETH breakdown
        eth_nl = self.eth_n_losses
        eth_nw = self.eth_wins
        eth_pf = min(self.eth_gross_win_r / max(self.eth_gross_loss_r, 1e-6), 99.99)
        eth_aw = self.eth_gross_win_r  / max(eth_nw, 1)
        eth_al = self.eth_gross_loss_r / max(eth_nl, 1)
        eth_rr = eth_aw / max(eth_al, 1e-6) if eth_nl > 0 else 0.0
        eth_dur = self.eth_total_duration / max(self.eth_n_dur, 1)

        # Trades per week normalised by the FULL training range, not elapsed episodes.
        # E.g. 98 trades over 6 years (312 weeks) = 0.31 trades/week.
        weeks = (n_training_days / 5.0) if n_training_days > 0 else max(self.n_episodes / 5.0, 1.0)
        trades_per_week = n / max(weeks, 1.0)

        return {
            "n_trades":             n,
            "n_wins":               nw,
            "n_losses":             nl,
            "trades_per_week":      trades_per_week,
            "win_rate":             win_rate,
            "total_pnl_r":          self.total_pnl_r,
            "total_pnl_dollars":    self.total_pnl_dollars,
            "profit_factor":        pf,
            "avg_win_r":            avg_win_r,
            "avg_loss_r":           avg_los_r,
            "win_loss_ratio":       win_loss_ratio,
            "sharpe_ratio":         sharpe,
            "max_drawdown_pct":     max_dd_pct,
            "rth_trades":           self.rth_trades,
            "rth_wins":             self.rth_wins,
            "rth_pf":               rth_pf,
            "rth_rr":               rth_rr,
            "rth_avg_duration":     rth_dur,
            "eth_trades":           self.eth_trades,
            "eth_wins":             self.eth_wins,
            "eth_pf":               eth_pf,
            "eth_rr":               eth_rr,
            "eth_avg_duration":     eth_dur,
            "max_win_dollars":      self.max_win_dollars,
            "max_loss_dollars":     self.max_loss_dollars,
            "avg_win_dollars":      avg_win_d,
            "avg_loss_dollars":     avg_los_d,
            "avg_rr":               avg_rr,
            "avg_duration_minutes": avg_dur,
            "min_duration_minutes": min_dur,
            "max_duration_minutes": self.max_duration_minutes,
        }
