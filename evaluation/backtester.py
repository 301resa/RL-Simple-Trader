"""
evaluation/backtester.py
=========================
Walk-forward backtesting engine.

Runs the trained agent deterministically across a held-out
test period, recording every trade and computing performance
metrics.

Key principles:
  - Test data is NEVER seen during training
  - Agent acts deterministically (greedy policy)
  - Full trade journal is written for post-mortem analysis
  - Overfitting check compares train vs test performance
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from environment.trading_env import TradingEnv
from evaluation.metrics_calculator import MetricsCalculator
from evaluation.trade_journal import TradeJournal
from utils.logger import get_logger

log = get_logger(__name__)


class Backtester:
    """
    Runs the agent on a test environment and records results.

    Parameters
    ----------
    env : TradingEnv
        Test environment (loaded with test-period trading days).
    agent : PPOAgent
        Trained agent (loaded from checkpoint).
    journal : TradeJournal
        Journal to record trades into.
    metrics_calculator : MetricsCalculator
    n_episodes : Optional[int]
        Number of episodes to run. None = run all test days once.
    deterministic : bool
        True = greedy policy (recommended for evaluation).
    """

    def __init__(
        self,
        env: TradingEnv,
        agent: Any,
        journal: TradeJournal,
        metrics_calculator: MetricsCalculator,
        n_episodes: Optional[int] = None,
        deterministic: bool = True,
    ) -> None:
        self.env = env
        self.agent = agent
        self.journal = journal
        self.metrics_calculator = metrics_calculator
        self.n_episodes = n_episodes or len(env.trading_days)
        self.deterministic = deterministic

    def run(self) -> Dict:
        """
        Execute the backtest.

        Returns
        -------
        dict
            Full performance report including metrics and journal analysis.
        """
        log.info(
            "Backtest started",
            n_episodes=self.n_episodes,
            deterministic=self.deterministic,
        )

        episode_summaries: List[dict] = []

        for ep_idx in range(self.n_episodes):
            obs, info = self.env.reset()
            ep_date = info.get("date", f"episode_{ep_idx}")
            terminated = truncated = False
            ep_reward = 0.0
            episode_trade_count = 0

            while not (terminated or truncated):
                # Get action mask from environment
                action_mask = np.array(self.env.action_masks(), dtype=np.float32)

                # Agent predicts action (deterministic for evaluation)
                action, _ = self.agent.predict(
                    obs,
                    action_masks=action_mask,
                    deterministic=self.deterministic,
                )

                obs, reward, terminated, truncated, step_info = self.env.step(int(action))
                ep_reward += reward

                # Record completed trades from this step
                new_trades = self.env.position_manager.completed_trades
                if len(new_trades) > episode_trade_count:
                    for trade in new_trades[episode_trade_count:]:
                        oz_state = getattr(self.env, "_entry_order_zone_state", None)
                        atr_state = getattr(self.env, "_entry_atr_state", None)
                        trend_snap = getattr(self.env, "_last_trend_snap", None)

                        self.journal.record(
                            trade=trade,
                            episode_date=ep_date,
                            episode_trade_number=episode_trade_count + 1,
                            trend_state=trend_snap.state.name if trend_snap else "UNDEFINED",
                            in_supply_demand_zone=oz_state.in_bearish_order_zone or oz_state.in_bullish_order_zone if oz_state else False,
                            in_order_zone=oz_state.trade_worthwhile if oz_state else False,
                            confluence_score=oz_state.confluence_score if oz_state else 0.0,
                            liquidity_sweep_present=False,    # sweep pillar removed
                            rejection_candle_present=False,  # Pillar 3 removed
                            atr_pct_used_at_entry=atr_state.atr_pct_used if atr_state else 0.0,
                            rr_ratio_at_entry=oz_state.rr_ratio if oz_state else 0.0,
                            peak_unrealised_r=self.env._peak_unrealised_r,
                        )
                    episode_trade_count = len(new_trades)

            summary = {
                "date": ep_date,
                "episode_reward": ep_reward,
                "n_trades": episode_trade_count,
                "daily_pnl_r": self.env.position_manager.state.realised_pnl_r,
            }
            episode_summaries.append(summary)

            if (ep_idx + 1) % 20 == 0:
                log.info(
                    "Backtest progress",
                    episodes_done=ep_idx + 1,
                    total=self.n_episodes,
                    avg_reward=round(np.mean([e["episode_reward"] for e in episode_summaries[-20:]]), 4),
                )

        log.info("Backtest complete", total_episodes=len(episode_summaries))

        # Compute metrics
        df = self.journal.to_dataframe()
        metrics = self.metrics_calculator.compute_from_dataframe(df) if not df.empty else {}

        return {
            "metrics": metrics,
            "episode_summaries": episode_summaries,
            "journal_analysis": self.journal.analyse(),
            "n_episodes": len(episode_summaries),
            "avg_episode_reward": float(np.mean([e["episode_reward"] for e in episode_summaries])),
        }

    def compare_with_train(
        self, train_metrics: dict, test_metrics: dict
    ) -> dict:
        """
        Run overfitting assessment between train and test performance.

        Parameters
        ----------
        train_metrics : dict
            Metrics from the training period.
        test_metrics : dict
            Metrics from this backtest.

        Returns
        -------
        dict with degradation ratios and overfitting_flag.
        """
        return self.metrics_calculator.overfitting_score(train_metrics, test_metrics)