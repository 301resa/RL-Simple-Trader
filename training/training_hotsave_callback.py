"""
training/training_hotsave_callback.py
=======================================
Saves the model mid-training when per-env cumulative metrics cross one of
three quality gates.  All metrics are accumulated across the ENTIRE training
run per env — no rolling windows.

Gate 1 — Standard PF/WR gate:
  At least min_envs_passing training envs simultaneously satisfy:
    • PF  > pf_threshold        (default 1.60)
    • WR  >= wr_threshold       (default 0.40)
    • total_pnl_r > 0           (never save on net loss)
    • trades >= min_trades      (scaled: max(10, n_trading_days * min_trades_per_week // 5))

Gate 2 — Win-rate 70 gate:
  Any single training env satisfies ALL of:
    • WR  >= 0.70
    • total_pnl_dollars >= 0.5% of initial_capital
    • total_pnl_r > 0
    • trades >= min_trades
  Saved with prefix "hotsave_wr70_".

Gate 3 — Elite gate (exceptional quality):
  Any single training env satisfies ALL of:
    • total_pnl_dollars > elite_pnl_multiplier × initial_capital
    • WR × PF > elite_wr_pf_threshold
    • Sharpe > elite_sharpe
    • trades >= min_trades
  Saved with prefix "hotsave_elite_".

Each gate has its own cooldown.
Model files  → models_dir/hotsaves/
Journal files → journal_dir/hotsaves/   (separate from models — clean layout)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from training.env_cumulative import EnvCumulative
from utils.logger import get_logger

log = get_logger(__name__)


class TrainingHotSaveCallback(BaseCallback):
    """
    Saves the model during training when a quality gate is cleared.

    All gate checks use cumulative stats since training step 0 per env —
    no rolling windows.

    Parameters
    ----------
    models_dir : str | Path
        Where model .zip and _vecnormalize.pkl files are written.
    journal_dir : str | Path | None
        Where journal snapshots (Excel + HTML) are written.
        Defaults to models_dir if None.
    pf_threshold : float          Gate 1 — minimum profit factor.
    wr_threshold : float          Gate 1 — minimum win rate (0–1).
    min_trades : int              All gates — minimum trades.
    min_envs_passing : int        Gate 1 — envs that must simultaneously pass.
    cooldown_steps : int          Gate 1 — minimum steps between saves.
    wr70_cooldown_steps : int     Gate 2 — minimum steps between saves.
    elite_pnl_multiplier : float  Gate 3 — PnL must exceed this × initial_capital.
    elite_wr_pf_threshold : float Gate 3 — WR × PF must exceed this.
    elite_sharpe : float          Gate 3 — minimum annualised Sharpe.
    elite_cooldown_steps : int    Gate 3 — minimum steps between saves.
    initial_capital : float       Account size — used for WR70 and Elite PnL thresholds.
    vec_normalize                 If provided, saves normalisation stats alongside model.
    journal_callback              TrainingJournalCallback — if set, saves HTML+Excel on each gate.
    """

    def __init__(
        self,
        models_dir: str | Path,
        journal_dir: str | Path | None = None,
        pf_threshold: float = 1.60,
        wr_threshold: float = 0.40,
        min_trades: int = 50,
        min_envs_passing: int = 2,
        cooldown_steps: int = 50_000,
        wr70_cooldown_steps: int = 50_000,
        elite_pnl_multiplier: float  = 1.5,
        elite_wr_pf_threshold: float = 1.5,
        elite_sharpe: float          = 3.0,
        elite_cooldown_steps: int    = 50_000,
        initial_capital: float = 2500.0,
        check_every_steps: int = 4_096,
        vec_normalize=None,
        journal_callback=None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.models_dir           = Path(models_dir)
        self.journal_dir          = Path(journal_dir) if journal_dir else self.models_dir
        self.pf_threshold         = pf_threshold
        self.wr_threshold         = wr_threshold
        self.min_trades           = min_trades
        self.min_envs_passing     = min_envs_passing
        self.cooldown_steps       = cooldown_steps
        self.wr70_cooldown_steps  = wr70_cooldown_steps
        self.wr70_min_pnl_dollars = 0.005 * initial_capital
        self.elite_pnl_threshold  = elite_pnl_multiplier * initial_capital
        self.elite_wr_pf_threshold= elite_wr_pf_threshold
        self.elite_sharpe         = elite_sharpe
        self.elite_cooldown_steps = elite_cooldown_steps
        self.check_every_steps    = check_every_steps
        self.vec_normalize        = vec_normalize
        self._journal_callback    = journal_callback

        self._env_cumulative: Dict[int, EnvCumulative] = {}
        self._last_save_step:        int = -cooldown_steps
        self._last_wr70_save_step:   int = -wr70_cooldown_steps
        self._last_elite_save_step:  int = -elite_cooldown_steps
        self._trades: list = []

    # ── Episode capture ───────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, (info, done) in enumerate(zip(infos, dones)):
            if done and "profit_factor" in info:
                if i not in self._env_cumulative:
                    self._env_cumulative[i] = EnvCumulative()
                self._env_cumulative[i].update(info)
                for t in info.get("trades_list", []):
                    t["global_step"] = self.num_timesteps
                    self._trades.append(t)

        if self.num_timesteps % self.check_every_steps < self.n_envs:
            self._check_pf_gate()
            self._check_wr70_gate()
            self._check_elite_gate()

        return True

    # ── Gate 1 — PF / WR ─────────────────────────────────────────────────────

    def _check_pf_gate(self) -> None:
        step = self.num_timesteps
        if step - self._last_save_step < self.cooldown_steps:
            return
        if not self._env_cumulative:
            return

        agg_list = [c.to_info_dict() for c in self._env_cumulative.values()]
        passing  = [
            d for d in agg_list
            if (
                d.get("profit_factor",   0.0) > self.pf_threshold
                and d.get("win_rate",    0.0) >= self.wr_threshold
                and d.get("n_trades",    0)   >= self.min_trades
                and d.get("total_pnl_r", 0.0) > 0.0
            )
        ]
        if len(passing) < self.min_envs_passing:
            return

        avg_pf = float(np.mean([d["profit_factor"] for d in passing]))
        avg_wr = float(np.mean([d["win_rate"]       for d in passing]))
        self._save(
            step=step,
            prefix="hotsave",
            tag="PF gate",
            metrics={"avg_PF": avg_pf, "avg_WR": avg_wr, "envs": len(passing)},
        )
        self._last_save_step = step

    # ── Gate 2 — WR70 ────────────────────────────────────────────────────────

    def _check_wr70_gate(self) -> None:
        step = self.num_timesteps
        if step - self._last_wr70_save_step < self.wr70_cooldown_steps:
            return
        if not self._env_cumulative:
            return

        agg_list   = [c.to_info_dict() for c in self._env_cumulative.values()]
        qualifying = [
            d for d in agg_list
            if (
                d.get("win_rate",            0.0) >= 0.70
                and d.get("total_pnl_dollars", 0.0) >= self.wr70_min_pnl_dollars
                and d.get("n_trades",          0)   >= self.min_trades
                and d.get("total_pnl_r",       0.0) > 0.0
            )
        ]
        if not qualifying:
            return

        best = max(qualifying, key=lambda d: d.get("win_rate", 0.0))
        self._save(
            step=step,
            prefix="hotsave_wr70",
            tag="WR70 gate",
            metrics={
                "WR":     best.get("win_rate", 0.0),
                "PnL_$":  best.get("total_pnl_dollars", 0.0),
                "trades": best.get("n_trades", 0),
                "envs":   len(qualifying),
            },
        )
        self._last_wr70_save_step = step

    # ── Gate 3 — Elite ───────────────────────────────────────────────────────

    def _check_elite_gate(self) -> None:
        """Save when any env achieves elite return, WR×PF threshold, and Sharpe."""
        step = self.num_timesteps
        if step - self._last_elite_save_step < self.elite_cooldown_steps:
            return
        if not self._env_cumulative:
            return

        agg_list   = [c.to_info_dict() for c in self._env_cumulative.values()]
        qualifying = [
            d for d in agg_list
            if (
                d.get("total_pnl_dollars", 0.0)                          > self.elite_pnl_threshold
                and d.get("win_rate", 0.0) * d.get("profit_factor", 0.0) > self.elite_wr_pf_threshold
                and d.get("sharpe_ratio", 0.0)                           > self.elite_sharpe
                and d.get("n_trades", 0)                                 >= self.min_trades
                and d.get("total_pnl_r", 0.0)                            > 0.0
            )
        ]
        if not qualifying:
            return

        best = max(qualifying, key=lambda d: d.get("sharpe_ratio", 0.0))
        self._save(
            step=step,
            prefix="hotsave_elite",
            tag="Elite gate",
            metrics={
                "SH":     best.get("sharpe_ratio",    0.0),
                "WR*PF":  best.get("win_rate", 0.0) * best.get("profit_factor", 0.0),
                "PnL_$":  best.get("total_pnl_dollars", 0.0),
                "trades": best.get("n_trades", 0),
                "envs":   len(qualifying),
            },
        )
        self._last_elite_save_step = step

    # ── Shared save helper ────────────────────────────────────────────────────

    def _save(self, step: int, prefix: str, tag: str, metrics: dict) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        name       = f"{prefix}_{step:010d}"
        model_path = self.models_dir / name

        self.model.save(str(model_path))

        if self.vec_normalize is not None:
            vn_path = self.models_dir / f"{name}_vecnormalize.pkl"
            self.vec_normalize.save(str(vn_path))

        if self._journal_callback is not None and self._trades:
            try:
                self.journal_dir.mkdir(parents=True, exist_ok=True)
                self._journal_callback.write_snapshot(
                    output_dir=self.journal_dir,
                    stem=name,
                    trades=self._trades,
                )
            except Exception as exc:
                if self.verbose:
                    print(f"[HotSave] Journal snapshot failed: {exc}")

        if self.verbose >= 1:
            vn_note = " + VecNormalize" if self.vec_normalize is not None else ""
            met_str = "  ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            line = "=" * 70
            print(f"\n{line}")
            print(f"  HOTSAVE [{tag}]  |  step {step:,}")
            print(f"  {met_str}")
            print(f"  Model  : {model_path}.zip{vn_note}")
            print(f"  Journal: {self.journal_dir / name}")
            print(line)

        log.info(
            "Training hot-save written",
            gate=tag,
            step=step,
            model_path=str(model_path),
            journal_dir=str(self.journal_dir),
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()},
        )

    # ── n_envs helper ─────────────────────────────────────────────────────────

    @property
    def n_envs(self) -> int:
        try:
            return self.training_env.num_envs
        except AttributeError:
            return 1
