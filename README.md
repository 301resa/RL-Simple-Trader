# RL Simple Trader — R1

A Reinforcement Learning trading agent trained on ES/NQ futures using a single-pillar
Order Zone entry strategy (Supply/Demand Zone confluence + ATR room filter).

Built on **RecurrentPPO** (LSTM) via SB3/sb3_contrib. The agent processes one 5-minute
bar at a time — the LSTM hidden state carries session memory across bars without needing
a multi-bar lookback window in the observation vector.

---

## ⚠️ IMPORTANT — For AI Assistants / Low-Context Sessions

> **Do NOT remove, rename, or stub out any existing function, class, method, callback,
> or file unless the user explicitly asks you to delete it.**
>
> This codebase is large. If you are running low on context or tokens:
> - Edit only the specific lines/functions you were asked to change.
> - Leave all surrounding code untouched.
> - Do not "clean up" imports, remove unused variables, or refactor adjacent code.
> - Do not replace working implementations with `pass` or `raise NotImplementedError`.
> - If unsure whether something is used, assume it IS used and leave it alone.
> - before you change the code read Readme.md and if you change code update the READme.md for the change that made

>- make sure you remove the redunant codes, variables and tidy up the code without losing functionalities or breaking the codes.  make sure codes are clean, readable, professional and comply with highest coding standards.
> - once you review the code, make a note of all the bottleknecks in the codes and let user know the effect of improving on the speed , perforamce and stabilty or anyother metric in the code execution.
---

## Project Structure

```
R1/
├── main.py                          # Entry point — train / evaluate / walk_forward / analyse
├── config/
│   ├── agent_config.yaml            # PPO hyperparameters, LSTM, eval schedule, walk-forward
│   ├── environment_config.yaml      # Instruments (ES/NQ/MES/MNQ), session, account
│   ├── features_config.yaml         # ATR, zone detection, liquidity sweep thresholds
│   ├── risk_config.yaml             # Stop loss, take profit, position sizing, daily limits
│   ├── reward_config.yaml           # Shaped reward weights (R-multiples)
│   └── logging_config.yaml          # Log level, file paths
│
├── agent/
│   ├── ppo_agent.py                 # PPOAgent wrapper around SB3 RecurrentPPO
│   └── network.py                   # Custom LSTM policy network
│
├── data/
│   ├── data_loader.py               # Loads 5-min OHLCV CSV + daily bars
│   ├── data_splitter.py             # Chronological train/val/test splits
│   ├── data_augmentor.py            # OHLCV jitter augmentation (training only)
│   └── data_validator.py            # Data quality checks
│
├── environment/
│   ├── trading_env.py               # Gymnasium TradingEnv — core episode loop
│   ├── position_manager.py          # Position sizing, trailing stops, daily risk limits
│   ├── reward_calculator.py         # R-multiple shaped reward function
│   └── action_space.py              # Action masker (entry conditions, session gates)
│
├── features/
│   ├── order_zone_engine.py         # Order Zone confluence score: zone (90%) + ATR room (10%)
│   ├── zone_detector.py             # Supply/Demand zone detection (consolidation + impulse)
│   ├── liquidity_detector.py        # Retained for reference — sweep detection now in zone_detector.py
│   ├── trend_classifier.py          # HH/HL/LH/LL trend structure
│   ├── atr_calculator.py            # ATR (daily) — gates entries and sizes stops
│   └── observation_builder.py       # Builds flat observation vector for the LSTM
│
├── training/
│   ├── trainer.py                   # Main training loop — wires all callbacks
│   ├── trading_eval_callback.py     # Eval-based model saving (phase-gated composite score)
│   ├── training_hotsave_callback.py # Training hot-save (V7-style two-tier PF/WR gate)
│   ├── metrics_logger_callback.py   # Per-env training table printed after each rollout
│   ├── checkpoint_manager.py        # Periodic checkpoint saving + rotation
│   ├── training_journal_callback.py # Excel + Plotly HTML trade journal (every 50k steps)
│   ├── fold_journal_callback.py     # Per-fold journal for walk-forward analysis
│   ├── shaping_decay_callback.py    # Reward shaping decay across training stages
│   └── curriculum.py                # Optional curriculum scheduler
│
├── evaluation/
│   ├── test_fold.py                 # Load all checkpoints; cleans previous results, parallel rollout, ranked leaderboard + per-model HTML/Excel
│   ├── backtester.py                # Deterministic backtest on test data
│   ├── metrics_calculator.py        # Sharpe, profit factor, drawdown, etc.
│   ├── trade_journal.py             # Trade-level logging (Excel + CSV)
│   └── journal_viewer.py            # Journal analysis viewer
│
├── utils/
│   ├── logger.py                    # Structured logging (structlog)
│   ├── metrics_printer.py           # Console + file log helpers
│   ├── normalizer.py                # Observation normalisation utilities
│   ├── validators.py                # Config validation
│   └── feature_exporter.py         # Export computed features to CSV
│
├── docs/
│   ├── RL_Order_Zone_Strategy_Summary.docx   # Full strategy document
│   ├── RL_Order_Zone_Strategy_Summary.pdf    # PDF version
│   └── generate_strategy_doc.py              # Script that regenerates the Word doc
│
├── test_environment.py              # Smoke-test for TradingEnv
└── test_features.py                 # Smoke-test for feature pipeline
```

---

## Trading Strategy

### Entry — Order Zone System

A trade is only entered when the **zone pillar** is present. Weighted factors:

| Factor | Weight | Condition |
|--------|--------|-----------|
| Supply/Demand Zone | 90% | Price is inside a valid consolidation-then-impulse zone |
| ATR Room | 10% | Directional ATR move < 85% of daily ATR in the entry direction |

A **liquidity sweep** is required before any entry is placed. Price must first
trade through the zone's liquidity level (supply: `zone.top`; demand: `zone.bottom`),
then re-enter the order block — only then is a pending limit placed.

- Minimum confluence score: **0.55** (configurable in `features_config.yaml`)
- Minimum R:R ratio: **1.5:1** before an entry is allowed

### Pending Limit Order Entry

Entry requires a **liquidity sweep** first. The zone's liquidity level must be traded through
(supply: price ≥ `zone.top`; demand: price ≤ `zone.bottom`) before any order is placed.
Once swept, a limit is placed at the sweep level to catch the re-entry into the order block:

- **LONG**: limit at `demand.bottom` — fills when `bar.high ≥ demand.bottom` (price rallies back up)
- **SHORT**: limit at `supply.top` — fills when `bar.low ≤ supply.top` (price drops back down)
- **Sweep gate**: orders are skipped if `zone.was_swept == False` — no order until liquidity taken.
- **Wide zone filter**: zones wider than **10 points** are skipped entirely.
- **Cancellation**: all pending orders are cancelled automatically at session end, or
  when the agent places a new entry signal in any direction (cancel-and-replace).
- **Agent EXIT action** while flat also cancels any open pending order.
- If no zone is detected, the limit falls back to the current price (immediate fill on the next bar).

The observation vector includes pending order context: `pending_active`, `pending_direction`,
`pending_dist_norm` (distance from current price to limit level, ATR-normalised).
Zone features also include `supply_swept` and `demand_swept` binary flags.

### Stop Loss

Stop placed **1.5 points beyond the sweep level** — the level that was just taken as
liquidity. If price continues through that level after filling, the thesis is invalidated.

```
LONG:  stop = demand.bottom − 1.5 pts   (just below the swept low)
SHORT: stop = supply.top   + 1.5 pts    (just above the swept high)
```

Stop widening is **never** allowed once placed.

### Take Profit

Target = the **high of the last up-leg** (for a LONG) or the **low of the last down-leg** (for a SHORT).

This is the swing extreme of the most recent directional move before the pullback/dip
that created the supply/demand zone. The agent targets that prior swing high or low
as the natural first profit objective.

### Trailing Stop

| Trigger | Action |
|---------|--------|
| Profit reaches 1.2R | Trailing stop activates; locks in 1.2R minimum |
| Profit reaches 3R | Trail tightens aggressively |

### Risk Per Trade

- **1% of account equity** per trade (capital-based sizing)
- **Fractional contract sizes**: `[0.5, 1.0, 1.5, 2.0, 2.5]` — snapped to the largest
  tier that doesn't exceed the 1% capital risk
- **Confluence-graded sizing**: the confluence score gates the maximum allowed tier:

  | Confluence Score | Max Contracts |
  |-----------------|---------------|
  | < 0.50          | 0.5           |
  | 0.50 – 0.65     | 1.0           |
  | 0.65 – 0.75     | 1.5           |
  | 0.75 – 0.85     | 2.0           |
  | ≥ 0.85          | 2.5           |

  Final size = `min(capital_tier, confluence_tier)` — never exceeds 1% risk.

- **Dual daily loss limit**: stops trading when **either** limit is reached:
  - **−3R** on the day, OR
  - **−$1,000** on the day (configurable via `max_daily_loss_dollars` in `risk_config.yaml`)

---

## Running the Agent

### Train from scratch
```bash
python main.py --mode train --config config/ --data data/
```

### Continue from checkpoint
```bash
python main.py --mode train --config config/ --data data/ \
    --checkpoint logs/models/checkpoint_00100000.zip --no-clean
```

### Backtest a trained model
```bash
python main.py --mode evaluate --config config/ --data data/ \
    --checkpoint logs/models/best_model.zip
```

### Walk-forward analysis (rolling 12-month train / 5-week val)
```bash
python main.py --mode walk_forward --config config/ --data data/
```

### Test-fold evaluation (all checkpoints vs. a date range)
```bash
python main.py --mode test_fold \
    --models-dir logs/walk_forward/fold_00/models \
    --config config/ --data data/ \
    --test-start 2026-03-01 --test-end 2026-04-09 \
    --out-dir logs/walk_forward/fold_00/test_results \
    --n-workers 8
```
Every run **automatically deletes** all `.xlsx` and `*_journal.html` files in
`--out-dir` before writing new results. Produces:
- Per-model Plotly HTML journal (candlestick + per-trade PnL bars + cumulative PnL)
- Per-model Excel journal (Trades / Daily / Metrics sheets)
- `leaderboard.xlsx` — all models ranked best-first by composite score

### Analyse a saved journal
```bash
python main.py --mode analyse --journal logs/journal/
```

> **Clean slate**: Every `train` and `walk_forward` run **automatically deletes** all
> previous output (models, checkpoints, logs, journals) under `logs/` before starting.
> Pass `--no-clean` to skip this when resuming.

---

## Output Layout

```
logs/
├── models/
│   ├── checkpoint_NNNNNNNNNN.zip           # Phase-gated eval checkpoints
│   ├── checkpoint_NNNNNNNNNN_vecnormalize.pkl
│   ├── final_model.zip                     # Safety-net checkpoint at end of training
│   ├── vecnormalize.pkl                    # VecNormalize stats (end of training)
│   └── hotsaves/
│       ├── hotsave_NNNNNNNNNN.zip          # Training hot-saves (PF/WR gate)
│       └── hotsave_NNNNNNNNNN_vecnormalize.pkl
├── checkpoints/                            # Periodic SB3 checkpoints (every 100k steps)
├── tensorboard/                            # TensorBoard event files
├── journal/                                # Excel + Plotly HTML trade journals
├── walk_forward/                           # Per-fold outputs
└── metrics.log                             # Console table mirror
```

---

## Model Saving Logic

### 1. Eval-based saves (`TradingEvalCallback`)
Fires every **100,000 steps** on the validation set. Saves when a **composite score**
beats the previous best AND the phase gate is cleared:

| Phase | Steps | Min composite score |
|-------|-------|---------------------|
| 1 | 0–400k | 0.05 |
| 2 | 400k–800k | 0.15 |
| 3 | 800k–1.2M | 0.25 |
| 4 | 1.2M–1.6M | 0.35 |
| 5 | 1.6M+ | 0.45 |

Composite score = weighted average of: Sharpe (40%), P&L in R (30%), Win/Loss ratio (15%), Max drawdown (15%).

A **FINAL_STEP** checkpoint is always written at the end of training as a safety net.

### 2. Training hot-saves (`TrainingHotSaveCallback`)
Checks every ~4,096 steps (one rollout) against three quality gates:

**Gate 1 — PF/WR** (`hotsave_NNNNNNNNNN.zip`): at least 2 envs simultaneously satisfy:

| Criterion | Threshold |
|-----------|-----------|
| Profit Factor | > 1.60 |
| Win Rate | ≥ 40% |
| Trades | ≥ 20 |

**Gate 2 — Sharpe** (`hotsave_sh_NNNNNNNNNN.zip`): at least 2 envs simultaneously satisfy:

| Criterion | Threshold |
|-----------|-----------|
| Sharpe ratio | > 1.2 |
| Profit Factor | > 1.85 |
| Win/Loss ratio | > 1.0 |
| Total PnL (R) | > 0 |
| Trades | ≥ 20 |

**Gate 3 — WR70** (`hotsave_wr70_NNNNNNNNNN.zip`): any single env satisfies all of:

| Criterion | Threshold |
|-----------|-----------|
| Win Rate | ≥ 70% |
| Total PnL ($) | > 0 |
| Trades | ≥ 20 |

All gates saved to `logs/models/hotsaves/`. Cooldown: 50,000 steps per gate.

### Disabling eval saves
Set `save_enabled: false` under `evaluation:` in `agent_config.yaml` to run eval
metrics and log results without writing any model files (useful for exploration runs).

---

## Configuration Files

| File | Controls |
|------|----------|
| `agent_config.yaml` | PPO hyperparameters, LSTM size, eval schedule, walk-forward settings |
| `environment_config.yaml` | Instrument (ES/NQ), session type (RTH/GLOBEX/FULL), account size |
| `features_config.yaml` | Zone detection thresholds, ATR settings, order zone weights |
| `risk_config.yaml` | Stop placement, take profit, position sizing, daily loss limits |
| `reward_config.yaml` | Shaped reward weights — hold penalty, entry bonuses/penalties, exit rewards |
| `logging_config.yaml` | Log verbosity and output paths |

---

## Key Design Decisions

- **60-candle sliding window + 11 engineered features**: `lookback_bars: 60` — the last
  60 bars of OHLCV (log-returns + volume ratio) form the price block. Structured features
  (36) cover: ATR (4), zone signals (10: dist_norm×2, in_zone×2, width_norm×2,
  age_norm×2, swept×2), order zone / confluence (10), portfolio state (8), session timing
  (4: session_time_pct, bars_remaining_pct, `is_rth`, `rth_time_pct`). An additional 11
  engineered price/volatility features complete the vector. Wide zones (> 10 pts) are
  zeroed in all zone features before building the obs. The zone detector uses its own
  500-bar history internally. Zone features include `supply_swept` and `demand_swept`
  binary flags. Observation vector size: `60 × 5 + 36 + 11 = 347` features.
  The LSTM (256 units) carries within-session state across steps.

- **OHLCV jitter augmentation** (`data/data_augmentor.py`): Applied to every training
  episode to prevent the agent memorising price-to-outcome mappings. Each bar receives
  an independent discrete offset sampled uniformly from `{−0.5, −0.25, 0, +0.25, +0.5}`
  points for OHLC and `{−10, −5, 0, +5, +10}` contracts for volume. OHLC integrity is
  enforced after jittering (`high ≥ max(open, close)`, `low ≤ min(open, close)`).
  Each of the 16 parallel workers is seeded differently so every env sees a unique
  jitter sequence. Augmentation is **training-only** — evaluation and backtest use
  clean unadjusted bars.

- **Annualised Sharpe**: All Sharpe calculations use `mean(pnl_r) / std(pnl_r) × √252`
  consistently across training table, eval callback, and test_fold.

- **Sweep-based stop**: `stop = sweep_level ± 1.5 pts`. Entry is at the sweep level
  (supply.top or demand.bottom); stop is 1.5 points beyond that level. A sustained
  break back through the sweep level invalidates the setup immediately. This keeps 1R
  tight and consistent regardless of zone width.

- **No hard trade cap**: `max_trades_per_day: 999` — the dual daily loss limit (−3R
  or −$1,000) is the natural brake. The reward's `hold_flat_penalty` is applied
  **only when a valid order zone setup is present AND no pending order is already active**
  — the agent is penalised for ignoring a good setup, not for patiently waiting when
  no setup exists or correctly waiting for a pending limit to fill.

- **Pending limit order entry**: agent places a limit at the **near edge** of the zone
  (demand.top for LONG, supply.bottom for SHORT) — the first level touched as price
  re-enters the zone. Filled on the bar that trades through the limit; cancelled at session
  end or on a new entry signal. Zones wider than 10 points are skipped.

- **Confluence-graded sizing**: position size scales from 0.5 to 2.5 contracts in 0.5
  increments based on the confluence score, capped by the 1%-of-capital risk constraint.

- **VecNormalize**: Normalises observations only (`norm_reward=False`). Stats are saved
  alongside every checkpoint so evaluation always uses matched normalisation.

- **ATR period = 1** (previous day's true range): zone detection thresholds and the ATR
  exhaustion gate are measured against yesterday's actual range. This adapts to current
  volatility — quiet days produce tight thresholds, volatile days produce wider ones.
  A 14-day smooth would lag regime changes and apply stale thresholds.

- **Zone selection by edge proximity**: `_build_state()` selects the supply zone whose
  `bottom` edge is nearest to current price (SHORT limit placed at supply.bottom), and the
  demand zone whose `top` edge is nearest (LONG limit placed at demand.top).
  Selection is by the entry-side edge so the closest actionable zone is always chosen.

- **Impulse-extreme take-profit**: each detected zone stores the `impulse_extreme` — the
  actual high (supply) or low (demand) of the impulse bar that confirmed the zone. This
  swing extreme is used as the primary take-profit target, replacing the ATR-projection
  approximation. Falls back to ATR projection only when `impulse_extreme` is absent.

- **Dollar-based Profit Factor** (`profit_factor_usd`): `test_fold` scoring uses
  gross-profit-dollars / gross-loss-dollars as the primary metric. R-based PF can show
  positive values (e.g. 1.27) on a net-negative dollar run when confluence-graded sizing
  puts larger positions on losing trades. Dollar PF correctly reflects actual P&L.

- **Performance optimisations** (applied to hot paths):
  - `ATRCalculator.get_atr_for_date()`: O(n) pandas filter → O(log n) `bisect` on sorted list.
  - `ATRCalculator.compute_all_session_states()`: vectorised O(n) running max/min replaces
    O(n²) per-step DataFrame slice in the precompute loop.
  - `ObservationBuilder.prepare_episode()`: numpy arrays and rolling-20 volume averages
    pre-extracted once per episode reset (cumsum trick); `build()` uses cached arrays.
  - `ZoneDetector`: invalid Zone objects pruned from lists once they exceed 3× the per-side cap.
  - `TradingEnv` precompute loop: `copy.deepcopy()` replaced with `dataclasses.replace()`
    (~10–50× faster) for Zone snapshots; `ATRState`/`OrderZoneState` stored directly.
  - `TradingEnv.step()`: `atr_series.iloc[]` lookup replaced with already-loaded
    `atr_state.atr_daily`; `current_price` reused for session-end force-close.
  - `DataLoader.get_bars_before()`: O(n\_total\_bars) full-dataset boolean scan replaced
    with bisect on sorted day list + day-index walk (~7 day lookups instead of 120k rows).
  - `TradingEnv.reset()` history ATR: 500-iteration Python loop replaced with vectorised
    `unique-date → bisect map` (≈7 bisect calls per reset instead of 500).
  - `TradingEnv` RTH time constants (`_rth_start_time`, `_rth_end_time`,
    `_rth_total_secs`) pre-computed once in `__init__` — eliminates 2 `pd.Timestamp`
    allocations and 1 `timedelta.total_seconds()` call every step.
  - `TradingEnv._compute_action_mask()` accepts optional `portfolio_state` — avoids
    a redundant `get_portfolio_state()` call and a pandas `.iloc` lookup per step.
  - `ZoneState` and `FIXED_STOP_BUFFER_PTS`/`MIN_STOP_PTS` moved to module-level imports
    in `trading_env.py`; removed repeated function-level import calls in hot paths.

---

## Instruments

| Symbol | Point Value | Tick Size |
|--------|------------|-----------|
| ES | $50 / point | 0.25 |
| MES | $5 / point | 0.25 |
| NQ | $20 / point | 0.25 |
| MNQ | $2 / point | 0.25 |

Default instrument: **ES** (configurable in `environment_config.yaml`).

---

## Dependencies

- `stable-baselines3` + `sb3-contrib` (RecurrentPPO)
- `gymnasium`
- `torch`
- `pandas`, `numpy`
- `plotly`, `openpyxl` (journals)
- `python-docx` (strategy document generation)
- `pyyaml`, `structlog`
