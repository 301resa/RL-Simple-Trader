# RL Simple Trader — R1

A Reinforcement Learning trading agent trained on ES/NQ futures using a single-pillar
Order Zone entry strategy (Supply/Demand Zone confluence + ATR room filter).

Built on **RecurrentPPO** (LSTM) via SB3/sb3_contrib. The agent processes one 5-minute
bar at a time — the LSTM hidden state carries session memory across bars without needing
a multi-bar lookback window in the observation vector.

---

## ⚠️ IMPORTANT — For AI Assistants / Low-Context Sessions
> always start any task by refering or carefully reading readme file.

> **Do NOT remove, rename, or stub out any existing function, class, method, callback,
> or file unless the user explicitly asks you to delete it.**

>work through this systematically, starting with a file-by-file plan. Set up the todo list and start working systematically.

> This codebase is large. If you are running low on context or tokens:
> - Edit only the specific lines/functions you were asked to change.
> - Leave all surrounding code untouched.
> - Do not "clean up" imports, remove unused variables, or refactor adjacent code.
> - Do not replace working implementations with `pass` or `raise NotImplementedError`.
> - If unsure whether something is used, assume it IS used and leave it alone.
> - before you change the code read Readme.md and if you change code update the READme.md for the change that made

 > -  work through this systematically, starting with a file-by-file plan. Set up the todo list and start working systematically.  Tidy up the codes without losing functionalities or breaking the codes, make sure the speed and accuracy of the code is achieved - double checks before you go to next part of the code. everything should be clean code, fast  and performance to  highest quality

>- make sure you remove the redunant codes, variables and tidy up the code without losing functionalities or breaking the codes.  make sure codes are clean, readable, professional and comply with highest coding standards.
> - once you review the code, make a note of all the bottleknecks in the codes and let user know the effect of improving on the speed , perforamce and stabilty or anyother metric in the code execution.
>- update the Readme file once finish 
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
│   ├── zone_detector.py             # Supply/Demand zone detection; sweep via bar high/low (wick-and-return)
│   ├── atr_calculator.py            # ATR (daily) — gates entries and sizes stops
│   └── observation_builder.py       # Builds flat observation vector for the LSTM
│
├── training/
│   ├── trainer.py                   # Main training loop — wires all callbacks
│   ├── trading_eval_callback.py     # Eval-based model saving (phase-gated composite score)
│   ├── training_hotsave_callback.py # Training hot-save — four quality gates (PF/WR, Sharpe, WR70, Elite); min_trades scales with date range
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
│   ├── logger.py                    # Structured logging (structlog) — tees stdout to file
│   ├── metrics_printer.py           # Console + file log helpers
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

A **liquidity sweep** is required before any entry is placed. Price must first trade
through the zone's liquidity level — sweep detection uses the **bar high/low** (not close)
so wick-and-return setups are captured correctly: supply is swept when `bar.high ≥ zone.top`;
demand is swept when `bar.low ≤ zone.bottom`. Once swept, a pending limit is placed to
catch the re-entry into the order block.

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
- **Three-constraint sizing** — smallest of all three wins:

  1. **Capital constraint**: `risk_dollars / (stop_pts × point_value)` → never exceeds 1% risk
  2. **Confluence constraint**: graduated score gates the maximum tier:

     | Confluence Score | Max Contracts |
     |-----------------|---------------|
     | < 0.60          | 0.5           |
     | 0.60 – 0.70     | 1.0           |
     | 0.70 – 0.80     | 1.5           |
     | 0.80 – 0.90     | 2.0           |
     | ≥ 0.90          | 2.5           |

     In-zone confluence is now **graduated** (0.55–1.00) based on zone quality
     (width × 40% + age × 35% + touches × 25%) so all five tiers are reachable.

  3. **Zone-width constraint**: wider zones carry more uncertainty.
     Size is scaled by `max(0.50, 1 − zone_width / 10)` — a 5-pt zone gets
     75% of the size a 0-pt zone would get; a 10-pt zone gets 50%.

- **Daily budget guard**: a single trade cannot consume more than **50% of the
  remaining daily loss budget** — if the day is already down $700 of the $1,000
  limit, the next trade is capped at $150 risk (50% × $300 remaining).

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

### Walk-forward analysis (rolling 12-month train / 2-week val)
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
├── models/                                 # Model files only — no journals here
│   ├── best_model.zip                      # Best eval-gated checkpoint
│   ├── best_model_vecnormalize.pkl
│   ├── final_model.zip                     # Safety-net checkpoint at end of training
│   ├── vecnormalize.pkl
│   └── hotsaves/                           # Hotsave model files only
│       ├── hotsave_NNNNNNNNNN.zip          # Gate 1 — PF/WR
│       ├── hotsave_wr70_NNNNNNNNNN.zip     # Gate 2 — WR ≥ 70%
│       ├── hotsave_elite_NNNNNNNNNN.zip    # Gate 3 — Elite
│       └── hotsave_*_vecnormalize.pkl      # VecNormalize stats (per save)
├── checkpoints/                            # SB3 periodic checkpoints (every 100k steps)
├── tensorboard/                            # TensorBoard event files
├── journal/                                # All journals — Excel + Plotly HTML
│   ├── training_journal.xlsx               # Aggregate training journal
│   ├── training_journal.html               # Aggregate Plotly HTML
│   ├── env00_trades.html                   # Per-env OHLC trade charts
│   ├── env01_trades.html
│   └── hotsaves/                           # Hotsave trade journals (separate from models)
│       ├── hotsave_NNNNNNNNNN.xlsx
│       └── hotsave_NNNNNNNNNN.html
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
| 0 | 0–400k | 0.05 |
| 1 | 400k–750k | 0.15 |
| 2 | 750k–1.1M | 0.25 |
| 3 | 1.1M–1.5M | 0.35 |
| 4 | 1.5M+ | 0.45 |

Composite score = weighted average of: Sharpe (30%), P&L in R (25%), Win/Loss ratio (25%), Max drawdown (20%).

A **FINAL_STEP** checkpoint is always written at the end of training as a safety net.

### 2. Training hot-saves (`TrainingHotSaveCallback`)
Checks every ~4,096 steps (one rollout) against three quality gates.
Model files go to `logs/models/hotsaves/`; trade journals (Excel + HTML) go to
`logs/journal/hotsaves/` — completely separate so the folders stay clean.

**Minimum trade count** scales automatically with the training date range:
```
min_trades = max(10, n_trading_days × min_trades_per_week ÷ 5)
```
Default: `min_trades_per_week = 3` (configured in `agent_config.yaml` → `hotsave:`).

| Date range | Trading days | min_trades |
|------------|-------------|-----------|
| 6 months | ~126 | 75 |
| 12 months | ~252 | 151 |
| 2 years | ~504 | 302 |

A model is **never saved if total PnL ≤ 0**.

**Gate 1 — PF/WR** (`hotsave_NNNNNNNNNN.zip`): at least 2 envs simultaneously satisfy:

| Criterion | Threshold |
|-----------|-----------|
| Profit Factor | > 1.60 |
| Win Rate | ≥ 40% |
| Total PnL (R) | > 0 |
| Trades | ≥ min_trades |

**Gate 2 — WR70** (`hotsave_wr70_NNNNNNNNNN.zip`): any single env satisfies all of:

| Criterion | Threshold |
|-----------|-----------|
| Win Rate | ≥ 70% |
| Total PnL ($) | > 0.5% of initial capital |
| Total PnL (R) | > 0 |
| Trades | ≥ min_trades |

**Gate 3 — Elite** (`hotsave_elite_NNNNNNNNNN.zip`): any single env satisfies all of:

| Criterion | Threshold |
|-----------|-----------|
| Cumulative PnL ($) | > 1.5 × initial capital |
| WR × PF | > 1.5 |
| Sharpe ratio | > 3.0 |
| Total PnL (R) | > 0 |
| Trades | ≥ min_trades |

Cooldown: configurable per gate in `agent_config.yaml` → `hotsave:`.

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

- **60-candle sliding window + 17 engineered features**: `lookback_bars: 60` — the last
  60 bars of OHLC log-returns form the price block (volume excluded — the strategy is
  price-structure based). Structured features (36) cover: ATR (4), zone signals (10:
  dist_norm×2, in_zone×2, width_norm×2, age_norm×2, swept×2), order zone / confluence
  (10), portfolio state (8), session timing (4: session_time_pct, bars_remaining_pct,
  `is_rth`, `rth_time_pct`). An additional 17 engineered features complete the vector:
  price location (5), momentum (4), volatility regime (2), bar character (1), HTF context
  (5: 1h/2h close-in-range, prior-day range position, multi-TF momentum coherence, HTF
  vol expansion). Wide zones (> 10 pts) are zeroed in all zone features before building
  the obs. The zone detector uses its own 500-bar history internally. Zone features
  include `supply_swept` and `demand_swept` binary flags. Observation vector size:
  `60 × 4 + 36 + 17 = 293` features. The LSTM (256 units) carries within-session state
  across steps.

- **OHLCV augmentation** (`data/data_augmentor.py`): Three-stage augmentation applied
  to every training episode to prevent the agent memorising price-to-outcome mappings:
  1. **Session-level return scaling** — a factor f ~ U(0.85, 1.15) is drawn per episode;
     all OHLC bars are re-expressed as ratios relative to the session open, raised to
     the power f, then converted back to prices.  This compresses or expands intraday
     moves (±15%) so the agent encounters different volatility regimes on each replay.
  2. **Bar-level OHLC jitter** — each bar's O/H/L/C receives an independent continuous
     offset drawn from U(−2.0, +2.0) pts (±8 ticks for ES).
  Volume is intentionally not augmented — it is excluded from the observation vector.
  OHLC integrity is enforced after every step (`high ≥ max(open, close)`,
  `low ≤ min(open, close)`).  Each of the 16 parallel workers receives its own
  `OHLCVAugmentor` seeded with `base_seed + worker_id` so every env sees a fully
  independent augmentation sequence.  Augmentation is **training-only** — evaluation
  and backtest always use clean unadjusted bars.  Parameters are configurable in
  `agent_config.yaml` under `augmentation:`.

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

- **Pending limit order entry**: agent places a limit at the **far edge** of the zone
  (demand.bottom for LONG, supply.top for SHORT) — the sweep level where liquidity
  was taken; this is where a re-entry into the zone begins. Filled on the bar that trades through the limit; cancelled at session
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
  - `DataLoader.load()`: processed CSV is cached as a sibling `.pkl` file on first run;
    subsequent loads skip D/M/YYYY string parsing and tz-localization (~4× faster).
  - `DataLoader.get_bars_before()`: O(n\_total\_bars) full-dataset boolean scan replaced
    with bisect on sorted day list + day-index walk (~7 day lookups instead of 120k rows).
  - `TradingEnv.reset()` history ATR: 500-iteration Python loop replaced with vectorised
    `unique-date → bisect map` (≈7 bisect calls per reset instead of 500).
  - `TradingEnv` RTH time constants (`_rth_start_time`, `_rth_end_time`,
    `_rth_total_secs`) pre-computed once in `__init__` — eliminates 2 `pd.Timestamp`
    allocations and 1 `timedelta.total_seconds()` call every step.
  - `TradingEnv._compute_action_mask()` accepts optional `portfolio_state` — avoids
    a redundant `get_portfolio_state()` call and a pandas `.iloc` lookup per step.
  - `ZoneState`, `FIXED_STOP_BUFFER_PTS`, `MIN_STOP_PTS`, and `MAX_ZONE_WIDTH_PTS` are
    module-level constants in `order_zone_engine.py`; imported once by all callers.
  - RTH elapsed-seconds computed via integer arithmetic (`bar_time.hour*3600 + min*60`)
    instead of 2 `pd.Timestamp` allocations per step.
  - Dead O(n×window) vol_ratios fallback loop in `ObservationBuilder.build()` removed;
    `prepare_episode()` always pre-caches ratios so the fallback was unreachable.
  - `ObservationBuilder.prepare_episode()`: precomputes all four OHLC log-return arrays
    (`_lr_open/high/low/close`) in one vectorised pass per episode; `build()` now slices
    precomputed arrays instead of recomputing log-returns and `np.diff(np.log(...))` per step.
  - `TradingEnv.step()`: replaces `session_bars.iloc[step]` with O(1) numpy lookups into
    `_cached_closes/highs/lows`; `_session_times` list pre-built in `reset()` eliminates
    `session_bars.index[step].time()` per step.
  - `TradingEnv._compute_action_mask()`: last `.iloc` close lookup replaced with numpy cache.
  - `OHLCVAugmentor.apply()`: all three augmentation stages now operate on a single `(n,4)`
    numpy array; 12 separate `.values`/column-assignment calls replaced with 1 extraction
    + 1 write-back — ~4× faster per episode reset.
  - `ATRCalculator.compute_all_session_states()`: Python loop replaced with
    `np.maximum.accumulate` / `np.minimum.accumulate` — fully vectorised O(n).
    Prior-day lookup also switched from pandas boolean filter to bisect (matching
    `get_atr_for_date`).
  - `TrainingHotSaveCallback`: three gate checks previously called `to_info_dict()` three
    times each; refactored to compute `agg_list` once and pass it to all gates.
  - `TrainingJournalCallback`: all file I/O (Excel, HTML, OHLC chart) disabled during
    training — `_on_step` accumulates trades only; `_save()` / `_on_training_end()` are
    no-ops. Call `write_snapshot()` / `_write_excel()` post-training for analysis.

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
