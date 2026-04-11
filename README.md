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
│   ├── liquidity_detector.py        # Retained file — sweep logic no longer used in pipeline
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
│   ├── test_fold.py                 # Load all checkpoints, ranked results table + Plotly HTML
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

The liquidity sweep pillar has been removed. The LSTM learns sweep context
directly from raw price observations.

- Minimum confluence score: **0.35** (configurable in `features_config.yaml`)
- Minimum R:R ratio: **1.5:1** before an entry is allowed

### Pending Limit Order Entry

When the agent decides to enter, it places a **pending limit order at the 50% midpoint
of the detected zone** (i.e. `zone.midpoint = (top + bottom) / 2`), rather than entering
at market. The order fills only when price trades at or through that level on a future bar.

- **LONG**: fills when `bar.low ≤ zone_demand_midpoint`
- **SHORT**: fills when `bar.high ≥ zone_supply_midpoint`
- **Cancellation**: all pending orders are cancelled automatically at session end, or
  when the agent places a new entry signal in any direction (cancel-and-replace).
- **Agent EXIT action** while flat also cancels any open pending order.
- If no zone is detected, the limit falls back to the current price (immediate fill on the next bar).

The three observation slots previously reserved for the removed rejection-candle features
are now used for pending order context: `pending_active`, `pending_direction`, `pending_dist_norm`
(distance from current price to limit level, ATR-normalised). Observation dimension is unchanged.

### Stop Loss

Stop placed **1.5 fixed points** beyond the zone boundary (not ATR-based), measured from
the far edge of the zone (not the entry midpoint).
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
Fires every **20,000 steps** on the validation set. Saves when a **composite score**
beats the previous best AND the phase gate is cleared:

| Phase | Steps | Min composite score |
|-------|-------|---------------------|
| 1 | 0–400k | 0.05 |
| 2 | 400k–800k | 0.15 |
| 3 | 800k–1.2M | 0.25 |
| 4 | 1.2M–1.6M | 0.35 |
| 5 | 1.6M+ | 0.45 |

Composite score = weighted average of: Sharpe (30%), P&L in R (25%), Win/Loss ratio (25%), Max drawdown (20%).

A **FINAL_STEP** checkpoint is always written at the end of training as a safety net.

### 2. Training hot-saves (`TrainingHotSaveCallback`)
Checks every ~4,096 steps (one rollout) against a **single unified gate** on live training envs:

At least **2 individual envs** must simultaneously satisfy **all** of:

| Criterion | Threshold |
|-----------|-----------|
| Profit Factor | > 1.60 |
| Win Rate | ≥ 40% |
| Trades | ≥ 20 |

Saved to `logs/models/hotsaves/`. Cooldown: 50,000 steps between saves.

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

- **500-candle sliding window**: `lookback_bars: 500` — the last 500 bars of OHLCV
  (log-returns + volume ratio) are included in every observation. The window shifts
  forward one bar at each step. Prior-session bars fill the window from the start of
  each episode so bar 0 never has zero-padding — the same 500-bar history context
  used by the zone detector. The LSTM (256 units) additionally carries within-session
  state across steps. Observation vector size: `500 × 5 + 28 = 2,528` features.

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

- **Fixed stop buffer**: 1.5 points beyond zone boundary (from zone edge, not midpoint).
  Not ATR-relative — gives precise, predictable risk definition on 5-min ES/NQ bars.

- **No hard trade cap**: `max_trades_per_day: 999` — the dual daily loss limit (−3R
  or −$1,000) is the natural brake. The reward's `hold_flat_penalty: −0.025/bar` forces
  the agent to seek entries rather than sit idle.

- **Pending limit order entry**: agent places a limit at the zone midpoint (50% depth);
  filled on the bar that trades through the limit price, cancelled at session end or on
  a new entry signal. Avoids chasing price and improves average fill quality.

- **Confluence-graded sizing**: position size scales from 0.5 to 2.5 contracts in 0.5
  increments based on the confluence score, capped by the 1%-of-capital risk constraint.

- **VecNormalize**: Normalises observations only (`norm_reward=False`). Stats are saved
  alongside every checkpoint so evaluation always uses matched normalisation.

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
