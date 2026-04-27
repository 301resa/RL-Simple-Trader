# RL Simple Trader — R1

A Reinforcement Learning trading agent trained on ES/NQ futures using a single-pillar
Order Zone entry strategy (Supply/Demand Zone confluence + ATR room filter).

Built on **RecurrentPPO** (LSTM) via SB3/sb3_contrib. The agent processes one 5-minute
bar at a time — the LSTM hidden state carries session memory across bars without needing
a multi-bar lookback window in the observation vector.

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
| Supply/Demand Zone | 90% | Price is inside a valid order block zone (sonarlab/wugamlo/consolidation) |
| ATR Room | 10% | Directional ATR move < 85% of daily ATR in the entry direction |

A **liquidity sweep** is required before any entry is placed. Price must first trade
through the zone's liquidity level — sweep detection uses the **bar high/low** (not close)
so wick-and-return setups are captured correctly: supply is swept when `bar.high ≥ zone.top`;
demand is swept when `bar.low ≤ zone.bottom`. Once swept, a pending limit is placed to
catch the re-entry into the order block.

- Minimum confluence score: **0.55** (configurable in `features_config.yaml`)
- Minimum R:R ratio: **1.75:1** (hard gate in `_place_pending_order`, reads `risk_config.yaml → take_profit.min_rr_ratio`) — pending order is rejected if the actual computed target/stop prices yield R:R below this threshold

### Pending Limit Order Entry

Entry requires a **liquidity sweep** first (see `was_swept` flag detected by
`zone_detector.py` via bar high/low). Once swept, a limit is placed at the **near edge**
of the zone — the first edge price touches as it returns into the block:

- **LONG**: limit at `demand.top` (near edge) — fills when `bar.low ≤ demand.top`
- **SHORT**: limit at `supply.bottom` (near edge) — fills when `bar.high ≥ supply.bottom`
- **Sweep gate**: orders are skipped if `zone.was_swept == False` — no order until liquidity taken.
- **Zone width filter**: zones outside `[min_zone_pts, max_zone_pts]` are skipped (from the instrument profile — e.g. ES: 1–10 pts, NQ: 4–40 pts).
- **50% midpoint guard**: a pending order is rejected when the current bar has already violated the inner half of the zone — LONG blocked if `bar.low < zone.midpoint`; SHORT blocked if `bar.high > zone.midpoint`. Configurable via `features_config.yaml → zones.detection_mode`.
- **Hard R:R gate**: after computing the actual limit, stop, and target prices, the pending order is rejected if `(target − limit) / (stop − limit) < min_rr_ratio` (reads `risk_config.yaml → take_profit.min_rr_ratio`, currently **1.75**).
- **Sonarlab Order Block detection** (`detection_mode: "sonarlab"`, **default**): ROC-triggered institutional OB method. A 4-bar open-to-open Rate of Change is computed each bar; when ROC crosses ±`ob_sensitivity/100`% (default 28 → ±0.28%), the detector walks back 4–15 bars to find the *first opposing candle* — that candle is the true institutional order block origin. Bullish cross (ROC crosses above +threshold) → look back for last bearish candle → demand zone `[low, high]`; Bearish cross → look back for last bullish candle → supply zone `[low, high]`. `was_swept=True` at creation (momentum cross confirms price left the zone). 5-bar cooldown prevents duplicate triggers on the same move. Configurable: `ob_sensitivity` (ROC threshold × 0.01). Active by default at `detection_mode: "sonarlab"` in `features_config.yaml`.
- **Wugamlo Order Block detection** (`detection_mode: "wugamlo"`): zones formed at the *last opposing candle before N consecutive same-direction candles*. Bullish OB = last red candle before `ob_length` consecutive green candles → demand zone `[low, open]`; Bearish OB = last green candle before `ob_length` consecutive red candles → supply zone `[open, high]`. An optional `ob_threshold_pct` filters for minimum % move. `ob_use_wicks: true` extends the zone to the full `[low, high]`. `was_swept=True` at creation. Produces ~50% more zones than consolidation mode with 100% sweep rate.
- **Cut-through invalidation**: a zone is killed the moment a bar pierces past its far
  edge by more than `stop_buffer_pts` (or closes past the far edge — no buffer). Demand
  dies on `bar.low < zone.bottom − stop_buffer_pts` *or* `bar.close < zone.bottom`; supply
  mirrors. `zone_detector.py` drops the zone from `nearest_supply/demand` immediately,
  and `trading_env.step()` also cancels the pending limit on the *same bar* if the
  current candle reaches the stop level on the wrong side — preventing the classic
  enter-and-stop-on-the-same-candle failure mode.
- **No duplicate pending at the same spot**: `_place_pending_order` rejects a new
  order that matches the existing pending's direction AND limit price (within one
  tick). The original pending is kept — its `placed_bar_idx` and expiry countdown
  are NOT reset. Re-issuing `ENTER_LONG`/`ENTER_SHORT` on the same setup cannot
  stack orders or indefinitely extend an old pending's life. A different direction
  or a different price replaces the existing pending.
- **In-bar fill invariant**: a pending limit can only fill when
  `bar_low ≤ limit_price ≤ bar_high`. `_check_pending_fill` gates the fill and
  `_open_from_pending` re-validates the same inequality before calling
  `PositionManager.enter()`. If the invariant is ever violated the fill is
  REFUSED (logged as a warning) and the pending is held for the next bar —
  no more "fills outside the candle" regressions.
- **Cancellation**: all pending orders are cancelled automatically at session end, or
  when the agent places a valid, different-spot entry signal (cancel-and-replace).
  An `ENTER_*` action on an invalid/unswept/out-of-range zone no longer silently
  cancels the existing pending — the original is preserved.
- **Agent EXIT action** while flat also cancels any open pending order.
- **No blind-entry fallback**: if no valid, swept, un-pierced zone exists in the chosen
  direction, `_place_pending_order` skips without placing a limit at current price.

The observation vector includes pending order context: `pending_active`, `pending_direction`,
`pending_dist_norm` (distance from current price to limit level, ATR-normalised).
Zone features also include `supply_swept` and `demand_swept` binary flags.

### Stop Loss

Stop is placed **beyond the far edge** of the zone with an instrument-profile buffer
(e.g. ES = 1.5 pts = 6 ticks; NQ = 6.0 pts = 24 ticks). Risk per trade = `zone_width + stop_buffer`.

```
LONG:  stop = demand.bottom − stop_buffer_pts   (below the far/lower edge of demand)
SHORT: stop = supply.top    + stop_buffer_pts   (above the far/upper edge of supply)
```

Stop widening is **never** allowed once placed.

### Take Profit

Three-priority target selection (`_compute_target_price`):

1. **Opposing zone near edge** *(primary)* — the structural liquidity level on the far side of the range:
   - LONG: `nearest_supply.bottom` (bottom of the nearest overhead supply zone)
   - SHORT: `nearest_demand.top` (top of the nearest underlying demand zone)
   This mirrors how the trade is entered: range low for shorts, range high for longs.

2. **Impulse extreme of the entry zone** *(fallback 1)* — the actual high (supply) or low (demand)
   of the impulse bar that confirmed the zone, used when no valid opposing zone is present.

3. **ATR projection** *(fallback 2)* — 1× ATR from entry, used only when both zone-based
   targets are unavailable.

### Trailing Stop

The trailing stop is a **two-phase mechanism**:

| Phase | Trigger | Behaviour |
|-------|---------|-----------|
| Activation | Agent selects `TRAIL_STOP` while unrealised P&L ≥ `trail_activate_r` (default 1.2R) | `trailing_active = True`; initial stop level set at `current_price − trail_dist` |
| Ratcheting | Every subsequent bar while `trailing_active` | Stop advances mechanically toward price — agent does **not** need to keep pressing `TRAIL_STOP` |

Once active, the stop is a one-way ratchet: it only moves in the profitable direction (`max` for LONG, `min` for SHORT). The agent's only remaining decision is when to activate it.

### Risk Per Trade

- **1% of account equity** per trade (capital-based sizing)
- **Per-instrument contract tiers** (from `environment_config.yaml → contracts.<SYMBOL>.contract_tiers`):
  - **Minis** (ES / NQ): `[1, 2]` — integer contracts only, hard cap of 2
  - **Micros** (MES / MNQ): `[1, 2, 3, 5, 8]` — integer contracts only, higher tiers allowed since $ per tick is smaller
- **Three-constraint sizing** — smallest of all three wins:

  1. **Capital constraint**: `risk_dollars / (stop_pts × point_value)` → never exceeds 1% risk
  2. **Confluence constraint**: graduated score gates the maximum tier (thresholds from the
     instrument profile — ES/NQ use `[0.75]` → 1 contract below, 2 contracts at/above).
  3. **Zone-width constraint**: wider zones carry more uncertainty. Size is scaled by
     `max(0.50, 1 − zone_width / max_zone_pts)` — a zone at half the instrument's max
     width gets 75% size; at the max gets 50%.

- **Daily budget guard**: a single trade cannot consume more than **50% of the
  remaining daily loss budget** — if the day is already down $2,100 of the $3,000
  limit, the next trade is capped at $450 risk (50% × $900 remaining).

- **Dual daily loss limit**: stops trading when **either** limit is reached:
  - **−3R** on the day, OR
  - **−$3,000** on the day (configurable via `max_daily_loss_dollars` in `risk_config.yaml`)

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
Every run **automatically deletes** all `.xlsx` and `*_<suffix>.html` files in
`--out-dir` before writing new results. Produces:
- Per-model Plotly HTML journal (candlestick + per-trade PnL bars + cumulative PnL)
- Per-model Excel journal (Trades / Daily / Metrics sheets)
- `leaderboard.xlsx` — all models ranked best-first by composite score

Extra flags (passed through to `evaluation/test_fold.py`):
- `--best-only` — only write a journal for the single best model.
- `--rank-by` — `score` (default) | `pnl_dollars` | `pnl_r` | `sharpe`.
- `--journal-suffix` — suffix used in journal filenames (default: `journal`).

### Training journal (pick best model on TRAIN data, one HTML)
```bash
python main.py --mode training_journal \
    --models-dir logs/walk_forward/fold_00/models \
    --config config/ --data data/ \
    --train-start 2020-01-02 --train-end 2025-12-31
```
Runs every saved model deterministically across the training date range with
**real, un-augmented OHLC bars**, ranks by total dollar PnL, and writes a
single comprehensive Plotly HTML + Excel journal for the winner:

```
logs/walk_forward/fold_00/training_journal/
├── <best_model>_training_journal.html   # candlestick + entry/SL/TP + cum-PnL + trade table
├── <best_model>_training_journal.xlsx   # Trades / Daily / Metrics sheets
├── leaderboard.xlsx                      # all models ranked
└── test_fold_results.txt
```

The HTML mirrors the test-fold journal layout: OHLC candles with LONG/SHORT
entry markers, per-trade SL and TP lines, exit markers coloured by win/loss,
per-trade PnL bar chart, cumulative PnL curve, and a sortable trade-by-trade
data grid with `#, Date, Dir, Entry/Exit time, Entry/Exit price, SL, TP,
Lots, PnL (R/pts/$), Exit reason, MAE (R)`.

Use this to verify that training produced a working policy and that order
zones/entries land at sensible locations on the actual training data. If
`--models-dir` is omitted, it defaults to the first `logs/walk_forward/fold_*/models`
directory found; `--out-dir` defaults to `<fold>/training_journal`.

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

- **Near-edge entry, far-edge stop**: limit fills at the **near edge** of the zone
  (demand.top for LONG, supply.bottom for SHORT); stop sits **beyond the far edge** plus
  an instrument-profile buffer (ES: 6 ticks, NQ: 24 ticks). Risk per trade is therefore
  `zone_width + stop_buffer` — a clean invalidation if price pushes straight through the
  block. Zone-width bounds (`min_zone_pts` / `max_zone_pts` in the instrument profile)
  keep 1R from collapsing (too-tight blocks) or ballooning (oversized blocks).

- **No hard trade cap**: `max_trades_per_day: 999` — the dual daily loss limit (−3R
  or −$3,000) is the natural brake. The reward's `hold_flat_penalty` is applied
  **only when a valid order zone setup is present AND no pending order is already active**
  — the agent is penalised for ignoring a good setup, not for patiently waiting when
  no setup exists or correctly waiting for a pending limit to fill.

- **Pending limit order entry**: agent places a limit at the **near edge** of the zone
  (demand.top for LONG, supply.bottom for SHORT) — first touch into the zone after an
  external liquidity sweep of the prior swing. Filled on the bar that trades through the
  limit; cancelled at session end or on a new entry signal. Zone widths outside
  `[min_zone_pts, max_zone_pts]` (instrument profile) are skipped — tiny blocks have
  no room to work, oversized blocks balloon the risk.

- **Confluence-graded sizing**: position size is graded by confluence and zone-width
  tier against per-instrument `contract_tiers` (Minis: `[1, 2]`, Micros: `[1, 2, 3, 5, 8]`
  — integer only) and capped by the 1%-of-capital risk constraint — the smallest of the
  three caps wins.

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

- **Opposing-zone take-profit** (three-priority): primary target is the near edge of the
  opposing zone (`nearest_supply.bottom` for LONG, `nearest_demand.top` for SHORT) — the
  structural liquidity level at the far side of the range. Falls back to the entry zone's
  `impulse_extreme` when no opposing zone is present, then to ATR projection as last resort.

- **Stationary core reward**: `core_reward = pnl_r` (raw R-multiple). A previous
  `pnl_r × win_rate` formulation was removed — it violated reward stationarity because
  the same (state, action) pair yielded different rewards depending on session history.
  It also amplified loss penalties as WR rose, incentivising premature Agent-Exit to
  protect the multiplier. Win rate is observable in `portfolio_state`; the critic learns
  its importance without a hardcoded multiplier.

- **Always-on discipline penalties**: `hold_flat_penalty` and `penalty_per_bar`
  (overstay) are **never** multiplied by `shaping_scale`. Both are behavioural
  guardrails, not training wheels. Decaying the overstay penalty while keeping
  `hold_flat_penalty` active creates a Stage 3 exploit where the agent camps in
  stale breakeven trades to avoid the flat penalty. Both always remain live.

- **Masked-action penalty**: `RecurrentPPO` has no native action-masking support.
  When the agent selects an invalid action (overridden to `HOLD`), a fixed −0.05
  step penalty is applied. This teaches the actor network to push masked-action
  logits toward zero without relying on the environment override alone.

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
  - `ZoneState` is a module-level dataclass in `order_zone_engine.py`; zone geometry
    constants (`stop_buffer_pts`, `min_zone_pts`, `max_zone_pts`, `fallback_stop_pts`)
    now live on the `InstrumentProfile` loaded from `environment_config.yaml` and are
    passed into `OrderZoneEngine` at construction time.
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

| Symbol | Point Value | Tick Size | Tick Value | Stop Buffer | Min Zone | Max Zone |
|--------|-------------|-----------|------------|-------------|----------|----------|
| ES     | $50 / pt    | 0.25      | $12.50     | 6 ticks     | 4 ticks  | 40 ticks |
| MES    | $5  / pt    | 0.25      | $1.25      | 6 ticks     | 4 ticks  | 40 ticks |
| NQ     | $20 / pt    | 0.25      | $5.00      | 24 ticks    | 16 ticks | 160 ticks|
| MNQ    | $2  / pt    | 0.25      | $0.50      | 24 ticks    | 16 ticks | 160 ticks|

Contract tiers (from confluence-graded sizing):
- **Minis (ES / NQ)**: `[1, 2]` — integer contracts only, capped at 2.
- **Micros (MES / MNQ)**: `[1, 2, 3, 5, 8]` — integer contracts only.

### Switching instruments

Instrument switching is a **one-line config change**. In `config/environment_config.yaml`:

```yaml
instruments:
  default: NQ        # ES | MES | NQ | MNQ
```

All geometry (stop buffer, zone width bounds, jitter, target, contract tiers, confluence
thresholds) is expressed in **ticks** under `contracts.<SYMBOL>` and materialised into
points by `utils/instrument.py → load_instrument_profile()`. Every consumer
(`OrderZoneEngine`, `PositionManager`, `ObservationBuilder`, `TradingEnv`,
`OHLCVAugmentor`) accepts an `InstrumentProfile` and pulls its numbers from there — so
ES↔NQ↔MES↔MNQ requires no code edits.

---

## Dependencies

- `stable-baselines3` + `sb3-contrib` (RecurrentPPO)
- `gymnasium`
- `torch`
- `pandas`, `numpy`
- `plotly`, `openpyxl` (journals)
- `python-docx` (strategy document generation)
- `pyyaml`, `structlog`
