# Claude Code Instructions — Simple Trader R1

## Project Overview

RL trading agent (RecurrentPPO / LSTM) trained on ES/NQ futures using Stable-Baselines3.
See README.md for full architecture, file layout, and training pipeline details.

---

## Critical ⚠️ Project Rules — DO NOT CHANGE

**Do NOT remove, rename, or stub out any function, class, method, callback, or file unless explicitly requested.**
ALWYAS USE CMD instead of PowerShell.

### Critical TP and SL Configuration

**Take-Profit and Stop-Loss settings are tightly coupled and must remain synchronized.**

**DO NOT modify** `take_profit.swing_multiplier` or stop-loss placement without explicit user request.

**Why:**
- **Take-Profit**: Controlled by `risk_config.yaml → take_profit.swing_multiplier` (default 0.6)
  - TP = entry + (swing - entry) × multiplier
  - Currently set to 60% of swing-high/low distance
  - Located in: `environment/trading_env.py:_compute_target_price()` (lines 1140–1148)

- **R:R Gate**: The `min_rr_ratio: 1.75` gate in `risk_config.yaml` uses the **FULL swing distance (no multiplier)**
  - TP target = entry + (swing - entry) × 0.6 (easier hits)
  - R:R validation = (full swing - entry) / (entry - stop) (conservative entry)
  - This DECOUPLES swing_multiplier from R:R validation
  - Gate calculation: `environment/trading_env.py:_place_pending_order()` (lines 949–976)

- **Stop-Loss**: Placed `stop_buffer_pts` beyond the zone far edge (never widened)
  - Located in: `environment/position_manager.py` and `trading_env.py:_place_pending_order()`
  - SL widening is hard-blocked with `allow_stop_widening: false`

**Note on swing_multiplier**: Changing it only affects TP distance, NOT entry acceptance (R:R gate uses full swing). Safer to tune TP without destabilizing entry validation.

### Conservative Same-Candle Exit Rule

**Location:** `environment/position_manager.py` — `check_exit()` method (lines 440–510)

**Rule:** When a position **enters and exits within the same 5-minute candle**, apply conservative penalties:
1. **Stop loss is skipped** — Stop cannot trigger on the entry candle (SL only active from next bar onward)
2. **Take-profit is forced to loss** — If TP is touched same-candle, exit at SL price (marked as loss) instead
3. **Ambiguous both-hit candle** — If both SL and TP touched in same candle, assume SL was hit first (worst-case)

**Rationale:** Prevents agent from exploiting unreliable intrabar fills and sweep-and-recover moves. One-candle trades are unlikely to be genuine; the conservative exit forces multi-candle holding periods.

**Code signature:**
```python
same_bar_entry = (current_bar_idx == s.entry_bar_idx)
if same_bar_entry and s.target_price > 0:
    exit_price = active_stop  # Exit at SL, not TP
    exit_reason = STOP_LOSS
```

**Do NOT remove, weaken, or bypass this rule without explicit user approval.**

---

## General Coding Principles

These principles directly address common issues with AI-assisted development:

### 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

LLMs often pick an interpretation silently and run with it. This principle forces explicit reasoning:
- State assumptions explicitly — If uncertain, ask rather than guess
- Present multiple interpretations — Don't pick silently when ambiguity exists
- Push back when warranted — If a simpler approach exists, say so
- Stop when confused — Name what's unclear and ask for clarification

### 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

Combat the tendency toward overengineering:
- No features beyond what was asked
- No abstractions for single-use code
- No "flexibility" or "configurability" that wasn't requested
- No error handling for impossible scenarios
- If 200 lines could be 50, rewrite it

The test: Would a senior engineer say this is overcomplicated? If yes, simplify.

### 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting
- Don't refactor things that aren't broken
- Match existing style, even if you'd do it differently
- **Dead code/unused variables**: Flag them for user review — don't delete without approval

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused
- Don't remove pre-existing dead code unless asked

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform imperative tasks into verifiable goals:
- Instead of "Add validation" → "Write tests for invalid inputs, then make them pass"
- Instead of "Fix the bug" → "Write a test that reproduces it, then make it pass"
- Instead of "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

---

## Session Start Workflow

At the beginning of every conversation, before doing anything else:
1. Read `README.md` to understand the current state of the project.
2. Think before coding — surface assumptions and tradeoffs before implementation.
3. Plan systematically — set up a todo list and work file-by-file.
4. Make surgical edits only — change what you're asked to change, match existing style.
5. No speculative features — just the code requested, nothing more.

## Code Quality & Maintenance

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting
- Don't refactor things that aren't broken
- Match existing style, even if you'd do it differently
- **Dead code/unused variables**: Flag them for user review — don't delete without approval

When low on context:
- Edit only the specific lines/functions you were asked to change
- Leave all surrounding code untouched
- Do not replace working implementations with `pass` or `raise NotImplementedError`
- If unsure whether something is used, assume it IS used and leave it alone

After changes:
- Update README.md to reflect what changed
- Run tests in **cmd** (not PowerShell)
- Review code for bottlenecks and report performance implications

Model/journal outputs:
- Maintain clean, tidy, intuitive directory structure — no messy files/folders
