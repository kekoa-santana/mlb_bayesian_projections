# Prompt for Claude Code — Batter hit widening follow-up (post-implementation)

Copy everything below the line into Claude Code, or @-reference this file.

---

## You are in

Repo: `player_profiles` (branch: use current branch; this work is related to `batter-hit` / hit-differentiation widening). Python project with game sims under `src/models/game_sim/` and precompute in `scripts/precompute/confident_picks.py`.

## Context (already implemented — do not redo unless you find bugs)

Recent uncommitted (or just-committed) work did three things:

1. **`confident_picks.py`:** Read `h_babip_adj` from `data/dashboard/park_factor_lifts.parquet` (via `h_babip_adj` in the venue dict), set `park_h_babip_adj` per game, and pass it into `simulate_lineup_game`, `simulate_batter_game` (TB path, which also received `park_k_lift`, `park_bb_lift`, `park_hr_lift`), and the full-game both-teams sim. Stale comment about "H/BB skipped" was replaced with accurate text.

2. **`src/models/game_sim/bip_model.py`:** Scaled quality-metric `_BABIP_COEFS` ~2×, retargeted intercept for ~0.295 league-average pred, `shrinkage_k` 300→150, widened `pred_babip` and post-adjustment hit-probability clamps.

3. **`src/utils/constants.py`:** `BABIP_LD_COEFF` 0.25 → 0.60.

`pytest tests/test_sim_consistency.py` passed after these changes.

## Your goals (execute in order)

1. **Docs sync**  
   Update `docs/batter_hits_signal_execution_plan.md` so it is not stale: Phase 1 items that are *done* (loading `h_babip_adj`, passing `park_h_babip_adj` into lineup + batter TB + full-game sim) should be checked off or marked complete. Remove or reword any sentence that still claims `h_babip` is "not read" in `confident_picks`. Keep the doc as a future roadmap for Phases 2–3+.

2. **Optional parity: pitcher `simulate_game` vs batter paths**  
   In `confident_picks.py`, the pitcher sim uses `simulate_game(..., game_context=GameContext(...))` with only a subset of fields; it does **not** set `park_k_lift`, `park_bb_lift`, `park_hr_lift`, or `park_h_babip_adj` on `GameContext` (see `src/models/game_sim/pa_outcome_model.py` — `GameContext` has these).  
   - Decide whether the **opposing lineup** in the **pitcher** game sim should receive the same **park** context as the **batter** sims for the same `game_pk`.  
   - If yes: pass the same per-game `park_k_lift`, `park_bb_lift`, `park_hr_lift`, and `park_h_babip_adj` into `GameContext` for the pitcher `simulate_game` call (and any `simulate_game` / shared paths that need it), without double-counting BABIP.  
   - If no: add a one-line code comment at that call site explaining why park is omitted for the pitcher sim.  
   Prefer minimal, focused diffs; do not refactor unrelated sim code.

3. **Evidence of impact (lightweight)**  
   Add a small, repeatable check — choose one:  
   - a short script under `scripts/` (e.g. one-off `qa_compare_h_distribution.py`) that loads pre/post `game_props` or runs a **tiny** n_sims with fixed seed, two venues (one with `h_babip_adj` near 0, one meaningfully non-zero) and prints mean `E[H]` and/or `p_over_1.5` for the same notional player row; **or**  
   - a focused unit test with mocked park lifts proving `park_h_babip_adj` changes BIP or hit output in the expected *direction* when everything else is fixed.  
   The deliverable is something a human can run to see that widening did not silently no-op.

4. **Calibration reminder (no full backtest required unless asked)**  
   In the PR description or a short `docs/` note (only if the user approves a second file — otherwise put it in the PR body only), record that the paper `docs/paper_v1_calibrated_game_props.md` cites strong batter-H calibration; after BIP + LD + clamp changes, **ECE for batter H should be re-check** on a walk-forward or held-out season — do *not* claim calibration is unchanged without running metrics.

## Constraints

- No drive-by refactors, no reformat of unrelated files.  
- If `park_factor_lifts.parquet` sometimes lacks `h_babip_adj`, keep `pr.get("h_babip_adj", 0.0)` behavior.  
- After edits, run: `python -m pytest tests/test_sim_consistency.py -q` and any tests you touch; fix failures you introduce.

## Definition of done

- [ ] `batter_hits_signal_execution_plan.md` reflects current reality.  
- [ ] Pitcher sim: either **park in GameContext** or **documented** intentional omission.  
- [ ] One QA script or test proving `park_h_babip` / widening affects output as intended.  
- [ ] Tests you rely on are green.  
- [ ] User gets a 3–5 sentence summary of files changed and how to run the QA check.

---

*Derived from the internal “audit of hit differentiation widening work.”*
