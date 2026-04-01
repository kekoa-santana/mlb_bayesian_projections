# Recommended improvements (end-to-end pipeline)

This document lists improvements discussed for the stack **mlb_fantasy_ETL → player_profiles → tdd-dashboard**, with **where** each change would be implemented. Paths are relative to each repo root unless noted.

---

## 1. Ingestion — incremental daily runs

**Goal:** Reduce redundant work on every daily run (e.g. transaction backfill from a fixed start date) while preserving correctness.

| Where | What |
|-------|------|
| `mlb_fantasy_ETL/full_pipeline.py` | Gate `fetch_and_load_transactions(txn_backfill_start, end_date)` behind a flag or “last successful txn watermark” so normal daily runs only ingest new/changed transactions; keep full backfill as an explicit mode. |
| `mlb_fantasy_ETL/ingestion/ingest_transactions.py` (or equivalent) | Implement idempotent incremental fetch by date range or high-water mark stored in DB/raw table metadata. |
| `mlb_fantasy_ETL/daily_refresh.py` (repo: `mlb_fantasy_ETL`, sibling to `player_profiles`) | Pass flags to `full_pipeline.py` for “daily” vs “full backfill” profiles. |

---

## 2. Ingestion — freshness and completeness checks

**Goal:** Detect silent partial failures (missing Statcast days, empty boxscore pulls) before production SQL runs.

| Where | What |
|-------|------|
| `mlb_fantasy_ETL/full_pipeline.py` | After ingestion, assert expected game count / pitch row counts vs schedule for `start_date`–`end_date`; fail fast or warn loudly. |
| `mlb_fantasy_ETL/ingestion/ingest_statcast.py` | Emit per-day row counts and optional checksums into a small `raw` or `staging` metadata table. |
| `mlb_fantasy_ETL/ingestion/ingest_boxscores.py` | Same pattern for boxscore coverage vs scheduled games. |
| `player_profiles/src/data/data_qa.py` (if extended) | Optional cross-repo read-only QA that queries Postgres and compares to yesterday’s counts (for dashboards / ops). |

---

## 3. Transformation / loading — Alembic and `CREATE INDEX CONCURRENTLY`

**Goal:** Migrations that use `CREATE INDEX CONCURRENTLY` must not run inside Alembic’s default transaction block (PostgreSQL error: *cannot run inside a transaction block*).

| Where | What |
|-------|------|
| `mlb_fantasy_ETL/alembic_fantasy/env.py` | Use `transaction_per_migration` / `autocommit` for migrations that issue `CONCURRENTLY`, or split those operations into manual SQL runbooks. |
| `mlb_fantasy_ETL/alembic_fantasy/versions/m8h9i0j1k2l3_add_pipeline_performance_indexes.py` (and similar) | Replace with non-concurrent indexes inside Alembic transactions, **or** move concurrent index creation to a separate script executed outside Alembic’s transactional DDL. |

Reference failure captured in `mlb_fantasy_ETL/output.md` (Alembic upgrade traceback).

---

## 4. Transformation / loading — roster rebuild performance and safety

**Goal:** Avoid full `TRUNCATE` + rebuild of `production.dim_roster` on every run if a cheaper incremental path is acceptable; improve operational safety.

| Where | What |
|-------|------|
| `mlb_fantasy_ETL/transformation/production/load_roster.py` | `_write_roster()` currently truncates `production.dim_roster` then inserts from temp; consider `DELETE`/`INSERT` by changed keys, or merge strategy keyed on `player_id` to shorten lock duration. |
| `mlb_fantasy_ETL/full_pipeline.py` | Optionally skip `build_roster()` when ingestion did not touch roster-driving tables (feature flag + dependency tracking). |

---

## 5. Modeling / precompute — known weak spots and deferred features

**Goal:** Close gaps called out in project docs (e.g. SB vs Marcel, in-season contact-quality / velocity trends).

| Where | What |
|-------|------|
| `player_profiles/scripts/precompute_dashboard_data.py` | Wire new sections or groups when new model outputs exist (`--include` groups in `_SECTION_GROUPS`). |
| `player_profiles/src/models/counting_projections.py` (and related SB logic) | Rework stolen-base projection / era adjustment (per CLAUDE.md “SB loses to Marcel”). |
| `player_profiles/src/models/*` (new or existing season models) | Add or defer “velocity trend acceleration” and “stabilized contact quality” as documented in CLAUDE.md Advanced Feature Triage. |
| `player_profiles/src/evaluation/counting_backtest.py` / `scripts/run_counting_backtest.py` | Validate SB (and any new stats) vs Marcel after changes. |

---

## 6. In-season updates — beyond K / BB / HR samples

**Goal:** Conjugate updating stays fast but reflects more of the season model (or dynamic league context).

| Where | What |
|-------|------|
| `player_profiles/scripts/update_in_season.py` | Extend `HITTER_RATE_STATS` / `PITCHER_RATE_STATS` and corresponding `update_rate_samples_step` / `update_hitter_rate_samples_step` calls for additional binomial outcomes you trust at season-to-date resolution (e.g. extra outcomes already in `fact_pa` or boxscore staging). |
| `player_profiles/src/models/in_season_updater.py` | Core Beta-Binomial / conjugate helpers; add functions or parameters for new stats and for time-varying league prior. |
| `player_profiles/scripts/update_in_season.py` | Replace hardcoded `league_avg=...` in `update_rate_samples_step` / `update_hitter_rate_samples_step` with values computed from DB league totals for the current season (or rolling window). |

---

## 7. Simulation — align calibration and context across two code paths

**Goal:** Daily K-only sim path and hourly PA-by-PA path should document and, where possible, share the same adjustments (umpire, weather, lineup source) and validation.

| Where | What |
|-------|------|
| `player_profiles/scripts/update_in_season.py` | `simulate_todays_games()` — add umpire/weather lifts if you want parity with dashboard PA sim; or explicitly document as “K-only fast path” in code comments and user-facing methodology. |
| `tdd-dashboard/scripts/update_in_season.py` | `run_schedule_refresh()` — already applies umpire/weather, TTO, pitch-count features, exit model; keep this as source of truth for “full prop” calibration or extract shared helpers. |
| `player_profiles/src/evaluation/game_prop_validation.py` | Extend or split validation so both artifact shapes (`todays_sims.parquet` from engine vs dashboard) are covered if both are used in production. |
| `player_profiles/scripts/run_game_prop_backtest.py` | Backtest configuration to include both sim modes if applicable. |

---

## 8. Dashboard — partial failure UX

**Goal:** Streamlit app degrades gracefully when some parquets are missing or stale instead of only hard-blocking.

| Where | What |
|-------|------|
| `tdd-dashboard/app.py` | After `check_data_exists()`, allow subset of `PAGES` when only optional artifacts fail; show banner with manifest warnings. |
| `tdd-dashboard/utils/helpers.py` | `check_data_exists()` — optional “minimum viable” vs “full” artifact sets. |
| `tdd-dashboard/views/data_health.py` | Surface missing optional artifacts and link to required precompute groups. |

---

## 9. Dashboard — observability for artifact size and load latency

**Goal:** Catch regressions when Parquet files grow or loaders slow down.

| Where | What |
|-------|------|
| `tdd-dashboard/services/data_loader.py` | Log or metric hooks (file size, load ms) behind debug flag or structured logging. |
| `tdd-dashboard/services/manifest.py` | Extend `generate_manifest()` to store file sizes and optional `pandas` dtypes hash; `validate_manifest()` strict mode for CI. |
| `tdd-dashboard/scripts/update_in_season.py` | Already writes `manifest.json`; could append timing section for each step. |

---

## 10. Cross-repo orchestration

**Goal:** Single place documents which repo owns which step (already mostly true); optional hardening.

| Where | What |
|-------|------|
| `mlb_fantasy_ETL/daily_refresh.py` | Document or enforce order: ETL → precompute groups → dashboard `update_in_season.py`; add exit-code policy when precompute is skipped. |
| `player_profiles/CLAUDE.md` | Keep “Dashboard Relationship” section in sync when artifact contracts change. |

---

## Quick reference: repo roots

| Repo | Typical root on this machine |
|------|------------------------------|
| ETL | `.../data_analytics/mlb_fantasy_ETL/` |
| Projection engine | `.../data_analytics/player_profiles/` |
| Dashboard | `.../data_analytics/tdd-dashboard/` |

Artifact output for the dashboard is written under **`tdd-dashboard/data/dashboard/`** by `player_profiles/scripts/precompute_dashboard_data.py` and by `player_profiles/scripts/update_in_season.py` (absolute `DASHBOARD_REPO` in that script should stay aligned with your machine or be made configurable via env).
