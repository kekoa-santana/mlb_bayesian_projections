# One-Stop Baseball Analytics Plan (ETL + Modeling + Dashboard)

## Summary
Build a unified, production-ready analytics platform across three repos:
1. `mlb_fantasy_ETL` as the canonical data platform.
2. `player_profiles` as the canonical modeling/backtest platform.
3. `tdd-dashboard` as the canonical product/UI platform.

The roadmap prioritizes reliability and contracts first, then feature expansion. This order is required because current growth is bottlenecked by pipeline hardening gaps (execution edge cases, missing automated tests/CI, and manual sync drift).

## Scope
In scope:
1. End-to-end data-to-product reliability.
2. Shared data contracts and season/version configuration.
3. Automated update operations (daily + intraday).
4. Product expansion for “one-stop shop” analytics pages.
5. Measurable QA and observability.

Out of scope for this plan window:
1. Replacing Streamlit with a new frontend framework.
2. Replacing PostgreSQL or replatforming infrastructure.
3. Rewriting Bayesian model families from scratch.

## Target Architecture Decisions
1. **Source-of-truth boundaries**
- ETL truth: `mlb_fantasy_ETL`.
- Model truth: `player_profiles`.
- UI truth: `tdd-dashboard`.
- Dashboard must not add private model logic; it consumes precomputed artifacts.

2. **Contract-first integration**
- Introduce versioned artifact contracts for all files in `data/dashboard/`.
- Every producer run emits schema/version metadata.
- Every consumer run validates contract before use.

3. **Season/config centralization**
- Remove hardcoded season constants from runtime logic.
- Add centralized season/runtime config loaded by all repos.

4. **No manual drift**
- Replace manual “sync-by-copy” habits with scripted sync + verification checks for shared modules.

## Public APIs / Interfaces / Type Changes
1. Add `config/runtime.yaml` in each repo (same schema).
- Fields: `current_season`, `train_start_season`, `train_end_season`, `supported_historical_seasons`, `game_types_in_scope`, `refresh_windows`, `schema_version`.
- Default: `current_season=2026`.

2. Add artifact manifest contract `data/dashboard/manifest.json`.
- Fields: `artifact_name`, `schema_version`, `row_count`, `column_hash`, `generated_at`, `producer_repo`, `producer_commit`.
- Produced by `player_profiles` precompute and `update_in_season`.
- Validated by `tdd-dashboard` loaders on startup.

3. Add ETL run metadata table `production.etl_run_log` and DQ table `production.etl_dq_results`.
- Types: `run_id`, `pipeline_step`, `status`, `started_at`, `ended_at`, `row_count`, `error_text`, `dq_check_name`, `dq_status`, `dq_details`.

4. Add CLI contracts:
- `mlb_fantasy_ETL`: `python full_pipeline.py --mode {daily,intraday,backfill} --validate-only`.
- `player_profiles`: `python scripts/precompute_dashboard_data.py --contract-out data/dashboard/manifest.json`.
- `tdd-dashboard`: startup flag `--strict-contracts`.

5. Add dashboard page interfaces:
- `Model Performance` page: consumes backtest outputs and contract metadata.
- `Data Health` page: consumes ETL/model run logs and freshness checks.
- `Stats` page added to nav as a first-class route.

## Workstreams
1. **ETL Hardening (`mlb_fantasy_ETL`)**
- Fix `full_pipeline.py` skip-ingestion variable pathing.
- Fix DQ import path mismatch in analysis scripts.
- Convert print-based diagnostics to structured logging.
- Add deterministic batch size settings for staging upserts.
- Add explicit upsert policy switch (`prefer_incoming` vs `fill_null_only`) and set default to `prefer_incoming` for correction-friendly reloads.

2. **Model Pipeline Contracting (`player_profiles`)**
- Emit manifest for every dashboard artifact output.
- Add contract tests that fail on breaking schema drift.
- Include backtest summary export in stable machine-readable format for dashboard ingestion.

3. **Dashboard Productization (`tdd-dashboard`)**
- Break `app.py` into modules by page + shared services.
- Introduce typed data-access layer with contract validation.
- Add missing Stats nav route.
- Add Model Performance and Data Health pages.
- Remove hardcoded 2025/2026 references from logic and labels.

4. **Operations & Automation**
- Scheduled daily ETL + model updates.
- Intraday schedule/lineup/game sim refresh every 15 minutes during game windows.
- Alerting for stale artifacts, failed contracts, and failed ETL/model steps.

## Delivery Plan (8 Weeks, 4 Milestones)

### Milestone 1 (Weeks 1-2): Reliability Baseline
1. ETL fixes and contract scaffolding.
2. Add ETL unit tests and smoke integration tests.
3. Add CI workflows in all repos (lint + tests + contract checks).
4. Deliverable: green CI on PR; reproducible daily run from clean environment.
5. Exit criteria:
- ETL runs succeed in `daily` and `validate-only` modes.
- DQ scripts execute without import errors.
- No hardcoded path dependencies for core run modes.

### Milestone 2 (Weeks 3-4): Contracted Integration
1. Implement manifest production in `player_profiles`.
2. Implement manifest validation in `tdd-dashboard`.
3. Add shared runtime config loading.
4. Scripted sync/verification for shared modules used by dashboard.
5. Deliverable: end-to-end run with strict contract gate.
6. Exit criteria:
- Dashboard startup fails fast on contract mismatch.
- All required artifacts validated before page rendering.
- Season/year labels sourced from config, not constants.

### Milestone 3 (Weeks 5-6): One-Stop Product Layer v1
1. Add `Model Performance` page.
2. Add `Data Health` page.
3. Add `Stats` page to navigation.
4. Implement baseline mobile layout improvements for key pages.
5. Deliverable: unified analytics + model accountability + pipeline status in app.
6. Exit criteria:
- Predicted-vs-actual metrics visible by stat and timeframe.
- Freshness and run status visible to end users.
- App remains performant under strict contracts.

### Milestone 4 (Weeks 7-8): Decision Tools and Intraday Ops
1. Ship intraday scheduler for live updates.
2. Add optimizer sandbox prototype (lineup/trade what-if) backed by existing projections.
3. Add team-level outlook cards (projected team wins proxy based on roster aggregation).
4. Deliverable: first version of “decision support” layer.
5. Exit criteria:
- Intraday updates execute automatically during configured windows.
- Optimizer scenarios return stable outputs with guardrails.
- Team-level output is reproducible and documented.

---

## Phase 3: Advanced Modeling

### Milestone A: Expanded Projections

**Workstream: Contact Quality Model**
- [ ] Bayesian model for hitter xwOBA / barrel rate (extends beyond K%, BB%)
- [ ] Enables full offensive projections: AVG, OBP, SLG, HR with uncertainty
- [ ] Precompute and export to dashboard parquets

**Workstream: ERA Component Model**
- [ ] Model pitcher ERA components: HR/FB%, BABIP-allowed, strand rate
- [ ] Projected ERA/FIP/xFIP with uncertainty intervals
- [ ] Export to dashboard parquets

**Workstream: SB Projection Fix**
- [ ] Replace uniform SB era adjustment with speed-tier-specific factors
- [ ] Validate against Marcel on 2023-2025 holdout
- [ ] Update counting projections export

Exit criteria:
- Contact quality projections available for all qualified hitters
- ERA component projections available for all qualified pitchers
- SB projections competitive with Marcel baseline

### Milestone B: Intelligence Layer

**Workstream: Player Similarity & Comps**
- [ ] Build PECOTA-style comp system
  - Cosine similarity on (age, K%, BB%, contact quality, speed, role)
  - Historical comps: "Player X at age 27 looks like Player Y at age 27"
- [ ] Export comp data to dashboard parquets
- [ ] Use comp trajectories for long-range projection context

**Workstream: Fielding Integration**
- [ ] Ingest OAA (Outs Above Average) from Baseball Savant data
- [ ] Add to ETL pipeline or as standalone fetch
- [ ] Export to dashboard parquets

Exit criteria:
- Player comps available for all players with 2+ seasons
- OAA data ingested and exported

> **Note:** Dashboard surfacing of these features is tracked in `tdd-dashboard/PLAN.md` (Milestones 6-7).

---

## Testing Plan and Scenarios
1. **ETL tests**
- Unit: spec coercion, bounds handling, PK dedup behavior, retry behavior.
- Integration: one-day run into test DB schemas, verify row counts and keys.
- Contract: schema and nullability checks on raw/staging/production handoffs.

2. **Model tests (`player_profiles`)**
- Existing test suite remains required.
- Add artifact contract tests for every dashboard output file.
- Add regression tests for backtest summary output schema.
- Add calibration regression test: Brier score must not degrade > 5% vs baseline.

3. **Dashboard tests**
- Loader tests: missing/invalid manifest, schema mismatch, stale data detection.
- Page smoke tests: each route renders with fixture artifacts.
- E2E: startup -> load page -> inspect core metrics/cards.

4. **Ops tests**
- Scheduled job dry-run in non-production mode.
- Alert simulation on failed ETL step, missing artifact, stale timestamp.
- Recovery scenario: resume from failed intraday run.

## Rollout and Monitoring
1. Environments: local dev -> staging -> production.
2. Deployment gate: CI green + contract validation + smoke tests.
3. Monitoring metrics:
- ETL success rate by step.
- Artifact freshness lag.
- Dashboard render errors by page.
- Contract failure counts.
4. Alert thresholds:
- Daily pipeline failure: immediate alert.
- Intraday lag > 20 minutes during game windows: alert.
- Contract mismatch: blocking alert.

## Risks and Mitigations
1. Risk: Schema drift between repos.
- Mitigation: strict manifest contracts + CI contract checks.

2. Risk: Intraday API instability.
- Mitigation: retry policy + partial-refresh fallback + stale-data labeling.

3. Risk: Large monolith slows delivery.
- Mitigation: modularize app by page/data services in Milestone 3.

4. Risk: Hidden ETL correctness issues.
- Mitigation: formalize DQ scripts into CI-grade tests and run logs.

## Assumptions and Defaults
1. Database remains PostgreSQL `mlb_fantasy` on `localhost:5433`.
2. Core game scope defaults to regular season for projections; postseason remains optional context input.
3. Primary update cadence defaults:
- Daily full pipeline overnight.
- Intraday updates every 15 minutes during configured game windows.
4. Existing parquet artifact naming is preserved initially; versioning is added via manifest.
5. Existing Bayesian model families remain in place; this plan focuses on pipeline/product reliability and integration.

## Immediate Next Sprint Backlog (Decision-Complete)
1. Fix ETL orchestration edge cases and DQ import pathing.
2. Add CI workflow files to all three repos.
3. Add shared runtime config and remove hardcoded season constants from dashboard entry points.
4. Add manifest generation/validation path with strict mode.
5. Add dashboard `Stats` page to sidebar navigation.
6. Add initial `Data Health` page showing freshness and last successful update state.

Success criteria for next sprint:
1. Fresh clone can run daily pipeline and dashboard with contract validation enabled.
2. CI enforces tests + contracts on every PR.
3. Dashboard displays current season from config and surfaces data freshness status.
