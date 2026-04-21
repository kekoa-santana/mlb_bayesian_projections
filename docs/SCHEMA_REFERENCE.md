# MLB Fantasy ETL — Full Schema Reference

> Generated 2026-03-13. Database configured via `.env` (see project root).

---

## Table of Contents

- [Overview](#overview)
- [Schema: raw](#schema-raw)
- [Schema: staging](#schema-staging)
- [Schema: production](#schema-production)
- [Schema: fantasy](#schema-fantasy)

---

## Overview

| Schema | Table | Approx Rows | PK |
|--------|-------|-------------|-----|
| **raw** | landing_boxscores | — | load_id |
| **raw** | dim_game | ~19.5k | game_pk |
| **raw** | batting_boxscores | ~458k | (batter_id, game_pk, team_id) |
| **raw** | pitching_boxscores | ~173k | (pitcher_id, game_pk, team_id) |
| **raw** | landing_statcast_files | — | run_id |
| **raw** | transactions | — | transaction_id |
| **staging** | statcast_pitches | ~5.6M | (game_pk, game_counter, pitch_number) |
| **staging** | statcast_at_bats | ~1.5M | (game_pk, game_counter) |
| **staging** | statcast_batted_balls | ~985k | (game_pk, game_counter, pitch_number) |
| **staging** | statcast_sprint_speed | ~3k | (player_id, season) |
| **staging** | batting_boxscores | ~458k | (batter_id, game_pk, team_id) |
| **staging** | pitching_boxscores | ~173k | (pitcher_id, game_pk, team_id) |
| **staging** | milb_batting_game_logs | — | (batter_id, game_pk, team_id) |
| **staging** | milb_pitching_game_logs | — | (pitcher_id, game_pk, team_id) |
| **staging** | live_pitches | 0 | (game_pk, at_bat_index, pitch_number) |
| **staging** | live_batted_balls | 0 | (game_pk, at_bat_index) |
| **production** | dim_game | 19,555 | game_pk |
| **production** | dim_player | 3,290 | player_id |
| **production** | dim_team | 30 | — |
| **production** | dim_prospects | 57,924 | (player_id, season) |
| **production** | dim_transaction | 417,644 | transaction_id |
| **production** | dim_park_factor | 566 | (venue_id, season, batter_stand) |
| **production** | dim_umpire | 19,553 | game_pk |
| **production** | dim_weather | 19,523 | game_pk |
| **production** | dim_model_run | 0 | run_id (UUID) |
| **production** | fact_pa | 1,402,858 | pa_id (BIGSERIAL); UK: (game_pk, game_counter) |
| **production** | fact_pitch | 5,449,981 | pitch_id (BIGSERIAL); UK: (game_pk, game_counter, pitch_number) |
| **production** | sat_pitch_shape | 5,449,981 | pitch_id |
| **production** | sat_batted_balls | 946,068 | pitch_id |
| **production** | fact_lineup | 458,657 | (game_pk, player_id) |
| **production** | fact_game_totals | 38,994 | (game_pk, team_id) |
| **production** | fact_player_game_mlb | 571,829 | (player_id, game_pk, player_role) |
| **production** | fact_milb_player_game | 2,056,586 | (player_id, game_pk, player_role) |
| **production** | fact_player_form_rolling | 571,829 | (player_id, game_pk, player_role) |
| **production** | fact_streak_indicator | 571,829 | (player_id, game_pk, player_role) |
| **production** | fact_platoon_splits | 25,280 | (player_id, season, player_role, platoon_side) |
| **production** | fact_matchup_history | 449,523 | (batter_id, pitcher_id) |
| **production** | fact_player_status_timeline | 237,741 | (player_id, status_start_date, status_type) |
| **production** | fact_prospect_snapshot | 57,924 | (player_id, season) |
| **production** | fact_prospect_transition | 15,489 | (player_id, event_date, from_level, to_level) |
| **production** | fact_player_projection | 0 | (run_id, player_id, as_of_date, horizon, scenario) |
| **production** | fact_projection_backtest | 0 | (run_id, player_id, target_date, metric) |
| **fantasy** | dk_batter_game_scores | 457,972 | (batter_id, game_pk) |
| **fantasy** | dk_pitcher_game_scores | 173,372 | (pitcher_id, game_pk) |
| **fantasy** | espn_batter_game_scores | 470,509 | (batter_id, game_pk) |
| **fantasy** | espn_pitcher_game_scores | 173,372 | (pitcher_id, game_pk) |

---

## Schema: raw

All raw tables store API values as text (`_text` suffix columns) before type casting in staging.

### raw.landing_boxscores

| Column | Type | Nullable |
|--------|------|----------|
| load_id | UUID | NO |
| ingested_at | TIMESTAMPTZ | YES |
| source | TEXT | NO |
| game_pk | BIGINT | NO |
| payload | TEXT | NO |

### raw.dim_game

| Column | Type | Nullable |
|--------|------|----------|
| game_pk | BIGINT | NO |
| home_team_id | INT | NO |
| away_team_id | INT | NO |
| game_date | DATE | NO |
| game_type | TEXT | NO |
| season | INT | YES |
| home_team_name | TEXT | YES |
| away_team_name | TEXT | YES |
| home_wins_text | TEXT | YES |
| home_losses_text | TEXT | YES |
| away_wins_text | TEXT | YES |
| away_losses_text | TEXT | YES |
| venue_id_text | TEXT | YES |
| doubleheader_text | TEXT | YES |
| day_night_text | TEXT | YES |
| games_in_series_text | TEXT | YES |
| series_in_game_number_text | TEXT | YES |

### raw.batting_boxscores

| Column | Type | Nullable |
|--------|------|----------|
| row_num | INT | YES |
| batter_id | BIGINT | NO |
| batter_name | TEXT | YES |
| game_pk | BIGINT | NO |
| team_id | INT | NO |
| team_name | TEXT | YES |
| position | TEXT | YES |
| ground_outs_text | TEXT | YES |
| air_outs_text | TEXT | YES |
| runs_text | TEXT | YES |
| doubles_text | TEXT | YES |
| triples_text | TEXT | YES |
| home_runs_text | TEXT | YES |
| strikeouts_text | TEXT | YES |
| walks_text | TEXT | YES |
| intentional_walks_text | TEXT | YES |
| hits_text | TEXT | YES |
| hit_by_pitch_text | TEXT | YES |
| at_bats_text | TEXT | YES |
| caught_stealing_text | TEXT | YES |
| sb_text | TEXT | YES |
| sb_pct_text | TEXT | YES |
| plate_appearances_text | TEXT | YES |
| total_bases_text | TEXT | YES |
| rbi_text | TEXT | YES |
| errors_text | TEXT | YES |
| source | TEXT | YES |
| load_id | UUID | YES |
| ingested_at | TIMESTAMPTZ | YES |

### raw.pitching_boxscores

| Column | Type | Nullable |
|--------|------|----------|
| row_num | INT | YES |
| pitcher_id | BIGINT | NO |
| pitcher_name | TEXT | YES |
| team_id | INT | NO |
| game_pk | BIGINT | NO |
| team_name | TEXT | YES |
| is_starter_text | TEXT | YES |
| fly_outs_text | TEXT | YES |
| ground_outs_text | TEXT | YES |
| air_outs_text | TEXT | YES |
| runs_text | TEXT | YES |
| doubles_text | TEXT | YES |
| triples_text | TEXT | YES |
| home_runs_text | TEXT | YES |
| strike_outs_text | TEXT | YES |
| walks_text | TEXT | YES |
| intentional_walks_text | TEXT | YES |
| hits_text | TEXT | YES |
| hit_by_pitch_text | TEXT | YES |
| at_bats_text | TEXT | YES |
| caught_stealing_text | TEXT | YES |
| stolen_bases_text | TEXT | YES |
| stolen_base_percentage_text | TEXT | YES |
| number_of_pitches_text | TEXT | YES |
| innings_pitched_text | TEXT | YES |
| wins_text | TEXT | YES |
| losses_text | TEXT | YES |
| saves_text | TEXT | YES |
| save_opportunities_text | TEXT | YES |
| holds_text | TEXT | YES |
| blown_saves_text | TEXT | YES |
| earned_runs_text | TEXT | YES |
| batters_faced_text | TEXT | YES |
| outs_text | TEXT | YES |
| complete_game_text | TEXT | YES |
| shutout_text | TEXT | YES |
| pitches_thrown_text | TEXT | YES |
| balls_text | TEXT | YES |
| strikes_text | TEXT | YES |
| strike_percentage_text | TEXT | YES |
| hit_batsmen_text | TEXT | YES |
| balks_text | TEXT | YES |
| wild_pitches_text | TEXT | YES |
| pickoffs_text | TEXT | YES |
| rbi_text | TEXT | YES |
| games_finished_text | TEXT | YES |
| runs_scored_per_9_text | TEXT | YES |
| home_runs_per_9_text | TEXT | YES |
| inherited_runners_text | TEXT | YES |
| inherited_runners_scored_text | TEXT | YES |
| catchers_interference_text | TEXT | YES |
| sac_bunts_text | TEXT | YES |
| sac_flies_text | TEXT | YES |
| passed_ball_text | TEXT | YES |
| pop_outs_text | TEXT | YES |
| line_outs_text | TEXT | YES |
| source | TEXT | YES |
| load_id | UUID | YES |
| ingested_at | TIMESTAMPTZ | YES |

### raw.landing_statcast_files

| Column | Type | Nullable |
|--------|------|----------|
| run_id | UUID | NO |
| pulled_at | TIMESTAMPTZ | NO |
| start_date | DATE | NO |
| end_date | DATE | NO |
| row_count | INT | NO |
| schema_hash | TEXT | NO |
| file_path | TEXT | NO |
| query_params | TEXT | NO |

### raw.transactions

| Column | Type | Nullable |
|--------|------|----------|
| transaction_id | BIGINT | NO |
| player_id | BIGINT | YES |
| player_name | TEXT | YES |
| to_team_id | INT | YES |
| to_team_name | TEXT | YES |
| from_team_id | INT | YES |
| from_team_name | TEXT | YES |
| date | DATE | NO |
| effective_date | DATE | YES |
| resolution_date | DATE | YES |
| type_code | TEXT | NO |
| type_desc | TEXT | YES |
| description | TEXT | YES |
| source | TEXT | YES |
| load_id | UUID | YES |
| ingested_at | TIMESTAMPTZ | YES |

---

## Schema: staging

Cleaned and typed versions of raw data. Statcast tables are loaded from parquet files.

### staging.statcast_pitches

| Column | Type | Nullable |
|--------|------|----------|
| game_pk | BIGINT | NO |
| game_counter | BIGINT | NO |
| game_date | DATE | YES |
| pitcher | BIGINT | NO |
| batter | BIGINT | NO |
| pitch_number | SMALLINT | NO |
| pitch_type | VARCHAR(10) | YES |
| pitch_name | TEXT | YES |
| description | TEXT | YES |
| release_speed | FLOAT4 | YES |
| release_pos_x | FLOAT4 | YES |
| release_pos_y | FLOAT4 | YES |
| release_pos_z | FLOAT4 | YES |
| release_spin_rate | FLOAT4 | YES |
| release_extension | FLOAT4 | YES |
| spin_axis | FLOAT4 | YES |
| effective_speed | FLOAT4 | YES |
| pfx_x | FLOAT4 | YES |
| pfx_z | FLOAT4 | YES |
| vy0 | FLOAT4 | YES |
| vx0 | FLOAT4 | YES |
| vz0 | FLOAT4 | YES |
| ax | FLOAT4 | YES |
| ay | FLOAT4 | YES |
| az | FLOAT4 | YES |
| zone | SMALLINT | YES |
| plate_x | FLOAT4 | YES |
| plate_z | FLOAT4 | YES |
| sz_top | FLOAT4 | YES |
| sz_bot | FLOAT4 | YES |
| p_throws | VARCHAR(1) | YES |
| stand | VARCHAR(1) | YES |
| balls | SMALLINT | YES |
| strikes | SMALLINT | YES |
| inning | SMALLINT | YES |
| on_3b | BIGINT | YES |
| on_2b | BIGINT | YES |
| on_1b | BIGINT | YES |
| outs_when_up | SMALLINT | YES |
| home_score | SMALLINT | YES |
| away_score | SMALLINT | YES |
| bat_score | SMALLINT | YES |
| fld_score | SMALLINT | YES |
| home_score_diff | SMALLINT | YES |
| bat_score_diff | SMALLINT | YES |
| if_fielding_alignment | TEXT | YES |
| of_fielding_alignment | TEXT | YES |
| arm_angle | FLOAT4 | YES |
| home_team | VARCHAR(3) | YES |
| away_team | VARCHAR(3) | YES |
| game_type | VARCHAR(1) | YES |
| pitch_result_type | TEXT | YES |
| is_bip | BOOL | YES |
| is_whiff | BOOL | YES |
| is_called_strike | BOOL | YES |
| is_ball | BOOL | YES |
| is_swing | BOOL | YES |
| is_foul | BOOL | YES |

### staging.statcast_at_bats

| Column | Type | Nullable |
|--------|------|----------|
| game_pk | BIGINT | NO |
| game_counter | BIGINT | NO |
| game_date | DATE | YES |
| pitcher | BIGINT | NO |
| batter | BIGINT | NO |
| inning | SMALLINT | YES |
| inning_topbot | VARCHAR(3) | YES |
| last_pitch_number | SMALLINT | YES |
| events | TEXT | YES |
| balls | SMALLINT | YES |
| strikes | SMALLINT | YES |
| outs_when_up | SMALLINT | YES |
| times_through_order | SMALLINT | YES |
| bat_score | SMALLINT | YES |
| fld_score | SMALLINT | YES |
| bat_score_diff | SMALLINT | YES |
| post_bat_score | SMALLINT | YES |
| outs_on_ab | SMALLINT | YES |
| is_bip | BOOL | YES |
| is_walk | BOOL | YES |
| is_strikeout | BOOL | YES |
| rbi | SMALLINT | YES |
| total_whiffs | SMALLINT | YES |
| total_pitches | SMALLINT | YES |
| total_called_strikes | SMALLINT | YES |
| total_swings | SMALLINT | YES |
| total_fouls | SMALLINT | YES |
| pitcher_pa_number | SMALLINT | YES |

### staging.statcast_batted_balls

| Column | Type | Nullable |
|--------|------|----------|
| game_pk | BIGINT | NO |
| game_counter | BIGINT | NO |
| game_date | DATE | YES |
| pitcher | BIGINT | NO |
| batter | BIGINT | NO |
| pitch_number | SMALLINT | NO |
| pitch_type | VARCHAR(10) | YES |
| bb_type | TEXT | YES |
| launch_speed | FLOAT4 | YES |
| launch_angle | FLOAT4 | YES |
| hit_distance_sc | FLOAT4 | YES |
| estimated_ba_using_speedangle | FLOAT4 | YES |
| estimated_woba_using_speedangle | FLOAT4 | YES |
| estimated_slg_using_speedangle | FLOAT4 | YES |
| babip_value | SMALLINT | YES |
| iso_value | SMALLINT | YES |
| woba_value | FLOAT4 | YES |
| hit_location | SMALLINT | YES |
| hc_x | FLOAT4 | YES |
| hc_y | FLOAT4 | YES |
| description | TEXT | YES |
| events | TEXT | YES |
| is_homerun | BOOL | YES |

### staging.statcast_sprint_speed

| Column | Type | Nullable |
|--------|------|----------|
| player_id | BIGINT | NO |
| season | INT | NO |
| player_name | TEXT | YES |
| team | TEXT | YES |
| position | TEXT | YES |
| age | SMALLINT | YES |
| competitive_runs | SMALLINT | YES |
| bolts | FLOAT4 | YES |
| hp_to_1b | FLOAT4 | YES |
| sprint_speed | FLOAT4 | YES |

### staging.batting_boxscores

| Column | Type | Nullable |
|--------|------|----------|
| batter_id | BIGINT | NO |
| game_pk | BIGINT | NO |
| team_id | INT | NO |
| batter_name | TEXT | YES |
| team_name | TEXT | YES |
| position | TEXT | YES |
| ground_outs | SMALLINT | YES |
| air_outs | SMALLINT | YES |
| runs | SMALLINT | YES |
| doubles | SMALLINT | YES |
| triples | SMALLINT | YES |
| home_runs | SMALLINT | YES |
| strikeouts | SMALLINT | YES |
| walks | SMALLINT | YES |
| intentional_walks | SMALLINT | YES |
| hits | SMALLINT | YES |
| hit_by_pitch | SMALLINT | YES |
| at_bats | SMALLINT | YES |
| caught_stealing | SMALLINT | YES |
| sb | SMALLINT | YES |
| sb_pct | FLOAT4 | YES |
| plate_appearances | SMALLINT | YES |
| total_bases | SMALLINT | YES |
| rbi | SMALLINT | YES |
| errors | SMALLINT | YES |
| source | TEXT | YES |
| load_id | UUID | YES |
| ingested_at | TIMESTAMPTZ | YES |

### staging.pitching_boxscores

| Column | Type | Nullable |
|--------|------|----------|
| game_pk | BIGINT | NO |
| pitcher_id | BIGINT | NO |
| team_id | BIGINT | NO |
| team_name | TEXT | YES |
| pitcher_name | TEXT | YES |
| is_starter | BOOL | YES |
| fly_outs | SMALLINT | YES |
| ground_outs | SMALLINT | YES |
| air_outs | SMALLINT | YES |
| runs | SMALLINT | YES |
| doubles | SMALLINT | YES |
| triples | SMALLINT | YES |
| home_runs | SMALLINT | YES |
| strike_outs | SMALLINT | YES |
| walks | SMALLINT | YES |
| intentional_walks | SMALLINT | YES |
| hits | SMALLINT | YES |
| hit_by_pitch | SMALLINT | YES |
| at_bats | SMALLINT | YES |
| caught_stealing | SMALLINT | YES |
| stolen_bases | SMALLINT | YES |
| stolen_base_pct | FLOAT4 | YES |
| number_of_pitches | SMALLINT | YES |
| innings_pitched | FLOAT4 | YES |
| wins | SMALLINT | YES |
| losses | SMALLINT | YES |
| saves | SMALLINT | YES |
| save_opportunities | SMALLINT | YES |
| holds | SMALLINT | YES |
| blown_saves | SMALLINT | YES |
| earned_runs | SMALLINT | YES |
| batters_faced | SMALLINT | YES |
| outs | SMALLINT | YES |
| complete_game | BOOL | YES |
| shutout | BOOL | YES |
| balls | SMALLINT | YES |
| strikes | SMALLINT | YES |
| strike_pct | FLOAT4 | YES |
| hit_batsmen | SMALLINT | YES |
| balks | SMALLINT | YES |
| wild_pitches | SMALLINT | YES |
| pickoffs | SMALLINT | YES |
| rbi | SMALLINT | YES |
| games_finished | BOOL | YES |
| runs_scored_per_9 | FLOAT4 | YES |
| home_runs_per_9 | FLOAT4 | YES |
| inherited_runners | SMALLINT | YES |
| inherited_runners_scored | SMALLINT | YES |
| catchers_interference | SMALLINT | YES |
| sac_bunts | SMALLINT | YES |
| sac_flies | SMALLINT | YES |
| passed_ball | SMALLINT | YES |
| pop_outs | SMALLINT | YES |
| line_outs | SMALLINT | YES |
| source | TEXT | YES |
| ingested_at | TIMESTAMP | YES |
| load_id | UUID | YES |

### staging.milb_batting_game_logs

| Column | Type | Nullable |
|--------|------|----------|
| batter_id | BIGINT | NO |
| game_pk | BIGINT | NO |
| team_id | INT | NO |
| batter_name | TEXT | YES |
| team_name | TEXT | YES |
| position | TEXT | YES |
| sport_id | SMALLINT | NO |
| level | TEXT | NO |
| parent_org_id | INT | YES |
| game_date | DATE | NO |
| season | INT | NO |
| ground_outs | SMALLINT | YES |
| air_outs | SMALLINT | YES |
| runs | SMALLINT | YES |
| doubles | SMALLINT | YES |
| triples | SMALLINT | YES |
| home_runs | SMALLINT | YES |
| strikeouts | SMALLINT | YES |
| walks | SMALLINT | YES |
| intentional_walks | SMALLINT | YES |
| hits | SMALLINT | YES |
| hit_by_pitch | SMALLINT | YES |
| at_bats | SMALLINT | YES |
| caught_stealing | SMALLINT | YES |
| sb | SMALLINT | YES |
| sb_pct | FLOAT4 | YES |
| plate_appearances | SMALLINT | YES |
| total_bases | SMALLINT | YES |
| rbi | SMALLINT | YES |
| errors | SMALLINT | YES |
| source | TEXT | YES |
| load_id | UUID | YES |
| ingested_at | TIMESTAMPTZ | YES |

### staging.milb_pitching_game_logs

| Column | Type | Nullable |
|--------|------|----------|
| game_pk | BIGINT | NO |
| pitcher_id | BIGINT | NO |
| team_id | BIGINT | NO |
| team_name | TEXT | YES |
| pitcher_name | TEXT | YES |
| sport_id | SMALLINT | NO |
| level | TEXT | NO |
| parent_org_id | INT | YES |
| game_date | DATE | NO |
| season | INT | NO |
| is_starter | BOOL | YES |
| fly_outs | SMALLINT | YES |
| ground_outs | SMALLINT | YES |
| air_outs | SMALLINT | YES |
| runs | SMALLINT | YES |
| doubles | SMALLINT | YES |
| triples | SMALLINT | YES |
| home_runs | SMALLINT | YES |
| strike_outs | SMALLINT | YES |
| walks | SMALLINT | YES |
| intentional_walks | SMALLINT | YES |
| hits | SMALLINT | YES |
| hit_by_pitch | SMALLINT | YES |
| at_bats | SMALLINT | YES |
| caught_stealing | SMALLINT | YES |
| stolen_bases | SMALLINT | YES |
| stolen_base_pct | FLOAT4 | YES |
| number_of_pitches | SMALLINT | YES |
| innings_pitched | FLOAT4 | YES |
| wins | SMALLINT | YES |
| losses | SMALLINT | YES |
| saves | SMALLINT | YES |
| save_opportunities | SMALLINT | YES |
| holds | SMALLINT | YES |
| blown_saves | SMALLINT | YES |
| earned_runs | SMALLINT | YES |
| batters_faced | SMALLINT | YES |
| outs | SMALLINT | YES |
| complete_game | BOOL | YES |
| shutout | BOOL | YES |
| balls | SMALLINT | YES |
| strikes | SMALLINT | YES |
| strike_pct | FLOAT4 | YES |
| hit_batsmen | SMALLINT | YES |
| balks | SMALLINT | YES |
| wild_pitches | SMALLINT | YES |
| pickoffs | SMALLINT | YES |
| rbi | SMALLINT | YES |
| games_finished | BOOL | YES |
| inherited_runners | SMALLINT | YES |
| inherited_runners_scored | SMALLINT | YES |
| catchers_interference | SMALLINT | YES |
| sac_bunts | SMALLINT | YES |
| sac_flies | SMALLINT | YES |
| passed_ball | SMALLINT | YES |
| source | TEXT | YES |
| ingested_at | TIMESTAMP | YES |
| load_id | UUID | YES |

### staging.live_pitches
**PK**: `(game_pk, at_bat_index, pitch_number)` — Live game feed pitch data (schema only, 0 rows).

| Column | Type | Nullable |
|--------|------|----------|
| game_pk | BIGINT | NO |
| at_bat_index | INT | NO |
| pitch_number | SMALLINT | NO |
| game_date | DATE | NO |
| pitcher_id | BIGINT | NO |
| batter_id | BIGINT | NO |
| inning | SMALLINT | YES |
| half_inning | TEXT | YES |
| pitch_type | TEXT | YES |
| pitch_name | TEXT | YES |
| release_speed | REAL | YES |
| spin_rate | REAL | YES |
| pfx_x | REAL | YES |
| pfx_z | REAL | YES |
| plate_x | REAL | YES |
| plate_z | REAL | YES |
| zone | SMALLINT | YES |
| sz_top | REAL | YES |
| sz_bot | REAL | YES |
| call_code | VARCHAR | YES |
| call_description | TEXT | YES |
| is_whiff | BOOLEAN | YES |
| is_called_strike | BOOLEAN | YES |
| is_swing | BOOLEAN | YES |
| is_bip | BOOLEAN | YES |
| is_foul | BOOLEAN | YES |
| balls | SMALLINT | YES |
| strikes | SMALLINT | YES |
| outs_when_up | SMALLINT | YES |
| bat_side | VARCHAR | YES |
| pitch_hand | VARCHAR | YES |
| launch_speed | REAL | YES |
| launch_angle | REAL | YES |
| hit_distance | REAL | YES |
| event_type | TEXT | YES |
| event | TEXT | YES |
| created_at | TIMESTAMPTZ | YES |

### staging.live_batted_balls
**PK**: `(game_pk, at_bat_index)` — Live game feed batted ball data (schema only, 0 rows).

| Column | Type | Nullable |
|--------|------|----------|
| game_pk | BIGINT | NO |
| at_bat_index | INT | NO |
| game_date | DATE | NO |
| batter_id | BIGINT | NO |
| pitcher_id | BIGINT | NO |
| batter_name | TEXT | YES |
| pitcher_name | TEXT | YES |
| team_id | INT | YES |
| inning | SMALLINT | YES |
| half_inning | TEXT | YES |
| launch_speed | REAL | YES |
| launch_angle | REAL | YES |
| hit_distance | REAL | YES |
| event_type | TEXT | YES |
| event | TEXT | YES |
| pitch_speed | REAL | YES |
| pitch_type | TEXT | YES |
| bat_side | VARCHAR | YES |
| pitch_hand | VARCHAR | YES |
| created_at | TIMESTAMPTZ | YES |

---

## Schema: production

### Dimensions

#### production.dim_game
**PK**: `game_pk`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| game_pk | BIGINT | NO | Unique game identifier |
| game_date | DATE | NO | |
| home_team_id | INT | NO | |
| away_team_id | INT | NO | |
| home_team_name | TEXT | YES | |
| away_team_name | TEXT | YES | |
| game_type | TEXT | YES | R=regular, P=playoff, W=wildcard, D=division, L=league, F=WS. Excludes E (exhibition), S (spring) |
| season | INT | YES | |
| home_team_wins | SMALLINT | YES | |
| home_team_losses | SMALLINT | YES | |
| away_team_wins | SMALLINT | YES | |
| away_team_losses | SMALLINT | YES | |
| venue_id | INT | YES | |
| doubleheader | TEXT | YES | |
| day_night | TEXT | YES | |
| games_in_series | SMALLINT | YES | |
| series_in_game_number | SMALLINT | YES | |

#### production.dim_player
**PK**: `player_id`

| Column | Type | Nullable |
|--------|------|----------|
| player_id | BIGINT | NO |
| player_name | TEXT | NO |
| team_id | BIGINT | NO |
| first_name | TEXT | YES |
| last_name | TEXT | YES |
| birth_date | DATE | YES |
| age | SMALLINT | YES |
| height | TEXT | YES |
| weight | SMALLINT | YES |
| active | BOOL | YES |
| primary_position_code | SMALLINT | YES |
| primary_position | VARCHAR(4) | YES |
| draft_year | SMALLINT | YES |
| mlb_debut_date | DATE | YES |
| bat_side | VARCHAR(1) | YES |
| pitch_hand | VARCHAR(1) | YES |
| sz_top | FLOAT4 | YES |
| sz_bot | FLOAT4 | YES |

#### production.dim_team

| Column | Type | Nullable |
|--------|------|----------|
| team_id | BIGINT | YES |
| team_name | TEXT | YES |
| full_name | TEXT | YES |
| abbreviation | TEXT | YES |
| venue | TEXT | YES |
| division | TEXT | YES |
| divsion_id | BIGINT | YES |
| location | TEXT | YES |

#### production.dim_prospects
**PK**: `(player_id, season)`

| Column | Type | Nullable |
|--------|------|----------|
| player_id | BIGINT | NO |
| full_name | TEXT | NO |
| first_name | TEXT | YES |
| last_name | TEXT | YES |
| primary_position | TEXT | YES |
| bat_side | TEXT | YES |
| pitch_hand | TEXT | YES |
| birth_date | DATE | YES |
| current_age | SMALLINT | YES |
| height | TEXT | YES |
| weight | SMALLINT | YES |
| milb_team_id | INT | YES |
| milb_team_name | TEXT | YES |
| parent_org_id | INT | YES |
| parent_org_name | TEXT | YES |
| sport_id | SMALLINT | NO |
| level | TEXT | NO |
| status_code | TEXT | YES |
| status_description | TEXT | YES |
| mlb_debut_date | DATE | YES |
| draft_year | SMALLINT | YES |
| season | SMALLINT | NO |
| jersey_number | TEXT | YES |
| updated_at | TIMESTAMPTZ | YES |

#### production.dim_transaction
**PK**: `transaction_id`

| Column | Type | Nullable |
|--------|------|----------|
| transaction_id | BIGINT | NO |
| player_id | BIGINT | NO |
| player_name | TEXT | YES |
| to_team_id | INT | YES |
| to_team_name | TEXT | YES |
| from_team_id | INT | YES |
| from_team_name | TEXT | YES |
| transaction_date | DATE | NO |
| effective_date | DATE | YES |
| resolution_date | DATE | YES |
| type_code | TEXT | NO |
| type_desc | TEXT | YES |
| description | TEXT | YES |
| is_il_placement | BOOL | YES |
| is_il_activation | BOOL | YES |
| is_il_transfer | BOOL | YES |
| il_type | TEXT | YES |
| injury_description | TEXT | YES |
| created_at | TIMESTAMPTZ | YES |

#### production.dim_park_factor
**PK**: `(venue_id, season, batter_stand)`

| Column | Type | Nullable |
|--------|------|----------|
| venue_id | INT | NO |
| season | INT | NO |
| batter_stand | VARCHAR(1) | NO |
| venue_name | TEXT | YES |
| hr_pf_season | FLOAT4 | YES |
| pa_season | INT | YES |
| hr_season | INT | YES |
| hr_pf_3yr | FLOAT4 | YES |
| pa_3yr | INT | YES |
| hr_3yr | INT | YES |
| created_at | TIMESTAMPTZ | YES |

#### production.dim_umpire
**PK**: `game_pk`

| Column | Type | Nullable |
|--------|------|----------|
| game_pk | BIGINT | NO |
| hp_umpire_name | TEXT | YES |
| created_at | TIMESTAMPTZ | YES |

#### production.dim_weather
**PK**: `game_pk`

| Column | Type | Nullable |
|--------|------|----------|
| game_pk | BIGINT | NO |
| temperature | SMALLINT | YES |
| condition | VARCHAR(20) | YES |
| is_dome | BOOL | YES |
| wind_speed | SMALLINT | YES |
| wind_direction | VARCHAR(20) | YES |
| wind_category | VARCHAR(5) | YES |
| created_at | TIMESTAMPTZ | YES |

#### production.dim_model_run
**PK**: `run_id` (UUID) — Empty, used by projection system

| Column | Type | Nullable |
|--------|------|----------|
| run_id | UUID | NO |
| model_name | TEXT | NO |
| model_version | TEXT | YES |
| feature_cutoff_date | DATE | YES |
| train_start_date | DATE | YES |
| train_end_date | DATE | YES |
| target_variable | TEXT | YES |
| hyperparameters | TEXT | YES |
| notes | TEXT | YES |
| created_at | TIMESTAMPTZ | YES |

---

### Core Facts

#### production.fact_pa
**PK**: `pa_id` (BIGSERIAL) | **UK**: `(game_pk, game_counter)`

| Column | Type | Nullable |
|--------|------|----------|
| pa_id | BIGINT | NO |
| game_pk | BIGINT | NO |
| pitcher_id | BIGINT | NO |
| batter_id | BIGINT | NO |
| game_counter | INT | NO |
| pitcher_pa_number | INT | YES |
| times_through_order | SMALLINT | YES |
| balls | SMALLINT | YES |
| strikes | SMALLINT | YES |
| outs_when_up | SMALLINT | YES |
| inning | INT | YES |
| inning_topbot | TEXT | YES |
| events | TEXT | YES |
| description | TEXT | YES |
| bat_score | SMALLINT | YES |
| fld_score | SMALLINT | YES |
| post_bat_score | SMALLINT | YES |
| bat_score_diff | SMALLINT | YES |
| last_pitch_number | SMALLINT | YES |
| created_at | TIMESTAMPTZ | YES |

#### production.fact_pitch
**PK**: `pitch_id` (BIGSERIAL) | **UK**: `(game_pk, game_counter, pitch_number)`

| Column | Type | Nullable |
|--------|------|----------|
| pitch_id | BIGINT | NO |
| pa_id | BIGINT | NO |
| game_pk | BIGINT | NO |
| pitcher_id | BIGINT | NO |
| batter_id | BIGINT | NO |
| game_counter | INT | NO |
| pitch_number | INT | NO |
| pitch_type | TEXT | YES |
| pitch_name | TEXT | YES |
| description | TEXT | YES |
| release_speed | FLOAT4 | YES |
| effective_speed | FLOAT4 | YES |
| release_spin_rate | FLOAT4 | YES |
| release_extension | FLOAT4 | YES |
| spin_axis | FLOAT4 | YES |
| pfx_x | FLOAT4 | YES |
| pfx_z | FLOAT4 | YES |
| zone | SMALLINT | YES |
| plate_x | FLOAT4 | YES |
| plate_z | FLOAT4 | YES |
| balls | SMALLINT | YES |
| strikes | SMALLINT | YES |
| outs_when_up | SMALLINT | YES |
| bat_score_diff | SMALLINT | YES |
| is_whiff | BOOL | YES |
| is_called_strike | BOOL | YES |
| is_bip | BOOL | YES |
| is_swing | BOOL | YES |
| is_foul | BOOL | YES |
| created_at | TIMESTAMPTZ | YES |
| batter_stand | VARCHAR(1) | YES |

---

### Satellites

#### production.sat_pitch_shape
**PK**: `pitch_id` (FK to fact_pitch)

| Column | Type | Nullable |
|--------|------|----------|
| pitch_id | BIGINT | NO |
| release_pos_x | FLOAT4 | YES |
| release_pos_y | FLOAT4 | YES |
| release_pos_z | FLOAT4 | YES |
| release_spin_rate | FLOAT4 | YES |
| release_extension | FLOAT4 | YES |
| release_speed | FLOAT4 | YES |
| spin_axis | FLOAT4 | YES |
| pfx_x | FLOAT4 | YES |
| pfx_z | FLOAT4 | YES |
| vx0 | FLOAT4 | YES |
| vy0 | FLOAT4 | YES |
| vz0 | FLOAT4 | YES |
| ax | FLOAT4 | YES |
| ay | FLOAT4 | YES |
| az | FLOAT4 | YES |
| plate_x | FLOAT4 | YES |
| plate_z | FLOAT4 | YES |
| sz_top | FLOAT4 | YES |
| sz_bot | FLOAT4 | YES |
| created_at | TIMESTAMPTZ | NO |

#### production.sat_batted_balls
**PK**: `pitch_id` (FK to fact_pitch)

| Column | Type | Nullable |
|--------|------|----------|
| pitch_id | BIGINT | NO |
| pa_id | BIGINT | YES |
| bb_type | TEXT | YES |
| events | TEXT | YES |
| launch_speed | FLOAT4 | YES |
| launch_angle | FLOAT4 | YES |
| hit_distance_sc | FLOAT4 | YES |
| hc_x | FLOAT4 | YES |
| hc_y | FLOAT4 | YES |
| hc_x_centered | FLOAT4 | YES |
| is_homerun | BOOL | YES |
| xba | FLOAT4 | YES |
| xslg | FLOAT4 | YES |
| xwoba | FLOAT4 | YES |
| woba_value | FLOAT4 | YES |
| babip_value | SMALLINT | YES |
| iso_value | SMALLINT | YES |
| hit_location | SMALLINT | YES |
| hard_hit | BOOL | YES |
| sweet_spot | BOOL | YES |
| ideal_contact | BOOL | YES |
| la_band | TEXT | YES |
| ev_band | TEXT | YES |
| spray_bucket | TEXT | YES |
| created_at | TIMESTAMPTZ | NO |

---

### Game-Level Facts

#### production.fact_lineup
**PK**: `(game_pk, player_id)`

| Column | Type | Nullable |
|--------|------|----------|
| game_pk | BIGINT | NO |
| player_id | BIGINT | NO |
| team_id | INT | NO |
| batting_order | SMALLINT | NO |
| is_starter | BOOL | NO |
| position | VARCHAR(4) | YES |
| home_away | VARCHAR(4) | YES |
| season | INT | YES |
| created_at | TIMESTAMPTZ | YES |

#### production.fact_game_totals
**PK**: `(game_pk, team_id)`

| Column | Type | Nullable |
|--------|------|----------|
| game_pk | BIGINT | NO |
| team_id | INT | NO |
| season | INT | YES |
| home_away | VARCHAR(4) | YES |
| runs | INT | YES |
| hits | INT | YES |
| doubles | INT | YES |
| triples | INT | YES |
| home_runs | INT | YES |
| walks | INT | YES |
| strikeouts | INT | YES |
| hit_by_pitch | INT | YES |
| sb | INT | YES |
| caught_stealing | INT | YES |
| at_bats | INT | YES |
| plate_appearances | INT | YES |
| total_bases | INT | YES |
| rbi | INT | YES |
| errors | INT | YES |
| created_at | TIMESTAMPTZ | YES |

#### production.fact_player_game_mlb
**PK**: `(player_id, game_pk, player_role)` — Unified batter+pitcher game log

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| player_id | BIGINT | NO | |
| game_pk | BIGINT | NO | |
| player_role | VARCHAR(7) | NO | 'batter' or 'pitcher' |
| game_date | DATE | YES | |
| season | INT | YES | |
| team_id | INT | YES | |
| bat_pa | SMALLINT | YES | Batter stats (NULL for pitchers) |
| bat_ab | SMALLINT | YES | |
| bat_h | SMALLINT | YES | |
| bat_2b | SMALLINT | YES | |
| bat_3b | SMALLINT | YES | |
| bat_hr | SMALLINT | YES | |
| bat_r | SMALLINT | YES | |
| bat_rbi | SMALLINT | YES | |
| bat_bb | SMALLINT | YES | |
| bat_k | SMALLINT | YES | |
| bat_hbp | SMALLINT | YES | |
| bat_sb | SMALLINT | YES | |
| bat_cs | SMALLINT | YES | |
| bat_tb | SMALLINT | YES | |
| bat_errors | SMALLINT | YES | |
| bat_avg | FLOAT4 | YES | |
| bat_obp | FLOAT4 | YES | |
| bat_slg | FLOAT4 | YES | |
| pit_ip | FLOAT4 | YES | Pitcher stats (NULL for batters) |
| pit_er | SMALLINT | YES | |
| pit_r | SMALLINT | YES | |
| pit_h | SMALLINT | YES | |
| pit_bb | SMALLINT | YES | |
| pit_k | SMALLINT | YES | |
| pit_hr | SMALLINT | YES | |
| pit_bf | SMALLINT | YES | |
| pit_w | SMALLINT | YES | |
| pit_l | SMALLINT | YES | |
| pit_sv | SMALLINT | YES | |
| pit_hld | SMALLINT | YES | |
| pit_bs | SMALLINT | YES | |
| pit_pitches | SMALLINT | YES | |
| pit_strikes | SMALLINT | YES | |
| pit_is_starter | BOOL | YES | |
| pit_era | FLOAT4 | YES | |
| pit_whip | FLOAT4 | YES | |
| created_at | TIMESTAMPTZ | YES | |

#### production.fact_milb_player_game
**PK**: `(player_id, game_pk, player_role)` — MiLB game log (same layout)

| Column | Type | Nullable |
|--------|------|----------|
| player_id | BIGINT | NO |
| game_pk | BIGINT | NO |
| player_role | VARCHAR(7) | NO |
| game_date | DATE | YES |
| season | INT | YES |
| team_id | INT | YES |
| team_name | TEXT | YES |
| sport_id | SMALLINT | YES |
| level | TEXT | YES |
| parent_org_id | INT | YES |
| bat_pa | SMALLINT | YES |
| bat_ab | SMALLINT | YES |
| bat_h | SMALLINT | YES |
| bat_2b | SMALLINT | YES |
| bat_3b | SMALLINT | YES |
| bat_hr | SMALLINT | YES |
| bat_r | SMALLINT | YES |
| bat_rbi | SMALLINT | YES |
| bat_bb | SMALLINT | YES |
| bat_k | SMALLINT | YES |
| bat_hbp | SMALLINT | YES |
| bat_sb | SMALLINT | YES |
| bat_cs | SMALLINT | YES |
| bat_tb | SMALLINT | YES |
| bat_errors | SMALLINT | YES |
| pit_ip | FLOAT4 | YES |
| pit_er | SMALLINT | YES |
| pit_r | SMALLINT | YES |
| pit_h | SMALLINT | YES |
| pit_bb | SMALLINT | YES |
| pit_k | SMALLINT | YES |
| pit_hr | SMALLINT | YES |
| pit_bf | SMALLINT | YES |
| pit_w | SMALLINT | YES |
| pit_l | SMALLINT | YES |
| pit_sv | SMALLINT | YES |
| pit_hld | SMALLINT | YES |
| pit_bs | SMALLINT | YES |
| pit_pitches | SMALLINT | YES |
| pit_strikes | SMALLINT | YES |
| pit_is_starter | BOOL | YES |
| created_at | TIMESTAMPTZ | YES |

---

### Analytics Facts

#### production.fact_player_form_rolling
**PK**: `(player_id, game_pk, player_role)` — 15-game and 30-game rolling windows

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| player_id | BIGINT | NO | |
| game_pk | BIGINT | NO | |
| player_role | VARCHAR(7) | NO | 'batter' or 'pitcher' |
| game_date | DATE | NO | |
| season | INT | YES | |
| bat_pa_15 | INT | YES | Batter 15-game rolling stats |
| bat_ab_15 | INT | YES | |
| bat_h_15 | INT | YES | |
| bat_2b_15 | INT | YES | |
| bat_3b_15 | INT | YES | |
| bat_hr_15 | INT | YES | |
| bat_bb_15 | INT | YES | |
| bat_k_15 | INT | YES | |
| bat_sb_15 | INT | YES | |
| bat_rbi_15 | INT | YES | |
| bat_tb_15 | INT | YES | |
| bat_hbp_15 | INT | YES | |
| bat_avg_15 | FLOAT4 | YES | |
| bat_obp_15 | FLOAT4 | YES | |
| bat_slg_15 | FLOAT4 | YES | |
| bat_ops_15 | FLOAT4 | YES | |
| bat_pa_30 | INT | YES | Batter 30-game rolling stats |
| bat_ab_30 | INT | YES | |
| bat_h_30 | INT | YES | |
| bat_2b_30 | INT | YES | |
| bat_3b_30 | INT | YES | |
| bat_hr_30 | INT | YES | |
| bat_bb_30 | INT | YES | |
| bat_k_30 | INT | YES | |
| bat_sb_30 | INT | YES | |
| bat_rbi_30 | INT | YES | |
| bat_tb_30 | INT | YES | |
| bat_hbp_30 | INT | YES | |
| bat_avg_30 | FLOAT4 | YES | |
| bat_obp_30 | FLOAT4 | YES | |
| bat_slg_30 | FLOAT4 | YES | |
| bat_ops_30 | FLOAT4 | YES | |
| pit_ip_15 | FLOAT4 | YES | Pitcher 15-game rolling stats |
| pit_er_15 | INT | YES | |
| pit_h_15 | INT | YES | |
| pit_bb_15 | INT | YES | |
| pit_k_15 | INT | YES | |
| pit_hr_15 | INT | YES | |
| pit_bf_15 | INT | YES | |
| pit_era_15 | FLOAT4 | YES | |
| pit_whip_15 | FLOAT4 | YES | |
| pit_k9_15 | FLOAT4 | YES | |
| pit_bb9_15 | FLOAT4 | YES | |
| pit_ip_30 | FLOAT4 | YES | Pitcher 30-game rolling stats |
| pit_er_30 | INT | YES | |
| pit_h_30 | INT | YES | |
| pit_bb_30 | INT | YES | |
| pit_k_30 | INT | YES | |
| pit_hr_30 | INT | YES | |
| pit_bf_30 | INT | YES | |
| pit_era_30 | FLOAT4 | YES | |
| pit_whip_30 | FLOAT4 | YES | |
| pit_k9_30 | FLOAT4 | YES | |
| pit_bb9_30 | FLOAT4 | YES | |
| created_at | TIMESTAMPTZ | YES | |

#### production.fact_streak_indicator
**PK**: `(player_id, game_pk, player_role)` — Hot/cold streak detection via z-scores

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| player_id | BIGINT | NO | |
| game_pk | BIGINT | NO | |
| player_role | VARCHAR(7) | NO | 'batter' or 'pitcher' |
| game_date | DATE | YES | |
| season | INT | YES | |
| games_in_window | SMALLINT | YES | Games with 5+ required for z-score |
| bat_avg_10g | FLOAT4 | YES | 10-game rolling avg |
| bat_avg_season | FLOAT4 | YES | Season-to-date avg |
| bat_avg_zscore | FLOAT4 | YES | Z-score: (10g - season) / stddev |
| bat_obp_10g | FLOAT4 | YES | |
| bat_obp_season | FLOAT4 | YES | |
| bat_obp_zscore | FLOAT4 | YES | |
| bat_slg_10g | FLOAT4 | YES | |
| bat_slg_season | FLOAT4 | YES | |
| bat_slg_zscore | FLOAT4 | YES | |
| pit_era_10g | FLOAT4 | YES | |
| pit_era_season | FLOAT4 | YES | |
| pit_era_zscore | FLOAT4 | YES | Inverted: negative z = hot for pitchers |
| pit_whip_10g | FLOAT4 | YES | |
| pit_whip_season | FLOAT4 | YES | |
| pit_whip_zscore | FLOAT4 | YES | |
| streak_flag | VARCHAR(4) | YES | 'hot', 'cold', or NULL |
| created_at | TIMESTAMPTZ | YES | |

#### production.fact_platoon_splits
**PK**: `(player_id, season, player_role, platoon_side)` — Season-level vLH/vRH splits

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| player_id | BIGINT | NO | |
| season | INT | NO | |
| player_role | VARCHAR(7) | NO | 'batter' or 'pitcher' |
| platoon_side | VARCHAR(3) | NO | 'vLH' or 'vRH' |
| pa | INT | YES | |
| ab | INT | YES | |
| h | INT | YES | |
| doubles | INT | YES | |
| triples | INT | YES | |
| hr | INT | YES | |
| bb | INT | YES | |
| k | INT | YES | |
| hbp | INT | YES | |
| sf | INT | YES | |
| avg | FLOAT4 | YES | |
| obp | FLOAT4 | YES | |
| slg | FLOAT4 | YES | |
| ops | FLOAT4 | YES | |
| woba | FLOAT4 | YES | |
| k_pct | FLOAT4 | YES | |
| bb_pct | FLOAT4 | YES | |
| total_pitches | INT | YES | |
| whiff_rate | FLOAT4 | YES | |
| chase_rate | FLOAT4 | YES | |
| hard_hit_pct | FLOAT4 | YES | |
| sweet_spot_pct | FLOAT4 | YES | |
| xwoba_avg | FLOAT4 | YES | |
| created_at | TIMESTAMPTZ | YES | |

#### production.fact_matchup_history
**PK**: `(batter_id, pitcher_id)` — Career batter-vs-pitcher aggregates

| Column | Type | Nullable |
|--------|------|----------|
| batter_id | BIGINT | NO |
| pitcher_id | BIGINT | NO |
| pa | INT | YES |
| ab | INT | YES |
| h | INT | YES |
| doubles | INT | YES |
| triples | INT | YES |
| hr | INT | YES |
| bb | INT | YES |
| k | INT | YES |
| hbp | INT | YES |
| sf | INT | YES |
| avg | FLOAT4 | YES |
| obp | FLOAT4 | YES |
| slg | FLOAT4 | YES |
| ops | FLOAT4 | YES |
| total_pitches | INT | YES |
| whiff_rate | FLOAT4 | YES |
| chase_rate | FLOAT4 | YES |
| zone_contact_rate | FLOAT4 | YES |
| avg_exit_velo | FLOAT4 | YES |
| avg_launch_angle | FLOAT4 | YES |
| hard_hit_pct | FLOAT4 | YES |
| xwoba_avg | FLOAT4 | YES |
| first_matchup_date | DATE | YES |
| last_matchup_date | DATE | YES |
| created_at | TIMESTAMPTZ | YES |

---

### Player Status & Prospects

#### production.fact_player_status_timeline
**PK**: `(player_id, status_start_date, status_type)` — IL/DFA/trade/option status history

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| player_id | BIGINT | NO | |
| status_start_date | DATE | NO | |
| status_type | VARCHAR(20) | NO | 'IL-10', 'IL-15', 'IL-60', 'active', 'option_minors', 'designate', 'release', 'trade' |
| status_end_date | DATE | YES | From LEAD() over next status change |
| team_id | INT | YES | |
| team_name | TEXT | YES | |
| injury_description | TEXT | YES | Only for IL placements |
| days_on_status | INT | YES | status_end_date - status_start_date |
| season | INT | YES | |
| created_at | TIMESTAMPTZ | YES | |

#### production.fact_prospect_snapshot
**PK**: `(player_id, season)` — Prospect profile per season

| Column | Type | Nullable |
|--------|------|----------|
| player_id | BIGINT | NO |
| season | SMALLINT | NO |
| full_name | TEXT | YES |
| primary_position | TEXT | YES |
| bat_side | TEXT | YES |
| pitch_hand | TEXT | YES |
| birth_date | DATE | YES |
| age_at_season_start | SMALLINT | YES |
| level | TEXT | YES |
| sport_id | SMALLINT | YES |
| parent_org_id | INT | YES |
| parent_org_name | TEXT | YES |
| milb_team_id | INT | YES |
| milb_team_name | TEXT | YES |
| status_code | TEXT | YES |
| mlb_debut_date | DATE | YES |
| draft_year | SMALLINT | YES |
| games_played | INT | YES |
| created_at | TIMESTAMPTZ | YES |

#### production.fact_prospect_transition
**PK**: `(player_id, event_date, from_level, to_level)` — Level promotions/demotions

| Column | Type | Nullable |
|--------|------|----------|
| player_id | BIGINT | NO |
| event_date | DATE | NO |
| from_level | TEXT | NO |
| to_level | TEXT | NO |
| from_sport_id | SMALLINT | YES |
| to_sport_id | SMALLINT | YES |
| from_team_name | TEXT | YES |
| to_team_name | TEXT | YES |
| transition_type | VARCHAR(10) | YES |
| season | INT | YES |
| created_at | TIMESTAMPTZ | YES |

---

### Projection System (empty — future use)

#### production.fact_player_projection
**PK**: `(run_id, player_id, as_of_date, horizon, scenario)`

| Column | Type | Nullable |
|--------|------|----------|
| run_id | UUID | NO |
| player_id | BIGINT | NO |
| as_of_date | DATE | NO |
| horizon | VARCHAR(10) | NO |
| scenario | VARCHAR(10) | NO |
| player_role | VARCHAR(7) | YES |
| projected_pts_p10 | FLOAT4 | YES |
| projected_pts_p50 | FLOAT4 | YES |
| projected_pts_p90 | FLOAT4 | YES |
| projected_pa | FLOAT4 | YES |
| projected_ip | FLOAT4 | YES |
| projected_avg | FLOAT4 | YES |
| projected_obp | FLOAT4 | YES |
| projected_slg | FLOAT4 | YES |
| projected_era | FLOAT4 | YES |
| projected_whip | FLOAT4 | YES |
| projected_k9 | FLOAT4 | YES |
| confidence_score | FLOAT4 | YES |
| created_at | TIMESTAMPTZ | YES |

#### production.fact_projection_backtest
**PK**: `(run_id, player_id, target_date, metric)`

| Column | Type | Nullable |
|--------|------|----------|
| run_id | UUID | NO |
| player_id | BIGINT | NO |
| target_date | DATE | NO |
| metric | VARCHAR(20) | NO |
| predicted_value | FLOAT4 | YES |
| actual_value | FLOAT4 | YES |
| error | FLOAT4 | YES |
| abs_error | FLOAT4 | YES |
| pct_error | FLOAT4 | YES |
| within_p10_p90 | BOOL | YES |
| created_at | TIMESTAMPTZ | YES |

---

## Schema: fantasy

### fantasy.dk_batter_game_scores
**PK**: `(batter_id, game_pk)` — DraftKings batter scoring

| Column | Type | Nullable | DK Points |
|--------|------|----------|-----------|
| batter_id | BIGINT | NO | |
| game_pk | BIGINT | NO | |
| team_id | BIGINT | YES | |
| batter_name | TEXT | YES | |
| game_date | DATE | YES | |
| season | FLOAT8 | YES | |
| dk_pts_1B | BIGINT | YES | × 3.0 |
| dk_pts_2B | BIGINT | YES | × 5.0 |
| dk_pts_3B | BIGINT | YES | × 8.0 |
| dk_pts_HR | BIGINT | YES | × 10.0 |
| dk_pts_RBI | BIGINT | YES | × 2.0 |
| dk_pts_R | BIGINT | YES | × 2.0 |
| dk_pts_BB | BIGINT | YES | × 2.0 |
| dk_pts_HBP | BIGINT | YES | × 2.0 |
| dk_pts_SB | BIGINT | YES | × 5.0 |
| dk_points | FLOAT8 | YES | Total |

### fantasy.dk_pitcher_game_scores
**PK**: `(pitcher_id, game_pk)` — DraftKings pitcher scoring

| Column | Type | Nullable | DK Points |
|--------|------|----------|-----------|
| pitcher_id | BIGINT | NO | |
| game_pk | BIGINT | NO | |
| team_id | BIGINT | YES | |
| pitcher_name | TEXT | YES | |
| is_starter | BOOL | YES | |
| game_date | DATE | YES | |
| season | FLOAT8 | YES | |
| dk_pts_IP | FLOAT8 | YES | IP × 2.25 |
| dk_pts_K | BIGINT | YES | × 2.0 |
| dk_pts_W | BIGINT | YES | × 4.0 |
| dk_pts_ER | BIGINT | YES | × -2.0 |
| dk_pts_H | FLOAT8 | YES | × -0.6 |
| dk_pts_BB | FLOAT8 | YES | × -0.6 |
| dk_pts_HBP | FLOAT8 | YES | × -0.6 |
| dk_pts_CG | FLOAT8 | YES | +2.5 |
| dk_pts_CGSO | FLOAT8 | YES | +2.5 |
| dk_pts_NH | BIGINT | YES | +5.0 |
| dk_points | FLOAT8 | YES | Total |

### fantasy.espn_batter_game_scores
**PK**: `(batter_id, game_pk)` — ESPN batter scoring

| Column | Type | Nullable | ESPN Points |
|--------|------|----------|-------------|
| batter_id | BIGINT | NO | |
| game_pk | BIGINT | NO | |
| team_id | BIGINT | YES | |
| batter_name | TEXT | YES | |
| game_date | DATE | YES | |
| season | FLOAT8 | YES | |
| pts_H | BIGINT | YES | +1 |
| pts_R | BIGINT | YES | +1 |
| pts_TB | BIGINT | YES | +1 |
| pts_RBI | BIGINT | YES | +1 |
| pts_BB | BIGINT | YES | +1 |
| pts_K | BIGINT | YES | -1 |
| pts_SB | BIGINT | YES | +1 |
| pts_E | BIGINT | YES | -1 |
| pts_CYC | BIGINT | YES | +10 (hitting for the cycle) |
| pts_GWRBI | BIGINT | YES | Not computed |
| pts_GSHR | BIGINT | YES | Not computed |
| fantasy_points | BIGINT | YES | Total |

### fantasy.espn_pitcher_game_scores
**PK**: `(pitcher_id, game_pk)` — ESPN pitcher scoring

| Column | Type | Nullable | ESPN Points |
|--------|------|----------|-------------|
| pitcher_id | BIGINT | NO | |
| game_pk | BIGINT | NO | |
| team_id | BIGINT | YES | |
| pitcher_name | TEXT | YES | |
| is_starter | BOOL | YES | |
| game_date | DATE | YES | |
| season | FLOAT8 | YES | |
| pts_H | BIGINT | YES | -1 |
| pts_RA | BIGINT | YES | -1 |
| pts_ER | BIGINT | YES | -1 |
| pts_BB | BIGINT | YES | -1 |
| pts_K | BIGINT | YES | +1 |
| pts_PKO | BIGINT | YES | +1 |
| pts_W | BIGINT | YES | +2 |
| pts_L | BIGINT | YES | -2 |
| pts_SV | BIGINT | YES | +2 |
| pts_BS | BIGINT | YES | -2 |
| pts_IP | BIGINT | YES | +3 per full inning |
| pts_CG | BIGINT | YES | +3 |
| pts_SO | BIGINT | YES | +3 (CG shutout) |
| pts_NH | BIGINT | YES | +5 (no-hitter) |
| pts_PG | BIGINT | YES | +5 (perfect game) |
| fantasy_points | BIGINT | YES | Total |

---

## Connection Details

```
Host: (see .env)
Port: (see .env)
Database: (see .env)
Driver: postgresql+psycopg
Schemas: raw, staging, production, fantasy
```

Seasons covered: 2018–2025 (2026 daily pipeline active).
