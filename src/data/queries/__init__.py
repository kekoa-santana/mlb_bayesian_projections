"""Query functions organized by domain.

Explicit re-exports for discoverability and static analysis.
"""
from src.data.queries.hitter import (  # noqa: F401
    get_hitter_pitch_type_profile,
    get_season_totals,
    get_season_totals_by_pitcher_hand,
    get_league_platoon_baselines,
    get_season_totals_with_age,
    get_hitter_observed_profile,
    get_batter_bip_profile,
    get_hitter_aggressiveness,
    get_hitter_zone_grid,
)
from src.data.queries.pitcher import (  # noqa: F401
    get_pitcher_arsenal_profile,
    get_pitcher_season_totals,
    get_pitcher_season_totals_with_age,
    get_pitcher_outcomes_by_stand,
    get_pitch_shape_offerings,
    get_pitcher_fly_ball_data,
    get_pitcher_observed_profile,
    get_pitcher_season_totals_extended,
    get_pitcher_efficiency,
)
from src.data.queries.game import (  # noqa: F401
    get_pitcher_game_logs,
    get_game_batter_ks,
    get_batter_game_logs,
    get_game_batter_stats,
    get_game_lineups,
    get_batter_game_actuals,
    get_catcher_game_assignments,
    get_game_starter_teams,
)
from src.data.queries.environment import (  # noqa: F401
    get_park_factors,
    get_hitter_team_venue,
    get_pitcher_team_venue,
    get_umpire_tendencies,
    get_umpire_k_tendencies,
    get_weather_effects,
    get_days_rest,
    get_team_defense_lifts,
    get_catcher_framing_effects,
)
from src.data.queries.simulator import (  # noqa: F401
    get_tto_adjustment_profiles,
    get_pitcher_pitch_count_features,
    get_batter_pitch_count_features,
    get_exit_model_training_data,
    get_pitcher_exit_tendencies,
    get_team_bullpen_rates,
    get_team_reliever_roster,
    get_bullpen_trailing_workload,
    get_reliever_role_history,
    get_reliever_stats_by_team,
)
from src.data.queries.breakout import (  # noqa: F401
    get_pa_outcomes,
    get_hitter_breakout_features,
    get_pitcher_breakout_features,
    get_postseason_batter_stats,
    get_postseason_pitcher_stats,
    get_rolling_form,
    get_rolling_hard_hit,
)
from src.data.queries.prospect import (  # noqa: F401
    get_prospect_snapshots_for_org_depth,
    get_mlb_batter_first_seasons,
    get_mlb_batters_with_min_pa_season,
    get_mlb_pitchers_with_min_bf,
    get_established_mlb_pitcher_ids,
    get_fg_prospect_rankings,
    get_prospect_fv_grades,
    get_pitcher_player_ids,
    get_mlb_debut_pitcher_rates,
    get_mlb_debut_batter_rates,
)
from src.data.queries.traditional import (  # noqa: F401
    get_hitter_traditional_stats,
    get_pitcher_traditional_stats,
    get_hitter_recent_form,
    get_pitcher_recent_form,
    get_hitter_season_totals_extended,
    get_batted_ball_spray,
    get_sprint_speed,
    get_pitcher_location_grid,
    get_pitcher_pitch_locations,
    get_player_teams,
    get_pitcher_run_values,
    get_prospect_info,
    get_prospect_transitions,
    get_lineup_priors,
    get_lineup_priors_by_hand,
    get_milb_batter_season_totals,
    get_milb_pitcher_season_totals,
    get_hitter_daily_standouts,
    get_pitcher_daily_standouts,
)

# Shared constants
from src.data.queries._common import _WOBA_WEIGHTS  # noqa: F401

# Private constant re-exported for backward compat
_PS_ROUND_ORDER = {"F": 1, "D": 2, "L": 3, "W": 4}
