"""
Pitch-type mappings, event definitions, zone boundaries, and league-average
priors for the Bayesian projection system.

All constants derived from the actual mlb_fantasy.production schema.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Pitch-type abbreviation → readable name  (matches production.fact_pitch)
# ---------------------------------------------------------------------------
PITCH_TYPE_MAP: dict[str, str] = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "SL": "Slider",
    "CH": "Changeup",
    "CU": "Curveball",
    "FC": "Cutter",
    "ST": "Sweeper",
    "KC": "Knuckle Curve",
    "FS": "Split-Finger",
    "SV": "Slurve",
    "FA": "Other",
    "EP": "Eephus",
    "KN": "Knuckleball",
    "FO": "Forkball",
    "CS": "Slow Curve",
    "PO": "Pitch Out",
    "SC": "Screwball",
    "UN": "Unknown",
}

# Pitch types that are too rare / non-competitive to model individually.
# These get grouped into an "Other" bucket or excluded from matchup calcs.
EXCLUDED_PITCH_TYPES: set[str] = {"PO", "UN", "SC", "FA"}

# Broad families for partial-pooling correlation structure.
# Pitches in the same family share information in the hierarchical model.
PITCH_FAMILIES: dict[str, list[str]] = {
    "fastball": ["FF", "SI", "FC"],
    "breaking": ["SL", "CU", "ST", "KC", "SV", "CS"],
    "offspeed": ["CH", "FS", "FO", "EP", "KN"],
}

# Reverse lookup: pitch_type → family
PITCH_TO_FAMILY: dict[str, str] = {
    pt: family for family, pts in PITCH_FAMILIES.items() for pt in pts
}

# ---------------------------------------------------------------------------
# Pitch-outcome definitions  (description column in production.fact_pitch)
# ---------------------------------------------------------------------------
# NOTE: The production schema already has pre-computed boolean columns
# (is_whiff, is_swing, is_called_strike, is_bip, is_foul).  These string
# sets are kept for reference, validation, and any raw-data work.

WHIFF_DESCRIPTIONS: set[str] = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",               # foul tips are strikes / outs — count as whiff
}

SWING_DESCRIPTIONS: set[str] = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "foul",
    "foul_bunt",
    "bunt_foul_tip",
    "hit_into_play",
    "missed_bunt",
}

CALLED_STRIKE_DESCRIPTIONS: set[str] = {
    "called_strike",
}

BALL_DESCRIPTIONS: set[str] = {
    "ball",
    "blocked_ball",
    "automatic_ball",
    "pitchout",
    "hit_by_pitch",
}

BIP_DESCRIPTIONS: set[str] = {
    "hit_into_play",
}

# CSW = Called Strike + Whiff (key quality metric)
CSW_DESCRIPTIONS: set[str] = CALLED_STRIKE_DESCRIPTIONS | WHIFF_DESCRIPTIONS

# ---------------------------------------------------------------------------
# PA-level event groupings  (events column in production.fact_pa)
# ---------------------------------------------------------------------------
STRIKEOUT_EVENTS: set[str] = {"strikeout", "strikeout_double_play"}

WALK_EVENTS: set[str] = {"walk", "intent_walk"}

HIT_EVENTS: set[str] = {"single", "double", "triple", "home_run"}

OUT_EVENTS: set[str] = {
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "double_play",
    "fielders_choice",
    "fielders_choice_out",
    "sac_fly",
    "sac_bunt",
    "sac_fly_double_play",
    "sac_bunt_double_play",
    "triple_play",
}

# ---------------------------------------------------------------------------
# Zone definitions (Statcast zone numbers + plate location boundaries)
# ---------------------------------------------------------------------------
# Statcast zones 1-9 are the strike zone grid; 11-14 are chase quadrants.
STRIKE_ZONES: set[int] = {1, 2, 3, 4, 5, 6, 7, 8, 9}
CHASE_ZONES: set[int] = {11, 12, 13, 14}

# Plate location boundaries (feet from center of plate)
# Used when zone column is missing or for continuous-location analysis.
ZONE_BOUNDARIES = {
    "plate_x_left": -0.8333,   # left edge of zone (batter's perspective)
    "plate_x_right": 0.8333,   # right edge of zone
    "plate_z_top_avg": 3.5,    # average top of zone (use per-batter sz_top when available)
    "plate_z_bot_avg": 1.5,    # average bottom of zone (use per-batter sz_bot when available)
}

# 5x5 zone grid for location visualization
ZONE_GRID = {
    "n_cols": 5,
    "n_rows": 5,
    "x_min": -1.33,     # left edge (extends past zone)
    "x_max": 1.33,      # right edge
    "z_min": 1.0,       # bottom edge (below avg zone bottom)
    "z_max": 4.0,       # top edge (above avg zone top)
    "x_step": 0.532,    # (1.33 - (-1.33)) / 5
    "z_step": 0.6,      # (4.0 - 1.0) / 5
}

# ---------------------------------------------------------------------------
# Batted-ball thresholds
# ---------------------------------------------------------------------------
HARD_HIT_THRESHOLD: float = 95.0      # exit velo (mph)
BARREL_EV_MIN: float = 98.0           # minimum EV for barrel
BARREL_LA_SWEET_SPOT: tuple[float, float] = (26.0, 30.0)  # ideal LA range at 98 mph
# Barrel zone widens as EV increases — simplified here; full barrel def is complex.

# ---------------------------------------------------------------------------
# League-average rates per pitch type (2022-2024 pooled)
# Used as population-level priors in the hierarchical model.
# These are approximate; will be refined from actual data in feature_eng.
# ---------------------------------------------------------------------------
LEAGUE_AVG_BY_PITCH_TYPE: dict[str, dict[str, float]] = {
    # pitch_type: {whiff_rate, chase_rate, csw_pct, barrel_rate, xwoba_contact, hard_hit_rate}
    "FF": {
        "whiff_rate": 0.22,
        "chase_rate": 0.26,
        "csw_pct": 0.29,
        "barrel_rate": 0.085,
        "xwoba_contact": 0.370,
        "hard_hit_rate": 0.38,
    },
    "SI": {
        "whiff_rate": 0.15,
        "chase_rate": 0.30,
        "csw_pct": 0.25,
        "barrel_rate": 0.055,
        "xwoba_contact": 0.340,
        "hard_hit_rate": 0.35,
    },
    "SL": {
        "whiff_rate": 0.33,
        "chase_rate": 0.34,
        "csw_pct": 0.31,
        "barrel_rate": 0.045,
        "xwoba_contact": 0.290,
        "hard_hit_rate": 0.28,
    },
    "CH": {
        "whiff_rate": 0.32,
        "chase_rate": 0.33,
        "csw_pct": 0.29,
        "barrel_rate": 0.050,
        "xwoba_contact": 0.320,
        "hard_hit_rate": 0.32,
    },
    "CU": {
        "whiff_rate": 0.28,
        "chase_rate": 0.30,
        "csw_pct": 0.29,
        "barrel_rate": 0.040,
        "xwoba_contact": 0.280,
        "hard_hit_rate": 0.25,
    },
    "FC": {
        "whiff_rate": 0.24,
        "chase_rate": 0.28,
        "csw_pct": 0.30,
        "barrel_rate": 0.065,
        "xwoba_contact": 0.340,
        "hard_hit_rate": 0.34,
    },
    "ST": {
        "whiff_rate": 0.36,
        "chase_rate": 0.36,
        "csw_pct": 0.32,
        "barrel_rate": 0.035,
        "xwoba_contact": 0.260,
        "hard_hit_rate": 0.24,
    },
    "KC": {
        "whiff_rate": 0.30,
        "chase_rate": 0.32,
        "csw_pct": 0.30,
        "barrel_rate": 0.038,
        "xwoba_contact": 0.275,
        "hard_hit_rate": 0.24,
    },
    "FS": {
        "whiff_rate": 0.35,
        "chase_rate": 0.38,
        "csw_pct": 0.31,
        "barrel_rate": 0.042,
        "xwoba_contact": 0.290,
        "hard_hit_rate": 0.28,
    },
    "SV": {
        "whiff_rate": 0.30,
        "chase_rate": 0.32,
        "csw_pct": 0.29,
        "barrel_rate": 0.040,
        "xwoba_contact": 0.280,
        "hard_hit_rate": 0.26,
    },
}

# Overall league averages (all pitch types combined, 2022-2024)
LEAGUE_AVG_OVERALL: dict[str, float] = {
    "k_rate": 0.224,
    "bb_rate": 0.083,
    "barrel_rate": 0.068,
    "xwoba": 0.315,
    "woba": 0.315,
    "whiff_rate": 0.25,
    "chase_rate": 0.30,
    "csw_pct": 0.29,
    "hard_hit_rate": 0.33,
    # Batted ball type rates (2022-2024 pooled, LA-based)
    "gb_rate": 0.446,   # ground ball: LA < 10°
    "fb_rate": 0.321,   # fly ball: LA > 25°
    "hr_per_fb": 0.098, # HR per fly ball
}

# ---------------------------------------------------------------------------
# Clip bounds for logit transforms (avoid infinities)
# ---------------------------------------------------------------------------
CLIP_LO: float = 1e-6
CLIP_HI: float = 1 - 1e-6

# ---------------------------------------------------------------------------
# Game-sim league baselines (used for logit centering)
# ---------------------------------------------------------------------------
SIM_LEAGUE_K_RATE: float = 0.226
SIM_LEAGUE_BB_RATE: float = 0.082
SIM_LEAGUE_HR_RATE: float = 0.031

# ---------------------------------------------------------------------------
# BABIP constants
# ---------------------------------------------------------------------------
LEAGUE_BABIP_PITCHER: float = 0.292
LEAGUE_BABIP_BATTER: float = 0.300

# ---------------------------------------------------------------------------
# HBP rate
# ---------------------------------------------------------------------------
LEAGUE_HBP_RATE: float = 0.011

# ---------------------------------------------------------------------------
# Bullpen baseline rates (game sim defaults)
# ---------------------------------------------------------------------------
BULLPEN_K_RATE: float = 0.253
BULLPEN_BB_RATE: float = 0.084
BULLPEN_HR_RATE: float = 0.024

# ---------------------------------------------------------------------------
# Branding — The Data Diamond color palette
# ---------------------------------------------------------------------------
COLORS = {
    "gold": "#C8A96E",
    "teal": "#4FC3C8",
    "slate": "#7B8FA6",
    "cream": "#F5F2EE",
    "dark_bg": "#0F1117",
    "white": "#FFFFFF",
}
