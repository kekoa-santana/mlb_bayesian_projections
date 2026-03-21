"""
Platoon-adjusted matchup lifts and contact quality scoring.

Adds two enhancements to the base matchup model:

1. Platoon adjustment: modifies K/BB/HR matchup lifts based on
   batter_stand x pitcher_hand interaction per pitch type.

2. Contact quality matchup: scores xwOBA-on-contact per pitch type
   for H/TB matchup lifts (the base model only scores K/BB/HR).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)

_CLIP_LO = 1e-4
_CLIP_HI = 1.0 - 1e-4


def _logit(p: float) -> float:
    p = np.clip(p, _CLIP_LO, _CLIP_HI)
    return float(np.log(p / (1.0 - p)))


# ===================================================================
# Platoon adjustment table
# ===================================================================

def compute_platoon_adjustments(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute platoon whiff/chase/barrel adjustments by pitch type.

    Returns the logit-scale adjustment to apply on top of the base
    matchup score, by (batter_stand, pitcher_hand, pitch_type).

    A positive K adjustment means this platoon matchup produces MORE K
    than the overall average for that pitch type.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to compute from. Defaults to last 3.

    Returns
    -------
    pd.DataFrame
        Columns: batter_stand, pitcher_hand, pitch_type,
        platoon_k_adj, platoon_bb_adj, platoon_hr_adj.
        All on logit scale.
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]
    season_list = ", ".join(str(s) for s in seasons)

    raw = read_sql(f"""
        SELECT dp.bat_side as batter_stand,
               dp2.pitch_hand as pitcher_hand,
               fp.pitch_type,
               -- Whiff (for K)
               COUNT(CASE WHEN fp.is_whiff THEN 1 END)::float /
                   NULLIF(COUNT(CASE WHEN fp.is_swing THEN 1 END), 0) as whiff_rate,
               COUNT(CASE WHEN fp.is_swing THEN 1 END) as swings,
               -- Chase (for BB, inverted)
               COUNT(CASE WHEN fp.is_swing AND NOT fp.is_called_strike
                          AND fp.plate_x IS NOT NULL
                          AND (ABS(fp.plate_x) > 0.83 OR fp.plate_z > 3.5 OR fp.plate_z < 1.5)
                     THEN 1 END)::float /
                   NULLIF(COUNT(CASE WHEN fp.plate_x IS NOT NULL
                                     AND (ABS(fp.plate_x) > 0.83 OR fp.plate_z > 3.5 OR fp.plate_z < 1.5)
                     THEN 1 END), 0) as chase_rate,
               -- Barrel (for HR)
               COUNT(CASE WHEN fp.is_bip THEN 1 END) as bip
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        JOIN production.dim_player dp ON fp.batter_id = dp.player_id
        JOIN production.dim_player dp2 ON fp.pitcher_id = dp2.player_id
        WHERE dg.game_type = 'R' AND dg.season IN ({season_list})
              AND fp.pitch_type IS NOT NULL
        GROUP BY dp.bat_side, dp2.pitch_hand, fp.pitch_type
        HAVING COUNT(CASE WHEN fp.is_swing THEN 1 END) >= 500
    """, {})

    if raw.empty:
        return pd.DataFrame()

    # Compute overall averages per pitch type (across all platoon combos)
    overall = raw.groupby("pitch_type").apply(
        lambda g: pd.Series({
            "overall_whiff": (g["whiff_rate"] * g["swings"]).sum() / g["swings"].sum()
            if g["swings"].sum() > 0 else 0.25,
            "overall_chase": g["chase_rate"].mean(),
        }),
        include_groups=False,
    ).to_dict("index")

    # Compute logit adjustments: platoon_rate - overall_rate on logit scale
    rows = []
    for _, r in raw.iterrows():
        pt = r["pitch_type"]
        ovr = overall.get(pt, {"overall_whiff": 0.25, "overall_chase": 0.30})

        k_adj = _logit(r["whiff_rate"]) - _logit(ovr["overall_whiff"])

        chase = r.get("chase_rate")
        if pd.notna(chase) and chase > 0:
            bb_adj = -(_logit(chase) - _logit(ovr["overall_chase"]))  # inverted
        else:
            bb_adj = 0.0

        rows.append({
            "batter_stand": r["batter_stand"],
            "pitcher_hand": r["pitcher_hand"],
            "pitch_type": pt,
            "platoon_k_adj": round(k_adj, 4),
            "platoon_bb_adj": round(bb_adj, 4),
            "platoon_hr_adj": 0.0,  # HR platoon effect is small, skip for now
        })

    result = pd.DataFrame(rows)
    logger.info("Platoon adjustments: %d combinations", len(result))
    return result


def get_platoon_lift(
    platoon_table: pd.DataFrame,
    batter_stand: str,
    pitcher_hand: str,
    pitch_types: list[str],
    usage_weights: list[float],
) -> dict[str, float]:
    """Get weighted platoon lifts for a specific matchup.

    Parameters
    ----------
    platoon_table : pd.DataFrame
        From ``compute_platoon_adjustments``.
    batter_stand : str
        'L', 'R', or 'S'.
    pitcher_hand : str
        'L' or 'R'.
    pitch_types : list[str]
        Pitcher's pitch types.
    usage_weights : list[float]
        Usage fractions per pitch type (sum to ~1).

    Returns
    -------
    dict[str, float]
        platoon_k_lift, platoon_bb_lift, platoon_hr_lift on logit scale.
    """
    # Switch hitters: use opposite stand vs pitcher hand
    if batter_stand == "S":
        batter_stand = "R" if pitcher_hand == "L" else "L"

    k_lift = 0.0
    bb_lift = 0.0
    hr_lift = 0.0

    for pt, w in zip(pitch_types, usage_weights):
        row = platoon_table[
            (platoon_table["batter_stand"] == batter_stand)
            & (platoon_table["pitcher_hand"] == pitcher_hand)
            & (platoon_table["pitch_type"] == pt)
        ]
        if not row.empty:
            r = row.iloc[0]
            k_lift += w * r["platoon_k_adj"]
            bb_lift += w * r["platoon_bb_adj"]
            hr_lift += w * r["platoon_hr_adj"]

    return {
        "platoon_k_lift": round(k_lift, 4),
        "platoon_bb_lift": round(bb_lift, 4),
        "platoon_hr_lift": round(hr_lift, 4),
    }


# ===================================================================
# Contact quality matchup
# ===================================================================

def score_contact_quality_matchup(
    pitcher_id: int,
    batter_id: int,
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Score contact quality matchup using xwOBA-on-contact.

    Produces a logit lift for H (hits) based on how hard a hitter
    makes contact against this pitcher's pitch types.

    Parameters
    ----------
    pitcher_arsenal : pd.DataFrame
        With xwoba_against per pitch type.
    hitter_vuln : pd.DataFrame
        With xwoba_contact per pitch type.
    baselines_pt : dict
        With xwoba_contact league avg per pitch type.

    Returns
    -------
    dict
        matchup_h_logit_lift, matchup_xwoba_contact.
    """
    p_arsenal = pitcher_arsenal[pitcher_arsenal["pitcher_id"] == pitcher_id].copy()
    p_arsenal = p_arsenal[p_arsenal["pitches"] >= 20].copy()

    if p_arsenal.empty:
        return {"matchup_h_logit_lift": 0.0, "matchup_xwoba_contact": np.nan}

    total_usage = p_arsenal["usage_pct"].sum()
    if total_usage > 0:
        p_arsenal["usage_norm"] = p_arsenal["usage_pct"] / total_usage
    else:
        p_arsenal["usage_norm"] = 1.0 / len(p_arsenal)

    batter_rows = hitter_vuln[hitter_vuln["batter_id"] == batter_id]

    matchup_xwoba = 0.0
    baseline_xwoba = 0.0

    for _, row in p_arsenal.iterrows():
        pt = row["pitch_type"]
        usage = row["usage_norm"]

        # Pitcher's xwoba against
        pitcher_xwoba = row.get("xwoba_against", np.nan)
        league_xwoba = baselines_pt.get(pt, {}).get("xwoba_contact", 0.320)

        if pd.isna(pitcher_xwoba):
            pitcher_xwoba = league_xwoba

        # Hitter's xwoba on contact for this pitch type
        direct = batter_rows[batter_rows["pitch_type"] == pt]
        if not direct.empty and pd.notna(direct.iloc[0].get("xwoba_contact")):
            bip = direct.iloc[0].get("bip", 0)
            if pd.notna(bip) and bip >= 10:
                hitter_xwoba = float(direct.iloc[0]["xwoba_contact"])
            else:
                hitter_xwoba = league_xwoba
        else:
            hitter_xwoba = league_xwoba

        # Additive on raw scale (xwoba is not a probability, skip logit)
        pitcher_delta = pitcher_xwoba - league_xwoba
        hitter_delta = hitter_xwoba - league_xwoba
        matchup_xwoba_pt = league_xwoba + pitcher_delta + hitter_delta

        matchup_xwoba += usage * matchup_xwoba_pt
        baseline_xwoba += usage * pitcher_xwoba

    # Convert xwoba difference to a logit-scale H lift
    # ~0.050 xwoba difference ≈ ~0.15 logit lift on H rate
    xwoba_diff = matchup_xwoba - baseline_xwoba
    matchup_h_logit_lift = xwoba_diff * 3.0  # empirical scaling

    return {
        "matchup_h_logit_lift": round(float(matchup_h_logit_lift), 4),
        "matchup_xwoba_contact": round(float(matchup_xwoba), 4),
    }
