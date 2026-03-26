"""
AutoResearch game simulator experiment — AGENT EDITS THIS FILE.

Contains the CONFIG dict with all tunable simulator constants.
Modify CONFIG values to run experiments. The scoring function in
prepare.py is immutable and handles evaluation.

Usage:
    python autoresearch/train.py
    python autoresearch/train.py --max-games 500   # faster iteration
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
# CONFIG — All tunable simulator constants live here.
# The hill climber or AI agent modifies these values.
# ======================================================================

CONFIG = {
    # --- Fatigue model (pa_outcome_model.py) ---
    "fatigue_pitch_threshold": 85,     # pitches before fatigue kicks in
    "fatigue_k_slope": -0.003,         # K logit drops per pitch above threshold
    "fatigue_bb_slope": 0.002,         # BB logit increases per pitch
    "fatigue_hr_slope": 0.001,         # HR logit increases per pitch

    # --- HBP rate ---
    "league_hbp_rate": 0.011,

    # --- BIP outcome probabilities (bip_model.py) ---
    "bip_out": 0.700,
    "bip_single": 0.222,
    "bip_double": 0.065,
    "bip_triple": 0.005,

    # --- BABIP model (bip_model.py) ---
    "pop_babip": 0.300,
    "babip_shrinkage_k": 500,          # BIP needed for full weight

    # --- BABIP regression coefficients ---
    "babip_coef_avg_ev": 0.00035,
    "babip_coef_avg_la": -0.00396,
    "babip_coef_gb_pct": -0.08537,
    "babip_coef_sprint_speed": 0.00433,
    "babip_coef_intercept": 0.21759,

    # --- Hit distribution adjustments ---
    "hit_share_single": 0.755,
    "hit_share_double": 0.222,
    "hit_share_triple": 0.023,
    "ev_double_sensitivity": 0.03,     # EV factor effect on double share
    "gb_double_sensitivity": -0.02,    # GB factor effect on double share
    "speed_triple_sensitivity": 0.01,  # speed factor effect on triple share

    # --- Pitch count model (pitch_count_model.py) ---
    "pc_shrinkage_pitcher": 150,       # PA for full weight on pitcher adj
    "pc_shrinkage_batter": 150,        # PA for full weight on batter adj
    "pc_putaway_sensitivity": -3.0,    # higher putaway -> fewer pitches
    "pc_foul_rate_sensitivity": 2.5,   # higher foul rate -> more pitches
    "pc_contact_sensitivity": -1.0,    # higher contact -> fewer pitches

    # --- TTO model (tto_model.py) ---
    "tto_min_reliability_pa": 100.0,   # PA for full weight on pitcher-specific TTO
    "tto_bf_per_block": 9,             # batters per TTO block

    # --- Exit model (exit_model.py) ---
    "exit_default_avg_pitches": 88.0,
    "exit_shrinkage_k": 10,            # starts for full weight on pitcher exit rate

    # --- Exit calibration (simulator.py) ---
    "exit_calibration_offset": -0.35,  # logit offset to reduce exit prob (negative = stay longer)

    # --- Per-PA calibration offsets (pa_outcome_model.py) ---
    "calibration_k_offset": -0.02,     # K logit adjustment
    "calibration_bb_offset": 0.01,     # BB logit adjustment

    # --- Simulator (simulator.py) ---
    "max_pa_per_game": 45,             # safety valve

    # --- Monte Carlo settings ---
    "n_sims": 2000,                    # sims per game (lower = faster)
}


# ======================================================================
# Module patching — applies CONFIG to the actual simulator modules
# ======================================================================

def apply_config(cfg: dict) -> None:
    """Monkey-patch game simulator module constants from CONFIG."""
    import src.models.game_sim.pa_outcome_model as pa_mod
    import src.models.game_sim.bip_model as bip_mod
    import src.models.game_sim.pitch_count_model as pc_mod
    import src.models.game_sim.tto_model as tto_mod
    import src.models.game_sim.exit_model as exit_mod
    import src.models.game_sim.simulator as sim_mod

    # Fatigue
    pa_mod._FATIGUE_PITCH_THRESHOLD = cfg["fatigue_pitch_threshold"]
    pa_mod._FATIGUE_K_SLOPE = cfg["fatigue_k_slope"]
    pa_mod._FATIGUE_BB_SLOPE = cfg["fatigue_bb_slope"]
    pa_mod._FATIGUE_HR_SLOPE = cfg["fatigue_hr_slope"]
    pa_mod.LEAGUE_HBP_RATE = cfg["league_hbp_rate"]

    # BIP model
    bip_mod._DEFAULT_BIP_PROBS = {
        "out": cfg["bip_out"],
        "single": cfg["bip_single"],
        "double": cfg["bip_double"],
        "triple": cfg["bip_triple"],
    }
    bip_mod.POP_BABIP = cfg["pop_babip"]
    bip_mod._SHRINKAGE_K = cfg["babip_shrinkage_k"]
    bip_mod._BABIP_COEFS = {
        "avg_ev": cfg["babip_coef_avg_ev"],
        "avg_la": cfg["babip_coef_avg_la"],
        "gb_pct": cfg["babip_coef_gb_pct"],
        "sprint_speed": cfg["babip_coef_sprint_speed"],
        "intercept": cfg["babip_coef_intercept"],
    }

    # Pitch count
    pc_mod._SHRINKAGE_K_PITCHER = cfg["pc_shrinkage_pitcher"]
    pc_mod._SHRINKAGE_K_BATTER = cfg["pc_shrinkage_batter"]
    pc_mod._PUTAWAY_SENSITIVITY = cfg["pc_putaway_sensitivity"]
    pc_mod._FOUL_RATE_SENSITIVITY = cfg["pc_foul_rate_sensitivity"]
    pc_mod._CONTACT_SENSITIVITY = cfg["pc_contact_sensitivity"]

    # TTO
    tto_mod.BF_PER_TTO = cfg["tto_bf_per_block"]

    # Exit model
    exit_mod.DEFAULT_AVG_EXIT_PITCHES = cfg["exit_default_avg_pitches"]
    exit_mod._SHRINKAGE_K = cfg["exit_shrinkage_k"]

    # Exit calibration + per-PA offsets
    sim_mod._DEFAULT_EXIT_CALIBRATION_OFFSET = cfg["exit_calibration_offset"]
    pa_mod._CALIBRATION_K_OFFSET = cfg["calibration_k_offset"]
    pa_mod._CALIBRATION_BB_OFFSET = cfg["calibration_bb_offset"]

    # Simulator
    sim_mod.MAX_PA_PER_GAME = cfg["max_pa_per_game"]


# ======================================================================
# Experiment runner
# ======================================================================

def run_experiment(
    cfg: dict,
    max_games: int = 0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run the game simulator with CONFIG and return predictions DataFrame.

    Parameters
    ----------
    cfg : dict
        CONFIG dict with all tunable constants.
    max_games : int
        Max games to simulate (0 = all). Use 500-1500 for faster iteration.
    random_seed : int
        For reproducibility.
    """
    from autoresearch.prepare import load_cache
    from src.models.game_sim.simulator import simulate_game

    # Apply config before any model objects are created
    apply_config(cfg)

    cache = load_cache()
    games = cache["games"]
    exit_model = cache["exit_model"]
    n_sims = cfg["n_sims"]

    # Subsample if requested
    if max_games > 0 and len(games) > max_games:
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(len(games), size=max_games, replace=False)
        games = [games[i] for i in sorted(indices)]

    logger.info("Running %d games x %d sims...", len(games), n_sims)

    results = []
    for game in games:
        try:
            sim_result = simulate_game(
                pitcher_k_rate_samples=game["k_posterior"],
                pitcher_bb_rate_samples=game["bb_posterior"],
                pitcher_hr_rate_samples=game["hr_posterior"],
                lineup_matchup_lifts=game["matchup_lifts"],
                tto_lifts=game["tto_lifts"],
                pitcher_ppa_adj=game["pitcher_ppa_adj"],
                batter_ppa_adjs=game["batter_ppa_adjs"],
                exit_model=exit_model,
                pitcher_avg_pitches=game["pitcher_avg_pitches"],
                umpire_k_lift=game["umpire_k_lift"],
                umpire_bb_lift=game["umpire_bb_lift"],
                park_hr_lift=game["park_hr_lift"],
                weather_k_lift=game["weather_k_lift"],
                n_sims=n_sims,
                random_seed=random_seed + game["game_pk"] % 10000,
            )
        except Exception:
            continue

        summary = sim_result.summary()
        k_over = sim_result.over_probs("k", [3.5, 4.5, 5.5, 6.5, 7.5])
        outs_over = sim_result.over_probs("outs", [14.5, 15.5, 16.5, 17.5, 18.5])

        # Derive actual outs from IP (baseball notation)
        actual_ip = game["actual_ip"]
        actual_outs = int(actual_ip) * 3 + round((actual_ip % 1) * 10)

        rec = {
            "game_pk": game["game_pk"],
            "pitcher_id": game["pitcher_id"],
            "expected_k": summary["k"]["mean"],
            "std_k": summary["k"]["std"],
            "expected_bb": summary["bb"]["mean"],
            "std_bb": summary["bb"]["std"],
            "expected_h": summary["h"]["mean"],
            "std_h": summary["h"]["std"],
            "expected_hr": summary["hr"]["mean"],
            "expected_bf": summary["bf"]["mean"],
            "expected_outs": summary["outs"]["mean"],
            "std_outs": summary["outs"]["std"],
            "expected_ip": float(np.mean(sim_result.ip_samples())),
            "actual_k": game["actual_k"],
            "actual_bb": game["actual_bb"],
            "actual_h": game["actual_h"],
            "actual_hr": game["actual_hr"],
            "actual_bf": game["actual_bf"],
            "actual_outs": actual_outs,
            "actual_ip": game["actual_ip"],
        }

        for _, prow in outs_over.iterrows():
            col = f"p_outs_over_{prow['line']:.1f}".replace(".", "_")
            rec[col] = prow["p_over"]

        for _, prow in k_over.iterrows():
            col = f"p_over_{prow['line']:.1f}".replace(".", "_")
            rec[col] = prow["p_over"]

        results.append(rec)

    return pd.DataFrame(results)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-games", type=int, default=0,
                        help="Max games (0=all, 500-1500 for faster iteration)")
    args = parser.parse_args()

    t0 = time.time()

    predictions = run_experiment(CONFIG, max_games=args.max_games)

    from autoresearch.prepare import evaluate, format_metrics
    metrics = evaluate(predictions)
    elapsed = time.time() - t0

    print("\n" + "=" * 50)
    print(format_metrics(metrics))
    print(f"ELAPSED: {elapsed:.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
