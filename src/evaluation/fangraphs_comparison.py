"""
Compare TDD player rankings against FanGraphs Depth Charts projected fWAR.

Produces Spearman/Kendall rank correlations, tier agreement rates,
and identifies the biggest ranking disagreements for investigation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Results from a TDD vs FanGraphs ranking comparison."""

    player_type: str  # 'hitter' or 'pitcher'
    n_compared: int
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float
    tier_agreement: dict[str, float]  # tier_name -> agreement %
    biggest_overranks: pd.DataFrame  # TDD ranks much higher than FG
    biggest_underranks: pd.DataFrame  # TDD ranks much lower than FG
    merged: pd.DataFrame  # full merged data


def compare_hitter_rankings(
    tdd_rankings: pd.DataFrame,
    fg_projections: pd.DataFrame,
    top_n: int = 150,
) -> ComparisonResult:
    """Compare TDD hitter rankings against FanGraphs projected fWAR.

    Parameters
    ----------
    tdd_rankings : pd.DataFrame
        Output from rank_all()['hitters']. Must have 'batter_id' and
        'overall_rank' columns.
    fg_projections : pd.DataFrame
        Output from load_hitter_projections(). Must have 'player_id',
        'war', and 'fg_rank' columns.
    top_n : int
        Compare top N players from each system.

    Returns
    -------
    ComparisonResult
    """
    # Merge on MLBAM ID
    tdd = tdd_rankings.copy()
    fg = fg_projections.copy()

    # Standardize ID column name
    if "batter_id" in tdd.columns:
        tdd = tdd.rename(columns={"batter_id": "player_id"})

    # Use name columns for display
    tdd_name_col = "batter_name" if "batter_name" in tdd.columns else "player_name"
    tdd = tdd.rename(columns={tdd_name_col: "tdd_name"})

    # Keep relevant TDD columns
    tdd_cols = ["player_id", "tdd_name", "overall_rank", "current_value_score"]
    optional_tdd = [
        "position", "age", "projected_k_rate", "projected_bb_rate",
        "projected_woba", "offense_score", "fielding_combined",
        "trajectory_score", "grade_hit", "grade_power",
    ]
    for col in optional_tdd:
        if col in tdd.columns:
            tdd_cols.append(col)
    tdd = tdd[tdd_cols].copy()
    tdd = tdd.rename(columns={"overall_rank": "tdd_rank"})

    # Filter to top N from each system
    tdd_top = set(tdd.nsmallest(top_n, "tdd_rank")["player_id"].dropna())
    fg_top = set(fg.nsmallest(top_n, "fg_rank")["player_id"].dropna())
    universe = tdd_top | fg_top

    merged = tdd.merge(
        fg[["player_id", "fg_name", "fg_rank", "war", "team"]].rename(
            columns={"war": "fg_war"}
        ),
        on="player_id",
        how="inner",
    )
    # Only keep players in the top-N universe
    merged = merged[merged["player_id"].isin(universe)].copy()

    return _compute_comparison(merged, "hitter", top_n)


def compare_pitcher_rankings(
    tdd_rankings: pd.DataFrame,
    fg_projections: pd.DataFrame,
    top_n: int = 100,
) -> ComparisonResult:
    """Compare TDD pitcher rankings against FanGraphs projected fWAR.

    Parameters
    ----------
    tdd_rankings : pd.DataFrame
        Output from rank_all()['pitchers']. Must have 'pitcher_id' and
        'overall_rank' columns.
    fg_projections : pd.DataFrame
        Output from load_pitcher_projections(). Must have 'player_id',
        'war', and 'fg_rank' columns.
    top_n : int
        Compare top N players from each system.

    Returns
    -------
    ComparisonResult
    """
    tdd = tdd_rankings.copy()
    fg = fg_projections.copy()

    if "pitcher_id" in tdd.columns:
        tdd = tdd.rename(columns={"pitcher_id": "player_id"})

    tdd_name_col = "pitcher_name" if "pitcher_name" in tdd.columns else "player_name"
    tdd = tdd.rename(columns={tdd_name_col: "tdd_name"})

    tdd_cols = ["player_id", "tdd_name", "overall_rank", "current_value_score"]
    optional_tdd = [
        "role", "age", "projected_k_rate", "projected_bb_rate",
        "projected_era", "stuff_score", "command_score",
        "trajectory_score", "grade_stuff", "grade_command",
    ]
    for col in optional_tdd:
        if col in tdd.columns:
            tdd_cols.append(col)
    tdd = tdd[tdd_cols].copy()
    tdd = tdd.rename(columns={"overall_rank": "tdd_rank"})

    tdd_top = set(tdd.nsmallest(top_n, "tdd_rank")["player_id"].dropna())
    fg_top = set(fg.nsmallest(top_n, "fg_rank")["player_id"].dropna())
    universe = tdd_top | fg_top

    fg_cols = ["player_id", "fg_name", "fg_rank", "war", "team"]
    for col in ["ip", "era", "fip", "gs"]:
        if col in fg.columns:
            fg_cols.append(col)

    merged = tdd.merge(
        fg[fg_cols].rename(columns={"war": "fg_war"}),
        on="player_id",
        how="inner",
    )
    merged = merged[merged["player_id"].isin(universe)].copy()

    return _compute_comparison(merged, "pitcher", top_n)


def _compute_comparison(
    merged: pd.DataFrame,
    player_type: str,
    top_n: int,
) -> ComparisonResult:
    """Compute rank correlation, tier agreement, and disagreements."""
    n = len(merged)
    if n < 10:
        logger.warning(
            "Only %d %ss matched — results will be unreliable", n, player_type
        )

    # Rank correlations
    rho, rho_p = spearmanr(merged["tdd_rank"], merged["fg_rank"])
    tau, tau_p = kendalltau(merged["tdd_rank"], merged["fg_rank"])

    logger.info(
        "%s comparison (n=%d): Spearman ρ=%.3f (p=%.4f), Kendall τ=%.3f (p=%.4f)",
        player_type.title(), n, rho, rho_p, tau, tau_p,
    )

    # Tier agreement: are players in same quartile/tier?
    tier_agreement = {}
    for tier_name, tier_size in [
        ("top_10", 10), ("top_25", 25), ("top_50", 50), ("top_100", 100),
    ]:
        if tier_size > top_n:
            continue
        tdd_in_tier = set(
            merged.nsmallest(tier_size, "tdd_rank")["player_id"]
        )
        fg_in_tier = set(
            merged.nsmallest(tier_size, "fg_rank")["player_id"]
        )
        overlap = len(tdd_in_tier & fg_in_tier)
        agreement = overlap / tier_size
        tier_agreement[tier_name] = agreement
        logger.info(
            "  %s tier agreement: %d/%d (%.0f%%)",
            tier_name, overlap, tier_size, agreement * 100,
        )

    # Rank difference: positive = TDD ranks higher (lower number) than FG
    merged["rank_diff"] = merged["fg_rank"] - merged["tdd_rank"]
    merged["abs_rank_diff"] = merged["rank_diff"].abs()

    # Biggest disagreements
    overranks = (
        merged[merged["rank_diff"] > 0]
        .nlargest(15, "rank_diff")
        .copy()
    )
    underranks = (
        merged[merged["rank_diff"] < 0]
        .nsmallest(15, "rank_diff")
        .copy()
    )

    merged = merged.sort_values("tdd_rank").reset_index(drop=True)

    return ComparisonResult(
        player_type=player_type,
        n_compared=n,
        spearman_rho=rho,
        spearman_p=rho_p,
        kendall_tau=tau,
        kendall_p=tau_p,
        tier_agreement=tier_agreement,
        biggest_overranks=overranks,
        biggest_underranks=underranks,
        merged=merged,
    )


def format_comparison_report(result: ComparisonResult) -> str:
    """Format a ComparisonResult as a readable text report."""
    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append(
        f"  TDD vs FanGraphs Depth Charts — {result.player_type.title()} Rankings"
    )
    lines.append(sep)
    lines.append(f"Players compared: {result.n_compared}")
    lines.append("")

    # Rank correlation
    lines.append("RANK CORRELATION")
    lines.append(f"  Spearman rho = {result.spearman_rho:+.3f}  (p = {result.spearman_p:.4f})")
    lines.append(f"  Kendall  tau = {result.kendall_tau:+.3f}  (p = {result.kendall_p:.4f})")
    quality = (
        "Excellent" if result.spearman_rho > 0.85
        else "Good" if result.spearman_rho > 0.70
        else "Moderate" if result.spearman_rho > 0.50
        else "Weak"
    )
    lines.append(f"  Quality: {quality}")
    lines.append("")

    # Tier agreement
    lines.append("TIER AGREEMENT (overlap %)")
    for tier, pct in result.tier_agreement.items():
        bar = "#" * int(pct * 20) + "." * (20 - int(pct * 20))
        lines.append(f"  {tier:>8s}: {bar} {pct:.0%}")
    lines.append("")

    # Biggest overranks (TDD higher than FG)
    lines.append("TDD RANKS HIGHER THAN FANGRAPHS (potential breakout / upside calls)")
    lines.append(f"  {'Player':<25s} {'TDD':>5s} {'FG':>5s} {'Diff':>6s} {'fWAR':>5s}")
    lines.append(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*6} {'-'*5}")
    for _, row in result.biggest_overranks.head(10).iterrows():
        name = row.get("tdd_name", row.get("fg_name", "???"))
        lines.append(
            f"  {name:<25s} {int(row['tdd_rank']):>5d} {int(row['fg_rank']):>5d} "
            f"{int(row['rank_diff']):>+6d} {row['fg_war']:>5.1f}"
        )
    lines.append("")

    # Biggest underranks (FG higher than TDD)
    lines.append("FANGRAPHS RANKS HIGHER THAN TDD (potential gaps / undervaluation)")
    lines.append(f"  {'Player':<25s} {'TDD':>5s} {'FG':>5s} {'Diff':>6s} {'fWAR':>5s}")
    lines.append(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*6} {'-'*5}")
    for _, row in result.biggest_underranks.head(10).iterrows():
        name = row.get("tdd_name", row.get("fg_name", "???"))
        lines.append(
            f"  {name:<25s} {int(row['tdd_rank']):>5d} {int(row['fg_rank']):>5d} "
            f"{int(row['rank_diff']):>+6d} {row['fg_war']:>5.1f}"
        )
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


def save_comparison_csv(
    result: ComparisonResult,
    output_dir: str = "outputs",
) -> str:
    """Save merged comparison data to CSV for further analysis."""
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    path = out / f"fangraphs_comparison_{result.player_type}s.csv"
    result.merged.to_csv(path, index=False)
    logger.info("Saved comparison CSV to %s", path)
    return str(path)
