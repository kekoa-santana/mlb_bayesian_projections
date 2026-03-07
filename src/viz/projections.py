"""
Pre-season projection utilities and chart generators.

Forward-projects Bayesian posteriors one step, identifies movers (biggest
K% changes), and renders cream/navy Twitter/X content cards.
"""
from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src.data.db import read_sql
from src.viz.theme import (
    CREAM,
    DARK_BG,
    GOLD,
    SAGE,
    SLATE,
    add_brand_footer,
    add_watermark,
    format_pct,
    save_card,
)

logger = logging.getLogger(__name__)

# Text color shortcuts for cream background
_TEXT = DARK_BG      # primary text on cream
_TEXT_SEC = SLATE    # secondary text


# ===================================================================
# Data utilities
# ===================================================================


def project_posteriors_forward(
    trace: Any,
    data: dict[str, Any],
    season: int,
    id_col: str = "pitcher_id",
    name_col: str = "pitcher_name",
    hand_col: str = "pitch_hand",
    trials_col: str = "batters_faced",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Project posteriors one step forward via random-walk innovation.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted trace from the K% model.
    data : dict
        Model data dict (must contain ``"df"`` DataFrame).
    season : int
        Season to project *from* (most recent in training data).
    id_col : str
        Player ID column (``"pitcher_id"`` or ``"batter_id"``).
    name_col : str
        Player name column.
    hand_col : str
        Handedness column (``"pitch_hand"`` or ``"batter_stand"``).
    trials_col : str
        Trial count column (``"batters_faced"`` or ``"pa"``).
    random_seed : int
        For reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per player with observed/projected K% and delta.
    """
    df = data["df"].reset_index(drop=True)
    k_rate_post = np.asarray(trace.posterior["k_rate"].values, dtype=float)
    k_rate_flat = k_rate_post.reshape(-1, k_rate_post.shape[-1])
    n_draws = k_rate_flat.shape[0]

    # sigma_season posterior for innovation noise
    if "sigma_season" in trace.posterior:
        sigma_flat = trace.posterior["sigma_season"].values.flatten()
    else:
        sigma_flat = np.full(n_draws, 0.05)

    rng = np.random.default_rng(random_seed)

    mask = df["season"] == season
    positions = np.where(mask.values)[0]

    records: list[dict[str, Any]] = []
    for pos in positions:
        row = df.iloc[pos]
        samples = k_rate_flat[:, pos].copy()

        # Forward project: add one innovation step on logit scale
        eps = np.clip(samples, 1e-6, 1 - 1e-6)
        logit_samples = np.log(eps / (1.0 - eps))
        innovation = rng.normal(0, sigma_flat[:n_draws])
        projected = 1.0 / (1.0 + np.exp(-(logit_samples + innovation)))

        obs_k = float(row["k_rate"])
        proj_mean = float(np.mean(projected))

        records.append({
            id_col: row[id_col],
            name_col: row.get(name_col, ""),
            hand_col: row.get(hand_col, ""),
            "season": season,
            trials_col: int(row[trials_col]),
            "observed_k_rate": obs_k,
            "projected_k_rate_mean": proj_mean,
            "projected_k_rate_sd": float(np.std(projected)),
            "projected_k_rate_2_5": float(np.percentile(projected, 2.5)),
            "projected_k_rate_97_5": float(np.percentile(projected, 97.5)),
            "delta": proj_mean - obs_k,
        })

    return pd.DataFrame(records)


def find_movers(
    projected: pd.DataFrame,
    n_top: int = 5,
    min_trials: int = 200,
    player_type: str = "pitcher",
    role_filter: str | None = None,
    id_col: str | None = None,
    trials_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find players with the biggest projected K% changes.

    Parameters
    ----------
    projected : pd.DataFrame
        Output of ``project_posteriors_forward``.
    n_top : int
        Number of movers per direction.
    min_trials : int
        Minimum trials in the observed season.
    player_type : str
        ``"pitcher"`` or ``"hitter"`` — determines semantics.
    role_filter : str | None
        If ``"starter"``, filter to ``is_starter == 1``.
    id_col : str | None
        Override player ID column.
    trials_col : str | None
        Override trials column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(improvers, decliners)`` sorted by |delta|.

        - Pitcher: improver = K% goes UP (better), decliner = goes DOWN.
        - Hitter: improver = K% goes DOWN (better), decliner = goes UP.
    """
    if id_col is None:
        id_col = "pitcher_id" if player_type == "pitcher" else "batter_id"
    if trials_col is None:
        trials_col = "batters_faced" if player_type == "pitcher" else "pa"

    df = projected.copy()

    # Filter by minimum trials
    df = df[df[trials_col] >= min_trials]

    # Filter by role
    if role_filter == "starter" and "is_starter" in df.columns:
        df = df[df["is_starter"] == 1]

    if player_type == "pitcher":
        # Pitcher: higher K% = better → improver has positive delta
        improvers = df[df["delta"] > 0].nlargest(n_top, "delta")
        decliners = df[df["delta"] < 0].nsmallest(n_top, "delta")
    else:
        # Hitter: lower K% = better → improver has negative delta
        improvers = df[df["delta"] < 0].nsmallest(n_top, "delta")
        decliners = df[df["delta"] > 0].nlargest(n_top, "delta")

    return improvers.reset_index(drop=True), decliners.reset_index(drop=True)


def enrich_with_team_info(
    df: pd.DataFrame,
    id_col: str = "pitcher_id",
) -> pd.DataFrame:
    """Add ``team`` column from dim_player → dim_team.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``id_col`` column.
    id_col : str
        Player ID column name.

    Returns
    -------
    pd.DataFrame
        With ``team`` column added.
    """
    player_ids = df[id_col].unique().tolist()
    if not player_ids:
        df["team"] = ""
        return df

    placeholders = ", ".join(str(int(pid)) for pid in player_ids)
    query = f"""
        SELECT p.player_id AS {id_col},
               COALESCE(t.abbreviation, '') AS team
        FROM production.dim_player p
        LEFT JOIN production.dim_team t ON p.team_id = t.team_id
        WHERE p.player_id IN ({placeholders})
    """
    team_df = read_sql(query)
    return df.merge(team_df, on=id_col, how="left").fillna({"team": ""})


# ===================================================================
# Chart generators
# ===================================================================


def _format_name(row: pd.Series, name_col: str, hand_col: str) -> str:
    """Build display name like 'G. Cole (NYY) RHP'."""
    full_name = str(row.get(name_col, ""))
    parts = full_name.split(", ")
    if len(parts) == 2:
        last, first = parts
        display = f"{first[0]}. {last}"
    else:
        display = full_name

    team = row.get("team", "")
    hand = row.get(hand_col, "")
    suffix = ""
    if team:
        suffix += f" ({team})"
    if hand:
        suffix += f" {hand}HP"
    return display + suffix


def _draw_mover_column(
    ax: plt.Axes,
    players: pd.DataFrame,
    bar_color: str,
    direction_label: str,
    name_col: str = "pitcher_name",
    hand_col: str = "pitch_hand",
) -> None:
    """Draw one column of mover bars on the given axes."""
    ax.set_xlim(0.05, 0.45)
    n = len(players)
    if n == 0:
        ax.text(0.25, 0.5, "No data", color=_TEXT_SEC, ha="center",
                va="center", fontsize=14, transform=ax.transAxes)
        ax.set_yticks([])
        return

    # Use 1.4 units per row for breathing room
    ROW_STEP = 1.4
    y_positions = [(n - 1 - i) * ROW_STEP for i in range(n)]

    for i, (_, row) in enumerate(players.iterrows()):
        y = y_positions[i]
        proj = row["projected_k_rate_mean"]
        obs = row["observed_k_rate"]
        delta = row["delta"]

        # Horizontal bar at projected K%
        ax.barh(y, proj, height=0.55, color=bar_color, alpha=0.85, zorder=3)

        # Observed marker (dashed vertical line)
        ax.plot([obs, obs], [y - 0.27, y + 0.27], color=_TEXT_SEC,
                linestyle="--", linewidth=1.5, zorder=4)

        # Name + team
        label = _format_name(row, name_col, hand_col)
        ax.text(0.052, y + 0.33, label, color=_TEXT, fontsize=11,
                va="bottom", fontweight="bold", zorder=5)

        # Projected K% — dynamically placed to avoid the dashed line
        # The text needs ~0.035 of x-space. Place it right of the
        # rightmost element (bar tip or observed line) when there's room,
        # otherwise tuck it inside the bar.
        right_edge = max(proj, obs)
        if right_edge + 0.045 < 0.40:
            # Room to the right of both bar and dashed line
            pct_x = right_edge + 0.008
            pct_ha = "left"
            pct_color = _TEXT
        else:
            # Tight — place inside the bar where there's solid color
            pct_x = min(proj, obs) - 0.005
            pct_ha = "right"
            pct_color = CREAM  # readable on teal/slate bar
        ax.text(pct_x, y + 0.01, format_pct(proj),
                color=pct_color, fontsize=11, va="center", ha=pct_ha,
                fontweight="bold", zorder=5)

        # Delta pp — gold, right-aligned
        sign = "+" if delta > 0 else ""
        ax.text(0.44, y + 0.01, f"{sign}{delta * 100:.1f}pp",
                color=GOLD, fontsize=11, va="center", ha="right",
                fontweight="bold", zorder=5)

        # 2025 observed below bar
        ax.text(0.052, y - 0.38, f"2025: {format_pct(obs)}",
                color=_TEXT_SEC, fontsize=9, va="top", zorder=5)

    y_max = y_positions[0]
    ax.set_ylim(-0.7, y_max + 0.8)
    ax.set_yticks([])
    ax.set_xticks([])

    # Column title
    ax.set_title(direction_label, color=_TEXT, fontsize=14,
                 fontweight="bold", pad=10, loc="left")


def plot_pitcher_k_movers(
    improvers: pd.DataFrame,
    decliners: pd.DataFrame,
    season_label: str = "2026",
) -> plt.Figure:
    """Two-column 16:9 card showing pitcher K% movers.

    Parameters
    ----------
    improvers : pd.DataFrame
        Pitchers with biggest K% increase.
    decliners : pd.DataFrame
        Pitchers with biggest K% decrease.
    season_label : str
        Season being projected.

    Returns
    -------
    plt.Figure
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(top=0.80, bottom=0.1, left=0.04, right=0.96,
                        wspace=0.15)

    # Title
    fig.text(0.5, 0.93,
             f"PITCHER K% PROJECTIONS \u2014 {season_label}",
             ha="center", va="center", fontsize=22, color=GOLD,
             fontweight="bold")
    fig.text(0.5, 0.885,
             "Bayesian model: 2025 observed vs 2026 projected",
             ha="center", va="center", fontsize=13, color=_TEXT_SEC)

    _draw_mover_column(ax_l, improvers, SAGE,
                       "PROJECTED K% INCREASE",
                       name_col="pitcher_name", hand_col="pitch_hand")
    _draw_mover_column(ax_r, decliners, _TEXT_SEC,
                       "PROJECTED K% DECREASE",
                       name_col="pitcher_name", hand_col="pitch_hand")

    add_watermark(fig)
    add_brand_footer(fig, "Bayesian projection model")
    return fig


def plot_hitter_k_movers(
    improvers: pd.DataFrame,
    decliners: pd.DataFrame,
    season_label: str = "2026",
) -> plt.Figure:
    """Two-column 16:9 card showing hitter K% movers.

    Parameters
    ----------
    improvers : pd.DataFrame
        Hitters with biggest K% decrease (good for hitter).
    decliners : pd.DataFrame
        Hitters with biggest K% increase (bad for hitter).
    season_label : str
        Season being projected.

    Returns
    -------
    plt.Figure
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(top=0.80, bottom=0.1, left=0.04, right=0.96,
                        wspace=0.15)

    fig.text(0.5, 0.93,
             f"HITTER K% PROJECTIONS \u2014 {season_label}",
             ha="center", va="center", fontsize=22, color=GOLD,
             fontweight="bold")
    fig.text(0.5, 0.885,
             "Bayesian model: 2025 observed vs 2026 projected",
             ha="center", va="center", fontsize=13, color=_TEXT_SEC)

    _draw_mover_column(ax_l, improvers, SAGE,
                       "WILL STRIKE OUT LESS",
                       name_col="batter_name", hand_col="batter_stand")
    _draw_mover_column(ax_r, decliners, _TEXT_SEC,
                       "WILL STRIKE OUT MORE",
                       name_col="batter_name", hand_col="batter_stand")

    add_watermark(fig)
    add_brand_footer(fig, "Bayesian projection model")
    return fig


def plot_pitcher_card(
    pitcher_name: str,
    team: str,
    hand: str,
    observed_k_rate: float,
    projected_samples: np.ndarray,
    projected_summary: dict[str, float],
    season_label: str = "2026",
) -> plt.Figure:
    """1:1 individual pitcher projection card with KDE + credible interval.

    Parameters
    ----------
    pitcher_name : str
        Display name.
    team : str
        Team abbreviation.
    hand : str
        Pitching hand (R/L).
    observed_k_rate : float
        2025 observed K%.
    projected_samples : np.ndarray
        Forward-projected posterior samples.
    projected_summary : dict
        Keys: ``mean``, ``ci_lo``, ``ci_hi``.
    season_label : str
        Season being projected.

    Returns
    -------
    plt.Figure
    """
    proj_mean = projected_summary["mean"]
    ci_lo = projected_summary["ci_lo"]
    ci_hi = projected_summary["ci_hi"]

    fig = plt.figure(figsize=(10, 10))
    # Use 4 rows: title, stats, KDE, CI — with explicit spacing
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[0.8, 0.8, 3, 1.2],
        top=0.92, bottom=0.08, left=0.1, right=0.9,
        hspace=0.35,
    )

    # --- Row 0: Name + team ---
    ax_name = fig.add_subplot(gs[0])
    ax_name.set_xlim(0, 1)
    ax_name.set_ylim(0, 1)
    ax_name.axis("off")

    display_name = pitcher_name.upper()
    parts = pitcher_name.split(", ")
    if len(parts) == 2:
        display_name = f"{parts[1].upper()} {parts[0].upper()}"

    hand_str = f"{hand}HP" if hand else ""
    ax_name.text(0.5, 0.65, display_name, color=_TEXT,
                 fontsize=26, fontweight="bold", ha="center", va="center")
    ax_name.text(0.5, 0.15,
                 f"{team}  |  {hand_str}",
                 color=_TEXT_SEC, fontsize=16, ha="center", va="center")

    # --- Row 1: Big K% numbers ---
    ax_stats = fig.add_subplot(gs[1])
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis("off")

    ax_stats.text(0.25, 0.75, "2025 K%", color=_TEXT_SEC, fontsize=13,
                  ha="center", va="center")
    ax_stats.text(0.25, 0.25, format_pct(observed_k_rate),
                  color=_TEXT, fontsize=32, fontweight="bold",
                  ha="center", va="center")
    ax_stats.text(0.75, 0.75, f"{season_label} PROJECTED",
                  color=_TEXT_SEC, fontsize=13, ha="center", va="center")
    ax_stats.text(0.75, 0.25, format_pct(proj_mean),
                  color=GOLD, fontsize=32, fontweight="bold",
                  ha="center", va="center")

    # --- Row 2: KDE plot ---
    ax_kde = fig.add_subplot(gs[2])
    kde = gaussian_kde(projected_samples, bw_method=0.25)
    x_grid = np.linspace(
        max(0, projected_samples.min() - 0.03),
        min(1, projected_samples.max() + 0.03),
        500,
    )
    density = kde(x_grid)

    ax_kde.fill_between(x_grid, density, alpha=0.3, color=GOLD, zorder=3)
    ax_kde.plot(x_grid, density, color=GOLD, linewidth=2, zorder=4)

    # Observed line
    ax_kde.axvline(observed_k_rate, color=_TEXT_SEC, linestyle="--",
                   linewidth=2, zorder=5, label="2025 observed")
    # Projected mean line
    ax_kde.axvline(proj_mean, color=GOLD, linestyle="-",
                   linewidth=2.5, zorder=5, label=f"{season_label} projected")

    ax_kde.set_xlabel("K%", color=_TEXT, fontsize=13)
    ax_kde.set_ylabel("")
    ax_kde.set_yticks([])
    ax_kde.tick_params(axis="x", colors=_TEXT, labelsize=11)
    ax_kde.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    )
    ax_kde.legend(fontsize=10, loc="upper right",
                  facecolor=CREAM, edgecolor=_TEXT_SEC, labelcolor=_TEXT)
    ax_kde.spines["bottom"].set_visible(True)
    ax_kde.spines["bottom"].set_color(_TEXT_SEC)

    # --- Row 3: Credible interval bar ---
    ax_ci = fig.add_subplot(gs[3])
    ax_ci.set_xlim(ci_lo - 0.02, ci_hi + 0.02)
    ax_ci.set_ylim(0, 1)
    ax_ci.axis("off")

    ax_ci.text(0.5, 0.92, "95% CREDIBLE INTERVAL",
               color=_TEXT, fontsize=13, fontweight="bold",
               ha="center", va="center", transform=ax_ci.transAxes)

    bar_y = 0.5
    ax_ci.plot([ci_lo, ci_hi], [bar_y, bar_y], color=GOLD,
               linewidth=8, solid_capstyle="round", zorder=3)
    ax_ci.plot(ci_lo, bar_y, "|", color=GOLD, markersize=20, zorder=4)
    ax_ci.plot(ci_hi, bar_y, "|", color=GOLD, markersize=20, zorder=4)

    ax_ci.text(ci_lo, 0.15, format_pct(ci_lo),
               color=_TEXT_SEC, fontsize=13, ha="center", va="center")
    ax_ci.text(ci_hi, 0.15, format_pct(ci_hi),
               color=_TEXT_SEC, fontsize=13, ha="center", va="center")
    ax_ci.plot(proj_mean, bar_y, "D", color=_TEXT, markersize=8, zorder=5)

    # Card title
    fig.text(0.5, 0.97,
             f"K% PROJECTION \u2014 {season_label}",
             ha="center", va="center", fontsize=18, color=GOLD,
             fontweight="bold")

    add_watermark(fig)
    add_brand_footer(fig, "Bayesian projection model")
    return fig
