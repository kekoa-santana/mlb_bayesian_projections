"""
Composite breakout/regression content cards for hitters and pitchers.

Renders 16:9 Twitter/X cards showing multi-stat projected deltas and
a weighted composite improvement score.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz.theme import (
    CREAM,
    DARK_BG,
    GOLD,
    SAGE,
    SLATE,
    add_brand_footer,
    add_watermark,
    format_pct,
)

_TEXT = DARK_BG
_TEXT_SEC = SLATE
_RED = "#D4654A"  # regression / negative delta accent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_delta(value: float, is_pct: bool = True) -> str:
    """Format a delta as +X.X pp or +.XXX."""
    if np.isnan(value):
        return "—"
    if is_pct:
        sign = "+" if value > 0 else ""
        return f"{sign}{value * 100:.1f}"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.3f}"


def _delta_color(value: float, sign_good: int) -> str:
    """Return SAGE if delta is good, _RED if bad, SLATE if neutral/NaN."""
    if np.isnan(value):
        return _TEXT_SEC
    # sign_good: +1 means higher is better, -1 means lower is better
    if sign_good * value > 0.001:
        return SAGE
    elif sign_good * value < -0.001:
        return _RED
    return _TEXT_SEC


def _short_name(full_name: str) -> str:
    """'Last, First' -> 'F. Last'."""
    parts = full_name.split(", ")
    if len(parts) == 2:
        return f"{parts[1][0]}. {parts[0]}"
    return full_name


# ---------------------------------------------------------------------------
# Hitter composite card
# ---------------------------------------------------------------------------

# Stat display configs: (stat_key, label, sign_for_hitter, is_pct)
_HITTER_STATS = [
    ("k_rate",  "K%",   -1, True),
    ("bb_rate", "BB%",  +1, True),
    ("hr_rate", "HR%",  +1, True),
    ("xwoba",   "xwOBA", +1, False),
]


def plot_hitter_composite(
    players: pd.DataFrame,
    card_type: str = "breakout",
    season_label: str = "2026",
) -> plt.Figure:
    """16:9 card showing hitter composite breakout or regression candidates.

    Parameters
    ----------
    players : pd.DataFrame
        Rows from ``project_forward`` output (hitter_projections).
    card_type : str
        ``"breakout"`` or ``"regression"``.
    season_label : str
        Season being projected.

    Returns
    -------
    plt.Figure
    """
    n = len(players)
    if n == 0:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=20, color=_TEXT_SEC)
        ax.axis("off")
        return fig

    fig = plt.figure(figsize=(16, 9))

    # Title
    if card_type == "breakout":
        title = f"HITTER BREAKOUT CANDIDATES — {season_label}"
        accent = SAGE
        subtitle = "Projected to improve across K%, BB%, HR%, xwOBA"
    else:
        title = f"HITTER REGRESSION RISKS — {season_label}"
        accent = _RED
        subtitle = "Projected to decline across K%, BB%, HR%, xwOBA"

    fig.text(0.50, 0.94, title,
             ha="center", va="center", fontsize=22, color=GOLD,
             fontweight="bold")
    fig.text(0.50, 0.895, subtitle,
             ha="center", va="center", fontsize=12, color=_TEXT_SEC)

    # Table area
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column positions (x)
    col_x = {
        "rank": 0.02,
        "name": 0.06,
        "age": 0.28,
        "k_rate": 0.36,
        "bb_rate": 0.48,
        "hr_rate": 0.60,
        "xwoba": 0.72,
        "score": 0.88,
    }

    # Header row
    header_y = 0.96
    header_kw = dict(fontsize=11, color=_TEXT_SEC, ha="center",
                     va="center", fontweight="bold")
    ax.text(col_x["rank"], header_y, "#", **header_kw)
    ax.text(col_x["name"] + 0.06, header_y, "PLAYER",
            fontsize=11, color=_TEXT_SEC, ha="left", va="center",
            fontweight="bold")
    ax.text(col_x["age"], header_y, "AGE", **header_kw)

    for stat_key, label, _, _ in _HITTER_STATS:
        ax.text(col_x[stat_key], header_y, f"\u0394{label}", **header_kw)

    ax.text(col_x["score"], header_y, "COMPOSITE", **header_kw)

    # Divider under header
    ax.plot([0.01, 0.99], [0.93, 0.93], color=_TEXT_SEC, linewidth=0.5,
            alpha=0.4)

    # Player rows
    row_height = 0.88 / max(n, 1)
    for i, (_, row) in enumerate(players.iterrows()):
        y = 0.90 - i * row_height

        # Alternating row background
        if i % 2 == 1:
            ax.axhspan(y - row_height * 0.45, y + row_height * 0.45,
                       color=DARK_BG, alpha=0.03, zorder=0)

        # Rank
        ax.text(col_x["rank"], y, str(i + 1),
                fontsize=12, color=_TEXT_SEC, ha="center", va="center")

        # Name
        name = _short_name(str(row.get("batter_name", "")))
        hand = row.get("batter_stand", "")
        name_display = f"{name}"
        if hand:
            name_display += f"  {hand}"
        ax.text(col_x["name"], y, name_display,
                fontsize=12, color=_TEXT, ha="left", va="center",
                fontweight="bold")

        # Age
        age = row.get("age", "")
        ax.text(col_x["age"], y, str(int(age)) if age else "",
                fontsize=12, color=_TEXT, ha="center", va="center")

        # Stat deltas
        for stat_key, _, sign_good, is_pct in _HITTER_STATS:
            delta_col = f"delta_{stat_key}"
            delta = row.get(delta_col, np.nan)
            color = _delta_color(delta, sign_good)
            text = _fmt_delta(delta, is_pct=is_pct)
            ax.text(col_x[stat_key], y, text,
                    fontsize=12, color=color, ha="center", va="center",
                    fontweight="bold")

        # Composite score
        score = row.get("composite_score", 0)
        score_color = accent
        ax.text(col_x["score"], y, f"{score:+.2f}",
                fontsize=13, color=score_color, ha="center", va="center",
                fontweight="bold")

    # Legend at bottom
    ax.text(0.01, -0.02,
            "\u0394 = projected change from 2025  |  "
            "K%/BB%/HR% in percentage points  |  "
            "Composite = z-score weighted across stats",
            fontsize=9, color=_TEXT_SEC, ha="left", va="top")

    add_watermark(fig)
    add_brand_footer(fig, "Bayesian composite projection model")
    return fig


# ---------------------------------------------------------------------------
# Pitcher composite card
# ---------------------------------------------------------------------------

_PITCHER_STATS = [
    ("k_rate",    "K%",    +1, True),
    ("bb_rate",   "BB%",   -1, True),
    ("hr_per_bf", "HR/BF", -1, True),
]


def plot_pitcher_composite(
    players: pd.DataFrame,
    card_type: str = "breakout",
    season_label: str = "2026",
) -> plt.Figure:
    """16:9 card showing pitcher composite breakout or regression candidates.

    Parameters
    ----------
    players : pd.DataFrame
        Rows from ``project_forward`` output (pitcher_projections).
    card_type : str
        ``"breakout"`` or ``"regression"``.
    season_label : str
        Season being projected.

    Returns
    -------
    plt.Figure
    """
    n = len(players)
    if n == 0:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=20, color=_TEXT_SEC)
        ax.axis("off")
        return fig

    fig = plt.figure(figsize=(16, 9))

    if card_type == "breakout":
        title = f"PITCHER BREAKOUT CANDIDATES — {season_label}"
        accent = SAGE
        subtitle = "Projected to improve across K%, BB%, HR/BF"
    else:
        title = f"PITCHER REGRESSION RISKS — {season_label}"
        accent = _RED
        subtitle = "Projected to decline across K%, BB%, HR/BF"

    fig.text(0.50, 0.94, title,
             ha="center", va="center", fontsize=22, color=GOLD,
             fontweight="bold")
    fig.text(0.50, 0.895, subtitle,
             ha="center", va="center", fontsize=12, color=_TEXT_SEC)

    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    col_x = {
        "rank": 0.02,
        "name": 0.06,
        "age": 0.30,
        "role": 0.37,
        "k_rate": 0.47,
        "bb_rate": 0.59,
        "hr_per_bf": 0.71,
        "score": 0.88,
    }

    header_y = 0.96
    header_kw = dict(fontsize=11, color=_TEXT_SEC, ha="center",
                     va="center", fontweight="bold")
    ax.text(col_x["rank"], header_y, "#", **header_kw)
    ax.text(col_x["name"] + 0.06, header_y, "PLAYER",
            fontsize=11, color=_TEXT_SEC, ha="left", va="center",
            fontweight="bold")
    ax.text(col_x["age"], header_y, "AGE", **header_kw)
    ax.text(col_x["role"], header_y, "ROLE", **header_kw)

    for stat_key, label, _, _ in _PITCHER_STATS:
        ax.text(col_x[stat_key], header_y, f"\u0394{label}", **header_kw)

    ax.text(col_x["score"], header_y, "COMPOSITE", **header_kw)

    ax.plot([0.01, 0.99], [0.93, 0.93], color=_TEXT_SEC, linewidth=0.5,
            alpha=0.4)

    row_height = 0.88 / max(n, 1)
    for i, (_, row) in enumerate(players.iterrows()):
        y = 0.90 - i * row_height

        if i % 2 == 1:
            ax.axhspan(y - row_height * 0.45, y + row_height * 0.45,
                       color=DARK_BG, alpha=0.03, zorder=0)

        ax.text(col_x["rank"], y, str(i + 1),
                fontsize=12, color=_TEXT_SEC, ha="center", va="center")

        name = _short_name(str(row.get("pitcher_name", "")))
        hand = row.get("pitch_hand", "")
        name_display = f"{name}"
        if hand:
            name_display += f"  {hand}HP"
        ax.text(col_x["name"], y, name_display,
                fontsize=12, color=_TEXT, ha="left", va="center",
                fontweight="bold")

        age = row.get("age", "")
        ax.text(col_x["age"], y, str(int(age)) if age else "",
                fontsize=12, color=_TEXT, ha="center", va="center")

        role = "SP" if row.get("is_starter", 0) == 1 else "RP"
        ax.text(col_x["role"], y, role,
                fontsize=11, color=_TEXT_SEC, ha="center", va="center")

        for stat_key, _, sign_good, is_pct in _PITCHER_STATS:
            delta_col = f"delta_{stat_key}"
            delta = row.get(delta_col, np.nan)
            color = _delta_color(delta, sign_good)
            text = _fmt_delta(delta, is_pct=is_pct)
            ax.text(col_x[stat_key], y, text,
                    fontsize=12, color=color, ha="center", va="center",
                    fontweight="bold")

        score = row.get("composite_score", 0)
        score_color = accent
        ax.text(col_x["score"], y, f"{score:+.2f}",
                fontsize=13, color=score_color, ha="center", va="center",
                fontweight="bold")

    ax.text(0.01, -0.02,
            "\u0394 = projected change from 2025  |  "
            "All deltas in percentage points  |  "
            "Composite = z-score weighted across stats",
            fontsize=9, color=_TEXT_SEC, ha="left", va="top")

    add_watermark(fig)
    add_brand_footer(fig, "Bayesian composite projection model")
    return fig
