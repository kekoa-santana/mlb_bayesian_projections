"""
The Data Diamond -- MLB Bayesian Projection Dashboard.

Interactive Streamlit app for exploring hierarchical Bayesian player
projections, posterior distributions, and game-level K predictions.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.viz.theme import (  # noqa: E402
    GOLD, EMBER, SAGE, SLATE, CREAM, DARK,
    add_watermark,
)

# ---------------------------------------------------------------------------
# Derived colors for dashboard dark theme
# ---------------------------------------------------------------------------
DARK_CARD = "#181b23"
DARK_BORDER = "#2a2e3a"
POSITIVE = SAGE
NEGATIVE = EMBER
DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"
ICON_PATH = PROJECT_ROOT / "iconTransparent.png"

st.set_page_config(
    page_title="The Data Diamond | MLB Projections",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    /* Brand header */
    .brand-header {{
        background: {DARK_CARD};
        border: 1px solid {DARK_BORDER};
        padding: 1.2rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .brand-title {{
        color: {GOLD};
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 1px;
    }}
    .brand-subtitle {{
        color: {SLATE};
        font-size: 0.9rem;
        margin-top: 2px;
    }}

    /* Metric cards */
    .metric-card {{
        background: {DARK_CARD};
        border: 1px solid {DARK_BORDER};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }}
    .metric-value {{
        color: {GOLD};
        font-size: 1.6rem;
        font-weight: 700;
    }}
    .metric-label {{
        color: {SLATE};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }}
    .metric-delta {{
        font-size: 0.85rem;
        margin-top: 4px;
    }}

    /* Insight cards */
    .insight-card {{
        background: {DARK_CARD};
        border: 1px solid {DARK_BORDER};
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .insight-title {{
        color: {GOLD};
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }}
    .insight-bullet {{
        color: {CREAM};
        font-size: 0.92rem;
        line-height: 1.7;
        padding-left: 0.5rem;
    }}
    .insight-bullet .dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }}

    /* Percentile bars */
    .pctile-row {{
        display: flex;
        align-items: center;
        margin: 10px 0;
    }}
    .pctile-label {{
        width: 65px;
        color: {SLATE};
        font-size: 0.85rem;
        font-weight: 600;
    }}
    .pctile-bar-bg {{
        flex: 1;
        background: {DARK};
        border-radius: 6px;
        height: 22px;
        position: relative;
        overflow: hidden;
        border: 1px solid {DARK_BORDER};
    }}
    .pctile-bar-fill {{
        height: 100%;
        border-radius: 5px;
        transition: width 0.3s ease;
    }}
    .pctile-info {{
        width: 220px;
        text-align: right;
        color: {SLATE};
        font-size: 0.8rem;
        padding-left: 12px;
    }}

    /* Delta colors */
    .delta-pos {{ color: {POSITIVE}; font-weight: 600; }}
    .delta-neg {{ color: {NEGATIVE}; font-weight: 600; }}
    .delta-neutral {{ color: {SLATE}; font-weight: 600; }}

    /* Table styling */
    .stDataFrame {{ font-size: 0.85rem; }}

    /* Sidebar branding */
    [data-testid="stSidebar"] {{
        padding-top: 0;
    }}
    [data-testid="stSidebar"] [data-testid="stImage"] {{
        display: flex;
        justify-content: center;
        padding-top: 1rem;
    }}
    .sidebar-brand {{
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }}
    .sidebar-brand-name {{
        color: {GOLD};
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 2px;
    }}
    .sidebar-brand-sub {{
        color: {SLATE};
        font-size: 0.75rem;
        margin-top: 4px;
    }}

    /* Section dividers */
    .section-header {{
        color: {GOLD};
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid {DARK_BORDER};
    }}

    /* Pitch profile tables */
    .pitch-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.85rem;
    }}
    .pitch-table th {{
        color: {SLATE};
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 6px 10px;
        border-bottom: 1px solid {DARK_BORDER};
        text-align: left;
    }}
    .pitch-table td {{
        color: {CREAM};
        padding: 8px 10px;
        border-bottom: 1px solid {DARK_BORDER}22;
        vertical-align: middle;
    }}
    .pitch-table tr:last-child td {{
        border-bottom: none;
    }}
    .pitch-table .pt-name {{
        font-weight: 600;
        white-space: nowrap;
    }}
    .pitch-table .pt-n {{
        color: {SLATE};
        font-size: 0.75rem;
    }}
    .spark-cell {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .spark-bar {{
        height: 14px;
        border-radius: 7px;
        min-width: 3px;
    }}
    .spark-val {{
        font-size: 0.82rem;
        font-weight: 500;
        min-width: 36px;
    }}

    /* Matchup edge indicator */
    .edge-dot {{
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin: 0 auto;
    }}
    .matchup-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.85rem;
    }}
    .matchup-table th {{
        color: {SLATE};
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 6px 8px;
        border-bottom: 1px solid {DARK_BORDER};
        text-align: left;
    }}
    .matchup-table td {{
        color: {CREAM};
        padding: 7px 8px;
        border-bottom: 1px solid {DARK_BORDER}22;
        vertical-align: middle;
    }}
    .matchup-table tr:last-child td {{
        border-bottom: none;
    }}
    .matchup-table .pt-name {{
        font-weight: 600;
        white-space: nowrap;
    }}
    .matchup-table .pt-n {{
        color: {SLATE};
        font-size: 0.75rem;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_projections(player_type: str) -> pd.DataFrame:
    """Load pre-computed projection parquet."""
    path = DASHBOARD_DIR / f"{player_type}_projections.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_k_samples() -> dict[str, np.ndarray]:
    """Load pitcher K% posterior samples."""
    path = DASHBOARD_DIR / "pitcher_k_samples.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


@st.cache_data
def load_bf_priors() -> pd.DataFrame:
    """Load BF priors."""
    path = DASHBOARD_DIR / "bf_priors.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_pitcher_arsenal() -> pd.DataFrame:
    """Load pitcher arsenal profiles."""
    path = DASHBOARD_DIR / "pitcher_arsenal.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_vulnerability(career: bool = False) -> pd.DataFrame:
    """Load hitter vulnerability profiles (career-aggregated or single season)."""
    if career:
        path = DASHBOARD_DIR / "hitter_vuln_career.parquet"
        if path.exists():
            return pd.read_parquet(path)
    # Fallback to single-season
    path = DASHBOARD_DIR / "hitter_vuln.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_strength(career: bool = False) -> pd.DataFrame:
    """Load hitter strength profiles (career-aggregated or single season)."""
    if career:
        path = DASHBOARD_DIR / "hitter_str_career.parquet"
        if path.exists():
            return pd.read_parquet(path)
    # Fallback to single-season
    path = DASHBOARD_DIR / "hitter_str.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _check_data_exists() -> bool:
    """Check if pre-computed data exists."""
    required = [
        DASHBOARD_DIR / "hitter_projections.parquet",
        DASHBOARD_DIR / "pitcher_projections.parquet",
    ]
    return all(p.exists() for p in required)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_pct(val: float, decimals: int = 1) -> str:
    """Format a 0-1 rate as percentage string."""
    return f"{val * 100:.{decimals}f}%"


def _fmt_xwoba(val: float) -> str:
    """Format xwOBA as .XXX."""
    return f".{val * 1000:.0f}"


def _fmt_stat(val: float, key: str, decimals: int = 1) -> str:
    """Format a stat value based on its key."""
    if key == "xwoba":
        return _fmt_xwoba(val)
    return _fmt_pct(val, decimals)


def _delta_html(val: float, higher_is_better: bool = True) -> str:
    """Format a delta as colored HTML span."""
    pct = val * 100
    improving = (pct > 0 and higher_is_better) or (pct < 0 and not higher_is_better)
    if abs(pct) < 0.05:
        return f'<span class="delta-neutral">0.0pp</span>'
    elif improving:
        return f'<span class="delta-pos">{pct:+.1f}pp</span>'
    else:
        return f'<span class="delta-neg">{pct:+.1f}pp</span>'


def _metric_card(label: str, value: str, delta_html: str = "") -> str:
    """Render a styled metric card with optional delta."""
    delta_div = f'<div class="metric-delta">{delta_html}</div>' if delta_html else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_div}
    </div>
    """


# ---------------------------------------------------------------------------
# Percentile rank helpers
# ---------------------------------------------------------------------------
def _percentile_rank(
    series: pd.Series, value: float, higher_is_better: bool,
) -> float:
    """Compute percentile rank (0-100) of value within series."""
    valid = series.dropna()
    if len(valid) == 0:
        return 50.0
    if higher_is_better:
        return float((valid < value).sum() / len(valid) * 100)
    else:
        return float((valid > value).sum() / len(valid) * 100)


def _pctile_color(pctile: float) -> str:
    """Color for percentile bar fill."""
    if pctile >= 80:
        return SAGE
    elif pctile >= 60:
        return GOLD
    elif pctile >= 40:
        return SLATE
    elif pctile >= 20:
        return EMBER
    else:
        return NEGATIVE


def _pctile_bar_html(
    label: str,
    pctile: float,
    ci_lo: float,
    ci_hi: float,
    key: str,
    pctile_prev: float | None = None,
) -> str:
    """Render a percentile bar row with optional 2025 dashed reference line."""
    color = _pctile_color(pctile)
    ci_str = f"{_fmt_stat(ci_lo, key)} - {_fmt_stat(ci_hi, key)}"

    # Dashed line for previous-season percentile
    prev_line = ""
    prev_label = ""
    if pctile_prev is not None:
        # Direction arrow
        diff = pctile - pctile_prev
        if abs(diff) >= 1:
            arrow = "&#9650;" if diff > 0 else "&#9660;"
            arrow_color = SAGE if diff > 0 else EMBER
        else:
            arrow = ""
            arrow_color = SLATE
        prev_line = (
            f'<div style="position:absolute; left:{pctile_prev:.0f}%; top:0; '
            f'height:100%; width:2px; border-left:2px dashed {SLATE}; opacity:0.7; z-index:2;"></div>'
        )
        prev_label = (
            f'<span style="color:{arrow_color}; font-size:0.75rem; margin-left:8px;">'
            f'{arrow} 2025: {pctile_prev:.0f}th</span>'
        )

    return (
        f'<div style="margin:12px 0;">'
        f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">'
        f'<span style="color:{SLATE}; font-size:0.85rem; font-weight:600;">{label}{prev_label}</span>'
        f'<span style="color:{SLATE}; font-size:0.8rem;">{pctile:.0f}th percentile | Range: {ci_str}</span>'
        f'</div>'
        f'<div style="position:relative; width:100%; background:{DARK}; border-radius:6px; height:22px; '
        f'overflow:hidden; border:1px solid {DARK_BORDER};">'
        f'<div style="height:100%; width:{pctile:.0f}%; background:{color}; border-radius:5px;"></div>'
        f'{prev_line}'
        f'</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Plain English insight generator
# ---------------------------------------------------------------------------
_STAT_NAMES = {
    "k_rate": "strikeout rate",
    "bb_rate": "walk rate",
    "hr_rate": "home run rate",
    "hr_per_bf": "home run rate",
    "xwoba": "expected wOBA",
}

_GOOD_DIRECTION_LABEL = {
    ("k_rate", True): "miss more bats",          # pitcher: higher K% good
    ("k_rate", False): "make more contact",       # hitter: lower K% good
    ("bb_rate", True): "draw more walks",          # hitter: higher BB% good
    ("bb_rate", False): "improve control",         # pitcher: lower BB% good
    ("hr_rate", True): "hit for more power",       # hitter: higher HR good
    ("hr_per_bf", False): "keep the ball in the park",  # pitcher: lower HR good
    ("xwoba", True): "produce better at the plate",     # hitter: higher xwOBA good
}


def _generate_scouting_bullets(
    stat_configs: list[tuple[str, str, bool, str]],
    player_row: pd.Series,
    all_df: pd.DataFrame,
    player_type: str,
) -> list[tuple[str, str]]:
    """Generate plain English scouting report bullets.

    Returns list of (color_hex, text) tuples.
    """
    bullets: list[tuple[str, str]] = []

    for label, key, higher_better, _desc in stat_configs:
        obs_col = f"observed_{key}"
        proj_col = f"projected_{key}"
        ci_lo_col = f"projected_{key}_2_5"
        ci_hi_col = f"projected_{key}_97_5"

        if obs_col not in player_row.index or pd.isna(player_row.get(obs_col)):
            continue

        observed = player_row[obs_col]
        projected = player_row[proj_col]
        delta_pp = (projected - observed) * 100

        ci_lo = player_row.get(ci_lo_col, projected)
        ci_hi = player_row.get(ci_hi_col, projected)
        ci_width = (ci_hi - ci_lo) * 100

        pctile = _percentile_rank(all_df[proj_col], projected, higher_better)

        # Determine if this is a good or bad move
        improving = (delta_pp > 0 and higher_better) or (delta_pp < 0 and not higher_better)

        # Direction sentence
        stat_name = _STAT_NAMES.get(key, label)
        obs_str = _fmt_stat(observed, key)
        proj_str = _fmt_stat(projected, key)

        if abs(delta_pp) < 0.5:
            direction_text = (
                f"{label} projected to hold steady at {proj_str}"
            )
            dot_color = SLATE
        elif improving:
            good_label = _GOOD_DIRECTION_LABEL.get((key, higher_better), "improve")
            if abs(delta_pp) > 3:
                direction_text = (
                    f"{label} jumps from {obs_str} to {proj_str} "
                    f"({delta_pp:+.1f}pp) -- expect him to {good_label}"
                )
            else:
                direction_text = (
                    f"{label} ticks from {obs_str} to {proj_str} "
                    f"({delta_pp:+.1f}pp) -- slight improvement"
                )
            dot_color = POSITIVE
        else:
            if abs(delta_pp) > 3:
                direction_text = (
                    f"{label} projected to slide from {obs_str} to {proj_str} "
                    f"({delta_pp:+.1f}pp) -- notable regression risk"
                )
            else:
                direction_text = (
                    f"{label} may slip from {obs_str} to {proj_str} "
                    f"({delta_pp:+.1f}pp) -- minor adjustment"
                )
            dot_color = NEGATIVE

        # Confidence + context
        if ci_width < 6:
            conf_text = "high confidence"
        elif ci_width < 12:
            conf_text = "moderate confidence"
        else:
            conf_text = "wide range of outcomes"

        # Percentile context
        if pctile >= 90:
            rank_text = f"elite ({pctile:.0f}th percentile)"
        elif pctile >= 75:
            rank_text = f"above-average ({pctile:.0f}th pctile)"
        elif pctile >= 40:
            rank_text = f"mid-tier ({pctile:.0f}th pctile)"
        else:
            rank_text = f"below-average ({pctile:.0f}th pctile)"

        full_text = f"{direction_text}. {conf_text.capitalize()}, {rank_text}."
        bullets.append((dot_color, full_text))

    return bullets


# ---------------------------------------------------------------------------
# Matplotlib helpers (dark-themed for dashboard)
# ---------------------------------------------------------------------------
def _apply_dark_mpl() -> None:
    """Apply dark-themed matplotlib settings for dashboard charts."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": DARK,
        "axes.facecolor": DARK,
        "savefig.facecolor": DARK,
        "text.color": CREAM,
        "axes.labelcolor": SLATE,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "font.size": 12,
        "font.family": "sans-serif",
    })


_apply_dark_mpl()


def _create_posterior_fig(
    samples: np.ndarray,
    observed: float | None = None,
    stat_label: str = "K%",
    color: str = SAGE,
) -> plt.Figure:
    """Create a posterior KDE plot with brand styling."""
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    pct_samples = samples * 100
    kde = gaussian_kde(pct_samples, bw_method=0.3)
    x = np.linspace(pct_samples.min() - 2, pct_samples.max() + 2, 300)
    y = kde(x)

    ax.fill_between(x, y, alpha=0.3, color=color)
    ax.plot(x, y, color=color, linewidth=2)

    # 95% credible interval shading
    ci_lo, ci_hi = np.percentile(pct_samples, [2.5, 97.5])
    ci_mask = (x >= ci_lo) & (x <= ci_hi)
    ax.fill_between(x[ci_mask], y[ci_mask], alpha=0.15, color=color)

    # Mean line
    mean_val = np.mean(pct_samples)
    ax.axvline(mean_val, color=GOLD, linewidth=2, linestyle="--", alpha=0.9)
    ax.text(
        mean_val, ax.get_ylim()[1] * 0.92,
        f" {mean_val:.1f}%",
        color=GOLD, fontsize=11, fontweight="bold", va="top",
    )

    # Observed line
    if observed is not None:
        obs_pct = observed * 100
        ax.axvline(obs_pct, color=SLATE, linewidth=1.5, linestyle=":", alpha=0.8)
        ax.text(
            obs_pct, ax.get_ylim()[1] * 0.75,
            f" {obs_pct:.1f}% (2025)",
            color=SLATE, fontsize=9, va="top",
        )

    # CI annotation
    ax.text(
        0.98, 0.95,
        f"95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%]",
        transform=ax.transAxes,
        color=SLATE, fontsize=9, ha="right", va="top",
    )

    ax.set_xlabel(stat_label, color=SLATE, fontsize=10)
    ax.set_ylabel("", color=SLATE, fontsize=10)
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])

    add_watermark(fig)
    fig.tight_layout()
    return fig


def _create_game_k_fig(
    k_samples: np.ndarray,
    pitcher_name: str,
) -> plt.Figure:
    """Create a game K distribution histogram."""
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    max_k = int(k_samples.max()) + 1
    bins = np.arange(-0.5, max_k + 1.5, 1)
    counts, _, bars = ax.hist(
        k_samples, bins=bins, density=True,
        color=SAGE, alpha=0.7, edgecolor=DARK, linewidth=0.5,
    )

    # Color the mode bar gold
    mode_k = int(np.median(k_samples))
    for bar in bars:
        if abs(bar.get_x() + 0.5 - mode_k) < 0.5:
            bar.set_facecolor(GOLD)
            bar.set_alpha(0.9)

    mean_k = np.mean(k_samples)
    ax.axvline(mean_k, color=GOLD, linewidth=2, linestyle="--", alpha=0.9)
    ax.text(
        mean_k + 0.3, ax.get_ylim()[1] * 0.9,
        f"E[K] = {mean_k:.1f}",
        color=GOLD, fontsize=11, fontweight="bold", va="top",
    )

    ax.set_xlabel("Strikeouts", color=SLATE, fontsize=11)
    ax.set_ylabel("Probability", color=SLATE, fontsize=10)
    ax.set_title(
        f"{pitcher_name} -- Projected K Distribution (2026)",
        color=CREAM, fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_watermark(fig)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pitch type display config
# ---------------------------------------------------------------------------
PITCH_DISPLAY: dict[str, str] = {
    "FF": "4-Seam", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "CU": "Curveball", "ST": "Sweeper",
    "CH": "Changeup", "FS": "Splitter", "KC": "Knuckle-Curve",
    "SV": "Slurve", "CS": "Slow Curve", "FO": "Forkball",
    "EP": "Eephus", "KN": "Knuckleball",
}
# Fixed display order: fastballs → breaking → offspeed
PITCH_ORDER: list[str] = [
    "FF", "SI", "FC",
    "SL", "ST", "CU", "KC", "SV", "CS",
    "CH", "FS", "FO",
    "EP", "KN",
]
PITCH_FAMILY_COLORS: dict[str, str] = {
    "fastball": "#E8575A",   # warm red
    "breaking": "#5B9BD5",   # cool blue
    "offspeed": "#70AD47",   # green
}
PITCH_TYPE_TO_FAMILY: dict[str, str] = {
    "FF": "fastball", "SI": "fastball", "FC": "fastball",
    "SL": "breaking", "CU": "breaking", "ST": "breaking",
    "KC": "breaking", "SV": "breaking", "CS": "breaking",
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed",
    "EP": "offspeed", "KN": "offspeed",
}


def _whiff_quality_color(whiff_rate: float) -> str:
    """Color-code a whiff rate: green=elite, gold=avg, red=poor."""
    if whiff_rate >= 0.35:
        return SAGE
    elif whiff_rate >= 0.25:
        return GOLD
    elif whiff_rate >= 0.15:
        return SLATE
    else:
        return EMBER


def _xwoba_quality_color(xwoba: float) -> str:
    """Color-code xwOBA against: green=suppresses contact, red=gets hit."""
    if xwoba <= 0.280:
        return SAGE
    elif xwoba <= 0.340:
        return GOLD
    elif xwoba <= 0.400:
        return SLATE
    else:
        return EMBER


# ---------------------------------------------------------------------------
# Pitch profile stat tables (sparkbar style)
# ---------------------------------------------------------------------------
def _spark_color_rate(value: float, league_avg: float, higher_is_worse: bool) -> str:
    """Color a rate stat relative to league average.

    For hitter vulnerabilities (higher_is_worse=True): red if above avg, green if below.
    For pitcher skills (higher_is_worse=False): green if above avg, red if below.
    """
    ratio = value / league_avg if league_avg > 0 else 1.0
    if higher_is_worse:
        if ratio >= 1.25:
            return EMBER
        elif ratio >= 1.05:
            return GOLD
        elif ratio >= 0.85:
            return SLATE
        else:
            return SAGE
    else:
        if ratio >= 1.25:
            return SAGE
        elif ratio >= 1.05:
            return GOLD
        elif ratio >= 0.85:
            return SLATE
        else:
            return EMBER


def _spark_color_xwoba(value: float, for_pitcher: bool = False) -> str:
    """Color xwOBA — green = dangerous for hitters, green = suppresses for pitchers."""
    if for_pitcher:
        # Lower is better for pitchers
        if value <= 0.280:
            return SAGE
        elif value <= 0.340:
            return GOLD
        elif value <= 0.400:
            return SLATE
        else:
            return EMBER
    else:
        # Higher is better for hitters (more damage)
        if value >= 0.420:
            return SAGE
        elif value >= 0.350:
            return GOLD
        elif value >= 0.300:
            return SLATE
        else:
            return EMBER


def _spark_html(value: float, max_val: float, color: str,
                alpha: float = 0.85) -> str:
    """Render an inline pill-shaped sparkbar."""
    width_pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    return (
        f'<div class="spark-bar" style="width:{width_pct:.0f}%; '
        f'background:{color}; opacity:{alpha:.2f};"></div>'
    )


def _combine_platoon_vuln(vuln_df: pd.DataFrame) -> pd.DataFrame:
    """Combine L/R platoon splits into a single row per pitch type.

    Sums raw counts and recomputes rates.  xwOBA is BIP-weighted.
    """
    sum_cols = [
        "pitches", "swings", "whiffs", "out_of_zone_pitches",
        "chase_swings", "called_strikes", "csw", "bip",
        "hard_hits", "barrels_proxy",
    ]
    agg = {c: "sum" for c in sum_cols if c in vuln_df.columns}
    combined = vuln_df.groupby("pitch_type").agg(agg).reset_index()
    combined["whiff_rate"] = combined["whiffs"] / combined["swings"].replace(0, np.nan)
    combined["chase_rate"] = combined["chase_swings"] / combined["out_of_zone_pitches"].replace(0, np.nan)
    combined["csw_pct"] = combined["csw"] / combined["pitches"].replace(0, np.nan)
    if "pitch_family" in vuln_df.columns:
        fam_map = vuln_df.drop_duplicates("pitch_type").set_index("pitch_type")["pitch_family"].to_dict()
        combined["pitch_family"] = combined["pitch_type"].map(fam_map)
    if "xwoba_contact" in vuln_df.columns:
        xw = vuln_df[vuln_df["xwoba_contact"].notna() & (vuln_df["bip"] > 0)]
        if not xw.empty:
            xw_agg = xw.groupby("pitch_type").apply(
                lambda g: (g["xwoba_contact"] * g["bip"]).sum() / g["bip"].sum()
                if g["bip"].sum() > 0 else np.nan,
                include_groups=False,
            ).reset_index(name="xwoba_contact")
            combined = combined.merge(xw_agg, on="pitch_type", how="left")
        else:
            combined["xwoba_contact"] = np.nan
    # Keep batter_id for downstream filtering
    combined["batter_id"] = vuln_df["batter_id"].iloc[0]
    return combined


def _build_hitter_profile_table(vuln_df: pd.DataFrame) -> str:
    """Build HTML stat table for a hitter's pitch-type profile.

    Columns: Pitch | Whiff% | CStr% | Chase% | xwOBA | Pitches
    """
    from src.utils.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    df = vuln_df.copy()
    df = df[df["pitches"] >= 15]  # Minimum sample
    if df.empty:
        return ""

    # Compute rates from raw counts
    df["called_str_rate"] = df["called_strikes"] / df["pitches"].replace(0, np.nan)
    df["whiff_rate_raw"] = df["whiffs"] / df["swings"].replace(0, np.nan)
    df["chase_rate_raw"] = df["chase_swings"] / df["out_of_zone_pitches"].replace(0, np.nan)

    # Fixed order: fastballs → breaking → offspeed
    df["_order"] = df["pitch_type"].map({pt: i for i, pt in enumerate(PITCH_ORDER)}).fillna(99)
    df = df.sort_values("_order")

    # Max values for sparkbar scaling
    max_whiff = max(df["whiff_rate_raw"].dropna().max(), 0.01)
    max_cstr = max(df["called_str_rate"].dropna().max(), 0.01)
    max_chase = max(df["chase_rate_raw"].dropna().max(), 0.01)
    max_xwoba = max(df["xwoba_contact"].dropna().max(), 0.01) if "xwoba_contact" in df.columns else 0.5

    rows_html = ""
    for _, row in df.iterrows():
        pt = row["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        family = PITCH_TYPE_TO_FAMILY.get(pt, "offspeed")
        family_color = PITCH_FAMILY_COLORS.get(family, SLATE)
        n_pitches = int(row["pitches"])

        # League averages for this pitch type
        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)
        lg_chase = lg.get("chase_rate", 0.30)

        # Reliability alpha (fade small samples)
        swings = row.get("swings", 0) or 0
        alpha = min(1.0, 0.45 + 0.55 * (min(swings, 80) / 80))

        # Whiff%
        whiff = row.get("whiff_rate_raw", np.nan)
        if pd.notna(whiff):
            w_color = _spark_color_rate(whiff, lg_whiff, higher_is_worse=True)
            whiff_cell = (
                f'<div class="spark-cell">'
                f'{_spark_html(whiff, max_whiff, w_color, alpha)}'
                f'<span class="spark-val" style="color:{w_color};">{whiff*100:.0f}%</span>'
                f'</div>'
            )
        else:
            whiff_cell = f'<span style="color:{SLATE};">--</span>'

        # Called Strike%
        cstr = row.get("called_str_rate", np.nan)
        if pd.notna(cstr):
            # Higher called strike rate = pitcher is freezing the hitter = bad for hitter
            c_color = _spark_color_rate(cstr, 0.14, higher_is_worse=True)
            cstr_cell = (
                f'<div class="spark-cell">'
                f'{_spark_html(cstr, max_cstr, c_color, alpha)}'
                f'<span class="spark-val" style="color:{c_color};">{cstr*100:.0f}%</span>'
                f'</div>'
            )
        else:
            cstr_cell = f'<span style="color:{SLATE};">--</span>'

        # Chase%
        chase = row.get("chase_rate_raw", np.nan)
        if pd.notna(chase):
            ch_color = _spark_color_rate(chase, lg_chase, higher_is_worse=True)
            chase_cell = (
                f'<div class="spark-cell">'
                f'{_spark_html(chase, max_chase, ch_color, alpha)}'
                f'<span class="spark-val" style="color:{ch_color};">{chase*100:.0f}%</span>'
                f'</div>'
            )
        else:
            chase_cell = f'<span style="color:{SLATE};">--</span>'

        # xwOBA on contact
        xwoba = row.get("xwoba_contact", np.nan)
        if pd.notna(xwoba) and xwoba > 0 and xwoba < 2.0:
            x_color = _spark_color_xwoba(xwoba, for_pitcher=False)
            xwoba_cell = (
                f'<div class="spark-cell">'
                f'{_spark_html(xwoba, max_xwoba, x_color, alpha)}'
                f'<span class="spark-val" style="color:{x_color};">.{int(xwoba*1000):03d}</span>'
                f'</div>'
            )
        else:
            xwoba_cell = f'<span style="color:{SLATE};">--</span>'

        rows_html += (
            f'<tr>'
            f'<td><span class="pt-name" style="color:{family_color};">{pt_name}</span></td>'
            f'<td>{whiff_cell}</td>'
            f'<td>{cstr_cell}</td>'
            f'<td>{chase_cell}</td>'
            f'<td>{xwoba_cell}</td>'
            f'<td class="pt-n">{n_pitches:,}</td>'
            f'</tr>'
        )

    return (
        f'<table class="pitch-table">'
        f'<thead><tr>'
        f'<th>Pitch</th><th>Whiff%</th><th>CStr%</th>'
        f'<th>Chase%</th><th>xwOBA</th><th>Pitches</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )


def _build_pitcher_profile_table(arsenal_df: pd.DataFrame) -> str:
    """Build HTML stat table for a pitcher's arsenal profile.

    Columns: Pitch | Whiff% | CSW% | xwOBA Ag | Velo | Usage
    """
    from src.utils.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    df = arsenal_df.copy()
    df = df[df["pitches"] >= 20]
    if df.empty:
        return ""

    # CSW% = (whiffs + called_strikes-equivalent) — approximate from whiff_rate * swings + csw
    # Arsenal df has whiff_rate, usage, velo, xwoba_against
    # Need to compute CSW if not present
    if "csw_pct" not in df.columns:
        # Approximate: csw_pct ≈ whiff_rate * (swings/pitches) + called_strike_rate
        # But we may not have called_strikes, so derive from what we have
        if "swings" in df.columns and "whiffs" in df.columns:
            df["csw_pct"] = (df["whiffs"] + df.get("called_strikes", 0)) / df["pitches"].replace(0, np.nan)
        else:
            df["csw_pct"] = np.nan

    # Fixed order: fastballs → breaking → offspeed
    df["_order"] = df["pitch_type"].map({pt: i for i, pt in enumerate(PITCH_ORDER)}).fillna(99)
    df = df.sort_values("_order")

    # Max values for sparkbar scaling
    max_whiff = max(df["whiff_rate"].dropna().max(), 0.01)
    max_csw = max(df["csw_pct"].dropna().max(), 0.01) if "csw_pct" in df.columns else 0.40
    max_xwoba = 0.500  # Fixed scale for pitcher xwOBA against

    rows_html = ""
    for _, row in df.iterrows():
        pt = row["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        family = PITCH_TYPE_TO_FAMILY.get(pt, "offspeed")
        family_color = PITCH_FAMILY_COLORS.get(family, SLATE)

        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)
        lg_csw = lg.get("csw_pct", 0.29)

        usage = row.get("usage_pct", 0)
        velo = row.get("avg_velo", np.nan)

        # Whiff%
        whiff = row.get("whiff_rate", np.nan)
        if pd.notna(whiff):
            w_color = _spark_color_rate(whiff, lg_whiff, higher_is_worse=False)
            whiff_cell = (
                f'<div class="spark-cell">'
                f'{_spark_html(whiff, max_whiff, w_color)}'
                f'<span class="spark-val" style="color:{w_color};">{whiff*100:.0f}%</span>'
                f'</div>'
            )
        else:
            whiff_cell = f'<span style="color:{SLATE};">--</span>'

        # CSW%
        csw = row.get("csw_pct", np.nan)
        if pd.notna(csw):
            csw_color = _spark_color_rate(csw, lg_csw, higher_is_worse=False)
            csw_cell = (
                f'<div class="spark-cell">'
                f'{_spark_html(csw, max_csw, csw_color)}'
                f'<span class="spark-val" style="color:{csw_color};">{csw*100:.0f}%</span>'
                f'</div>'
            )
        else:
            csw_cell = f'<span style="color:{SLATE};">--</span>'

        # xwOBA against
        xwoba = row.get("xwoba_against", np.nan)
        if pd.notna(xwoba):
            x_color = _spark_color_xwoba(xwoba, for_pitcher=True)
            xwoba_cell = (
                f'<div class="spark-cell">'
                f'{_spark_html(xwoba, max_xwoba, x_color)}'
                f'<span class="spark-val" style="color:{x_color};">.{int(xwoba*1000):03d}</span>'
                f'</div>'
            )
        else:
            xwoba_cell = f'<span style="color:{SLATE};">--</span>'

        # Velo + Usage annotations
        velo_str = f'{velo:.1f}' if pd.notna(velo) else '--'
        usage_str = f'{usage*100:.0f}%'

        rows_html += (
            f'<tr>'
            f'<td><span class="pt-name" style="color:{family_color};">{pt_name}</span></td>'
            f'<td>{whiff_cell}</td>'
            f'<td>{csw_cell}</td>'
            f'<td>{xwoba_cell}</td>'
            f'<td style="color:{CREAM}; font-size:0.82rem; font-weight:600;">{velo_str}</td>'
            f'<td style="color:{CREAM}; font-size:0.82rem; font-weight:600;">{usage_str}</td>'
            f'</tr>'
        )

    return (
        f'<table class="pitch-table">'
        f'<thead><tr>'
        f'<th>Pitch</th><th>Whiff%</th><th>CSW%</th>'
        f'<th>xwOBA Ag</th><th>Velo</th><th>Usage</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )


def _build_matchup_table(
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    str_df: pd.DataFrame,
) -> str:
    """Build combined pitcher-vs-hitter matchup table with sparkbars.

    Columns: Pitch | Usage | P Whiff% | H Whiff% | H Chase% | H xwOBA | Edge
    """
    from src.utils.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    p_df = arsenal_df.copy()
    p_df = p_df[p_df["pitches"] >= 20]
    if p_df.empty:
        return ""

    # Filter vuln/str to meaningful samples and deduplicate by keeping
    # the row with the most pitches per pitch_type (career parquets may
    # have duplicate rows from multi-season aggregation).
    v_df = vuln_df.copy()
    v_df = v_df[v_df["pitches"] >= 15]
    if not v_df.empty:
        v_df = v_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first"
        )
    s_df = str_df.copy() if not str_df.empty else pd.DataFrame()
    if not s_df.empty and "pitches" in s_df.columns:
        s_df = s_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first"
        )

    # Fixed pitch order
    p_df["_order"] = p_df["pitch_type"].map(
        {pt: i for i, pt in enumerate(PITCH_ORDER)}
    ).fillna(99)
    p_df = p_df.sort_values("_order")

    # Precompute max values for sparkbar scaling
    max_p_whiff = max(p_df["whiff_rate"].dropna().max(), 0.01)
    h_whiffs = []
    h_chases = []
    h_xwobas = []
    h_hard_hit_rates = []
    for _, row in p_df.iterrows():
        pt = row["pitch_type"]
        h_row = v_df[v_df["pitch_type"] == pt]
        if len(h_row) > 0:
            sw = h_row["swings"].iloc[0] if "swings" in h_row.columns else 0
            wh = h_row["whiffs"].iloc[0] if "whiffs" in h_row.columns else 0
            whiff_r = wh / sw if pd.notna(sw) and sw > 0 else np.nan
            h_whiffs.append(whiff_r)
            if "chase_swings" in h_row.columns and "out_of_zone_pitches" in h_row.columns:
                cs = h_row["chase_swings"].iloc[0]
                oz = h_row["out_of_zone_pitches"].iloc[0]
                h_chases.append(cs / oz if pd.notna(oz) and oz > 0 else np.nan)
            else:
                h_chases.append(np.nan)
        else:
            h_whiffs.append(np.nan)
            h_chases.append(np.nan)
        s_row = s_df[s_df["pitch_type"] == pt] if not s_df.empty else pd.DataFrame()
        if len(s_row) > 0 and "xwoba_contact" in s_row.columns:
            h_xwobas.append(s_row["xwoba_contact"].iloc[0])
        elif len(h_row) > 0 and "xwoba_contact" in h_row.columns:
            h_xwobas.append(h_row["xwoba_contact"].iloc[0])
        else:
            h_xwobas.append(np.nan)
        if len(s_row) > 0 and "hard_hit_rate" in s_row.columns:
            h_hard_hit_rates.append(s_row["hard_hit_rate"].iloc[0])
        else:
            h_hard_hit_rates.append(np.nan)

    max_h_whiff = max(pd.Series(h_whiffs).dropna().max(), 0.01) if any(pd.notna(v) for v in h_whiffs) else 0.40
    max_h_chase = max(pd.Series(h_chases).dropna().max(), 0.01) if any(pd.notna(v) for v in h_chases) else 0.40
    max_h_xwoba = max(pd.Series(h_xwobas).dropna().max(), 0.01) if any(pd.notna(v) for v in h_xwobas) else 0.50

    rows_html = ""
    for idx, (_, row) in enumerate(p_df.iterrows()):
        pt = row["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        family = PITCH_TYPE_TO_FAMILY.get(pt, "offspeed")
        family_color = PITCH_FAMILY_COLORS.get(family, SLATE)

        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)
        lg_chase = lg.get("chase_rate", 0.30)

        usage = row.get("usage_pct", 0)
        usage_str = f'{usage * 100:.0f}%'

        # Pitcher whiff%
        p_whiff = row.get("whiff_rate", np.nan)
        if pd.notna(p_whiff):
            pw_color = _spark_color_rate(p_whiff, lg_whiff, higher_is_worse=False)
            pw_cell = (
                f'<div class="spark-cell">'
                f'{_spark_html(p_whiff, max_p_whiff, pw_color)}'
                f'<span class="spark-val" style="color:{pw_color};">{p_whiff*100:.0f}%</span>'
                f'</div>'
            )
        else:
            pw_cell = f'<span style="color:{SLATE};">--</span>'

        # Hitter whiff%
        h_whiff = h_whiffs[idx]
        if pd.notna(h_whiff):
            hw_color = _spark_color_rate(h_whiff, lg_whiff, higher_is_worse=True)
            hw_cell = (
                f'<div class="spark-cell">'
                f'{_spark_html(h_whiff, max_h_whiff, hw_color)}'
                f'<span class="spark-val" style="color:{hw_color};">{h_whiff*100:.0f}%</span>'
                f'</div>'
            )
        else:
            hw_cell = f'<span style="color:{SLATE};">--</span>'

        # Hitter chase%
        h_chase = h_chases[idx]
        if pd.notna(h_chase):
            hc_color = _spark_color_rate(h_chase, lg_chase, higher_is_worse=True)
            hc_cell = (
                f'<div class="spark-cell">'
                f'{_spark_html(h_chase, max_h_chase, hc_color)}'
                f'<span class="spark-val" style="color:{hc_color};">{h_chase*100:.0f}%</span>'
                f'</div>'
            )
        else:
            hc_cell = f'<span style="color:{SLATE};">--</span>'

        # Hitter xwOBA on contact
        h_xwoba = h_xwobas[idx]
        if pd.notna(h_xwoba) and 0 < h_xwoba < 2.0:
            hx_color = _spark_color_xwoba(h_xwoba, for_pitcher=False)
            hx_cell = (
                f'<div class="spark-cell">'
                f'{_spark_html(h_xwoba, max_h_xwoba, hx_color)}'
                f'<span class="spark-val" style="color:{hx_color};">.{int(h_xwoba*1000):03d}</span>'
                f'</div>'
            )
        else:
            hx_cell = f'<span style="color:{SLATE};">--</span>'

        # Edge indicator: balanced pitcher skill vs hitter damage
        lg_xwoba = lg.get("xwoba_contact", 0.320)
        lg_hh = lg.get("hard_hit_rate", 0.33)
        h_hh = h_hard_hit_rates[idx]

        edge_score = 0.0
        # Pitcher skill: how much better is this pitch at generating whiffs?
        if pd.notna(p_whiff):
            edge_score += (p_whiff - lg_whiff) * 2.0
        # Hitter vulnerability: whiff + chase tendencies on this pitch type
        if pd.notna(h_whiff):
            edge_score += (h_whiff - lg_whiff) * 1.5
        if pd.notna(h_chase):
            edge_score += (h_chase - lg_chase) * 1.0
        # Hitter damage: xwOBA and hard-hit rate vs per-pitch-type baselines
        if pd.notna(h_xwoba):
            edge_score -= (h_xwoba - lg_xwoba) * 4.0
        if pd.notna(h_hh):
            edge_score -= (h_hh - lg_hh) * 2.0

        if edge_score > 0.10:
            edge_color = SAGE
        elif edge_score < -0.10:
            edge_color = EMBER
        else:
            edge_color = SLATE
        edge_cell = f'<span class="edge-dot" style="background:{edge_color};"></span>'

        rows_html += (
            f'<tr>'
            f'<td><span class="pt-name" style="color:{family_color};">{pt_name}</span></td>'
            f'<td style="color:{SLATE}; font-size:0.82rem;">{usage_str}</td>'
            f'<td>{pw_cell}</td>'
            f'<td>{hw_cell}</td>'
            f'<td>{hc_cell}</td>'
            f'<td>{hx_cell}</td>'
            f'<td style="text-align:center;">{edge_cell}</td>'
            f'</tr>'
        )

    return (
        f'<table class="matchup-table">'
        f'<thead><tr>'
        f'<th>Pitch</th><th>Usage</th><th>P Whiff%</th>'
        f'<th>H Whiff%</th><th>H Chase%</th><th>H xwOBA</th><th>Edge</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )


def _build_matchup_scouting_bullets(
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    str_df: pd.DataFrame,
    pitcher_name: str,
    hitter_name: str,
) -> list[tuple[str, str]]:
    """Generate per-pitch scouting bullets for the matchup.

    Returns list of (color, text) tuples.
    """
    from src.utils.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    p_df = arsenal_df.copy()
    p_df = p_df[p_df["pitches"] >= 20]
    if p_df.empty:
        return []

    # Deduplicate vuln/str — keep row with most pitches per pitch_type
    v_df = vuln_df.copy()
    v_df = v_df[v_df["pitches"] >= 15] if "pitches" in v_df.columns else v_df
    if not v_df.empty:
        v_df = v_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first"
        )
    s_df = str_df.copy() if not str_df.empty else pd.DataFrame()
    if not s_df.empty and "pitches" in s_df.columns:
        s_df = s_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first"
        )

    bullets: list[tuple[str, str]] = []

    # Gather per-pitch edges
    pitch_edges: list[tuple[str, float, str]] = []  # (pitch_name, edge_score, detail)

    for _, row in p_df.iterrows():
        pt = row["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)

        p_whiff = row.get("whiff_rate", np.nan)
        usage = row.get("usage_pct", 0)

        h_row = v_df[v_df["pitch_type"] == pt]
        h_whiff = np.nan
        if len(h_row) > 0 and "swings" in h_row.columns and "whiffs" in h_row.columns:
            sw = h_row["swings"].iloc[0]
            wh = h_row["whiffs"].iloc[0]
            h_whiff = wh / sw if pd.notna(sw) and sw > 0 else np.nan

        s_row = s_df[s_df["pitch_type"] == pt] if not s_df.empty else pd.DataFrame()
        h_xwoba = np.nan
        if len(s_row) > 0 and "xwoba_contact" in s_row.columns:
            h_xwoba = s_row["xwoba_contact"].iloc[0]
        elif len(h_row) > 0 and "xwoba_contact" in h_row.columns:
            h_xwoba = h_row["xwoba_contact"].iloc[0]

        # Score this pitch's edge (balanced: pitcher skill vs hitter damage)
        lg_xwoba = lg.get("xwoba_contact", 0.320)
        lg_hh = lg.get("hard_hit_rate", 0.33)
        lg_chase = lg.get("chase_rate", 0.30)

        h_chase = np.nan
        if len(h_row) > 0 and "chase_swings" in h_row.columns and "out_of_zone_pitches" in h_row.columns:
            cs = h_row["chase_swings"].iloc[0]
            oz = h_row["out_of_zone_pitches"].iloc[0]
            h_chase = cs / oz if pd.notna(oz) and oz > 0 else np.nan

        h_hh = np.nan
        if len(s_row) > 0 and "hard_hit_rate" in s_row.columns:
            h_hh = s_row["hard_hit_rate"].iloc[0]

        edge = 0.0
        detail_parts = []
        if pd.notna(p_whiff):
            edge += (p_whiff - lg_whiff) * 2.0
            if p_whiff > lg_whiff * 1.1:
                detail_parts.append(f"pitcher's whiff rate {p_whiff*100:.0f}%")
        if pd.notna(h_whiff):
            edge += (h_whiff - lg_whiff) * 1.5
            if h_whiff > lg_whiff * 1.15:
                detail_parts.append(f"hitter whiffs {h_whiff*100:.0f}% of the time")
        if pd.notna(h_chase):
            edge += (h_chase - lg_chase) * 1.0
        if pd.notna(h_xwoba):
            edge -= (h_xwoba - lg_xwoba) * 4.0
            if h_xwoba >= 0.400:
                detail_parts.append(f"but hitter does damage on contact (.{int(h_xwoba*1000):03d} xwOBA)")
            elif h_xwoba <= 0.280:
                detail_parts.append(f"weak contact (.{int(h_xwoba*1000):03d} xwOBA)")
        if pd.notna(h_hh):
            edge -= (h_hh - lg_hh) * 2.0

        detail = ", ".join(detail_parts) if detail_parts else ""
        pitch_edges.append((pt_name, edge, detail, usage))

    # Sort by absolute edge to find most interesting matchups
    pitch_edges.sort(key=lambda x: abs(x[1]), reverse=True)

    for pt_name, edge, detail, usage in pitch_edges[:3]:
        if abs(edge) < 0.02:
            continue
        if edge > 0:
            color = SAGE
            direction = "Pitcher advantage"
        else:
            color = EMBER
            direction = "Hitter advantage"

        text = f"<b>{pt_name}</b> ({usage*100:.0f}% usage) — {direction}"
        if detail:
            text += f": {detail}"
        bullets.append((color, text))

    if not bullets:
        bullets.append((GOLD, "No strong pitch-level edges in this matchup"))

    return bullets


# ---------------------------------------------------------------------------
# Matplotlib chart builders (used in matchup explorer)
# ---------------------------------------------------------------------------
def _create_arsenal_fig(
    arsenal_df: pd.DataFrame,
    pitcher_name: str,
) -> plt.Figure:
    """Pitcher arsenal chart.

    Bar length = velocity (fixed 0-103 axis).
    Bar thickness = usage %.
    Bar color = whiff rate quality.
    Dot at end = xwOBA against quality.
    Ordered by usage % (highest at top).
    """
    df = arsenal_df.sort_values("usage_pct", ascending=True).copy()
    df["label"] = (
        df["pitch_type"].map(PITCH_DISPLAY).fillna(df["pitch_type"])
        + "  " + (df["usage_pct"] * 100).round(0).astype(int).astype(str) + "%"
    )

    n = len(df)
    # Bar thickness proportional to usage — min 0.25, max 0.85
    max_usage = df["usage_pct"].max()
    min_thickness, max_thickness = 0.25, 0.85
    df["thickness"] = min_thickness + (
        (df["usage_pct"] / max_usage) * (max_thickness - min_thickness)
    )

    fig, ax = plt.subplots(figsize=(7, max(2.5, n * 0.6)))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    y_positions = np.arange(n)

    for i, (_, row) in enumerate(df.iterrows()):
        velo = row.get("avg_velo", np.nan)
        whiff = row.get("whiff_rate", np.nan)
        xwoba = row.get("xwoba_against", np.nan)
        bar_len = velo if pd.notna(velo) else 0
        color = _whiff_quality_color(whiff) if pd.notna(whiff) else SLATE
        thickness = row["thickness"]

        # Draw rounded velocity bar
        if bar_len > 0:
            rsize = min(thickness * 0.7, 2.0)
            bar = FancyBboxPatch(
                (0, y_positions[i] - thickness / 2),
                bar_len, thickness,
                boxstyle=f"round,pad=0,rounding_size={rsize:.2f}",
                facecolor=color, alpha=0.85, zorder=2,
            )
            ax.add_patch(bar)

        # xwOBA dot — small, positioned inside the bar near the tip
        if pd.notna(xwoba) and bar_len > 0:
            dot_color = _xwoba_quality_color(xwoba)
            ax.scatter(
                bar_len - 3, y_positions[i],
                s=20, color=dot_color, edgecolors="none",
                zorder=3,
            )

        # Annotation: whiff% and xwOBA text to the right
        parts = []
        if pd.notna(whiff):
            parts.append(f"Whiff {whiff*100:.0f}%")
        if pd.notna(xwoba):
            parts.append(f"xwOBA .{int(xwoba*1000):03d}")
        if parts:
            ax.text(
                104, y_positions[i], "  " + " | ".join(parts),
                color=CREAM, fontsize=8.5, va="center", ha="left",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["label"])
    ax.set_xlim(0, 103)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Velocity (mph)", color=SLATE, fontsize=10)
    ax.set_title(
        f"{pitcher_name} -- Pitch Arsenal (2025)",
        color=CREAM, fontsize=12, fontweight="bold", pad=10,
    )
    ax.tick_params(axis="y", colors=CREAM, labelsize=10)
    ax.tick_params(axis="x", colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_watermark(fig)
    fig.tight_layout()
    # Make room for annotations after tight_layout
    fig.subplots_adjust(right=0.60)
    return fig


def _blend_whiff_rate(row: pd.Series) -> float:
    """Blend raw whiff rate toward league baseline by sample reliability.

    reliability = min(swings, 50) / 50
    blended = reliability * raw + (1 - reliability) * league_avg
    """
    from src.utils.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    raw = row.get("whiff_rate", 0) or 0
    swings = row.get("swings", 0) or 0
    pt = row.get("pitch_type", "")
    league = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, {}).get(
        "whiff_rate", LEAGUE_AVG_OVERALL.get("whiff_rate", 0.25),
    )
    reliability = min(swings, 50) / 50
    return reliability * raw + (1 - reliability) * league


def _create_hitter_vuln_fig(
    vuln_df: pd.DataFrame,
    strength_df: pd.DataFrame,
    hitter_name: str,
) -> plt.Figure:
    """Dual bar chart: vulnerabilities (whiff rate, sample-blended) and
    strengths (xwOBA on contact)."""
    # Keep only pitch types with at least 10 swings
    merged = vuln_df[vuln_df["swings"] >= 10].copy() if "swings" in vuln_df.columns else vuln_df.copy()
    merged["label"] = merged["pitch_type"].map(PITCH_DISPLAY).fillna(merged["pitch_type"])

    # Reliability-blend whiff rate toward league baseline
    merged["blended_whiff"] = merged.apply(_blend_whiff_rate, axis=1)

    # xwoba_contact lives on the vuln df already; if missing, pull from strength_df
    if "xwoba_contact" not in merged.columns or merged["xwoba_contact"].isna().all():
        if strength_df is not None and not strength_df.empty and "xwoba_contact" in strength_df.columns:
            xwoba_map = strength_df.set_index("pitch_type")["xwoba_contact"].to_dict()
            merged["xwoba_contact"] = merged["pitch_type"].map(xwoba_map)

    merged = merged.sort_values("blended_whiff", ascending=True)

    n_rows = len(merged)
    fig, axes = plt.subplots(1, 2, figsize=(7, max(2.2, n_rows * 0.5)))
    fig.patch.set_facecolor(DARK)

    # Left: Vulnerability (blended whiff rate) with rounded bars
    ax1 = axes[0]
    ax1.set_facecolor(DARK)
    y_pos = np.arange(n_rows)
    for i, (_, row) in enumerate(merged.iterrows()):
        w = row["blended_whiff"]
        swings = row.get("swings", 50) or 50
        color = EMBER if w >= 0.30 else GOLD if w >= 0.20 else SAGE
        # Fade bars with fewer swings (alpha 0.4 at 10 swings → 1.0 at 100+)
        alpha = min(1.0, 0.4 + 0.6 * (min(swings, 100) - 10) / 90)
        bar_w = w * 100
        rsize = min(0.3, bar_w / 5) if bar_w > 0 else 0.1
        bar = FancyBboxPatch(
            (0, y_pos[i] - 0.3), bar_w, 0.6,
            boxstyle=f"round,pad=0,rounding_size={rsize:.2f}",
            facecolor=color, alpha=alpha, zorder=2,
        )
        ax1.add_patch(bar)
        # Pitch count annotation
        ax1.text(
            w * 100 + 1, y_pos[i], f"n={int(swings)}",
            color=SLATE, fontsize=7, va="center", alpha=0.7,
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(merged["label"])
    ax1.set_xlim(0, max(merged["blended_whiff"].max() * 100 + 12, 45))
    ax1.set_ylim(-0.6, n_rows - 0.4)
    ax1.set_xlabel("Whiff Rate %", color=SLATE, fontsize=9)
    ax1.set_title("Vulnerability", color=EMBER, fontsize=11, fontweight="bold")
    ax1.tick_params(colors=CREAM, labelsize=9)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Right: Contact quality — xwOBA on contact
    ax2 = axes[1]
    ax2.set_facecolor(DARK)
    xlabel = "xwOBA on Contact"

    if "xwoba_contact" in merged.columns:
        vals = merged["xwoba_contact"].fillna(0)
    else:
        vals = pd.Series([0.0] * n_rows, index=merged.index)

    for i, (_, row) in enumerate(merged.iterrows()):
        v = vals.iloc[i]
        if v > 0 and pd.notna(v):
            bar_width = v * 100  # Scale for display (0.400 → 40)
            color = SAGE if v >= 0.400 else GOLD if v >= 0.340 else SLATE
            rsize = min(0.3, bar_width / 5)
            bar = FancyBboxPatch(
                (0, y_pos[i] - 0.3), bar_width, 0.6,
                boxstyle=f"round,pad=0,rounding_size={rsize:.2f}",
                facecolor=color, alpha=0.85, zorder=2,
            )
            ax2.add_patch(bar)

    ax2.set_yticks(y_pos)
    ax2.set_ylim(-0.6, n_rows - 0.4)
    max_val = vals.max() if vals.max() > 0 else 0.5
    x_max = min(max_val * 100 + 8, 100)  # Cap at 1.000
    ax2.set_xlim(0, x_max)
    # Adaptive ticks based on data range
    if max_val > 0.6:
        tick_vals = [0, 0.200, 0.400, 0.600, 0.800]
    else:
        tick_vals = [0, 0.100, 0.200, 0.300, 0.400, 0.500]
    ax2.set_xticks([t * 100 for t in tick_vals])
    ax2.set_xticklabels([f".{int(t*1000):03d}" for t in tick_vals])
    ax2.set_xlabel(xlabel, color=SLATE, fontsize=9)
    ax2.set_title("Contact Quality", color=SAGE, fontsize=11, fontweight="bold")
    ax2.tick_params(colors=CREAM, labelsize=9)
    ax2.set_yticklabels([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        f"{hitter_name} -- Pitch-Type Profile",
        color=CREAM, fontsize=12, fontweight="bold", y=1.02,
    )

    add_watermark(fig)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Stat configs per player type
# ---------------------------------------------------------------------------
PITCHER_STATS = [
    ("K%", "k_rate", True, "Higher K% = more strikeout stuff"),
    ("BB%", "bb_rate", False, "Lower BB% = better control"),
    ("HR/BF", "hr_per_bf", False, "Lower HR/BF = fewer dingers allowed"),
]
HITTER_STATS = [
    ("K%", "k_rate", False, "Lower K% = better contact ability"),
    ("BB%", "bb_rate", True, "Higher BB% = better plate discipline"),
    ("HR/PA", "hr_rate", True, "Higher HR/PA = more power"),
    ("xwOBA", "xwoba", True, "Higher xwOBA = better expected production"),
]


# ---------------------------------------------------------------------------
# Page: Projections
# ---------------------------------------------------------------------------
def page_projections() -> None:
    """Sortable projection tables for pitchers and hitters."""
    st.markdown('<div class="section-header">2026 Projections</div>',
                unsafe_allow_html=True)

    player_type = st.radio(
        "Player type",
        ["Pitcher", "Hitter"],
        horizontal=True,
        key="proj_type",
    )

    df = load_projections(player_type.lower())
    if df.empty:
        st.warning(
            "No projection data found. "
            "Run `python scripts/precompute_dashboard_data.py` first."
        )
        return

    if player_type == "Pitcher":
        id_col, name_col, hand_col = "pitcher_id", "pitcher_name", "pitch_hand"
        stat_configs = PITCHER_STATS
    else:
        id_col, name_col, hand_col = "batter_id", "batter_name", "batter_stand"
        stat_configs = HITTER_STATS

    # --- Filters ---
    filter_cols = st.columns([2, 1, 1, 1])
    with filter_cols[0]:
        search = st.text_input(
            "Search player", "", placeholder="Type a name...", key="proj_search",
        )
    with filter_cols[1]:
        if player_type == "Pitcher":
            role = st.selectbox("Role", ["All", "Starters", "Relievers"], key="proj_role")
        else:
            role = "All"
    with filter_cols[2]:
        hand_options = ["All"] + sorted(df[hand_col].dropna().unique().tolist())
        hand_filter = st.selectbox("Hand", hand_options, key="proj_hand")
    with filter_cols[3]:
        sort_options = ["Composite Score"] + [s[0] for s in stat_configs]
        sort_by = st.selectbox("Sort by", sort_options, key="proj_sort")

    # Apply filters
    if search:
        df = df[df[name_col].str.contains(search, case=False, na=False)]
    if player_type == "Pitcher":
        if role == "Starters":
            df = df[df["is_starter"] == 1]
        elif role == "Relievers":
            df = df[df["is_starter"] == 0]
    if hand_filter != "All":
        df = df[df[hand_col] == hand_filter]

    # Sort
    if sort_by == "Composite Score":
        sort_col = "composite_score"
        ascending = False
    else:
        stat_key = next(s[1] for s in stat_configs if s[0] == sort_by)
        sort_col = f"delta_{stat_key}"
        higher_is_better = next(s[2] for s in stat_configs if s[0] == sort_by)
        ascending = not higher_is_better

    df_sorted = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    # Build compact display table
    display_rows = []
    for _, row in df_sorted.iterrows():
        r: dict[str, object] = {
            "Rank": len(display_rows) + 1,
            "Name": row[name_col],
            "Age": int(row["age"]) if pd.notna(row.get("age")) else "",
            "Hand": row.get(hand_col, ""),
            "Score": round(row["composite_score"], 2),
        }
        for label, key, higher_better, _ in stat_configs:
            proj_col = f"projected_{key}"
            delta_col = f"delta_{key}"
            if proj_col in row.index and pd.notna(row.get(proj_col)):
                proj_val = _fmt_stat(row[proj_col], key)
                delta_pp = row[delta_col] * 100
                improving = (delta_pp > 0 and higher_better) or (delta_pp < 0 and not higher_better)
                if abs(delta_pp) < 0.05:
                    arrow = ""
                elif improving:
                    arrow = f" ({delta_pp:+.1f})"
                else:
                    arrow = f" ({delta_pp:+.1f})"
                r[label] = f"{proj_val}{arrow}"
            else:
                r[label] = "--"
        display_rows.append(r)

    display_df = pd.DataFrame(display_rows)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    st.caption(
        f"Showing {len(display_df)} {player_type.lower()}s. "
        "Composite score weights stat deltas (normalized, direction-aware). "
        "Positive = projected improvement. Deltas shown in parentheses (pp)."
    )


# ---------------------------------------------------------------------------
# Page: Player Profile
# ---------------------------------------------------------------------------
def page_player_profile() -> None:
    """Deep dive into a single player's projections."""
    st.markdown('<div class="section-header">Player Profile</div>',
                unsafe_allow_html=True)

    player_type = st.radio(
        "Player type",
        ["Pitcher", "Hitter"],
        horizontal=True,
        key="profile_type",
    )

    df = load_projections(player_type.lower())
    if df.empty:
        st.warning(
            "No projection data found. "
            "Run `python scripts/precompute_dashboard_data.py` first."
        )
        return

    if player_type == "Pitcher":
        name_col, id_col, hand_col = "pitcher_name", "pitcher_id", "pitch_hand"
        stat_configs = PITCHER_STATS
    else:
        name_col, id_col, hand_col = "batter_name", "batter_id", "batter_stand"
        stat_configs = HITTER_STATS

    # Player selector
    names = sorted(df[name_col].unique().tolist())
    selected_name = st.selectbox("Select player", names, key="profile_player")

    player_row = df[df[name_col] == selected_name].iloc[0]
    player_id = int(player_row[id_col])

    # --- Header card ---
    hand = player_row.get(hand_col, "")
    age = int(player_row["age"]) if pd.notna(player_row.get("age")) else "?"
    role = ""
    if player_type == "Pitcher" and "is_starter" in player_row.index:
        role = "SP" if player_row["is_starter"] else "RP"

    # Skill tier label
    _TIER_LABELS = {0: "Below-Avg", 1: "Average", 2: "Above-Avg", 3: "Elite"}
    skill_tier = int(player_row.get("skill_tier", 1)) if pd.notna(player_row.get("skill_tier")) else None
    tier_label = _TIER_LABELS.get(skill_tier, "") if skill_tier is not None else ""

    header_parts = [f"Age {age}"]
    if hand:
        if player_type == "Pitcher":
            header_parts.append("LHP" if hand == "L" else "RHP")
        else:
            header_parts.append(f"Bats {'L' if hand == 'L' else 'R'}")
    if role:
        header_parts.append(role)
    if tier_label:
        header_parts.append(f"Skill Tier: {tier_label}")

    composite = player_row["composite_score"]
    comp_color = POSITIVE if composite > 0 else NEGATIVE if composite < 0 else SLATE

    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="brand-title">{selected_name}</div>
            <div class="brand-subtitle">{' | '.join(header_parts)} | 2026 Projection</div>
        </div>
        <div style="color:{comp_color}; font-size:1.2rem; font-weight:600;">
            Composite: {composite:+.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Comparison baseline toggle ---
    compare_to = st.radio(
        "Compare projection to",
        ["Career Avg", "2025"],
        horizontal=True,
        key="compare_baseline",
    )

    # --- Stat metric cards ---
    cols = st.columns(len(stat_configs))
    for col, (label, key, higher_better, _) in zip(cols, stat_configs):
        obs_col = f"observed_{key}"
        career_col = f"career_{key}"
        proj_col = f"projected_{key}"

        if compare_to == "Career Avg" and career_col in player_row.index and pd.notna(player_row.get(career_col)):
            baseline = player_row[career_col]
            baseline_label = "Career"
        elif obs_col in player_row.index and pd.notna(player_row.get(obs_col)):
            baseline = player_row[obs_col]
            baseline_label = "2025"
        else:
            baseline = None
            baseline_label = ""

        if baseline is not None and proj_col in player_row.index and pd.notna(player_row.get(proj_col)):
            proj_str = _fmt_stat(player_row[proj_col], key)
            delta = player_row[proj_col] - baseline
            base_str = _fmt_stat(baseline, key)
            delta_str = (
                f"{baseline_label}: {base_str} ({_delta_html(delta, higher_better)})"
            )
            with col:
                st.markdown(
                    _metric_card(f"Proj. {label}", proj_str, delta_str),
                    unsafe_allow_html=True,
                )
        else:
            with col:
                st.markdown(
                    _metric_card(f"Proj. {label}", "--"),
                    unsafe_allow_html=True,
                )

    # --- Scouting Report (plain English) ---
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    bullets = _generate_scouting_bullets(stat_configs, player_row, df, player_type)
    if bullets:
        bullet_html = "".join(
            f'<div class="insight-bullet">'
            f'<span class="dot" style="background:{color};"></span>'
            f'{text}</div>'
            for color, text in bullets
        )
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">Scouting Report</div>
            {bullet_html}
        </div>
        """, unsafe_allow_html=True)

    # --- Percentile Rankings ---
    st.markdown('<div class="section-header">Percentile Rankings</div>',
                unsafe_allow_html=True)

    bars_html = ""
    for label, key, higher_better, _ in stat_configs:
        proj_col = f"projected_{key}"
        obs_col = f"observed_{key}"
        ci_lo_col = f"projected_{key}_2_5"
        ci_hi_col = f"projected_{key}_97_5"

        if proj_col not in player_row.index or pd.isna(player_row.get(proj_col)):
            continue

        pctile = _percentile_rank(df[proj_col], player_row[proj_col], higher_better)
        ci_lo = player_row.get(ci_lo_col, player_row[proj_col])
        ci_hi = player_row.get(ci_hi_col, player_row[proj_col])

        # 2025 observed percentile (rank among 2025 observed values)
        pctile_2025 = None
        if obs_col in player_row.index and pd.notna(player_row.get(obs_col)):
            pctile_2025 = _percentile_rank(
                df[obs_col], player_row[obs_col], higher_better,
            )

        bars_html += _pctile_bar_html(label, pctile, ci_lo, ci_hi, key, pctile_2025)

    if bars_html:
        st.markdown(
            f'<div class="insight-card">{bars_html}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "How this player ranks vs. all projected players (like Baseball Savant rankings). "
            "100th = best, 1st = worst. "
            "Dashed line = 2025 observed percentile. "
            "Green = elite (80+), gold = above-avg (60-79), "
            "gray = mid-tier (40-59), orange = below-avg (<40). "
            "Range = 95% credible interval."
        )

    # --- Pitch Profile Tables ---
    if player_type == "Pitcher":
        arsenal_df = load_pitcher_arsenal()
        if not arsenal_df.empty:
            p_arsenal = arsenal_df[arsenal_df["pitcher_id"] == player_id].copy()
            if not p_arsenal.empty:
                st.markdown('<div class="section-header">Pitch Arsenal</div>',
                            unsafe_allow_html=True)
                table_html = _build_pitcher_profile_table(p_arsenal)
                if table_html:
                    st.markdown(
                        f'<div class="insight-card">{table_html}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Colors relative to league average per pitch type. "
                        f'<span style="color:{SAGE};">Green</span> = above avg, '
                        f'<span style="color:{GOLD};">gold</span> = avg, '
                        f'<span style="color:{EMBER};">orange</span> = below avg. '
                        "CSW% = called strikes + whiffs / pitches.",
                        unsafe_allow_html=True,
                    )
    else:
        vuln_df = load_hitter_vulnerability(career=True)
        if not vuln_df.empty:
            h_vuln_all = vuln_df[vuln_df["batter_id"] == player_id].copy()
            if not h_vuln_all.empty:
                # Detect switch hitter: significant data on both sides
                side_counts = h_vuln_all.groupby("batter_stand")["pitches"].sum()
                is_switch = (
                    len(side_counts) > 1
                    and all(v >= 50 for v in side_counts.values)
                )

                section_label = "Pitch-Type Profile (Career)"
                if is_switch:
                    platoon_side = st.radio(
                        "Batter side",
                        ["vs RHP (bats L)", "vs LHP (bats R)", "Combined"],
                        horizontal=True,
                        key="profile_platoon",
                    )
                    if platoon_side.startswith("vs RHP"):
                        h_vuln = h_vuln_all[h_vuln_all["batter_stand"] == "L"].copy()
                        section_label = "Pitch-Type Profile (Career — vs RHP)"
                    elif platoon_side.startswith("vs LHP"):
                        h_vuln = h_vuln_all[h_vuln_all["batter_stand"] == "R"].copy()
                        section_label = "Pitch-Type Profile (Career — vs LHP)"
                    else:
                        # Combined: sum raw counts across sides, recompute rates
                        h_vuln = _combine_platoon_vuln(h_vuln_all)
                        section_label = "Pitch-Type Profile (Career — Combined)"
                else:
                    h_vuln = h_vuln_all

                st.markdown(f'<div class="section-header">{section_label}</div>',
                            unsafe_allow_html=True)
                table_html = _build_hitter_profile_table(h_vuln)
                if table_html:
                    st.markdown(
                        f'<div class="insight-card">{table_html}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Colors relative to league average per pitch type. "
                        f'<span style="color:{EMBER};">Orange</span> = exploitable, '
                        f'<span style="color:{SAGE};">green</span> = strength. '
                        "CStr% = called strikes / pitches. "
                        "Bar opacity reflects sample confidence.",
                        unsafe_allow_html=True,
                    )

    # --- Posterior KDE (for pitchers with K% samples) ---
    k_samples = load_k_samples()
    sample_key = str(player_id)

    if player_type == "Pitcher" and sample_key in k_samples:
        st.markdown('<div class="section-header">K% Posterior Distribution</div>',
                    unsafe_allow_html=True)
        samples = k_samples[sample_key]
        obs_k = player_row.get("observed_k_rate")
        fig = _create_posterior_fig(
            samples,
            observed=obs_k if pd.notna(obs_k) else None,
            stat_label="Projected K% (2026)",
        )
        _, chart_col, _ = st.columns([1, 3, 1])
        with chart_col:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        ci_lo, ci_hi = np.percentile(samples * 100, [2.5, 97.5])
        st.caption(
            f"Dashed gold = projected mean | Dotted gray = 2025 observed | "
            f"Shaded = 95% credible interval [{ci_lo:.1f}%, {ci_hi:.1f}%]"
        )

    # --- Stat detail table ---
    st.markdown('<div class="section-header">Stat Breakdown</div>',
                unsafe_allow_html=True)
    detail_rows = []
    for label, key, higher_better, desc in stat_configs:
        obs_col = f"observed_{key}"
        proj_col = f"projected_{key}"
        sd_col = f"projected_{key}_sd"
        lo_col = f"projected_{key}_2_5"
        hi_col = f"projected_{key}_97_5"
        if obs_col in player_row.index and pd.notna(player_row.get(obs_col)):
            detail_rows.append({
                "Stat": label,
                "2025 Observed": _fmt_stat(player_row[obs_col], key),
                "2026 Projected": _fmt_stat(player_row[proj_col], key),
                "Delta": f"{player_row[f'delta_{key}'] * 100:+.1f}pp",
                "95% CI": (
                    f"[{_fmt_stat(player_row[lo_col], key)}, "
                    f"{_fmt_stat(player_row[hi_col], key)}]"
                    if lo_col in player_row.index and pd.notna(player_row.get(lo_col))
                    else "--"
                ),
                "Description": desc,
            })
    if detail_rows:
        st.dataframe(
            pd.DataFrame(detail_rows),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Page: Game K Simulator
# ---------------------------------------------------------------------------
def page_game_k_sim() -> None:
    """Simulate game K totals for a selected pitcher."""
    from src.models.bf_model import get_bf_distribution
    from src.models.game_k_model import compute_k_over_probs, simulate_game_ks

    st.markdown('<div class="section-header">Game K Simulator</div>',
                unsafe_allow_html=True)

    k_samples_dict = load_k_samples()
    bf_priors = load_bf_priors()

    if not k_samples_dict:
        st.warning(
            "No K% posterior samples found. "
            "Run `python scripts/precompute_dashboard_data.py` first."
        )
        return

    pitcher_proj = load_projections("pitcher")
    if pitcher_proj.empty:
        st.warning("No pitcher projections found.")
        return

    # Filter to pitchers with K% samples
    available_ids = set(k_samples_dict.keys())
    pitchers_with_samples = pitcher_proj[
        pitcher_proj["pitcher_id"].astype(str).isin(available_ids)
    ].sort_values("pitcher_name")

    if pitchers_with_samples.empty:
        st.warning("No pitchers with K% samples found.")
        return

    # Pitcher selector
    name_to_id = dict(zip(
        pitchers_with_samples["pitcher_name"],
        pitchers_with_samples["pitcher_id"],
    ))
    selected_name = st.selectbox(
        "Select pitcher",
        sorted(name_to_id.keys()),
        key="gamek_pitcher",
    )
    pitcher_id = int(name_to_id[selected_name])
    k_rate_samples = k_samples_dict[str(pitcher_id)]

    # BF parameters
    bf_info = get_bf_distribution(pitcher_id, 2025, bf_priors)
    bf_mu = bf_info["mu_bf"]
    bf_sigma = bf_info["sigma_bf"]

    col1, col2 = st.columns(2)
    with col1:
        bf_mu_adj = st.slider(
            "Expected batters faced",
            min_value=10, max_value=35, value=int(round(bf_mu)),
            help="Adjust based on expected workload",
        )
    with col2:
        st.markdown(
            _metric_card("Projected K%", _fmt_pct(np.mean(k_rate_samples))),
            unsafe_allow_html=True,
        )

    # Simulate
    game_ks = simulate_game_ks(
        pitcher_k_rate_samples=k_rate_samples,
        bf_mu=float(bf_mu_adj),
        bf_sigma=bf_sigma,
        n_draws=10000,
        random_seed=42,
    )

    # K distribution chart
    fig = _create_game_k_fig(game_ks, selected_name)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # P(over X.5) table
    st.markdown('<div class="section-header">K Prop Lines</div>',
                unsafe_allow_html=True)
    k_over = compute_k_over_probs(game_ks)
    k_over = k_over[(k_over["line"] >= 2.5) & (k_over["line"] <= 10.5)].copy()

    display_lines = []
    for _, row in k_over.iterrows():
        line = row["line"]
        p_over = row["p_over"]
        if p_over > 0.65:
            signal = "Strong Over"
        elif p_over > 0.55:
            signal = "Lean Over"
        elif p_over < 0.35:
            signal = "Strong Under"
        elif p_over < 0.45:
            signal = "Lean Under"
        else:
            signal = "Toss-up"
        display_lines.append({
            "Line": f"Over {line:.1f}",
            "P(Over)": f"{p_over:.1%}",
            "P(Under)": f"{1 - p_over:.1%}",
            "Edge Signal": signal,
        })

    st.dataframe(
        pd.DataFrame(display_lines),
        use_container_width=True,
        hide_index=True,
    )

    # Summary stats
    st.markdown("---")
    summary_cols = st.columns(4)
    stats = [
        ("Expected K", f"{np.mean(game_ks):.1f}"),
        ("Std Dev", f"{np.std(game_ks):.1f}"),
        ("Median K", f"{np.median(game_ks):.0f}"),
        ("90th Pctile", f"{np.percentile(game_ks, 90):.0f}"),
    ]
    for col, (label, val) in zip(summary_cols, stats):
        with col:
            st.markdown(_metric_card(label, val), unsafe_allow_html=True)

    # Plain English summary
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    mean_k = np.mean(game_ks)
    p_over_5 = (game_ks >= 6).sum() / len(game_ks) * 100
    p_over_7 = (game_ks >= 8).sum() / len(game_ks) * 100

    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">What This Means</div>
        <div class="insight-bullet">
            <span class="dot" style="background:{GOLD};"></span>
            On an average night, expect around <strong>{mean_k:.0f} strikeouts</strong>
            (give or take {np.std(game_ks):.0f}).
        </div>
        <div class="insight-bullet">
            <span class="dot" style="background:{SAGE};"></span>
            There's a <strong>{p_over_5:.0f}% chance</strong> of 6+ Ks
            and a <strong>{p_over_7:.0f}% chance</strong> of 8+ Ks.
        </div>
        <div class="insight-bullet">
            <span class="dot" style="background:{SLATE};"></span>
            Based on {len(game_ks):,} Monte Carlo simulations using the Bayesian
            K% posterior (2018-2025 training data).
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Matchup Explorer
# ---------------------------------------------------------------------------
def page_matchup_explorer() -> None:
    """Head-to-head pitcher vs hitter matchup breakdown."""
    from src.models.matchup import score_matchup
    from src.utils.constants import LEAGUE_AVG_BY_PITCH_TYPE

    st.markdown('<div class="section-header">Matchup Explorer</div>',
                unsafe_allow_html=True)

    arsenal_df = load_pitcher_arsenal()
    vuln_df = load_hitter_vulnerability(career=True)
    str_df = load_hitter_strength(career=True)
    pitcher_proj = load_projections("pitcher")
    hitter_proj = load_projections("hitter")

    if arsenal_df.empty or vuln_df.empty:
        st.warning(
            "Matchup data not found. Re-run "
            "`python scripts/precompute_dashboard_data.py` to generate it."
        )
        return

    # --- Selectors ---
    col1, col2 = st.columns(2)
    with col1:
        pitcher_names = sorted(pitcher_proj["pitcher_name"].unique().tolist()) if not pitcher_proj.empty else []
        if not pitcher_names:
            st.warning("No pitcher projections available.")
            return
        selected_pitcher = st.selectbox("Select Pitcher", pitcher_names, key="mu_pitcher")
    with col2:
        hitter_names = sorted(hitter_proj["batter_name"].unique().tolist()) if not hitter_proj.empty else []
        if not hitter_names:
            st.warning("No hitter projections available.")
            return
        selected_hitter = st.selectbox("Select Hitter", hitter_names, key="mu_hitter")

    pitcher_row = pitcher_proj[pitcher_proj["pitcher_name"] == selected_pitcher].iloc[0]
    hitter_row = hitter_proj[hitter_proj["batter_name"] == selected_hitter].iloc[0]
    pitcher_id = int(pitcher_row["pitcher_id"])
    batter_id = int(hitter_row["batter_id"])
    pitcher_hand = pitcher_row.get("pitch_hand", "R")

    # --- Platoon-aware filtering for switch hitters ---
    # Determine which side the hitter bats from against this pitcher
    batter_vuln_all = vuln_df[vuln_df["batter_id"] == batter_id]
    batter_str_all = str_df[str_df["batter_id"] == batter_id] if not str_df.empty else pd.DataFrame()
    side_counts = batter_vuln_all.groupby("batter_stand")["pitches"].sum() if not batter_vuln_all.empty else pd.Series(dtype=float)
    is_switch = len(side_counts) > 1 and all(v >= 50 for v in side_counts.values)

    if is_switch:
        # Switch hitter bats from opposite side of pitcher
        platoon_side = "L" if pitcher_hand == "R" else "R"
        vuln_filtered = vuln_df[
            (vuln_df["batter_id"] != batter_id)
            | (vuln_df["batter_stand"] == platoon_side)
        ].copy()
        str_filtered = str_df[
            (str_df["batter_id"] != batter_id)
            | (str_df["batter_stand"] == platoon_side)
        ].copy() if not str_df.empty else pd.DataFrame()
        hitter_hand = platoon_side
    else:
        vuln_filtered = vuln_df
        str_filtered = str_df
        hitter_hand = hitter_row.get("batter_stand", "?")

    # Score the matchup (using platoon-filtered data)
    baselines_pt: dict[str, dict[str, float]] = {}
    for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items():
        baselines_pt[pt] = vals if isinstance(vals, dict) else {"whiff_rate": 0.25}

    matchup = score_matchup(
        pitcher_id, batter_id, arsenal_df, vuln_filtered, baselines_pt,
    )

    # --- Compute contact-quality adjustment ---
    # Usage-weighted xwOBA and hard-hit delta vs league baselines
    p_ars = arsenal_df[
        (arsenal_df["pitcher_id"] == pitcher_id) & (arsenal_df["pitches"] >= 20)
    ].copy()
    h_str_filt = str_filtered[str_filtered["batter_id"] == batter_id] if not str_filtered.empty else pd.DataFrame()
    damage_score = 0.0
    total_usage = p_ars["usage_pct"].sum() if not p_ars.empty else 0.0
    if total_usage > 0 and not h_str_filt.empty:
        for _, prow in p_ars.iterrows():
            pt = prow["pitch_type"]
            usage_w = prow["usage_pct"] / total_usage
            lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, {})
            lg_xwoba = lg.get("xwoba_contact", 0.320)
            lg_hh = lg.get("hard_hit_rate", 0.33)
            s_row = h_str_filt[h_str_filt["pitch_type"] == pt]
            if s_row.empty:
                continue
            h_xwoba = s_row["xwoba_contact"].iloc[0] if "xwoba_contact" in s_row.columns else np.nan
            h_hh = s_row["hard_hit_rate"].iloc[0] if "hard_hit_rate" in s_row.columns else np.nan
            if pd.notna(h_xwoba):
                damage_score += usage_w * (h_xwoba - lg_xwoba)
            if pd.notna(h_hh):
                damage_score += usage_w * (h_hh - lg_hh) * 0.5

    # --- Matchup header ---
    # Blend whiff lift (pitcher-favorable when positive) with damage score
    # (hitter-favorable when positive). Convert damage to same scale as logit lift.
    lift = matchup["matchup_k_logit_lift"]
    blended_edge = lift - damage_score * 6.0  # scale damage to logit-lift magnitude
    if blended_edge > 0.15:
        edge_label = "Pitcher Advantage"
        edge_color = SAGE
    elif blended_edge < -0.15:
        edge_label = "Hitter Advantage"
        edge_color = EMBER
    else:
        edge_label = "Neutral Matchup"
        edge_color = SLATE

    pitcher_label = "LHP" if pitcher_hand == "L" else "RHP"
    hitter_label = "LHH" if hitter_hand == "L" else "RHH"
    switch_tag = " (switch)" if is_switch else ""
    hand_str = f"{pitcher_label} vs {hitter_label}{switch_tag}"

    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="brand-title">{selected_pitcher} vs {selected_hitter}</div>
            <div class="brand-subtitle">{hand_str} | Career pitch-type profiles</div>
        </div>
        <div style="text-align:right;">
            <div style="color:{edge_color}; font-size:1.1rem; font-weight:600;">
                {edge_label}
            </div>
            <div style="color:{SLATE}; font-size:0.85rem;">
                K Lift: {lift:+.3f} logit
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Summary metrics ---
    mwhiff = matchup["matchup_whiff_rate"]
    bwhiff = matchup["baseline_whiff_rate"]
    whiff_delta = (mwhiff - bwhiff) if pd.notna(mwhiff) and pd.notna(bwhiff) else 0.0
    whiff_delta_html = _delta_html(whiff_delta, higher_is_better=True) if whiff_delta != 0 else ""

    m_cols = st.columns(4)
    with m_cols[0]:
        st.markdown(
            _metric_card(
                "Matchup Whiff",
                _fmt_pct(mwhiff) if pd.notna(mwhiff) else "--",
                whiff_delta_html,
            ),
            unsafe_allow_html=True,
        )
    with m_cols[1]:
        st.markdown(
            _metric_card("Baseline Whiff", _fmt_pct(bwhiff) if pd.notna(bwhiff) else "--"),
            unsafe_allow_html=True,
        )
    with m_cols[2]:
        st.markdown(
            _metric_card("Pitch Types", str(matchup["n_pitch_types"])),
            unsafe_allow_html=True,
        )
    with m_cols[3]:
        st.markdown(
            _metric_card("Data Reliability", f"{matchup['avg_reliability']:.0%}"),
            unsafe_allow_html=True,
        )

    # --- Combined matchup breakdown table ---
    st.markdown('<div class="section-header">Pitch-by-Pitch Matchup</div>',
                unsafe_allow_html=True)

    p_arsenal = arsenal_df[
        arsenal_df["pitcher_id"] == pitcher_id
    ].copy()
    h_vuln = vuln_filtered[vuln_filtered["batter_id"] == batter_id].copy()
    h_str = str_filtered[str_filtered["batter_id"] == batter_id].copy() if not str_filtered.empty else pd.DataFrame()

    if p_arsenal.empty:
        st.info("Insufficient arsenal data for detailed breakdown.")
    else:
        matchup_html = _build_matchup_table(p_arsenal, h_vuln, h_str)
        if matchup_html:
            st.markdown(
                f'<div class="insight-card">{matchup_html}</div>',
                unsafe_allow_html=True,
            )
            platoon_note = f" Hitter stats from {hitter_label} side." if is_switch else ""
            st.caption(
                "P Whiff% = pitcher's whiff rate with that pitch | "
                "H Whiff% = hitter's whiff rate against that pitch type | "
                "H Chase% = hitter's chase rate | H xwOBA = xwOBA on contact | "
                f"Edge: green = pitcher advantage, red = hitter advantage.{platoon_note}"
            )

    # --- Side-by-side profile tables ---
    st.markdown('<div class="section-header">Individual Profiles</div>',
                unsafe_allow_html=True)

    prof_col1, prof_col2 = st.columns(2)
    with prof_col1:
        st.markdown(
            f'<div style="color:{GOLD}; font-size:0.95rem; font-weight:600; '
            f'margin-bottom:0.5rem;">{selected_pitcher} — Arsenal</div>',
            unsafe_allow_html=True,
        )
        if not p_arsenal.empty:
            p_table = _build_pitcher_profile_table(p_arsenal)
            if p_table:
                st.markdown(
                    f'<div class="insight-card">{p_table}</div>',
                    unsafe_allow_html=True,
                )

    with prof_col2:
        vuln_label = f"{selected_hitter} — Vulnerabilities"
        if is_switch:
            vuln_label += f" (batting {hitter_hand})"
        st.markdown(
            f'<div style="color:{GOLD}; font-size:0.95rem; font-weight:600; '
            f'margin-bottom:0.5rem;">{vuln_label}</div>',
            unsafe_allow_html=True,
        )
        if not h_vuln.empty:
            h_table = _build_hitter_profile_table(h_vuln)
            if h_table:
                st.markdown(
                    f'<div class="insight-card">{h_table}</div>',
                    unsafe_allow_html=True,
                )

    # --- Scouting report ---
    st.markdown('<div class="section-header">Matchup Scouting Report</div>',
                unsafe_allow_html=True)

    # Overall summary — use blended edge (whiff lift + contact quality)
    bullets_html = ""
    if pd.notna(mwhiff) and pd.notna(bwhiff):
        whiff_delta_pp = (mwhiff - bwhiff) * 100
        if blended_edge > 0.15:
            summary = (
                f"Favorable matchup for <b>{selected_pitcher}</b>. "
                f"The hitter's pitch-type vulnerabilities align well with the arsenal"
            )
            if whiff_delta_pp > 1:
                summary += f", boosting the expected whiff rate by {whiff_delta_pp:.1f}pp above baseline."
            else:
                summary += "."
        elif blended_edge < -0.15:
            summary = (
                f"Tough matchup for <b>{selected_pitcher}</b>. "
            )
            if damage_score > 0.03:
                summary += (
                    f"<b>{selected_hitter}</b> does significant damage on contact against this arsenal"
                )
                if whiff_delta_pp > 1:
                    summary += f" despite an elevated whiff rate (+{whiff_delta_pp:.1f}pp)."
                else:
                    summary += f" and handles the pitch mix well."
            else:
                summary += (
                    f"<b>{selected_hitter}</b> handles this arsenal well, pulling the expected "
                    f"whiff rate {abs(whiff_delta_pp):.1f}pp below baseline."
                )
        else:
            summary = (
                f"Neutral matchup — no strong edge either way. "
                f"The whiff rate shifts by only {whiff_delta_pp:+.1f}pp from baseline."
            )

        bullets_html += (
            f'<div class="insight-bullet">'
            f'<span class="dot" style="background:{edge_color};"></span>'
            f'{summary}</div>'
        )

    # Per-pitch scouting bullets
    if not p_arsenal.empty and not h_vuln.empty:
        pitch_bullets = _build_matchup_scouting_bullets(
            p_arsenal, h_vuln, h_str, selected_pitcher, selected_hitter,
        )
        for color, text in pitch_bullets:
            bullets_html += (
                f'<div class="insight-bullet">'
                f'<span class="dot" style="background:{color};"></span>'
                f'{text}</div>'
            )

    # Reliability note
    bullets_html += (
        f'<div class="insight-bullet">'
        f'<span class="dot" style="background:{SLATE};"></span>'
        f'Data reliability: {matchup["avg_reliability"]:.0%} '
        f'(based on sample sizes across {matchup["n_pitch_types"]} pitch types)</div>'
    )

    st.markdown(
        f'<div class="insight-card">'
        f'<div class="insight-title">Matchup Summary</div>'
        f'{bullets_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Sidebar
    with st.sidebar:
        if ICON_PATH.exists():
            _, icon_col, _ = st.columns([1, 2, 1])
            with icon_col:
                st.image(str(ICON_PATH), width=110)
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-name">THE DATA DIAMOND</div>
            <div class="sidebar-brand-sub">KEKOA SANTANA</div>
            <div class="sidebar-brand-sub">Bayesian MLB Projections</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["Projections", "Player Profile", "Matchup Explorer", "Game K Simulator"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            f'<div style="color:{SLATE}; font-size:0.75rem; text-align:center;">'
            f'v1.2 | 2026 Season<br>'
            f'Trained on 2018-2025</div>',
            unsafe_allow_html=True,
        )

    if not _check_data_exists():
        st.error(
            "Dashboard data not found. Run the pre-computation first:\n\n"
            "```bash\n"
            "python scripts/precompute_dashboard_data.py\n"
            "```"
        )
        return

    if page == "Projections":
        page_projections()
    elif page == "Player Profile":
        page_player_profile()
    elif page == "Matchup Explorer":
        page_matchup_explorer()
    elif page == "Game K Simulator":
        page_game_k_sim()


if __name__ == "__main__":
    main()
