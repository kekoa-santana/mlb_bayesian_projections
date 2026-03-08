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
def load_hitter_vulnerability() -> pd.DataFrame:
    """Load hitter vulnerability profiles."""
    path = DASHBOARD_DIR / "hitter_vuln.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_strength() -> pd.DataFrame:
    """Load hitter strength profiles."""
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
) -> str:
    """Render a percentile bar row."""
    color = _pctile_color(pctile)
    ci_str = f"{_fmt_stat(ci_lo, key)} - {_fmt_stat(ci_hi, key)}"
    return (
        f'<div style="margin:12px 0;">'
        f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">'
        f'<span style="color:{SLATE}; font-size:0.85rem; font-weight:600;">{label}</span>'
        f'<span style="color:{SLATE}; font-size:0.8rem;">{pctile:.0f}th percentile | Range: {ci_str}</span>'
        f'</div>'
        f'<div style="width:100%; background:{DARK}; border-radius:6px; height:22px; '
        f'overflow:hidden; border:1px solid {DARK_BORDER};">'
        f'<div style="height:100%; width:{pctile:.0f}%; background:{color}; border-radius:5px;"></div>'
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


def _create_arsenal_fig(
    arsenal_df: pd.DataFrame,
    pitcher_name: str,
) -> plt.Figure:
    """Horizontal bar chart of a pitcher's arsenal, color-coded by whiff quality."""
    df = arsenal_df.sort_values("usage_pct", ascending=True).copy()
    df["label"] = df["pitch_type"].map(PITCH_DISPLAY).fillna(df["pitch_type"])

    fig, ax = plt.subplots(figsize=(7, max(2.2, len(df) * 0.55)))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    colors = [_whiff_quality_color(w) if pd.notna(w) else SLATE for w in df["whiff_rate"]]
    bars = ax.barh(df["label"], df["usage_pct"] * 100, color=colors, height=0.6, alpha=0.85)

    for bar, (_, row) in zip(bars, df.iterrows()):
        usage = row["usage_pct"] * 100
        velo = row.get("avg_velo", np.nan)
        whiff = row.get("whiff_rate", np.nan)
        parts = []
        if pd.notna(velo):
            parts.append(f"{velo:.0f} mph")
        if pd.notna(whiff):
            parts.append(f"Whiff: {whiff*100:.0f}%")
        annotation = "  " + " | ".join(parts) if parts else ""
        ax.text(
            usage + 0.8, bar.get_y() + bar.get_height() / 2,
            annotation, color=CREAM, fontsize=9, va="center",
        )

    ax.set_xlabel("Usage %", color=SLATE, fontsize=10)
    ax.set_title(
        f"{pitcher_name} -- Pitch Arsenal (2025)",
        color=CREAM, fontsize=12, fontweight="bold", pad=10,
    )
    ax.tick_params(colors=CREAM, labelsize=10)
    ax.set_xlim(0, df["usage_pct"].max() * 100 + 18)
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_watermark(fig)
    fig.tight_layout()
    return fig


def _create_hitter_vuln_fig(
    vuln_df: pd.DataFrame,
    strength_df: pd.DataFrame,
    hitter_name: str,
) -> plt.Figure:
    """Dual bar chart: vulnerabilities (whiff/chase) and strengths (barrel/hard-hit)."""
    # Merge vuln + strength on pitch_type, keep only types with enough swings
    vuln = vuln_df[vuln_df["swings"] >= 10].copy() if "swings" in vuln_df.columns else vuln_df.copy()
    vuln["label"] = vuln["pitch_type"].map(PITCH_DISPLAY).fillna(vuln["pitch_type"])

    if strength_df is not None and not strength_df.empty:
        str_cols = ["pitch_type", "barrel_rate_contact", "hard_hit_rate", "xwoba_contact"]
        str_cols = [c for c in str_cols if c in strength_df.columns]
        merged = vuln.merge(strength_df[str_cols], on="pitch_type", how="left")
    else:
        merged = vuln.copy()

    merged = merged.sort_values("whiff_rate", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(7, max(2.2, len(merged) * 0.5)))
    fig.patch.set_facecolor(DARK)

    # Left: Vulnerability (whiff rate)
    ax1 = axes[0]
    ax1.set_facecolor(DARK)
    colors_vuln = [EMBER if w >= 0.30 else GOLD if w >= 0.20 else SAGE
                   for w in merged["whiff_rate"].fillna(0)]
    ax1.barh(merged["label"], merged["whiff_rate"].fillna(0) * 100,
             color=colors_vuln, height=0.6, alpha=0.85)
    ax1.set_xlabel("Whiff Rate %", color=SLATE, fontsize=9)
    ax1.set_title("Vulnerability", color=EMBER, fontsize=11, fontweight="bold")
    ax1.tick_params(colors=CREAM, labelsize=9)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Right: Strength (barrel rate on contact or hard-hit rate)
    ax2 = axes[1]
    ax2.set_facecolor(DARK)
    if "barrel_rate_contact" in merged.columns:
        vals = merged["barrel_rate_contact"].fillna(0) * 100
        xlabel = "Barrel Rate %"
    elif "hard_hit_rate" in merged.columns:
        vals = merged["hard_hit_rate"].fillna(0) * 100
        xlabel = "Hard-Hit Rate %"
    else:
        vals = pd.Series([0] * len(merged))
        xlabel = "Contact Quality"

    colors_str = [SAGE if v >= 10 else GOLD if v >= 5 else SLATE for v in vals]
    ax2.barh(merged["label"], vals, color=colors_str, height=0.6, alpha=0.85)
    ax2.set_xlabel(xlabel, color=SLATE, fontsize=9)
    ax2.set_title("Contact Quality", color=SAGE, fontsize=11, fontweight="bold")
    ax2.tick_params(colors=CREAM, labelsize=9)
    ax2.set_yticklabels([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        f"{hitter_name} -- Pitch-Type Profile (2025)",
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

    # --- Stat metric cards ---
    cols = st.columns(len(stat_configs))
    for col, (label, key, higher_better, _) in zip(cols, stat_configs):
        obs_col = f"observed_{key}"
        proj_col = f"projected_{key}"
        if obs_col in player_row.index and pd.notna(player_row.get(obs_col)):
            proj_str = _fmt_stat(player_row[proj_col], key)
            delta = player_row[f"delta_{key}"]
            obs_str = _fmt_stat(player_row[obs_col], key)
            delta_str = (
                f"2025: {obs_str} ({_delta_html(delta, higher_better)})"
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
        ci_lo_col = f"projected_{key}_2_5"
        ci_hi_col = f"projected_{key}_97_5"

        if proj_col not in player_row.index or pd.isna(player_row.get(proj_col)):
            continue

        pctile = _percentile_rank(df[proj_col], player_row[proj_col], higher_better)
        ci_lo = player_row.get(ci_lo_col, player_row[proj_col])
        ci_hi = player_row.get(ci_hi_col, player_row[proj_col])

        bars_html += _pctile_bar_html(label, pctile, ci_lo, ci_hi, key)

    if bars_html:
        st.markdown(
            f'<div class="insight-card">{bars_html}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "How this player ranks vs. all projected players (like Baseball Savant rankings). "
            "100th = best, 1st = worst. "
            "Green = elite (80+), gold = above-avg (60-79), "
            "gray = mid-tier (40-59), orange = below-avg (<40). "
            "Range = 95% credible interval."
        )

    # --- Arsenal / Vulnerability Charts ---
    if player_type == "Pitcher":
        arsenal_df = load_pitcher_arsenal()
        if not arsenal_df.empty:
            p_arsenal = arsenal_df[arsenal_df["pitcher_id"] == player_id].copy()
            if not p_arsenal.empty:
                st.markdown('<div class="section-header">Pitch Arsenal</div>',
                            unsafe_allow_html=True)
                fig = _create_arsenal_fig(p_arsenal, selected_name)
                _, chart_col, _ = st.columns([1, 3, 1])
                with chart_col:
                    st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption(
                    "Bar color = whiff quality: "
                    f'<span style="color:{SAGE};">elite (35%+)</span> | '
                    f'<span style="color:{GOLD};">above-avg (25-35%)</span> | '
                    f'<span style="color:{SLATE};">avg (15-25%)</span> | '
                    f'<span style="color:{EMBER};">below-avg (<15%)</span>',
                    unsafe_allow_html=True,
                )
    else:
        vuln_df = load_hitter_vulnerability()
        str_df = load_hitter_strength()
        if not vuln_df.empty:
            h_vuln = vuln_df[vuln_df["batter_id"] == player_id].copy()
            h_str = str_df[str_df["batter_id"] == player_id].copy() if not str_df.empty else pd.DataFrame()
            if not h_vuln.empty:
                st.markdown('<div class="section-header">Pitch-Type Profile</div>',
                            unsafe_allow_html=True)
                fig = _create_hitter_vuln_fig(h_vuln, h_str, selected_name)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption(
                    "Left: whiff rate by pitch type "
                    f'(<span style="color:{EMBER};">red = exploitable 30%+</span>, '
                    f'<span style="color:{SAGE};">green = strong <20%</span>). '
                    "Right: barrel rate on contact "
                    f'(<span style="color:{SAGE};">green = dangerous 10%+</span>).',
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
    vuln_df = load_hitter_vulnerability()
    str_df = load_hitter_strength()
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

    # Score the matchup
    baselines_pt: dict[str, dict[str, float]] = {}
    for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items():
        baselines_pt[pt] = vals if isinstance(vals, dict) else {"whiff_rate": 0.25}

    matchup = score_matchup(
        pitcher_id, batter_id, arsenal_df, vuln_df, baselines_pt,
    )

    # --- Matchup header ---
    lift = matchup["matchup_k_logit_lift"]
    if lift > 0.15:
        edge_label = "Pitcher Advantage"
        edge_color = SAGE
    elif lift < -0.15:
        edge_label = "Hitter Advantage"
        edge_color = EMBER
    else:
        edge_label = "Neutral Matchup"
        edge_color = GOLD

    pitcher_hand = pitcher_row.get("pitch_hand", "?")
    hitter_hand = hitter_row.get("batter_stand", "?")
    hand_str = f"{'LHP' if pitcher_hand == 'L' else 'RHP'} vs {'LHH' if hitter_hand == 'L' else 'RHH'}"

    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="brand-title">{selected_pitcher} vs {selected_hitter}</div>
            <div class="brand-subtitle">{hand_str} | 2025 pitch-type profiles</div>
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
    m_cols = st.columns(4)
    mwhiff = matchup["matchup_whiff_rate"]
    bwhiff = matchup["baseline_whiff_rate"]
    with m_cols[0]:
        st.markdown(
            _metric_card("Matchup Whiff", _fmt_pct(mwhiff) if pd.notna(mwhiff) else "--"),
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

    # --- Pitch-by-pitch breakdown ---
    st.markdown('<div class="section-header">Pitch-by-Pitch Breakdown</div>',
                unsafe_allow_html=True)

    p_arsenal = arsenal_df[
        (arsenal_df["pitcher_id"] == pitcher_id) & (arsenal_df["pitches"] >= 20)
    ].sort_values("usage_pct", ascending=False).copy()

    if p_arsenal.empty:
        st.info("Insufficient arsenal data for detailed breakdown.")
    else:
        breakdown_rows = []
        for _, row in p_arsenal.iterrows():
            pt = row["pitch_type"]
            # Hitter's whiff against this pitch type
            h_row = vuln_df[
                (vuln_df["batter_id"] == batter_id) & (vuln_df["pitch_type"] == pt)
            ]
            h_whiff = float(h_row["whiff_rate"].iloc[0]) if len(h_row) > 0 and pd.notna(h_row["whiff_rate"].iloc[0]) else None
            h_chase = float(h_row["chase_rate"].iloc[0]) if len(h_row) > 0 and "chase_rate" in h_row.columns and pd.notna(h_row["chase_rate"].iloc[0]) else None

            # Hitter's strength against this pitch type
            s_row = str_df[
                (str_df["batter_id"] == batter_id) & (str_df["pitch_type"] == pt)
            ] if not str_df.empty else pd.DataFrame()
            h_barrel = float(s_row["barrel_rate_contact"].iloc[0]) if len(s_row) > 0 and pd.notna(s_row["barrel_rate_contact"].iloc[0]) else None

            breakdown_rows.append({
                "Pitch": PITCH_DISPLAY.get(pt, pt),
                "Usage": f"{row['usage_pct']*100:.0f}%",
                "Velo": f"{row['avg_velo']:.0f}" if pd.notna(row.get("avg_velo")) else "--",
                "P Whiff%": f"{row['whiff_rate']*100:.0f}%" if pd.notna(row.get("whiff_rate")) else "--",
                "H Whiff%": f"{h_whiff*100:.0f}%" if h_whiff is not None else "--",
                "H Chase%": f"{h_chase*100:.0f}%" if h_chase is not None else "--",
                "H Barrel%": f"{h_barrel*100:.1f}%" if h_barrel is not None else "--",
            })

        st.dataframe(
            pd.DataFrame(breakdown_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "P Whiff% = pitcher's whiff rate with that pitch. "
            "H Whiff% = hitter's whiff rate against that pitch type. "
            "H Chase% = hitter's chase rate. H Barrel% = hitter's barrel rate on contact."
        )

    # --- Side-by-side arsenal + vulnerability charts ---
    st.markdown('<div class="section-header">Visual Profiles</div>',
                unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if not p_arsenal.empty:
            fig = _create_arsenal_fig(p_arsenal, selected_pitcher)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with chart_col2:
        h_vuln = vuln_df[vuln_df["batter_id"] == batter_id].copy()
        h_str = str_df[str_df["batter_id"] == batter_id].copy() if not str_df.empty else pd.DataFrame()
        if not h_vuln.empty:
            fig = _create_hitter_vuln_fig(h_vuln, h_str, selected_hitter)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # --- Plain English summary ---
    if pd.notna(mwhiff) and pd.notna(bwhiff):
        whiff_delta_pp = (mwhiff - bwhiff) * 100
        if whiff_delta_pp > 2:
            summary = (
                f"This is a favorable matchup for **{selected_pitcher}**. "
                f"The hitter's pitch-type vulnerabilities align well with the pitcher's "
                f"arsenal, boosting the expected whiff rate by {whiff_delta_pp:.1f}pp above baseline."
            )
        elif whiff_delta_pp < -2:
            summary = (
                f"This is a tough matchup for **{selected_pitcher}**. "
                f"**{selected_hitter}** handles this arsenal well, pulling the expected "
                f"whiff rate {abs(whiff_delta_pp):.1f}pp below the pitcher's baseline."
            )
        else:
            summary = (
                f"A neutral matchup -- no strong edge either way. "
                f"The whiff rate shifts by only {whiff_delta_pp:+.1f}pp from baseline."
            )

        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">Matchup Summary</div>
            <div class="insight-bullet">
                <span class="dot" style="background:{edge_color};"></span>
                {summary}
            </div>
            <div class="insight-bullet">
                <span class="dot" style="background:{SLATE};"></span>
                Data reliability: {matchup['avg_reliability']:.0%} (based on sample sizes
                across {matchup['n_pitch_types']} pitch types).
            </div>
        </div>
        """, unsafe_allow_html=True)


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
