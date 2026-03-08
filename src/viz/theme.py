"""
The Data Diamond -- player_profiles theme (thin wrapper around tdd_theme).

Re-exports the shared brand package with project-specific paths
and backward-compatible aliases.  Falls back to inline definitions
when tdd_theme is not installed (e.g. deployed environments).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Shared brand package -- single source of truth (with fallback)
# ---------------------------------------------------------------------------
try:
    from tdd_theme import (                       # noqa: F401 -- re-exports
        GOLD, EMBER, SAGE, SLATE, CREAM, DARK,
        apply_theme, add_watermark, add_brand_footer, add_header,
        save_card, format_pct,
        ASPECT_SIZES, SUBTITLE_PRESETS, LOGO_PATH,
    )
except ImportError:
    # Inline fallback -- keeps deployed app running without the package
    GOLD = "#C8A96E"
    EMBER = "#D4562A"
    SAGE = "#6BA38E"
    SLATE = "#7B8FA6"
    CREAM = "#F5F2EE"
    DARK = "#0F1117"

    ASPECT_SIZES = {"16:9": (16, 9), "5:7": (10, 14), "1:1": (10, 10)}
    SUBTITLE_PRESETS = {
        "live": "Live Game Content",
        "postgame": "Post-Game Recap",
        "highlight": "Game Highlight",
        "projection": "Bayesian Projection Model",
    }
    LOGO_PATH = Path(__file__).resolve().parents[2] / "iconTransparent.png"

    def apply_theme() -> None:
        mpl.rcParams.update({
            "figure.facecolor": CREAM,
            "axes.facecolor": CREAM,
            "savefig.facecolor": CREAM,
            "text.color": DARK,
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "font.size": 14,
            "font.family": "sans-serif",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.3,
        })

    def add_watermark(fig) -> None:
        fig.text(
            0.5, 0.5, "TheDataDiamond",
            fontsize=60, color=SLATE, alpha=0.03,
            ha="center", va="center", rotation=30,
            transform=fig.transFigure, zorder=0,
        )

    def add_brand_footer(fig, subtitle="projection", aspect="16:9") -> None:
        resolved = SUBTITLE_PRESETS.get(subtitle, subtitle)
        fig.text(0.04, 0.02, "TheDataDiamond", fontsize=12, color=GOLD,
                 ha="left", va="bottom", fontweight="bold")
        fig.text(0.96, 0.02, resolved, fontsize=10, color=SLATE,
                 ha="right", va="bottom")

    def add_header(fig, title="", subtitle="", aspect="16:9") -> None:
        fig.text(0.5, 0.93, title, fontsize=22, color=GOLD,
                 ha="center", va="top", fontweight="bold")
        if subtitle:
            fig.text(0.5, 0.885, subtitle, fontsize=13, color=SLATE,
                     ha="center", va="top")

    def save_card(fig, name, output_dir=None, aspect="16:9"):
        from pathlib import Path
        import matplotlib.pyplot as plt
        out = Path(output_dir) if output_dir else Path("outputs/content")
        out.mkdir(parents=True, exist_ok=True)
        if aspect:
            w, h = ASPECT_SIZES.get(aspect, ASPECT_SIZES["16:9"])
            fig.set_size_inches(w, h)
        path = out / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def format_pct(value: float, decimals: int = 1) -> str:
        return f"{value * 100:.{decimals}f}%"

# ---------------------------------------------------------------------------
# Backward-compatible aliases (use new names in new code)
# ---------------------------------------------------------------------------
TEAL = SAGE          # legacy: cards that used TEAL now map to SAGE
DARK_BG = DARK       # legacy: old name for primary text color
WHITE = "#FFFFFF"

# ---------------------------------------------------------------------------
# Project-specific paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "content"

# Backward-compat alias
apply_dark_theme = apply_theme
