"""
The Data Diamond — player_profiles theme (thin wrapper around tdd_theme).

Re-exports the shared brand package with project-specific paths
and backward-compatible aliases.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Shared brand package — single source of truth
# ---------------------------------------------------------------------------
from tdd_theme import (                       # noqa: F401 — re-exports
    GOLD, EMBER, SAGE, SLATE, CREAM, DARK,
    apply_theme, add_watermark, add_brand_footer, add_header,
    save_card, format_pct,
    ASPECT_SIZES, SUBTITLE_PRESETS, LOGO_PATH,
)

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
