#!/usr/bin/env python
"""Validate frozen core rankings and weekly-form artifact contracts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


DASHBOARD_DIR = Path(r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard")


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> list[str]:
    missing = sorted(required - set(df.columns))
    if missing:
        return [f"{name}: missing columns {missing}"]
    return []


def main() -> int:
    errors: list[str] = []
    paths = {
        "hitters_live": DASHBOARD_DIR / "hitters_rankings.parquet",
        "pitchers_live": DASHBOARD_DIR / "pitchers_rankings.parquet",
        "hitters_core": DASHBOARD_DIR / "hitters_core_rankings.parquet",
        "pitchers_core": DASHBOARD_DIR / "pitchers_core_rankings.parquet",
        "hitters_weekly": DASHBOARD_DIR / "hitters_weekly_form.parquet",
        "pitchers_weekly": DASHBOARD_DIR / "pitchers_weekly_form.parquet",
    }
    for key, p in paths.items():
        if not p.exists():
            errors.append(f"{key}: missing file {p}")

    if errors:
        print("Contract check failed:")
        for e in errors:
            print(f"- {e}")
        return 1

    h_live = pd.read_parquet(paths["hitters_live"])
    p_live = pd.read_parquet(paths["pitchers_live"])
    h_core = pd.read_parquet(paths["hitters_core"])
    p_core = pd.read_parquet(paths["pitchers_core"])
    h_weekly = pd.read_parquet(paths["hitters_weekly"])
    p_weekly = pd.read_parquet(paths["pitchers_weekly"])

    errors += _require_columns(
        h_core, {"batter_id", "current_value_score", "rank_type", "core_anchor_season", "core_projection_season"},
        "hitters_core",
    )
    errors += _require_columns(
        p_core, {"pitcher_id", "current_value_score", "rank_type", "core_anchor_season", "core_projection_season"},
        "pitchers_core",
    )
    errors += _require_columns(
        h_weekly, {"batter_id", "weekly_form_score", "weekly_form_rank", "pa_14d"},
        "hitters_weekly",
    )
    errors += _require_columns(
        p_weekly, {"pitcher_id", "weekly_form_score", "weekly_form_rank", "bf_14d"},
        "pitchers_weekly",
    )

    if "rank_type" in h_core.columns and not (h_core["rank_type"] == "core_preseason").all():
        errors.append("hitters_core: rank_type must be core_preseason")
    if "rank_type" in p_core.columns and not (p_core["rank_type"] == "core_preseason").all():
        errors.append("pitchers_core: rank_type must be core_preseason")

    # Non-mutation guard: live rankings should not unexpectedly grow core-only metadata.
    for live_name, df in (("hitters_live", h_live), ("pitchers_live", p_live)):
        for forbidden in ("core_anchor_season", "core_projection_season"):
            if forbidden in df.columns:
                errors.append(f"{live_name}: contains core-only column {forbidden}")

    if errors:
        print("Contract check failed:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("Core + weekly-form contract checks passed.")
    print(f"Rows: hitters_core={len(h_core)}, pitchers_core={len(p_core)}")
    print(f"Rows: hitters_weekly={len(h_weekly)}, pitchers_weekly={len(p_weekly)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
