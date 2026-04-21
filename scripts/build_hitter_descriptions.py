"""Build defensible ranking descriptions for hitters.

Scope:
  - Top 60 overall
  - |TDD rank - FG rank| >= 25
  - Top 5 per position (for coverage)

Output: data/rankings_descriptions/hitters.parquet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.paths import dashboard_dir

RANKINGS_PARQUET = dashboard_dir() / "hitters_rankings.parquet"
FG_CSV = Path("outputs/fangraphs_comparison_hitters.csv")
OUT = Path("data/rankings_descriptions/hitters.parquet")


# --- helpers ---------------------------------------------------------------

POS_LABEL = {
    "C": "catcher",
    "1B": "first base",
    "2B": "second base",
    "3B": "third base",
    "SS": "shortstop",
    "LF": "left field",
    "CF": "center field",
    "RF": "right field",
    "DH": "designated hitter",
}

POS_FIELDING_SCALE = {
    "CF": 1.35, "SS": 1.20, "C": 1.05, "2B": 1.00, "3B": 1.00,
    "RF": 0.85, "LF": 0.85, "1B": 0.50, "DH": 0.00,
}


def fmt_pct(x: float | None, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{100 * float(x):.{digits}f}%"


def fmt_rate(x: float | None, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x):.{digits}f}".lstrip("0") if float(x) < 1 else f"{float(x):.{digits}f}"


def fmt_num(x: float | None, digits: int = 0) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x):.{digits}f}"


def ordinal(n: int) -> str:
    """Return '1st', '2nd', '3rd', '92nd', etc."""
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def pos_pct(df: pd.DataFrame, col: str, position: str, value: float) -> int | None:
    """Return percentile (0-100) of `value` within `position` for column `col`."""
    sub = df[df["position"] == position][col].dropna()
    if len(sub) < 2 or pd.isna(value):
        return None
    return int(round((sub < value).mean() * 100))


def overall_pct(df: pd.DataFrame, col: str, value: float) -> int | None:
    sub = df[col].dropna()
    if len(sub) < 2 or pd.isna(value):
        return None
    return int(round((sub < value).mean() * 100))


# --- description builder ---------------------------------------------------

def build_description(
    row: pd.Series,
    df: pd.DataFrame,
    fg_lookup: dict[int, dict],
) -> str:
    name = row["batter_name"]
    pos = row["position"]
    pos_name = POS_LABEL.get(pos, pos)
    rank = int(row["overall_rank"])
    pos_rank = int(row["pos_rank"])
    age_raw = int(row["age"]) if not pd.isna(row["age"]) else 0
    has_age = age_raw > 0
    age = age_raw
    cv = row["current_value_score"]
    tu = row["talent_upside_score"]

    off = row["offense_score"]
    field = row["fielding_combined"]
    traj = row["trajectory_score"]
    bsr = row["baserunning_score"]
    pos_adj = row["positional_adj"]
    health_label = row.get("health_label") or "Unknown"

    proj_woba = row.get("projected_woba")
    barrel = row.get("barrel_pct")
    hardhit = row.get("hard_hit_pct")
    wrc = row.get("wrc_plus")
    proj_k = row.get("projected_k_rate")
    proj_bb = row.get("projected_bb_rate")

    # Percentiles within position
    off_pct = pos_pct(df, "offense_score", pos, off)
    field_pct = pos_pct(df, "fielding_combined", pos, field)
    bsr_pct = pos_pct(df, "baserunning_score", pos, bsr)
    traj_pct = pos_pct(df, "trajectory_score", pos, traj)

    field_scale = POS_FIELDING_SCALE.get(pos, 1.0)

    # Sentence 1: lead
    s1 = f"Ranked #{rank} overall and #{pos_rank} at {pos} on a current-value score of {cv:.3f} (talent-upside {tu:.3f})."

    # Sentence 2: driver — cite numbers
    stat_bits: list[str] = []
    if proj_woba and not pd.isna(proj_woba):
        stat_bits.append(f"projected .{int(round(float(proj_woba)*1000)):03d} wOBA")
    if wrc and not pd.isna(wrc):
        stat_bits.append(f"{int(round(float(wrc)))} wRC+")
    if barrel and not pd.isna(barrel):
        stat_bits.append(f"{fmt_pct(barrel,1)} barrel rate")
    if hardhit and not pd.isna(hardhit):
        stat_bits.append(f"{fmt_pct(hardhit,1)} hard-hit")
    # Fallback: projected K/BB rates for young/prospect players with no traditional stats
    if not stat_bits:
        if proj_k and not pd.isna(proj_k):
            stat_bits.append(f"projected {fmt_pct(proj_k,1)} K rate")
        if proj_bb and not pd.isna(proj_bb):
            stat_bits.append(f"{fmt_pct(proj_bb,1)} BB rate")
    stats_phrase = ", ".join(stat_bits[:3]) if stat_bits else ""

    # Offense descriptor based on position percentile
    if off_pct is not None and off_pct >= 80:
        off_desc = f"The bat is a clear carry tool ({ordinal(off_pct)} percentile offense at {pos}"
    elif off_pct is not None and off_pct >= 55:
        off_desc = f"The bat is above-average for the position ({ordinal(off_pct)} pct offense at {pos}"
    elif off_pct is not None and off_pct >= 30:
        off_desc = f"Offense is roughly league-average for the position ({ordinal(off_pct)} pct at {pos}"
    elif off_pct is not None:
        off_desc = f"Offense grades below the position bar ({ordinal(off_pct)} pct at {pos}"
    else:
        off_desc = "Offense is the primary driver ("

    if stats_phrase:
        s2 = f"{off_desc}; {stats_phrase})"
    else:
        s2 = f"{off_desc})"

    # Attach fielding commentary
    if pos == "DH":
        s2 += ", and the bat has to carry the rank because positional adjustment is 0.000 and fielding is zero-weighted."
    elif field_pct is not None:
        verdict = (
            "a clear asset" if field_pct >= 80 else
            "above-average" if field_pct >= 60 else
            "roughly average" if field_pct >= 40 else
            "a drag" if field_pct >= 20 else
            "a significant drag"
        )
        s2 += f", while the glove is {verdict} ({ordinal(field_pct)}-percentile fielding at {pos}, position scale {field_scale:.2f}x)."
    else:
        s2 += "."

    # Sentence 3: trajectory / baserunning / position adj
    extras: list[str] = []
    if traj_pct is not None:
        age_note = f", age {age}" if has_age else ""
        extras.append(f"trajectory {ordinal(traj_pct)} pct{age_note}")
    if bsr_pct is not None and bsr_pct >= 70:
        extras.append(f"baserunning {ordinal(bsr_pct)} pct")
    elif bsr_pct is not None and bsr_pct <= 20:
        extras.append(f"baserunning only {ordinal(bsr_pct)} pct")
    if pos_adj and not pd.isna(pos_adj) and float(pos_adj) >= 0.5:
        extras.append(f"positional adjustment {float(pos_adj):.3f}")
    s3 = "The composite also credits " + ", ".join(extras) + "." if extras else ""

    # Sentence 4: FG tension if applicable.
    # rank_diff = fg_rank - tdd_rank (empirically verified from the CSV).
    #   positive => FG number is HIGHER (worse) than TDD => TDD likes him MORE than FG
    #   negative => FG number is LOWER (better) than TDD => FG likes him more than TDD
    fg = fg_lookup.get(int(row["batter_id"]))
    s4 = ""
    if fg is not None:
        diff = int(fg["rank_diff"])
        fg_rank = int(fg["fg_rank"])
        fg_war = float(fg["fg_war"]) if not pd.isna(fg["fg_war"]) else None
        war_str = f", {fg_war:.1f} fWAR" if fg_war is not None else ""
        if abs(diff) >= 25:
            if diff > 0:
                # TDD ranks him BETTER (lower number) than FG
                s4 = (
                    f"FanGraphs Depth Charts has him #{fg_rank}{war_str}, well below TDD; "
                    "the talent-first framework rewards his per-game rates without discounting for projected games played "
                    f"(health label: {health_label})."
                )
            else:
                # FG ranks him BETTER than TDD
                s4 = (
                    f"FanGraphs Depth Charts has him #{fg_rank}{war_str}, noticeably higher than TDD; "
                    "the composite down-weights him because position-relative fielding and projected talent don't support "
                    "the same per-game ceiling FG's fWAR credits."
                )

    # Health note if injury-flagged but still ranked high
    health_note = ""
    if health_label in ("Injury Prone", "Questionable") and rank <= 60:
        health_note = (
            f" Health label '{health_label}' is surfaced alongside the rank but excluded from the composite — "
            "TDD measures per-game talent, not durability."
        )

    pieces = [s1, s2]
    if s3:
        pieces.append(s3)
    if s4:
        pieces.append(s4)
    if health_note and not s4:
        pieces.append(health_note.strip())
    elif health_note and s4:
        pieces[-1] = pieces[-1] + health_note

    return " ".join(p for p in pieces if p)


# --- scope -----------------------------------------------------------------

def build_scope(df: pd.DataFrame, fg: pd.DataFrame | None) -> pd.DataFrame:
    """Return dataframe of players needing defense with a `needs_defense_reason` column."""
    scope_rows: list[dict] = []
    seen: set[int] = set()

    # 1. Top 60
    top60 = df[df["overall_rank"] <= 60]
    for _, r in top60.iterrows():
        bid = int(r["batter_id"])
        if bid in seen:
            continue
        seen.add(bid)
        scope_rows.append({"batter_id": bid, "needs_defense_reason": "top_60"})

    # 2. FG gap >= 25
    if fg is not None:
        gaps = fg[fg["abs_rank_diff"] >= 25]
        for _, r in gaps.iterrows():
            bid = int(r["player_id"])
            if bid in seen:
                continue
            if bid not in set(df["batter_id"].astype(int)):
                continue
            seen.add(bid)
            scope_rows.append({"batter_id": bid, "needs_defense_reason": "fg_gap"})

    # 3. Top 5 per position (non-obvious = not already in top 60)
    for pos in df["position"].dropna().unique():
        sub = df[df["position"] == pos].sort_values("pos_rank").head(5)
        for _, r in sub.iterrows():
            bid = int(r["batter_id"])
            if bid in seen:
                continue
            seen.add(bid)
            scope_rows.append({"batter_id": bid, "needs_defense_reason": "non_obvious_composite"})

    scope_df = pd.DataFrame(scope_rows)
    return scope_df


# --- main ------------------------------------------------------------------

def main() -> None:
    df = pd.read_parquet(RANKINGS_PARQUET)
    fg = pd.read_csv(FG_CSV) if FG_CSV.exists() else None
    fg_lookup: dict[int, dict] = {}
    if fg is not None:
        for _, r in fg.iterrows():
            fg_lookup[int(r["player_id"])] = r.to_dict()

    scope_df = build_scope(df, fg)
    merged = scope_df.merge(df, on="batter_id", how="left")

    # Generate descriptions
    rows_out: list[dict] = []
    for _, row in merged.iterrows():
        desc = build_description(row, df, fg_lookup)
        rows_out.append({
            "player_id": int(row["batter_id"]),
            "player_name": row["batter_name"],
            "overall_rank": int(row["overall_rank"]),
            "pos_rank": int(row["pos_rank"]),
            "position": row["position"],
            "description": desc,
            "needs_defense_reason": row["needs_defense_reason"],
        })

    out_df = pd.DataFrame(rows_out).sort_values("overall_rank").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT, index=False)

    # Report
    print(f"Wrote {len(out_df)} descriptions -> {OUT}")
    print("\nBreakdown by reason:")
    print(out_df["needs_defense_reason"].value_counts().to_string())
    print("\nBreakdown by position:")
    print(out_df["position"].value_counts().to_string())
    print("\nSample descriptions:")
    for i in [0, 5, 20, 55, 100]:
        if i < len(out_df):
            r = out_df.iloc[i]
            print(f"\n--- {r['player_name']} (#{r['overall_rank']}, {r['position']}, reason={r['needs_defense_reason']}) ---")
            print(r["description"])


if __name__ == "__main__":
    main()
