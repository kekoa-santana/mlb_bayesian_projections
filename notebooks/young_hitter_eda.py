import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.db import read_sql

D = "=" * 70

sql = (
    " SELECT"
    " ba.batter_id, dp.player_name, dp.birth_date,"
    " ba.season, ba.pa, ba.woba, ba.xwoba, ba.wrc_plus,"
    " ba.k_pct, ba.bb_pct, ba.barrel_pct, ba.hard_hit_pct, ba.bip_count,"
    " DATE_PART('year', AGE(MAKE_DATE(ba.season::int, 7, 1), dp.birth_date))::int AS season_age"
    " FROM production.fact_batting_advanced ba"
    " JOIN production.dim_player dp ON dp.player_id = ba.batter_id"
    " WHERE dp.birth_date IS NOT NULL"
    " AND ba.season BETWEEN 2018 AND 2025"
    " AND ba.pa > 0"
    " ORDER BY ba.season, dp.player_name"
)

print(D); print("LOADING DATA"); print(D)
df_raw = read_sql(sql)
print(f"Loaded {len(df_raw):,} player-seasons")

for col in ["woba", "xwoba", "wrc_plus", "pa", "hard_hit_pct"]:
    tmp = df_raw[["batter_id", "season", col]].copy()
    tmp = tmp.rename(columns={col: "next_" + col})
    tmp["season"] = tmp["season"] - 1
    df_raw = df_raw.merge(tmp[["batter_id", "season", "next_" + col]], on=["batter_id", "season"], how="left")
print("Columns built:", [c for c in df_raw.columns if c.startswith("next")])
