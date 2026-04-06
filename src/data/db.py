"""
Database connection helper for the mlb_fantasy PostgreSQL database.

Reads DB config from environment variables (and optional .env file),
with fallback to config/database.yaml for backward compatibility.
Provides a SQLAlchemy engine and convenience function for read_sql calls.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "database.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

DB_ENV_MAP = {
    "host": "DB_HOST",
    "port": "DB_PORT",
    "user": "DB_USER",
    "password": "DB_PASSWORD",
    "dbname": "DB_NAME",
    "schema": "SCHEMA",
}
REQUIRED_DB_KEYS = ("host", "port", "user", "password", "dbname")


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE pairs from a .env file."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            values[key] = value

    return values


def _load_from_env(dotenv_values: dict[str, str]) -> dict[str, Any]:
    """Load DB config from process env, falling back to .env parsed values."""
    cfg: dict[str, Any] = {}
    for field, env_key in DB_ENV_MAP.items():
        raw = os.getenv(env_key, dotenv_values.get(env_key, ""))
        if isinstance(raw, str):
            raw = raw.strip()
        if raw != "":
            cfg[field] = raw
    return cfg


def _load_db_config(
    path: Path = CONFIG_PATH,
    env_path: Path = ENV_PATH,
) -> dict[str, Any]:
    """Load database configuration from env/.env, then YAML fallback."""
    dotenv_values = _parse_dotenv(env_path)
    env_cfg = _load_from_env(dotenv_values)
    if all(k in env_cfg for k in REQUIRED_DB_KEYS):
        env_cfg["port"] = int(env_cfg["port"])
        return env_cfg

    if path.exists():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or "database" not in cfg:
            raise ValueError(f"Invalid database config format in {path}")
        file_cfg = cfg["database"]
        missing_yaml = [k for k in REQUIRED_DB_KEYS if k not in file_cfg]
        if missing_yaml:
            raise ValueError(
                f"Missing database keys in {path}: {', '.join(missing_yaml)}"
            )
        return file_cfg

    missing_env = [
        DB_ENV_MAP[key]
        for key in REQUIRED_DB_KEYS
        if key not in env_cfg
    ]
    raise FileNotFoundError(
        "Database config not found. Set env vars "
        f"({', '.join(missing_env)}) or provide {path}."
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine connected to mlb_fantasy."""
    cfg = _load_db_config()
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=int(cfg["port"]),
        database=cfg["dbname"],
    )
    engine = create_engine(url, pool_pre_ping=True, pool_size=5)
    logger.info("Connected to %s on %s:%s", cfg["dbname"], cfg["host"], cfg["port"])
    return engine


def load_or_build_parquet(
    cache_path: Path,
    builder: callable,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load a DataFrame from Parquet cache, or build and cache it.

    Parameters
    ----------
    cache_path : Path
        Full path to the ``.parquet`` cache file.
    builder : callable
        Zero-argument callable that returns the DataFrame to cache.
    force_rebuild : bool
        If True, ignore existing cache and rebuild.

    Returns
    -------
    pd.DataFrame
    """
    if cache_path.exists() and not force_rebuild:
        logger.info("Loading cached %s", cache_path.name)
        return pd.read_parquet(cache_path)

    logger.info("Building %s", cache_path.name)
    df = builder()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached %s (%d rows)", cache_path.name, len(df))
    return df


def read_sql(query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame.

    Parameters
    ----------
    query : str
        SQL query string. Use :param_name for bind parameters.
    params : dict, optional
        Bind-parameter values.

    Returns
    -------
    pd.DataFrame
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    logger.debug("Query returned %d rows", len(df))
    return df
