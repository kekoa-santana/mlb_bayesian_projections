"""Shared constants and imports for query modules."""
from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)

# Linear wOBA weights (2024 FanGraphs scale, close enough for 2018-2025)
_WOBA_WEIGHTS = {
    "ubb": 0.69,
    "hbp": 0.72,
    "single": 0.88,
    "double": 1.27,
    "triple": 1.62,
    "hr": 2.10,
}


def season_in_clause(seasons: Iterable[int]) -> str:
    """Build a comma-joined season list for ``WHERE season IN (...)``.

    Coerces each value to int to block SQL injection from untrusted input.
    """
    return ", ".join(str(int(s)) for s in seasons)
