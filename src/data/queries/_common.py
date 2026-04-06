"""Shared constants and imports for query modules."""
from __future__ import annotations

import logging

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
