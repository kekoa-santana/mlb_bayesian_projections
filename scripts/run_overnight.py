"""Overnight pipeline: Statcast projections → wait 2 hours → full precompute.

Usage:
    myenv/Scripts/python scripts/run_overnight.py
"""
import subprocess
import sys
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("overnight")

PYTHON = sys.executable

def run(cmd, label):
    logger.info("=" * 60)
    logger.info("STARTING: %s", label)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("=" * 60)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=".")
    elapsed = (time.time() - t0) / 60
    status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    logger.info("%s: %s in %.1f minutes", label, status, elapsed)
    return result.returncode

# Step 1: Statcast projections
rc = run(
    [PYTHON, "scripts/run_statcast_projections.py"],
    "Statcast Bayesian AR(1) Projections (10 metrics)",
)

# Wait 2 hours
wait_hours = 2
logger.info("=" * 60)
logger.info("Waiting %d hours before precompute... (until %s)",
            wait_hours,
            (datetime.now() + __import__("datetime").timedelta(hours=wait_hours)).strftime("%H:%M"))
logger.info("=" * 60)
time.sleep(wait_hours * 3600)

# Step 2: Full precompute
run(
    [PYTHON, "scripts/precompute_dashboard_data.py"],
    "Full Precompute (models + rankings + teams + sims)",
)

logger.info("=" * 60)
logger.info("OVERNIGHT PIPELINE COMPLETE")
logger.info("=" * 60)
