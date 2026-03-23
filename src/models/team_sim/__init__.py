"""Team-level season simulator with injury cascading.

ZiPS-style Monte Carlo: draw injuries per player, cascade PA/IP down
the depth chart, compute BaseRuns RS/RA, convert to Pythagorean wins.
Produces full win distributions per team (mean, p10, p90).
"""
