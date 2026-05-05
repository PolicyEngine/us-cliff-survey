"""Command-line entry points."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from .analysis import load_cliffs, render_findings
from .ecps_sweep import (
    DEFAULT_EARNINGS_LEVELS,
    run_ecps_sweep,
    save_outputs,
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def sweep_population_cli() -> None:
    """Run the ECPS sweep over the full Enhanced CPS."""
    _setup_logging()
    p = argparse.ArgumentParser(description="ECPS earnings sweep for cliff detection")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/ecps_sweep"),
        help="Output prefix (will produce <prefix>_curves.h5 and <prefix>_cliffs.parquet)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick timing test with 5 earnings levels.",
    )
    args = p.parse_args()

    if args.quick:
        levels = np.array([0, 50_000, 100_000, 1_000_000, 25_000_001], dtype=float)
    else:
        levels = DEFAULT_EARNINGS_LEVELS

    args.output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = run_ecps_sweep(year=args.year, earnings_levels=levels)
    elapsed = time.time() - t0

    save_outputs(result, str(args.output))

    n_levels = len(levels)
    logging.info(
        "Sweep done: %d households × %d earnings levels in %.1f s (%.1f s/level)",
        result.net_income.shape[0],
        n_levels,
        elapsed,
        elapsed / max(n_levels, 1),
    )


def analyze_cli() -> None:
    """Render a markdown findings document from a saved sweep."""
    _setup_logging()
    p = argparse.ArgumentParser(description="Summarise sweep results into findings.md")
    p.add_argument(
        "--prefix",
        type=Path,
        required=True,
        help="Path prefix for the saved sweep (matching --output of sweep-population)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/findings.md"),
        help="Where to write the markdown findings.",
    )
    args = p.parse_args()

    cliffs = load_cliffs(args.prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_findings(cliffs, args.output)
    logging.info("Wrote findings to %s", args.output)
