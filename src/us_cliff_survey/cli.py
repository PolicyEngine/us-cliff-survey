"""Command-line entry points."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from .aca_cliff import HouseholdComposition
from .aca_cliff import all_cliffs as all_aca_cliffs
from .aca_cliff import maximum_cliff as max_aca_cliff
from .analysis import load_cliffs, render_findings
from .cliff_map import (
    rating_area_dataframe,
    render_choropleth,
    state_max_dataframe,
)
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


def max_aca_cliff_cli() -> None:
    """Compute the theoretical maximum ACA PTC cliff for a given year."""
    _setup_logging()
    from datetime import date

    p = argparse.ArgumentParser(
        description="Theoretical maximum ACA PTC cliff at 400% FPL."
    )
    p.add_argument("--year", type=int, default=2026)
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show top N (state, composition) results.",
    )
    args = p.parse_args()

    target = date(args.year, 1, 1)
    top = max_aca_cliff(target=target)
    print(f"Theoretical maximum ACA cliff in {args.year}:")
    print(f"  state:                 {top.state}")
    print(f"  rating area:           {top.rating_area}")
    print(f"  composition:           {top.composition_label}")
    print(f"  base monthly premium:  ${top.base_monthly:,.2f}")
    print(f"  annual SLCSP:          ${top.annual_slcsp:,.2f}")
    print(f"  household size (FPL):  {top.fpl_household_size}")
    print(f"  FPL:                   ${top.fpl:,.2f}")
    print(f"  required contribution: ${top.required_contribution:,.2f}")
    print(f"  CLIFF:                 ${top.cliff:,.2f}")
    print()
    print(f"Top {args.top} (state, composition) by cliff:")
    for r in all_aca_cliffs(target=target)[: args.top]:
        print(
            f"  {r.state:>2} area {r.rating_area:>2} "
            f"{r.composition_label:>10}  cliff=${r.cliff:>9,.0f}"
        )


def aca_cliff_map_cli() -> None:
    """Render an ACA cliff choropleth + per-rating-area CSV."""
    _setup_logging()
    from datetime import date

    p = argparse.ArgumentParser(
        description="Map maximum ACA PTC cliff by state and rating area."
    )
    p.add_argument("--year", type=int, default=2026)
    p.add_argument(
        "--composition",
        choices=["2A64", "2A64+3K20", "2A64+3K14", "1A64"],
        default="2A64",
        help="Household composition to evaluate (default 2 adults age 64).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Where to write rating_areas.csv, state_max.csv, and the choropleth.",
    )
    args = p.parse_args()

    target = date(args.year, 1, 1)

    if args.composition == "2A64":
        comp = HouseholdComposition(2, ())
    elif args.composition == "1A64":
        comp = HouseholdComposition(1, ())
    elif args.composition == "2A64+3K14":
        comp = HouseholdComposition(2, (14, 14, 14))
    elif args.composition == "2A64+3K20":
        comp = HouseholdComposition(2, (20, 20, 20))
    else:
        raise ValueError(f"unknown composition {args.composition}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ra_df = rating_area_dataframe(comp, args.composition, target=target)
    ra_csv = (
        args.output_dir / f"aca_cliff_{args.year}_{args.composition}_rating_areas.csv"
    )
    ra_df.to_csv(ra_csv, index=False)
    logging.info("Wrote %s (%d rows)", ra_csv, len(ra_df))

    sm_df = state_max_dataframe(comp, args.composition, target=target)
    sm_csv = args.output_dir / f"aca_cliff_{args.year}_{args.composition}_state_max.csv"
    sm_df.to_csv(sm_csv, index=False)
    logging.info("Wrote %s (%d rows)", sm_csv, len(sm_df))

    html_path = (
        args.output_dir / f"aca_cliff_{args.year}_{args.composition}_choropleth.html"
    )
    png_path = (
        args.output_dir / f"aca_cliff_{args.year}_{args.composition}_choropleth.png"
    )
    render_choropleth(
        sm_df,
        composition_label=args.composition,
        year=args.year,
        output_html=html_path,
        output_png=png_path,
    )
    logging.info("Wrote %s and %s", html_path, png_path)
