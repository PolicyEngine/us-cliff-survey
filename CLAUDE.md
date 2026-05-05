# us-cliff-survey — agent guide

## Goal

Find the largest income tax cliffs in the US empirically, using PolicyEngine-US. The headline claim under test is whether New York's $25 million supplemental-tax recapture is the largest annual income-tax cliff in the United States.

## Stack

- Python 3.13 (≥3.13, <3.15)
- `policyengine-us` ≥ 1.500
- `uv` for env, `ruff` for lint
- pandas / numpy / parquet for data

## Commands

```bash
make install     # uv pip install -e ".[dev]"
make test        # pytest
make format      # ruff format
make sweep       # run synthetic + population sweeps
```

## Architecture

```
src/us_cliff_survey/
├── cliff_detection.py    # core algorithm: max cliff in (earnings, net_income) arrays
├── synthetic_sweep.py    # archetype × earnings sweep with axes
├── population_sweep.py   # ECPS households × earnings delta sweep
├── archetypes.py         # household situation builders
├── refinement.py         # adaptive refinement around discontinuities
└── cli.py                # entry points
```

## Key design decisions

- **Earnings as the swept variable**: `employment_income`. We do not sweep self-employment, capital gains, or other income types in v1. Adding them is straightforward but expands the search space significantly.
- **Net income definition**: `household_net_income` (PolicyEngine-US variable). Includes federal/state/local taxes and benefits.
- **Income tax definition**: `income_tax + state_income_tax` mapped to household. Used for income-tax-only cliff measurement when distinguishing tax cliffs from benefit cliffs.
- **Cliff detection**: a cliff at point i is `net_income[i] - net_income[i+1] > step_size[i]`. Adaptive refinement uses bisection to localize the discontinuity to ~$1.
- **Synthetic sweep range**: $0 to $30M (covers NY's $25M recapture). Coarse pass at $10K then refinement.
- **Population sweep**: per-household earnings delta of ±$5K from baseline at $250 steps; refines at candidate cliffs.

## Data

- ECPS dataset auto-downloads via `policyengine-us-data` on first microsim invocation.
- Cache lives in `~/.cache/policyengine-us-data/` (default).

## Common gotchas

- PolicyEngine `Simulation` (situation API) supports `axes` for vectorized sweeps. Use this for synthetic archetypes.
- `Microsimulation` operates on the full ECPS. Modifying `employment_income` requires rebuilding the dataset; it's faster to do separate runs at each delta.
- `household_net_income` includes refundable credits and benefits. For income-tax-only cliffs, also compute `income_tax_before_refundable_credits + state_income_tax`.

## When extending

- New cliff dimensions (self-employment, dividends, etc.) → add another sweep in `synthetic_sweep.py` with the corresponding variable name.
- Including non-PolicyEngine cliffs (e.g., housing vouchers) → would require external data sources; out of scope for v1.
