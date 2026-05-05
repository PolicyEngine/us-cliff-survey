# US cliff survey

Empirical survey of the largest income tax cliffs in the United States, using PolicyEngine-US.

## Question

Where in the US tax-and-transfer system does an additional dollar of earnings reduce a household's net income by more than that dollar — and where is the largest such cliff?

## Approach

Two complementary analyses:

1. **Synthetic sweep** (`sweep-synthetic`) — generates representative archetypes (state × filing status × number of dependents) and sweeps employment income from $0 to $30M with adaptive refinement. Catalogs theoretical cliff mechanisms. Confirms whether New York's $25 million supplemental-tax recapture is the largest income-tax cliff anywhere in the modeled tax code.

2. **Population sweep** (`sweep-population`) — uses Enhanced CPS households at 2026, varies primary-earner earnings around each household's baseline, and detects cliffs that depend on actual configurations (itemized deductions, benefit receipt, dependents, state of residence). Produces population-weighted prevalence estimates: how many US households face a cliff of $X or more at the margin.

The Enhanced CPS has very little weight above $1M of earnings, so the synthetic sweep is needed to confirm theoretical cliffs at the high-income tail. The population sweep is needed to find configuration-dependent cliffs (SALT-cap interactions, Medicaid loss thresholds, ACA subsidy edges, EITC phase-outs, dependent-care subsidy losses) that no archetype would naturally hit.

## Caveats

- Coverage is limited to mechanisms PolicyEngine-US models. State or local provisions not yet implemented are missed; PolicyEngine-US covers all 50 state income taxes plus DC, and major federal benefits.
- "Cliff" here means a discontinuity in household net income as a function of own earnings (Δnet_income / Δearnings < 0 across a small earnings step). It does not capture cliffs in non-modeled benefits (housing vouchers in some jurisdictions, employer benefits, etc.).
- All calculations use 2026 parameters and a static (no-behavioral-response) model.

## Setup

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Run

```bash
# Synthetic sweep across archetypes (~10-30 min)
sweep-synthetic --output results/synthetic.parquet

# Population sweep across Enhanced CPS (~30-90 min)
sweep-population --output results/population.parquet
```

## Outputs

- `results/synthetic.parquet` — one row per (state, filing_status, n_dependents, earnings) with computed cliff size and mechanism
- `results/population.parquet` — one row per ECPS household with detected cliff size, weight, demographic markers
- `results/findings.md` — ranked top cliffs with mechanism narratives

## Methodology

See `docs/methodology.md` for the cliff detection algorithm, sweep strategy, and definitions.
