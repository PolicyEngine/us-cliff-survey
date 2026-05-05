# US cliff survey — findings

Empirical and analytical investigation of the largest income-tax cliffs in
the modeled US tax-and-transfer system, motivated by the question:
**is New York's $25 million supplemental-tax recapture really the largest
annual income-tax cliff in America?**

## Headline

For realistic single tax filing units modeled in PolicyEngine-US:

| Cliff | Per-tax-unit max | Mechanism |
|-------|-----------------:|-----------|
| **NY $25M recapture** | **$149,455** | Crossing $25M of NY AGI applies the 10.9% top rate to all NY taxable income. |
| ACA PTC at 400% FPL (theoretical max) | $94,866 | 2 adults age 64 + 3 dependents age 20 in IL rating area 13 lose all marketplace subsidy. |
| ACA PTC at 400% FPL (couple-only) | $65,638 | 2 adults age 64 in IL rating area 13. |
| NJ Stay NJ | $35,984 | Property tax credit phase-out around $87.5K AGI. |
| Federal/state benefit cliffs | $20K-$50K range | Medicaid loss, EITC phase-outs, state mandate-related cliffs. |

NY's cliff is the largest per-tax-unit cliff for realistic household
shapes. Among the most extreme tax-unit compositions present in real
microdata (rare multi-adult tax units in ACS), the theoretical ACA cliff
in IL area 13 reaches $190,929 — but those compositions represent a
trivial weighted population.

## Methodology summary

Three lines of evidence:

1. **Empirical ECPS earnings sweep** — varied each Enhanced CPS tax unit's
   primary earner from $0 to $30M and detected drops in
   household_net_income.
2. **Analytical ACA cliff** — derived the theoretical max PTC cliff at
   400% FPL directly from PE-US parameter YAMLs (state rating area
   premiums, age curves, family-tier multipliers, FPL by household size).
3. **Microdata extreme-household scan** — for every tax unit in ECPS and
   ACS, computed the theoretical ACA cliff if placed in the highest-cost
   age-curve rating area.

## ECPS earnings sweep

### v1: every-head override (artifact diagnosed)

The first sweep set every tax-unit head's `employment_income` to the
swept level simultaneously. In multi-tax-unit households (~25% of ECPS),
this drove every head across the $25M threshold at once and produced
composite cliffs equal to N × per-tax-unit cliff. The headline drop was
$747,272 for a 5-tax-unit household — N × $149,455.

### v2: single-target override (correct)

Refactored to identify a single target person per household (highest
baseline earner; ties broken to tax-unit heads, then person index;
fallback to first tax-unit head if all earnings are zero) and override
only that person's earnings. Other household members — including other
tax-unit heads — stay at their ECPS baseline.

Results in `results/full_v2_cliffs.parquet`:

- **Every NY household with crossing earnings shows cliff = $149,455** —
  the clean per-tax-unit recapture cliff, matching the article's article
  figure to within $500 (the $999 step adds a small marginal-tax
  component).
- 522 weighted ECPS households face a detected cliff of any size.
- Largest non-NY cliff among weighted records: NJ Stay NJ at $35,984
  (rate ≈ 7×).

### Per-state max cliff (v2)

| State | Largest cliff (weighted) | Earnings | Marginal rate |
|------:|-------------------------:|---------:|--------------:|
| **NY** | **$149,455** | $25M | 150× |
| NJ | $35,984 | $65K | 7× |
| TX | $31,872 | $50K | 6× |
| NM | $31,540 | $65K | 6× |
| AZ | $25,929 | $60K | 5× |
| CA | $25,501 | $100K | 5× |

The non-NY cliffs are mostly state-Medicaid loss, ACA-related, or
state-credit phase-outs — not state-income-tax cliffs.

## Theoretical maximum ACA PTC cliff

Independent derivation from PE-US parameter YAMLs (verified by Codex
against the same sources):

```
2026 base monthly premium, IL rating area 13:        $789
Default age curve, age 64:                           3.9216 multiplier
Default age curve, age 20:                           1.268 multiplier
Default age curve, age <15:                          1.0 multiplier
2026 max child count (premium-rated):                3
2026 final required-contribution rate at 400% FPL:   9.96%
2026 5-person FPL (contiguous US):                   $38,680

Annual SLCSP (2A age 64 + 3 kids age 20):
  = 12 × 789 × (2 × 3.9216 + 3 × 1.268)
  = $110,275.69

Required contribution at 400% FPL:
  = 0.0996 × 4 × 38,680
  = $15,410.11

Cliff = $110,275.69 − $15,410.11 = $94,866
```

For 2A age 64 with kids under 15 (multiplier 1.0), the cliff is
$87,253. For 2 adults age 64 with no kids, $65,638.

Geographic distribution by state (couple aged 64, no kids):

- IL area 13: $65,638 (highest)
- FL area 44: $63,003
- WV area 5: $57,544
- WY area 3: $53,591
- AK area 2: $48,612
- … 498 rating areas total; mean $27,553, median $26,908

Choropleth in `results/aca_cliff_2026_2A64_choropleth.png` and per-state
data in `aca_cliff_2026_*_state_max.csv`.

NY and VT (family-tier rating, community pricing) show low cliffs
because they don't age-rate premiums; a 64-year-old pays the same
community rate as a 21-year-old. The cliff structure is also flatter
because NY/VT family-tier multipliers cap at 2-adults regardless of
household size.

## Extreme-household scan against microdata

For every tax unit in ECPS and ACS PUMS 2022, compute the theoretical
ACA cliff that would face that exact composition if relocated to IL
rating area 13.

### Enhanced CPS (43,188 tax units)

| n adults 21-64 | tax units | weighted households |
|---------------:|----------:|--------------------:|
| 0 | 1,684 | 5,610,337 |
| 1 | 27,124 | 106,039,300 |
| 2 | 13,722 | 38,983,680 |
| 3 | 624 | 2,145,317 |
| **4** | **34** | **143,650** |

Top extreme cliff in ECPS: **$92,020** for 3 adults (61, 55, 23) + 3
dependents (19, 19, 19) — weighted population of 4,661 households.
Below the synthetic max of $94,866 because no actual ECPS tax unit has
the exact "2 adults age 64 + 3 dependents age 20" composition.

### ACS PUMS 2022 with PR #890 tax-unit construction

[PolicyEngine/policyengine-us-data#890](https://github.com/PolicyEngine/policyengine-us-data/pull/890)
ported the CPS qualifying-child / qualifying-relative algorithm to ACS
(file by [issue #888](https://github.com/PolicyEngine/policyengine-us-data/issues/888)
spawned by this work). Before #890, ACS treated each housing unit as a
single tax unit, producing 20-adult "tax units" and a meaningless
$408,000 maximum cliff. After #890:

| n adults 21-64 | tax units | weighted households |
|---------------:|----------:|--------------------:|
| 0 | 102,041 | 5,909,155 |
| 1 | 901,023 | 89,700,445 |
| 2 | 438,352 | 43,386,675 |
| 3 | 18,310 | 1,792,436 |
| 4 | 1,013 | 101,441 |
| 5 | 37 | 3,861 |
| 6 | 3 | 320 |
| **11** | **1** | **45** |

Top extreme cliff in ACS (post-#890): **$190,929** for 11 adults aged
62, 62, 47, 46, 44, … with 0 kids — weighted population 45 households.
The next-largest is $112,378 for 5 adults aged 58, 56, 54, 50, 49.

The 11-adult tax unit is a real ACS record with a household composition
that the construction algorithm assigns to one filing unit (likely a
multi-generation household where many adults pass the qualifying-relative
test under one filer). Its weighted population is 45 — a vanishingly
small share of the US household distribution.

### Bottom line

NY's $149,455 per-tax-unit recapture cliff exceeds the realistic ACA
cliff ceiling for the overwhelming majority of household shapes. Two
caveats:

1. ACS contains rare tax-unit compositions whose theoretical ACA cliff
   exceeds NY's. These represent a trivial population (≤200 weighted
   households nationally for >$150K cliffs).
2. The ACA cliff applies to vastly more households — anywhere from
   millions of households near 400% FPL nationwide. NY's cliff applies
   to a few hundred filers crossing $25M of NY AGI in any given year.

For a defensible "largest income tax cliff in America" claim:

> New York's $25 million recapture is the largest per-tax-unit
> income-tax cliff in PolicyEngine-US for realistic household
> compositions. The ACA premium-tax-credit cliff at 400% FPL is the
> next-largest, and affects vastly more households.

## Reproducing the analysis

```bash
# Empirical ECPS sweep (~50 min)
uv run sweep-population --output results/full_v2

# Per-household cliff summary
uv run cliff-analyze --prefix results/full_v2 --output results/findings.md

# Theoretical max ACA cliff
uv run max-aca-cliff --year 2026 --top 10

# State choropleth + per-rating-area CSVs
uv run aca-cliff-map --year 2026 --composition 2A64
uv run aca-cliff-map --year 2026 --composition 2A64+3K20

# Extreme-household scan against any PE-US-data h5
uv run extreme-household --microdata path/to/enhanced_cps_2024.h5
uv run extreme-household --microdata path/to/acs_2022.h5
```

## Caveats

- Coverage is limited to mechanisms PolicyEngine-US models. State or
  local provisions not yet implemented are missed.
- The empirical ECPS sweep uses 2026 parameters and a static (no
  behavioural-response) model.
- Float32 precision in the simulation moves NY's cliff detection from
  the exact $25M threshold to the bracket [$25,000,001, $25,001,000];
  the cliff size is preserved.
- ACS extremes depend on PR #890's tax-unit construction algorithm,
  which uses heuristics for spouse and parent linkages that ACS does
  not provide directly. See issue #888 for the methodology trade-offs.
