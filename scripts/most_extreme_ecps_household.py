"""Find the most extreme-shaped tax unit in the ECPS (ignoring state and
income), evaluated against the highest-cost age-curve rating area
(IL area 13).

The question: which actual household composition in our microdata would
produce the largest ACA cliff if placed in the most premium-expensive
rating area? "Extreme" here means highest sum of age-curve multipliers
(more adults age ~64 + more kids age 20).

We use IL rating area 13 ($789/mo base) and the default age curve
(all states except AL/DC/MA/MN/MS/OR/UT and the family-tier states
NY/VT use it, though AL/MS/OR's higher 4.7244 age-64 multiplier could
push composite beyond IL — checked separately).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from us_cliff_survey.aca_cliff import (
    CUSTOM_AGE_CURVE_STATES,
    DEFAULT_2026_FINAL_RATE_400_FPL,
    FAMILY_TIER_STATES,
    MAX_CHILD_COUNT,
    _age_curve_multiplier,
    _value_at,
    fpl_for,
    load_pe_us_params,
    max_base_monthly_per_state,
)


def main() -> None:
    from policyengine_us import Microsimulation

    print("Loading Microsimulation (~60s on cold start)…")
    sim = Microsimulation()
    year = 2026
    target = date(year, 1, 1)

    ages = np.asarray(sim.calculate("age", period=year))
    person_tu = np.asarray(sim.calculate("person_tax_unit_id", period=year))
    person_hh = np.asarray(sim.calculate("person_household_id", period=year))
    hh_id = np.asarray(sim.calculate("household_id", period=year))
    hh_state = np.asarray(sim.calculate("state_code", period=year))
    hh_weight = np.asarray(sim.calculate("household_weight", period=year))

    state_by_hh = dict(zip(hh_id, hh_state, strict=False))
    weight_by_hh = dict(zip(hh_id, hh_weight, strict=False))

    persons_by_tu: dict[int, list[int]] = defaultdict(list)
    tu_to_hh: dict[int, int] = {}
    for i in range(len(ages)):
        persons_by_tu[int(person_tu[i])].append(int(ages[i]))
        tu_to_hh[int(person_tu[i])] = int(person_hh[i])

    params = load_pe_us_params()

    # Use highest-cost age-curve rating area: IL area 13 at $789/mo (2026).
    # AL/MS/OR have higher age-64 multipliers (4.7244) but lower bases; check
    # both and pick the larger annual SLCSP per adult.
    target_state = "IL"
    target_rating_area = 13
    target_base = 789.0
    default_curve = params["age_curves"]["default"]

    # Verify default IL is the per-adult max.
    contenders = [("IL", 13, 789.0, "default")]
    for s in ("AL", "MS", "OR"):
        max_per_state = max_base_monthly_per_state(params["rating"], target)
        if s in max_per_state:
            ra, base = max_per_state[s]
            adult_mult = _age_curve_multiplier(64, params["age_curves"][s.lower()], target)
            adult_il = _age_curve_multiplier(64, default_curve, target)
            if base * adult_mult > 789.0 * adult_il:
                contenders.append((s, ra, base, s.lower()))

    print(f"Per-adult max annual SLCSP candidates:")
    for s, ra, base, curve_name in contenders:
        curve = params["age_curves"][curve_name]
        adult_annual = 12 * base * _age_curve_multiplier(64, curve, target)
        print(
            f"  {s} area {ra}: ${base}/mo base, age-64 mult "
            f"{_age_curve_multiplier(64, curve, target):.4f}, "
            f"annual per-adult SLCSP ${adult_annual:,.0f}"
        )

    # Use IL area 13 (highest annual SLCSP per adult given its base × multiplier).
    print()
    print(f"Using {target_state} area {target_rating_area} for cliff evaluation.")
    print()

    rows = []
    for tu_id, person_ages in persons_by_tu.items():
        eligible_ages = [a for a in person_ages if 0 <= a <= 64]
        if not eligible_ages:
            continue
        adults = [a for a in eligible_ages if a >= 21]
        children = sorted([a for a in eligible_ages if a < 21], reverse=True)
        children_for_premium = children[:MAX_CHILD_COUNT]

        adult_total = sum(_age_curve_multiplier(a, default_curve, target) for a in adults)
        child_total = sum(
            _age_curve_multiplier(a, default_curve, target) for a in children_for_premium
        )
        slcsp = 12 * target_base * (adult_total + child_total)
        n_in_tu = len(person_ages)
        fpl = fpl_for(target_state, n_in_tu, params["fpg"], target)
        required = DEFAULT_2026_FINAL_RATE_400_FPL * 4 * fpl
        cliff = slcsp - required

        hh = tu_to_hh[tu_id]
        rows.append(
            {
                "tax_unit_id": tu_id,
                "household_weight": float(weight_by_hh.get(hh, 0.0)),
                "actual_state": str(state_by_hh.get(hh, "?")),
                "n_in_tu": n_in_tu,
                "n_adults_21_64": len(adults),
                "n_children_le20": len(children),
                "n_children_counted": len(children_for_premium),
                "adult_ages": tuple(sorted(adults, reverse=True)),
                "child_ages": tuple(sorted(children, reverse=True)),
                "child_ages_counted": tuple(children_for_premium),
                "multiplier_total": adult_total + child_total,
                "annual_slcsp_at_il13": slcsp,
                "fpl": fpl,
                "required_contrib": required,
                "cliff_at_il13": cliff,
            }
        )

    rows.sort(key=lambda r: r["cliff_at_il13"], reverse=True)

    print(f"Tax units evaluated: {len(rows):,}\n")
    print("Top 15 most extreme-shaped ECPS tax units (cliff if placed in IL area 13):")
    print()
    for r in rows[:15]:
        print(
            f"  cliff=${r['cliff_at_il13']:>10,.0f}  "
            f"adults={r['n_adults_21_64']:>2} {r['adult_ages']}  "
            f"kids_counted={r['child_ages_counted']!s:>20}  "
            f"actual_state={r['actual_state']:>2}  "
            f"weight={r['household_weight']:>10,.0f}"
        )

    weighted = [r for r in rows if r["household_weight"] > 0]
    print()
    print("Top 10 among weighted (population-representative) tax units:")
    for r in weighted[:10]:
        print(
            f"  cliff=${r['cliff_at_il13']:>10,.0f}  "
            f"adults={r['n_adults_21_64']:>2} {r['adult_ages']}  "
            f"kids_counted={r['child_ages_counted']!s:>20}  "
            f"actual_state={r['actual_state']:>2}  "
            f"weight={r['household_weight']:>10,.0f}"
        )

    print()
    adult_counts: dict[int, int] = defaultdict(int)
    weighted_adult_counts: dict[int, float] = defaultdict(float)
    for r in rows:
        adult_counts[r["n_adults_21_64"]] += 1
        weighted_adult_counts[r["n_adults_21_64"]] += r["household_weight"]

    print("Distribution of PTC-eligible adults (21-64) per tax unit:")
    print(f"  {'#adults':>8} {'tax_units':>10} {'weighted_count':>15}")
    for k in sorted(adult_counts):
        print(
            f"  {k:>8} {adult_counts[k]:>10,} {weighted_adult_counts[k]:>15,.0f}"
        )


if __name__ == "__main__":
    main()
