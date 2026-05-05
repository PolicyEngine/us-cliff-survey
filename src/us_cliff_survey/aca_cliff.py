"""Theoretical maximum ACA premium tax credit (PTC) cliff at 400% FPL.

PolicyEngine-US 2026 has the original (pre-IRA) ACA structure: households
above 400% FPL are ineligible for PTC, so PTC drops from a positive value
to zero at the threshold. The cliff equals annual SLCSP minus the
required contribution at exactly 400% FPL.

This module derives the maximum cliff analytically by reading PE-US's
parameter YAMLs:

- gov/aca/state_rating_area_cost.yaml — base age-0 monthly premium per
  state and rating area
- gov/aca/age_curves/{state}.yaml — age multiplier curves (default plus
  custom for AL, DC, MA, MN, MS, OR, UT)
- gov/aca/family_tier_states.yaml + family_tier_ratings/{ny,vt}.yaml —
  NY and VT use family-tier rating instead of summing per-person
- gov/aca/required_contribution_percentage/final.yaml — 2026 final rate
  at the 300-400 percent FPL bracket
- gov/hhs/fpg.yaml — federal poverty guideline by household size and
  state group (CONTIGUOUS_US, AK, HI)
- gov/aca/max_child_count.yaml — cap on the number of children younger
  than 21 counted toward premium (3)

Verified independently by Codex against the same parameter files.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

DEFAULT_PE_US_PARAMS = Path(
    "/Users/maxghenis/PolicyEngine/policyengine-us/policyengine_us/parameters"
)

CUSTOM_AGE_CURVE_STATES = {"AL", "DC", "MA", "MN", "MS", "OR", "UT"}
FAMILY_TIER_STATES = {"NY", "VT"}
STATE_GROUPS = {"AK": "AK", "HI": "HI"}  # everyone else is CONTIGUOUS_US
MAX_CHILD_COUNT = 3
DEFAULT_2026_FINAL_RATE_400_FPL = (
    0.0996  # gov/aca/required_contribution_percentage/final.yaml
)


@dataclass(frozen=True)
class HouseholdComposition:
    """Configuration for cliff calculation."""

    n_adults_age_64: int
    children_ages: tuple[
        int, ...
    ]  # caller chooses ages; cap at MAX_CHILD_COUNT applied

    @property
    def n_persons(self) -> int:
        return self.n_adults_age_64 + len(self.children_ages)


@dataclass
class CliffResult:
    state: str
    rating_area: int
    composition_label: str
    base_monthly: float
    annual_slcsp: float
    fpl_household_size: int
    fpl: float
    required_contribution: float
    cliff: float


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _value_at(node: dict, target: date) -> float | None:
    """Return the latest value at or before `target`, or None."""
    dated = sorted(
        ((d, v) for d, v in node.items() if isinstance(d, date)), reverse=True
    )
    for d, v in dated:
        if d <= target:
            return v
    return None


def _age_curve_multiplier(age: int, curve_yaml: dict, target: date) -> float:
    """Look up the multiplier for `age` in a bracket-style YAML."""
    brackets = curve_yaml["brackets"]
    # Each bracket has threshold (date->int) and amount (date->float).
    last_amount = 1.0
    for b in brackets:
        thresh = _value_at(b["threshold"], target)
        amt = _value_at(b["amount"], target)
        if thresh is None or amt is None:
            continue
        if age >= thresh:
            last_amount = amt
        else:
            break
    return last_amount


def fpl_for(state: str, n_persons: int, fpg_yaml: dict, target: date) -> float:
    """Return federal poverty guideline for given state and household size."""
    sg = STATE_GROUPS.get(state, "CONTIGUOUS_US")
    first = _value_at(fpg_yaml["first_person"][sg], target)
    addl = _value_at(fpg_yaml["additional_person"][sg], target)
    if first is None or addl is None:
        raise ValueError(f"FPL not available for {sg} at {target}")
    return first + (n_persons - 1) * addl


def max_base_monthly_per_state(
    rating_yaml: dict, target: date
) -> dict[str, tuple[int, float]]:
    """Return {state: (rating_area, max_base_monthly)} at `target` date."""
    out: dict[str, tuple[int, float]] = {}
    for state, ratings in rating_yaml.items():
        if state in ("description", "metadata") or not isinstance(ratings, dict):
            continue
        best_area = -1
        best_value = 0.0
        for ra_id, year_vals in ratings.items():
            if not isinstance(year_vals, dict):
                continue
            v = _value_at(year_vals, target)
            if v is not None and v > best_value:
                best_value = v
                best_area = ra_id
        if best_area >= 0:
            out[state] = (best_area, best_value)
    return out


def annual_slcsp(
    state: str,
    base_monthly: float,
    composition: HouseholdComposition,
    age_curves: dict[str, dict],
    family_tier: dict[str, dict],
    target: date,
) -> float:
    """Compute annual SLCSP given a state, base premium, and composition."""
    if state in FAMILY_TIER_STATES:
        # Family tier: choose largest applicable category given composition.
        ratings = family_tier[state]
        if composition.n_adults_age_64 == 0:
            return 0.0
        if composition.n_adults_age_64 >= 2 and composition.children_ages:
            cat = "TWO_ADULTS_AND_ONE_OR_MORE_CHILDREN"
        elif composition.n_adults_age_64 >= 2:
            cat = "TWO_ADULTS"
        elif composition.children_ages:
            cat = "ONE_ADULT_AND_ONE_OR_MORE_CHILDREN"
        else:
            cat = "ONE_ADULT"
        mult = _value_at(ratings[cat], target)
        return 12.0 * base_monthly * float(mult)

    curve_key = state.lower() if state in CUSTOM_AGE_CURVE_STATES else "default"
    curve = age_curves[curve_key]

    adult_mult = _age_curve_multiplier(64, curve, target)
    counted_children = composition.children_ages[:MAX_CHILD_COUNT]
    children_total = sum(
        _age_curve_multiplier(a, curve, target) for a in counted_children
    )
    persons_total_mult = composition.n_adults_age_64 * adult_mult + children_total
    return 12.0 * base_monthly * persons_total_mult


def cliff_for(
    state: str,
    rating_area: int,
    base_monthly: float,
    composition: HouseholdComposition,
    composition_label: str,
    age_curves: dict[str, dict],
    family_tier: dict[str, dict],
    fpg: dict,
    final_rate_400_fpl: float,
    target: date,
) -> CliffResult:
    slcsp = annual_slcsp(
        state, base_monthly, composition, age_curves, family_tier, target
    )
    fpl = fpl_for(state, composition.n_persons, fpg, target)
    required = final_rate_400_fpl * 4 * fpl
    return CliffResult(
        state=state,
        rating_area=rating_area,
        composition_label=composition_label,
        base_monthly=base_monthly,
        annual_slcsp=slcsp,
        fpl_household_size=composition.n_persons,
        fpl=fpl,
        required_contribution=required,
        cliff=slcsp - required,
    )


def load_pe_us_params(params_root: Path = DEFAULT_PE_US_PARAMS) -> dict:
    """Load all required PE-US parameter files into a dict."""
    aca = params_root / "gov" / "aca"
    return {
        "rating": _load_yaml(aca / "state_rating_area_cost.yaml"),
        "age_curves": {
            name: _load_yaml(aca / "age_curves" / f"{name}.yaml")
            for name in ("default", "al", "dc", "ma", "mn", "ms", "or", "ut")
        },
        "family_tier": {
            "NY": _load_yaml(aca / "family_tier_ratings" / "ny.yaml"),
            "VT": _load_yaml(aca / "family_tier_ratings" / "vt.yaml"),
        },
        "fpg": _load_yaml(params_root / "gov" / "hhs" / "fpg.yaml"),
    }


def standard_compositions() -> list[tuple[HouseholdComposition, str]]:
    """Return realistic household compositions to evaluate."""
    out: list[tuple[HouseholdComposition, str]] = []
    for n_adults in (1, 2):
        # Children ages 14 (default mult 1.0) and 20 (default mult 1.268).
        for n_kids in range(4):
            for kid_age in (14, 20):
                if n_kids == 0 and kid_age == 20:
                    continue  # avoid duplicate "no kids"
                comp = HouseholdComposition(n_adults, tuple([kid_age] * n_kids))
                label = (
                    f"{n_adults}A64+{n_kids}K{kid_age}" if n_kids else f"{n_adults}A64"
                )
                out.append((comp, label))
    # Deduplicate.
    seen = set()
    deduped = []
    for comp, label in out:
        key = (comp.n_adults_age_64, comp.children_ages)
        if key not in seen:
            seen.add(key)
            deduped.append((comp, label))
    return deduped


def maximum_cliff(
    target: date = date(2026, 1, 1),
    final_rate_400_fpl: float = DEFAULT_2026_FINAL_RATE_400_FPL,
    params_root: Path = DEFAULT_PE_US_PARAMS,
    compositions: Iterable[tuple[HouseholdComposition, str]] | None = None,
) -> CliffResult:
    """Find the household composition + rating area that maximises the cliff."""
    params = load_pe_us_params(params_root)
    max_per_state = max_base_monthly_per_state(params["rating"], target)
    compositions = list(compositions or standard_compositions())

    best = None
    for state, (ra, base) in max_per_state.items():
        for comp, label in compositions:
            r = cliff_for(
                state=state,
                rating_area=ra,
                base_monthly=base,
                composition=comp,
                composition_label=label,
                age_curves=params["age_curves"],
                family_tier=params["family_tier"],
                fpg=params["fpg"],
                final_rate_400_fpl=final_rate_400_fpl,
                target=target,
            )
            if best is None or r.cliff > best.cliff:
                best = r
    assert best is not None
    return best


def all_cliffs(
    target: date = date(2026, 1, 1),
    final_rate_400_fpl: float = DEFAULT_2026_FINAL_RATE_400_FPL,
    params_root: Path = DEFAULT_PE_US_PARAMS,
    compositions: Iterable[tuple[HouseholdComposition, str]] | None = None,
) -> list[CliffResult]:
    """Return cliff results for every (state, composition) pair, sorted descending."""
    params = load_pe_us_params(params_root)
    max_per_state = max_base_monthly_per_state(params["rating"], target)
    compositions = list(compositions or standard_compositions())
    rows = []
    for state, (ra, base) in max_per_state.items():
        for comp, label in compositions:
            rows.append(
                cliff_for(
                    state=state,
                    rating_area=ra,
                    base_monthly=base,
                    composition=comp,
                    composition_label=label,
                    age_curves=params["age_curves"],
                    family_tier=params["family_tier"],
                    fpg=params["fpg"],
                    final_rate_400_fpl=final_rate_400_fpl,
                    target=target,
                )
            )
    rows.sort(key=lambda r: r.cliff, reverse=True)
    return rows


def all_rating_area_cliffs(
    composition: HouseholdComposition,
    composition_label: str,
    target: date = date(2026, 1, 1),
    final_rate_400_fpl: float = DEFAULT_2026_FINAL_RATE_400_FPL,
    params_root: Path = DEFAULT_PE_US_PARAMS,
) -> list[CliffResult]:
    """One cliff per (state, rating_area) at `target` for a fixed household.

    Use to map the geographic distribution of ACA cliff exposure for a
    specific archetype (e.g. couple aged 64).
    """
    params = load_pe_us_params(params_root)
    rows: list[CliffResult] = []
    for state, ratings in params["rating"].items():
        if state in ("description", "metadata") or not isinstance(ratings, dict):
            continue
        for ra_id, year_vals in ratings.items():
            if not isinstance(year_vals, dict):
                continue
            base = _value_at(year_vals, target)
            if base is None:
                continue
            rows.append(
                cliff_for(
                    state=state,
                    rating_area=ra_id,
                    base_monthly=base,
                    composition=composition,
                    composition_label=composition_label,
                    age_curves=params["age_curves"],
                    family_tier=params["family_tier"],
                    fpg=params["fpg"],
                    final_rate_400_fpl=final_rate_400_fpl,
                    target=target,
                )
            )
    rows.sort(key=lambda r: r.cliff, reverse=True)
    return rows


def state_max_cliff(
    composition: HouseholdComposition,
    composition_label: str,
    target: date = date(2026, 1, 1),
    final_rate_400_fpl: float = DEFAULT_2026_FINAL_RATE_400_FPL,
    params_root: Path = DEFAULT_PE_US_PARAMS,
) -> dict[str, CliffResult]:
    """Return {state_code: highest-cliff CliffResult across that state's rating areas}."""
    rows = all_rating_area_cliffs(
        composition,
        composition_label,
        target=target,
        final_rate_400_fpl=final_rate_400_fpl,
        params_root=params_root,
    )
    out: dict[str, CliffResult] = {}
    for r in rows:
        if r.state not in out or r.cliff > out[r.state].cliff:
            out[r.state] = r
    return out
