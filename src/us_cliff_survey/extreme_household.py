"""Find the most extreme-shaped tax unit in PolicyEngine microdata.

For every tax unit in a microdata file (Enhanced CPS or ACS PUMS),
compute the theoretical ACA premium-tax-credit cliff that would face
that unit *if it were placed in the most premium-expensive age-curve
rating area* (IL area 13 — $789/mo age-0 base, default age curve, age-64
multiplier 3.9216). Return the tax units with the largest cliffs.

The point: which actual household compositions in our survey microdata
would generate the largest ACA cliff exposures, irrespective of where
they live or how much they earn? "Extreme" means the highest sum of
age-curve multipliers — most adults age 21-64 plus the 3 oldest kids
under 21.

This is a model-validity ceiling check. It uses each tax unit's actual
person ages but ignores the unit's actual state, rating area, and
income. It assumes proper tax-unit construction; for ACS this requires
PR PolicyEngine/policyengine-us-data#890 (which ports the CPS
construction algorithm to ACS).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .aca_cliff import (
    DEFAULT_2026_FINAL_RATE_400_FPL,
    DEFAULT_PE_US_PARAMS,
    MAX_CHILD_COUNT,
    _age_curve_multiplier,
    fpl_for,
    load_pe_us_params,
)

# IL rating area 13 has the highest 2026 base × age-64 multiplier among
# age-curve states ($789/mo × 3.9216 = $3,094/mo per age-64 adult).
DEFAULT_REFERENCE_STATE = "IL"
DEFAULT_REFERENCE_BASE_MONTHLY = 789.0


@dataclass
class ExtremeRow:
    tax_unit_id: int
    n_in_tu: int
    n_adults_21_64: int
    n_kids_le20: int
    adult_ages: tuple[int, ...]
    kid_ages_counted: tuple[int, ...]
    annual_slcsp: float
    fpl: float
    required_contribution: float
    cliff: float
    household_weight: float


def _load_microdata(h5_path: Path) -> dict[str, np.ndarray]:
    """Read the demographic columns we need from a PE-US-data h5 file.

    Supports both the flat layout used in ACS (`age` is a dataset) and the
    year-keyed layout used in Enhanced CPS (`age/2024` is a dataset, with
    a `<variable>/<year>` structure).
    """
    import h5py

    columns = (
        "age",
        "person_tax_unit_id",
        "person_household_id",
        "household_id",
        "household_weight",
    )

    out: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for col in columns:
            node = f[col]
            if isinstance(node, h5py.Group):
                # Year-keyed; pick the highest-numbered year subkey.
                years = sorted(node.keys())
                if not years:
                    raise ValueError(f"Empty group for {col} in {h5_path}")
                out[col] = node[years[-1]][:]
            else:
                out[col] = node[:]
    return out


def evaluate_microdata(
    h5_path: Path,
    target: date = date(2026, 1, 1),
    reference_base_monthly: float = DEFAULT_REFERENCE_BASE_MONTHLY,
    reference_state_for_fpl: str = DEFAULT_REFERENCE_STATE,
    final_rate_400_fpl: float = DEFAULT_2026_FINAL_RATE_400_FPL,
    params_root: Path = DEFAULT_PE_US_PARAMS,
) -> list[ExtremeRow]:
    """Return per-tax-unit ExtremeRow records sorted by cliff descending."""
    data = _load_microdata(h5_path)
    weight_by_hh = dict(
        zip(data["household_id"], data["household_weight"], strict=False)
    )
    persons_by_tu: dict[int, list[int]] = defaultdict(list)
    tu_to_hh: dict[int, int] = {}
    for i in range(len(data["age"])):
        persons_by_tu[int(data["person_tax_unit_id"][i])].append(int(data["age"][i]))
        tu_to_hh[int(data["person_tax_unit_id"][i])] = int(
            data["person_household_id"][i]
        )

    params = load_pe_us_params(params_root)
    default_curve = params["age_curves"]["default"]

    rows: list[ExtremeRow] = []
    for tu_id, person_ages in persons_by_tu.items():
        eligible = [a for a in person_ages if 0 <= a <= 64]
        if not eligible:
            continue
        adults = [a for a in eligible if a >= 21]
        children = sorted([a for a in eligible if a < 21], reverse=True)
        children_for_premium = children[:MAX_CHILD_COUNT]

        adult_total = sum(
            _age_curve_multiplier(a, default_curve, target) for a in adults
        )
        child_total = sum(
            _age_curve_multiplier(a, default_curve, target)
            for a in children_for_premium
        )
        slcsp = 12 * reference_base_monthly * (adult_total + child_total)
        n_in_tu = len(person_ages)
        fpl = fpl_for(reference_state_for_fpl, n_in_tu, params["fpg"], target)
        required = final_rate_400_fpl * 4 * fpl
        rows.append(
            ExtremeRow(
                tax_unit_id=tu_id,
                n_in_tu=n_in_tu,
                n_adults_21_64=len(adults),
                n_kids_le20=len(children),
                adult_ages=tuple(sorted(adults, reverse=True)),
                kid_ages_counted=tuple(children_for_premium),
                annual_slcsp=slcsp,
                fpl=fpl,
                required_contribution=required,
                cliff=slcsp - required,
                household_weight=float(weight_by_hh.get(tu_to_hh[tu_id], 0.0)),
            )
        )

    rows.sort(key=lambda r: r.cliff, reverse=True)
    return rows


def to_dataframe(rows: list[ExtremeRow]) -> pd.DataFrame:
    """Convert to a DataFrame for tabular output."""
    return pd.DataFrame(
        [
            {
                "tax_unit_id": r.tax_unit_id,
                "n_in_tu": r.n_in_tu,
                "n_adults_21_64": r.n_adults_21_64,
                "n_kids_le20": r.n_kids_le20,
                "adult_ages": ",".join(str(a) for a in r.adult_ages[:5]),
                "kid_ages_counted": ",".join(str(a) for a in r.kid_ages_counted),
                "annual_slcsp": r.annual_slcsp,
                "fpl": r.fpl,
                "required_contribution": r.required_contribution,
                "cliff": r.cliff,
                "household_weight": r.household_weight,
            }
            for r in rows
        ]
    )


def adult_count_distribution(rows: list[ExtremeRow]) -> pd.DataFrame:
    """Return a frequency table of n_adults_21_64 across tax units."""
    counts: dict[int, int] = defaultdict(int)
    weighted: dict[int, float] = defaultdict(float)
    for r in rows:
        counts[r.n_adults_21_64] += 1
        weighted[r.n_adults_21_64] += r.household_weight
    return pd.DataFrame(
        sorted(
            (
                {
                    "n_adults_21_64": k,
                    "tax_units": counts[k],
                    "weighted_count": weighted[k],
                }
                for k in counts
            ),
            key=lambda r: r["n_adults_21_64"],
        )
    )
