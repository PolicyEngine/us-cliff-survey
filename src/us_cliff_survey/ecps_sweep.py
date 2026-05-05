"""ECPS-with-axes sweep.

For each Enhanced CPS household, identify a single target person — the
highest baseline earner, with ties broken in favour of tax-unit heads —
and sweep that one person's employment_income from $0 to $30M while
keeping every other person's earnings and every household-level
configuration (state, dependents, itemized deductions, benefits, spouse
earnings) at the ECPS-recorded value. Compute household_net_income at
each earnings level and detect cliffs per household.

Implementation: a single Microsimulation loop over earnings levels.
At each level E, override employment_income to E for the target person
in each household while leaving everyone else (including non-target
tax-unit heads) at baseline. One Microsimulation calculate per level.
The output is a [household × earnings_level] matrix of
household_net_income, on which cliff detection runs row-wise.

Why this works
--------------
- Each ECPS household carries its real configuration into the sweep.
- All cliffs that depend on (state × deductions × dependents × benefit
  receipt) are surfaced for the households that actually have those
  configurations.
- The full earnings range $0–$30M ensures we cover the high-income tail
  (e.g. NY's $25M supplemental-tax recapture).
- Sweeping only one person avoids the multi-tax-unit composite effect:
  a multi-TU household sees its cliff once, at one tax unit's threshold,
  not magnified across every head.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cliff_detection import find_cliffs

log = logging.getLogger(__name__)

DEFAULT_EARNINGS_LEVELS = np.unique(
    np.concatenate(
        [
            # Dense at low/middle incomes where most benefit/credit cliffs live.
            np.arange(0, 100_000, 5_000, dtype=float),
            # Mid-income strata (SALT, AMT, NIIT thresholds).
            np.arange(100_000, 500_000, 25_000, dtype=float),
            # High income (millionaire surtaxes, recapture phase-ins).
            np.arange(500_000, 5_000_000, 250_000, dtype=float),
            # Very high.
            np.arange(5_000_000, 25_000_000, 5_000_000, dtype=float),
            # Bracket NY's $25M recapture: just below, just above.
            np.array([24_999_000, 24_999_999, 25_000_001, 25_001_000, 30_000_000]),
        ]
    )
)


@dataclass
class SweepOutput:
    """Result of an ECPS sweep, indexed by ECPS household.

    Filing status and dependent count live at the tax-unit entity in
    PolicyEngine-US, not the household, and roughly 25 percent of ECPS
    households contain more than one tax unit. Mapping them to the
    household requires picking a representative (typically the first
    tax unit). v1 omits these fields from the in-flight sweep output;
    they can be joined post-sweep from a baseline Microsimulation when
    needed.
    """

    earnings_levels: np.ndarray  # shape (E,)
    net_income: np.ndarray  # shape (H, E)
    income_tax: np.ndarray  # shape (H, E) — federal+state income tax
    household_weight: np.ndarray  # shape (H,)
    state_code: np.ndarray  # shape (H,) — string

    def cliffs(self, min_drop: float = 100.0) -> pd.DataFrame:
        """Return one row per detected cliff (largest per household)."""
        rows = []
        for h in range(self.net_income.shape[0]):
            ni = self.net_income[h, :]
            cliffs = find_cliffs(self.earnings_levels, ni, min_drop=min_drop)
            if not cliffs:
                continue
            top = cliffs[0]
            rows.append(
                {
                    "household_index": h,
                    "household_weight": float(self.household_weight[h]),
                    "state": str(self.state_code[h]),
                    "cliff_earnings": top.earnings_at_cliff,
                    "cliff_step": top.earnings_step,
                    "cliff_drop": top.net_income_drop,
                    "cliff_marginal_rate": top.implied_marginal_rate,
                    "n_cliffs_detected": len(cliffs),
                }
            )
        return pd.DataFrame(rows)


def _identify_target_person(
    baseline_emp: np.ndarray,
    is_head: np.ndarray,
    person_hh_id: np.ndarray,
) -> np.ndarray:
    """Mark exactly one target person per household.

    Selection rule: highest baseline employment_income. Ties broken in
    favour of tax-unit heads, then the lowest person index. If no one in
    the household earns and nobody is flagged as a head, the first person
    in the household becomes the target.
    """
    n = len(baseline_emp)
    df = pd.DataFrame(
        {
            "hh": np.asarray(person_hh_id),
            "emp": np.asarray(baseline_emp, dtype=float),
            "is_head": np.asarray(is_head).astype(int),
            "pidx": np.arange(n),
        }
    )
    # Sort each household so the target lands first: emp DESC, head DESC, pidx ASC.
    df_sorted = df.sort_values(
        by=["hh", "emp", "is_head", "pidx"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    target_pidx = df_sorted.groupby("hh", sort=False)["pidx"].first().to_numpy()
    out = np.zeros(n, dtype=bool)
    out[target_pidx] = True
    return out


def _override_target_earnings(
    baseline_emp: np.ndarray, is_target: np.ndarray, earnings: float
) -> np.ndarray:
    """Replace the target person's employment_income with `earnings`."""
    out = np.array(baseline_emp, dtype=float, copy=True)
    out[is_target.astype(bool)] = earnings
    return out


def run_ecps_sweep(
    year: int = 2026,
    earnings_levels: np.ndarray | None = None,
    progress: bool = True,
) -> SweepOutput:
    """Run the full ECPS sweep and return per-household curves."""
    from policyengine_us import Microsimulation

    if earnings_levels is None:
        earnings_levels = DEFAULT_EARNINGS_LEVELS

    # Baseline once — read demographics, locate heads, identify target person.
    # state_code and household_weight are natively household-level so we omit map_to.
    baseline = Microsimulation()
    baseline_emp = np.asarray(baseline.calculate("employment_income", period=year))
    is_head = np.asarray(baseline.calculate("is_tax_unit_head", period=year)).astype(
        bool
    )
    person_hh_id = np.asarray(baseline.calculate("person_household_id", period=year))
    hh_weight = np.asarray(baseline.calculate("household_weight", period=year))
    state_code = np.asarray(baseline.calculate("state_code", period=year))

    is_target = _identify_target_person(baseline_emp, is_head, person_hh_id)

    n_hh = len(hh_weight)
    n_levels = len(earnings_levels)

    net_income_matrix = np.zeros((n_hh, n_levels), dtype=np.float32)
    income_tax_matrix = np.zeros((n_hh, n_levels), dtype=np.float32)

    iterator = enumerate(earnings_levels)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=n_levels, desc="ECPS sweep")
        except ImportError:
            pass

    for i, earnings in iterator:
        sim = Microsimulation()
        new_emp = _override_target_earnings(baseline_emp, is_target, float(earnings))
        sim.set_input("employment_income", year, new_emp)
        ni = np.asarray(
            sim.calculate("household_net_income", period=year, map_to="household")
        )
        it = np.asarray(sim.calculate("income_tax", period=year, map_to="household"))
        sit = np.asarray(
            sim.calculate("state_income_tax", period=year, map_to="household")
        )
        net_income_matrix[:, i] = ni
        income_tax_matrix[:, i] = it + sit

    return SweepOutput(
        earnings_levels=earnings_levels,
        net_income=net_income_matrix,
        income_tax=income_tax_matrix,
        household_weight=hh_weight,
        state_code=state_code,
    )


def save_outputs(result: SweepOutput, prefix: str) -> None:
    """Persist the sweep matrix and per-household cliffs to parquet/h5."""
    import h5py

    h5_path = f"{prefix}_curves.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("earnings_levels", data=result.earnings_levels)
        f.create_dataset("net_income", data=result.net_income, compression="gzip")
        f.create_dataset("income_tax", data=result.income_tax, compression="gzip")
        f.create_dataset("household_weight", data=result.household_weight)
        f.create_dataset(
            "state_code",
            data=np.array(result.state_code, dtype="S2"),
        )

    cliffs_df = result.cliffs(min_drop=100.0)
    cliffs_df.to_parquet(f"{prefix}_cliffs.parquet", index=False)
    log.info("Wrote %s_curves.h5 and %s_cliffs.parquet", prefix, prefix)
