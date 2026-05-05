"""Time the candidate sweep approaches on real data.

Approaches:
A. Microsimulation loop — full ECPS, override head's employment_income, recalc.
B. Per-household Simulation with axes — one tax unit, axes sweep $0–$30M.
"""

from __future__ import annotations

import time

import numpy as np


def time_microsim_baseline() -> float:
    from policyengine_us import Microsimulation

    t0 = time.time()
    sim = Microsimulation()
    sim.calculate("household_net_income", period=2026, map_to="household")
    return time.time() - t0


def time_microsim_loop_iteration(reuse: bool = True) -> tuple[float, int, list[float]]:
    """Time approach A iterations.

    If reuse=True, one Microsimulation is created and reused; otherwise a
    fresh one is created each iteration. Returns the time per iteration
    (post-warmup) and the per-iteration breakdown.
    """
    from policyengine_us import Microsimulation

    sim = Microsimulation()
    baseline_emp = np.asarray(sim.calculate("employment_income", period=2026))
    is_head = np.asarray(sim.calculate("is_tax_unit_head", period=2026)).astype(bool)
    sim.calculate("household_net_income", period=2026, map_to="household")
    n_hh_per_iter = 0

    earnings_levels = [50_000.0, 100_000.0, 250_000.0, 1_000_000.0, 25_000_001.0]
    times = []
    for earnings in earnings_levels:
        t0 = time.time()
        if not reuse:
            sim = Microsimulation()
        new_emp = np.where(is_head, earnings, baseline_emp)
        sim.set_input("employment_income", 2026, new_emp)
        ni = sim.calculate("household_net_income", period=2026, map_to="household")
        times.append(time.time() - t0)
        n_hh_per_iter = len(ni)

    return float(np.mean(times)), n_hh_per_iter, times


def time_per_household_axes(n_axis_points: int = 100) -> float:
    """Time approach B: a per-household axis sweep."""
    from policyengine_us import Simulation

    year = 2026
    situation = {
        "people": {
            "head": {"age": {str(year): 40}, "employment_income": {str(year): 0}},
        },
        "tax_units": {
            "tu": {"members": ["head"], "filing_status": {str(year): "SINGLE"}},
        },
        "families": {"f": {"members": ["head"]}},
        "spm_units": {"s": {"members": ["head"]}},
        "marital_units": {"m": {"members": ["head"]}},
        "households": {"h": {"members": ["head"], "state_name": {str(year): "NY"}}},
        "axes": [
            [
                {
                    "name": "employment_income",
                    "count": n_axis_points,
                    "min": 0,
                    "max": 30_000_000,
                    "period": year,
                }
            ]
        ],
    }
    t0 = time.time()
    sim = Simulation(situation=situation)
    sim.calculate("household_net_income", period=year, map_to="household")
    return time.time() - t0


if __name__ == "__main__":
    print("== Approach A: Microsimulation baseline + override loop ==")
    t_baseline = time_microsim_baseline()
    print(f"  Microsim baseline (cold): {t_baseline:.1f}s")
    t_iter, n_hh, breakdown = time_microsim_loop_iteration(reuse=True)
    print(f"  Reuse=True per-iter mean: {t_iter:.1f}s ({n_hh:,} households)")
    print(f"  Per-iter breakdown: {[f'{t:.1f}s' for t in breakdown]}")
    print(f"  Projected for 30 earnings levels: {t_iter * 30 / 60:.1f} min")
    print(f"  Projected for 100 earnings levels: {t_iter * 100 / 60:.1f} min")

    print("\n== Approach B: Per-household Simulation with axes ==")
    t_axes_100 = time_per_household_axes(100)
    print(f"  100 axis points, single household: {t_axes_100:.2f}s")
    t_axes_500 = time_per_household_axes(500)
    print(f"  500 axis points, single household: {t_axes_500:.2f}s")
    print(
        f"  Projected for 1% sample (~850 households, 100 axis pts): "
        f"{t_axes_100 * 850 / 60:.1f} min"
    )
    print(
        f"  Projected for full ECPS (~85K households, 100 axis pts): "
        f"{t_axes_100 * 85000 / 60:.1f} min"
    )
