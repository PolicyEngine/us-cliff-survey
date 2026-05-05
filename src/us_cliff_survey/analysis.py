"""Post-sweep analysis: rank cliffs, summarise mechanisms, write findings.

Inputs: a *_curves.h5 (per-household sweep matrix) and *_cliffs.parquet
(largest detected cliff per household) produced by run_ecps_sweep.

Outputs:
- A summary DataFrame with the top-N cliffs, weighted by household_weight.
- Population-weighted prevalence: how many households face a cliff of $X+.
- A markdown findings file ranking top cliffs with mechanism context.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load_cliffs(prefix: str | Path) -> pd.DataFrame:
    """Load the per-household cliff parquet."""
    return pd.read_parquet(f"{prefix}_cliffs.parquet")


def load_curves(prefix: str | Path) -> dict[str, np.ndarray]:
    """Load the sweep curves from HDF5."""
    out: dict[str, np.ndarray] = {}
    with h5py.File(f"{prefix}_curves.h5", "r") as f:
        for k in f:
            arr = f[k][:]
            if arr.dtype.kind == "S":
                arr = np.array([s.decode("utf-8") for s in arr])
            out[k] = arr
    return out


def top_cliffs_by_size(cliffs: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Return the n largest cliffs by absolute drop, with descriptive context."""
    cols = [
        "state",
        "cliff_earnings",
        "cliff_drop",
        "cliff_step",
        "cliff_marginal_rate",
        "household_weight",
    ]
    return (
        cliffs.sort_values("cliff_drop", ascending=False)
        .head(n)[cols]
        .reset_index(drop=True)
    )


def cliff_prevalence(
    cliffs: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Population-weighted share of households facing cliffs at each threshold."""
    thresholds = thresholds or [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    total_weight = float(cliffs["household_weight"].sum())
    rows = []
    for t in thresholds:
        mask = cliffs["cliff_drop"] >= t
        n_hh = int(mask.sum())
        weight = float(cliffs.loc[mask, "household_weight"].sum())
        rows.append(
            {
                "min_cliff_drop": t,
                "households": n_hh,
                "weighted_count": weight,
                "weighted_share": weight / total_weight if total_weight else 0.0,
            }
        )
    return pd.DataFrame(rows)


def cliffs_by_state(cliffs: pd.DataFrame, n_top: int = 10) -> pd.DataFrame:
    """Largest cliff per state."""
    if cliffs.empty:
        return cliffs
    return (
        cliffs.sort_values("cliff_drop", ascending=False)
        .groupby("state", as_index=False)
        .first()[
            [
                "state",
                "cliff_earnings",
                "cliff_drop",
                "cliff_marginal_rate",
            ]
        ]
        .sort_values("cliff_drop", ascending=False)
        .head(n_top)
        .reset_index(drop=True)
    )


def render_findings(
    cliffs: pd.DataFrame,
    output_path: str | Path,
    title: str = "Largest US income tax cliffs (PolicyEngine-US, ECPS sweep)",
) -> None:
    """Write a markdown summary to disk."""
    weighted = cliffs[cliffs["household_weight"] > 0]
    top = top_cliffs_by_size(cliffs, n=15)
    top_weighted = top_cliffs_by_size(weighted, n=15)
    by_state = cliffs_by_state(cliffs, n_top=15)
    by_state_weighted = cliffs_by_state(weighted, n_top=15)
    prevalence = cliff_prevalence(weighted)

    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append(
        f"Households swept (drop > $100 detected): {len(cliffs):,} "
        f"(of these, {len(weighted):,} have positive survey weight)\n"
    )
    lines.append(
        "Notes:\n"
        "- A 'cliff' here is a drop in household_net_income as the head's "
        "employment_income increases. Marginal rate > 1.0 means the drop "
        "exceeds the earnings step (a true cliff).\n"
        "- The override applies to **every tax unit head** in each "
        "household at each earnings level. Households with multiple tax "
        "units therefore see composite cliffs that sum across heads. The "
        "per-tax-unit cliff is approximately drop / number_of_heads.\n"
    )

    lines.append("## Top 15 cliffs by absolute drop (any record)\n")
    lines.append(top.to_markdown(index=False, floatfmt=",.0f"))

    lines.append("\n\n## Top 15 cliffs among weighted households\n")
    lines.append(top_weighted.to_markdown(index=False, floatfmt=",.0f"))

    lines.append("\n\n## Largest cliff per state (any record, top 15)\n")
    lines.append(by_state.to_markdown(index=False, floatfmt=",.0f"))

    lines.append("\n\n## Largest cliff per state among weighted households (top 15)\n")
    lines.append(by_state_weighted.to_markdown(index=False, floatfmt=",.0f"))

    lines.append("\n\n## Population-weighted prevalence (weighted households only)\n")
    lines.append(prevalence.to_markdown(index=False, floatfmt=",.4f"))

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
