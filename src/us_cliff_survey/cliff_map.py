"""Render geographic maps of the maximum ACA PTC cliff.

Two outputs:

1. Per-rating-area table (CSV/parquet) — one row per (state, rating_area)
   with the cliff for a chosen household composition. PE-US doesn't ship
   rating-area boundaries, so this is the most granular view we can
   produce without additional data.

2. State-level choropleth — for each state, the maximum cliff across its
   rating areas, rendered as a Plotly choropleth. Tooltips show the
   maximum-cliff rating area, base premium, and cliff size.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .aca_cliff import (
    DEFAULT_2026_FINAL_RATE_400_FPL,
    DEFAULT_PE_US_PARAMS,
    HouseholdComposition,
    all_rating_area_cliffs,
    state_max_cliff,
)

POLICYENGINE_TEAL = "#39C6C0"


def rating_area_dataframe(
    composition: HouseholdComposition,
    composition_label: str,
    target: date = date(2026, 1, 1),
    final_rate_400_fpl: float = DEFAULT_2026_FINAL_RATE_400_FPL,
    params_root: Path = DEFAULT_PE_US_PARAMS,
) -> pd.DataFrame:
    """Wide table of cliffs at every rating area."""
    rows = all_rating_area_cliffs(
        composition,
        composition_label,
        target=target,
        final_rate_400_fpl=final_rate_400_fpl,
        params_root=params_root,
    )
    return pd.DataFrame(
        [
            {
                "state": r.state,
                "rating_area": r.rating_area,
                "base_monthly": r.base_monthly,
                "annual_slcsp": r.annual_slcsp,
                "fpl": r.fpl,
                "required_contribution": r.required_contribution,
                "cliff": r.cliff,
            }
            for r in rows
        ]
    )


def state_max_dataframe(
    composition: HouseholdComposition,
    composition_label: str,
    target: date = date(2026, 1, 1),
    final_rate_400_fpl: float = DEFAULT_2026_FINAL_RATE_400_FPL,
    params_root: Path = DEFAULT_PE_US_PARAMS,
) -> pd.DataFrame:
    """One row per state with its maximum cliff."""
    by_state = state_max_cliff(
        composition,
        composition_label,
        target=target,
        final_rate_400_fpl=final_rate_400_fpl,
        params_root=params_root,
    )
    return pd.DataFrame(
        [
            {
                "state": r.state,
                "rating_area": r.rating_area,
                "base_monthly": r.base_monthly,
                "annual_slcsp": r.annual_slcsp,
                "fpl": r.fpl,
                "required_contribution": r.required_contribution,
                "cliff": r.cliff,
            }
            for r in by_state.values()
        ]
    ).sort_values("cliff", ascending=False)


def render_choropleth(
    state_df: pd.DataFrame,
    composition_label: str,
    year: int,
    output_html: Path,
    output_png: Path | None = None,
) -> None:
    """Render a US state choropleth of maximum ACA cliff."""
    import plotly.express as px

    fig = px.choropleth(
        state_df,
        locations="state",
        locationmode="USA-states",
        color="cliff",
        scope="usa",
        color_continuous_scale=[
            (0.0, "#F2F4F7"),
            (0.5, "#7FE0DC"),
            (1.0, POLICYENGINE_TEAL),
        ],
        labels={"cliff": "Maximum ACA cliff ($)"},
        hover_data={
            "state": True,
            "rating_area": True,
            "base_monthly": ":$,.0f",
            "annual_slcsp": ":$,.0f",
            "cliff": ":$,.0f",
        },
        title=(
            f"Maximum {year} ACA premium tax credit cliff at 400% FPL, "
            f"by state — {composition_label}"
        ),
    )
    fig.update_layout(
        geo={"showlakes": True, "lakecolor": "white"},
        coloraxis_colorbar={"tickprefix": "$", "separatethousands": True},
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")
    if output_png is not None:
        _write_png(fig, output_png)


def _write_png(fig, output_png: Path) -> None:
    """Best-effort PNG export. New kaleido (>=1) uses write_fig_sync; older
    plotly uses fig.write_image. Failures are swallowed so callers always
    get the HTML output."""
    import contextlib

    with contextlib.suppress(Exception):
        import kaleido

        kaleido.write_fig_sync(
            fig,
            path=str(output_png),
            opts={"scale": 2, "width": 1200, "height": 700},
        )
        return
    with contextlib.suppress(Exception):
        fig.write_image(output_png, scale=2, width=1200, height=700)
