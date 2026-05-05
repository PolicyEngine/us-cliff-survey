"""Tests for ACA cliff map computation."""

from __future__ import annotations

from datetime import date

import pytest

from us_cliff_survey.aca_cliff import DEFAULT_PE_US_PARAMS, HouseholdComposition
from us_cliff_survey.cliff_map import (
    rating_area_dataframe,
    state_max_dataframe,
)

needs_pe_us = pytest.mark.skipif(
    not DEFAULT_PE_US_PARAMS.exists(),
    reason="PolicyEngine-US params not available locally",
)


@needs_pe_us
class TestRatingAreaDataframe:
    def test_returns_one_row_per_rating_area(self) -> None:
        comp = HouseholdComposition(n_adults_age_64=2, children_ages=())
        df = rating_area_dataframe(comp, "2A64", target=date(2026, 1, 1))
        # Each state has multiple rating areas; total in 2026 is around 498.
        assert len(df) > 400
        # Required columns present.
        for col in (
            "state",
            "rating_area",
            "base_monthly",
            "annual_slcsp",
            "cliff",
        ):
            assert col in df.columns

    def test_il_rating_area_13_is_highest(self) -> None:
        comp = HouseholdComposition(n_adults_age_64=2, children_ages=())
        df = rating_area_dataframe(comp, "2A64", target=date(2026, 1, 1))
        top = df.sort_values("cliff", ascending=False).iloc[0]
        assert top["state"] == "IL"
        assert top["rating_area"] == 13


@needs_pe_us
class TestStateMaxDataframe:
    def test_returns_one_row_per_state(self) -> None:
        comp = HouseholdComposition(n_adults_age_64=2, children_ages=())
        df = state_max_dataframe(comp, "2A64", target=date(2026, 1, 1))
        # 50 states + DC = 51, plus possibly territories with zero values
        # (GU, PR, VI may have $0 base premium in some years).
        assert len(df) >= 50
        # No duplicate state codes.
        assert df["state"].is_unique

    def test_il_is_at_top(self) -> None:
        comp = HouseholdComposition(n_adults_age_64=2, children_ages=())
        df = state_max_dataframe(comp, "2A64", target=date(2026, 1, 1))
        assert df.iloc[0]["state"] == "IL"

    def test_sorted_descending_by_cliff(self) -> None:
        comp = HouseholdComposition(n_adults_age_64=2, children_ages=())
        df = state_max_dataframe(comp, "2A64", target=date(2026, 1, 1))
        cliffs = df["cliff"].tolist()
        assert cliffs == sorted(cliffs, reverse=True)
