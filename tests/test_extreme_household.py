"""Tests for extreme_household.

The integration tests against real microdata are gated behind
@needs_pe_us; the unit tests below check the pure logic and the
DataFrame conversions.
"""

from __future__ import annotations

import numpy as np
import pytest

from us_cliff_survey.aca_cliff import DEFAULT_PE_US_PARAMS
from us_cliff_survey.extreme_household import (
    ExtremeRow,
    adult_count_distribution,
    to_dataframe,
)


def _row(
    tax_unit_id: int,
    n_adults: int,
    cliff: float,
    weight: float = 1.0,
) -> ExtremeRow:
    return ExtremeRow(
        tax_unit_id=tax_unit_id,
        n_in_tu=n_adults,
        n_adults_21_64=n_adults,
        n_kids_le20=0,
        adult_ages=tuple([60] * n_adults),
        kid_ages_counted=(),
        annual_slcsp=cliff + 10_000,
        fpl=20_000,
        required_contribution=10_000,
        cliff=cliff,
        household_weight=weight,
    )


class TestToDataframe:
    def test_columns_present(self) -> None:
        df = to_dataframe([_row(1, 2, 50_000)])
        assert set(df.columns) == {
            "tax_unit_id",
            "n_in_tu",
            "n_adults_21_64",
            "n_kids_le20",
            "adult_ages",
            "kid_ages_counted",
            "annual_slcsp",
            "fpl",
            "required_contribution",
            "cliff",
            "household_weight",
        }

    def test_truncates_to_top_5_adult_ages(self) -> None:
        rows = [
            ExtremeRow(
                tax_unit_id=1,
                n_in_tu=10,
                n_adults_21_64=10,
                n_kids_le20=0,
                adult_ages=(64, 63, 62, 61, 60, 59, 58, 57, 56, 55),
                kid_ages_counted=(),
                annual_slcsp=1.0,
                fpl=1.0,
                required_contribution=0.0,
                cliff=1.0,
                household_weight=1.0,
            )
        ]
        df = to_dataframe(rows)
        assert df["adult_ages"].iloc[0] == "64,63,62,61,60"


class TestAdultCountDistribution:
    def test_counts_and_weights_sum(self) -> None:
        rows = [
            _row(1, 1, 30_000, weight=10.0),
            _row(2, 1, 25_000, weight=20.0),
            _row(3, 2, 60_000, weight=15.0),
            _row(4, 3, 90_000, weight=5.0),
        ]
        df = adult_count_distribution(rows)
        df = df.sort_values("n_adults_21_64").reset_index(drop=True)

        assert list(df["n_adults_21_64"]) == [1, 2, 3]
        assert list(df["tax_units"]) == [2, 1, 1]
        assert list(df["weighted_count"]) == [30.0, 15.0, 5.0]


needs_pe_us = pytest.mark.skipif(
    not DEFAULT_PE_US_PARAMS.exists(),
    reason="PolicyEngine-US params not available locally",
)


@needs_pe_us
@pytest.mark.integration
class TestRealMicrodataSmoke:
    """Smoke test against actual microdata. Slow — gated behind @integration."""

    def test_evaluate_microdata_runs(self, tmp_path) -> None:
        # Construct a minimal h5 by writing two tax units: one couple at 64
        # and one single 30-year-old. Verify cliff for the couple > single.
        import h5py

        path = tmp_path / "tiny.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("age", data=np.array([64, 64, 30]))
            f.create_dataset("person_tax_unit_id", data=np.array([1, 1, 2]))
            f.create_dataset("person_household_id", data=np.array([1, 1, 2]))
            f.create_dataset("household_id", data=np.array([1, 2]))
            f.create_dataset("household_weight", data=np.array([100.0, 50.0]))

        from us_cliff_survey.extreme_household import evaluate_microdata

        rows = evaluate_microdata(path)
        assert len(rows) == 2
        # Couple at 64 should have larger cliff than single 30-year-old.
        couple = next(r for r in rows if r.n_in_tu == 2)
        single = next(r for r in rows if r.n_in_tu == 1)
        assert couple.cliff > single.cliff
        # Adult ages preserved.
        assert couple.adult_ages == (64, 64)
        assert single.adult_ages == (30,)
