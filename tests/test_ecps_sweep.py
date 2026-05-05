"""Tests for ecps_sweep.

The sweep module loops over earnings levels, overrides head's
employment_income, and gathers per-household net income. We test the
orchestration with a mock that stands in for PolicyEngine-US, plus an
@integration test that runs against the real Enhanced CPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from us_cliff_survey.ecps_sweep import (
    SweepOutput,
    _override_head_earnings,
    run_ecps_sweep,
)


@dataclass
class _FakeMicrosim:
    """Stand-in for policyengine_us.Microsimulation.

    Calculates a stylised cliff so the test exercises the cliff detector:
    every household's net income increases linearly with employment_income
    until $100,000, then drops by $5,000 (a clean cliff) and resumes
    the linear curve.
    """

    n_persons: int = 6
    n_households: int = 3
    states: tuple[str, ...] = ("CA", "NY", "TX")

    def __post_init__(self) -> None:
        self._head_earnings = 0.0
        # baseline_emp: each person earns 50_000; each household has 2 persons.
        self._baseline_emp = np.full(self.n_persons, 50_000.0)
        self._is_head = np.array([True, False] * self.n_households)

    def calculate(self, var: str, period: int = 2026, map_to: str | None = None):
        if var == "employment_income":
            return self._baseline_emp
        if var == "is_tax_unit_head":
            return self._is_head
        if var == "household_weight":
            return np.full(self.n_households, 100.0)
        if var == "state_code":
            return np.array(self.states)
        if var == "household_net_income":
            earnings = self._head_earnings
            base = earnings * 0.7  # 70% net of head's earnings
            cliff = 5_000.0 if earnings > 100_000 else 0.0
            return np.full(self.n_households, base - cliff + 10_000.0)
        if var == "income_tax":
            return np.full(self.n_households, max(0.0, self._head_earnings * 0.2))
        if var == "state_income_tax":
            return np.full(self.n_households, max(0.0, self._head_earnings * 0.05))
        raise NotImplementedError(f"unmocked variable: {var}")

    def set_input(self, var: str, period: int, values: np.ndarray) -> None:
        if var != "employment_income":
            raise NotImplementedError(f"only employment_income mocked, got {var}")
        # Capture head earnings so subsequent calculate() returns the right curve.
        head_values = np.asarray(values)[self._is_head]
        # All heads receive the same swept value in the loop, so all entries match.
        if not np.allclose(head_values, head_values[0]):
            raise AssertionError("expected all heads to have the same earnings")
        self._head_earnings = float(head_values[0])


class TestOverrideHeadEarnings:
    def test_only_heads_change(self) -> None:
        baseline = np.array([50_000.0, 30_000.0, 80_000.0, 0.0])
        is_head = np.array([True, False, True, False])
        out = _override_head_earnings(baseline, is_head, 250_000.0)

        assert out[0] == 250_000.0
        assert out[2] == 250_000.0
        assert out[1] == 30_000.0
        assert out[3] == 0.0

    def test_does_not_mutate_baseline(self) -> None:
        baseline = np.array([50_000.0, 30_000.0])
        is_head = np.array([True, False])
        before = baseline.copy()
        _override_head_earnings(baseline, is_head, 1_000.0)
        np.testing.assert_array_equal(baseline, before)


class TestRunEcpsSweep:
    @patch("policyengine_us.Microsimulation", new=_FakeMicrosim)
    def test_output_shapes(self) -> None:
        levels = np.array([0, 50_000, 100_000, 150_000], dtype=float)
        result = run_ecps_sweep(year=2026, earnings_levels=levels, progress=False)

        assert result.earnings_levels.shape == (4,)
        assert result.net_income.shape == (3, 4)  # 3 households × 4 levels
        assert result.income_tax.shape == (3, 4)
        assert result.household_weight.shape == (3,)
        assert result.state_code.shape == (3,)

    @patch("policyengine_us.Microsimulation", new=_FakeMicrosim)
    def test_different_earnings_produce_different_curves(self) -> None:
        levels = np.array([0, 50_000, 200_000], dtype=float)
        result = run_ecps_sweep(year=2026, earnings_levels=levels, progress=False)

        # Per row, net income should differ across columns.
        for hh in range(result.net_income.shape[0]):
            assert len(set(result.net_income[hh, :])) > 1

    @patch("policyengine_us.Microsimulation", new=_FakeMicrosim)
    def test_detects_cliff_in_swept_curve(self) -> None:
        # The fake simulator drops net income by $5,000 above $100,000.
        levels = np.array([0, 50_000, 100_000, 100_001], dtype=float)
        result = run_ecps_sweep(year=2026, earnings_levels=levels, progress=False)
        cliffs = result.cliffs(min_drop=100.0)

        assert len(cliffs) == 3  # one per household
        # The cliff drop should be ~$5,000 (the $0.70/$1 base income gain over
        # $1 of swept earnings is negligible against the $5,000 step).
        assert cliffs["cliff_drop"].iloc[0] == pytest.approx(5_000.0, rel=1e-3)
        assert (cliffs["cliff_earnings"] == 100_001).all()


class TestSweepOutputCliffs:
    def _make_output(self, net_income: np.ndarray) -> SweepOutput:
        n_hh, n_levels = net_income.shape
        return SweepOutput(
            earnings_levels=np.linspace(0, 1_000_000, n_levels),
            net_income=net_income,
            income_tax=np.zeros_like(net_income),
            household_weight=np.full(n_hh, 1.0),
            state_code=np.array(["NY"] * n_hh),
        )

    def test_no_cliffs_returns_empty_df(self) -> None:
        # Net income strictly increases — no cliffs.
        out = self._make_output(np.array([[0, 1000, 2000]], dtype=float))
        df = out.cliffs(min_drop=10.0)
        assert df.empty

    def test_returns_columns_in_expected_order(self) -> None:
        out = self._make_output(np.array([[0, -100, -50]], dtype=float))
        df = out.cliffs(min_drop=10.0)
        assert set(df.columns) == {
            "household_index",
            "household_weight",
            "state",
            "cliff_earnings",
            "cliff_step",
            "cliff_drop",
            "cliff_marginal_rate",
            "n_cliffs_detected",
        }

    def test_returns_largest_cliff_per_household(self) -> None:
        # Household 0: drops 100 then 200 — max = 200.
        # Household 1: monotonic — no cliff.
        ni = np.array(
            [
                [0, -100, -300],
                [0, 100, 200],
            ],
            dtype=float,
        )
        out = self._make_output(ni)
        df = out.cliffs(min_drop=10.0)
        assert len(df) == 1
        assert df["household_index"].iloc[0] == 0
        assert df["cliff_drop"].iloc[0] == pytest.approx(200.0)


@pytest.mark.integration
class TestRealEcps:
    """Integration test against the real Enhanced CPS — slow."""

    def test_smoke_three_levels(self) -> None:
        # Three earnings levels: clearly verifies the loop runs end-to-end.
        levels = np.array([0, 100_000, 25_000_001], dtype=float)
        result = run_ecps_sweep(
            year=2026,
            earnings_levels=levels,
            progress=False,
        )
        # Sanity: many ECPS households, three columns.
        assert result.net_income.shape[1] == 3
        assert result.net_income.shape[0] > 1000
        # Net income must vary across the three earnings levels for at least
        # most households.
        n_static = sum(
            len(set(result.net_income[h, :])) == 1
            for h in range(result.net_income.shape[0])
        )
        assert n_static / result.net_income.shape[0] < 0.5
