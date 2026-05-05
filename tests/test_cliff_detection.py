"""Tests for cliff_detection.

Cliff = a drop in net_income across an earnings step. This module finds
cliffs, ranks them, and bisects to localize a known cliff to ~$1.
"""

from __future__ import annotations

import numpy as np
import pytest

from us_cliff_survey.cliff_detection import (
    Cliff,
    find_cliffs,
    max_cliff,
    refine_cliff,
)


class TestFindCliffs:
    def test_monotonic_increasing_returns_empty(self) -> None:
        earnings = np.array([0, 1000, 2000, 3000], dtype=float)
        net_income = np.array([0, 800, 1600, 2400], dtype=float)
        assert find_cliffs(earnings, net_income) == []

    def test_flat_net_income_returns_empty(self) -> None:
        earnings = np.array([0, 1000, 2000], dtype=float)
        net_income = np.array([5000, 5000, 5000], dtype=float)
        assert find_cliffs(earnings, net_income) == []

    def test_single_cliff_detected(self) -> None:
        earnings = np.array([0, 1000, 2000, 3000], dtype=float)
        net_income = np.array([0, 800, 1600, -10000], dtype=float)
        cliffs = find_cliffs(earnings, net_income)

        assert len(cliffs) == 1
        cliff = cliffs[0]
        assert cliff.earnings_at_cliff == 3000
        assert cliff.earnings_step == 1000
        assert cliff.net_income_drop == 11600
        assert cliff.implied_marginal_rate == pytest.approx(11.6)

    def test_multiple_cliffs_sorted_by_drop_desc(self) -> None:
        earnings = np.array([0, 100, 200, 300, 400], dtype=float)
        # Drop of 50 between 100→200, drop of 200 between 300→400.
        net_income = np.array([0, 50, 0, 50, -150], dtype=float)
        cliffs = find_cliffs(earnings, net_income)

        assert len(cliffs) == 2
        assert cliffs[0].net_income_drop == 200
        assert cliffs[0].earnings_at_cliff == 400
        assert cliffs[1].net_income_drop == 50
        assert cliffs[1].earnings_at_cliff == 200

    def test_min_drop_filters_small_cliffs(self) -> None:
        earnings = np.array([0, 100, 200], dtype=float)
        net_income = np.array([0, -10, -20], dtype=float)

        assert find_cliffs(earnings, net_income, min_drop=0) == [
            Cliff(100, 100, 10, 0.1),
            Cliff(200, 100, 10, 0.1),
        ]
        assert find_cliffs(earnings, net_income, min_drop=15) == []

    def test_mismatched_shapes_raise(self) -> None:
        earnings = np.array([0, 100], dtype=float)
        net_income = np.array([0, 100, 200], dtype=float)
        with pytest.raises(ValueError, match="same shape"):
            find_cliffs(earnings, net_income)

    def test_unsorted_earnings_raise(self) -> None:
        earnings = np.array([0, 200, 100], dtype=float)
        net_income = np.array([0, 100, 50], dtype=float)
        with pytest.raises(ValueError, match="sorted ascending"):
            find_cliffs(earnings, net_income)

    def test_2d_earnings_raise(self) -> None:
        earnings = np.array([[0, 100], [200, 300]], dtype=float)
        net_income = np.array([[0, 100], [200, 300]], dtype=float)
        with pytest.raises(ValueError, match="1-D"):
            find_cliffs(earnings, net_income)

    def test_non_strict_increasing_earnings_allowed(self) -> None:
        # Equal earnings allowed (still ascending). Step = 0 → marginal rate = inf.
        earnings = np.array([0, 100, 100, 200], dtype=float)
        net_income = np.array([0, 100, 50, 100], dtype=float)
        cliffs = find_cliffs(earnings, net_income)
        assert len(cliffs) == 1
        assert cliffs[0].earnings_step == 0
        assert cliffs[0].net_income_drop == 50
        assert cliffs[0].implied_marginal_rate == np.inf


class TestCliffIsCliff:
    def test_drop_exceeds_step_is_cliff(self) -> None:
        cliff = Cliff(
            earnings_at_cliff=25_000_002,
            earnings_step=2,
            net_income_drop=149_953,
            implied_marginal_rate=74_976.5,
        )
        assert cliff.is_cliff is True

    def test_drop_below_step_is_not_cliff(self) -> None:
        # Marginal tax rate of 50% — a drop, but earning $1 more nets you 50¢.
        cliff = Cliff(
            earnings_at_cliff=200_000,
            earnings_step=1000,
            net_income_drop=500,
            implied_marginal_rate=0.5,
        )
        assert cliff.is_cliff is False

    def test_drop_equals_step_is_not_cliff(self) -> None:
        cliff = Cliff(
            earnings_at_cliff=100_000,
            earnings_step=1000,
            net_income_drop=1000,
            implied_marginal_rate=1.0,
        )
        assert cliff.is_cliff is False


class TestMaxCliff:
    def test_no_cliff_returns_none(self) -> None:
        earnings = np.array([0, 100, 200], dtype=float)
        net_income = np.array([0, 100, 200], dtype=float)
        assert max_cliff(earnings, net_income) is None

    def test_returns_largest_cliff(self) -> None:
        earnings = np.array([0, 100, 200, 300], dtype=float)
        net_income = np.array([0, -50, -10, -200], dtype=float)
        # Drops: 50, 0, 190 → largest = 190.
        result = max_cliff(earnings, net_income)
        assert result is not None
        assert result.net_income_drop == 190
        assert result.earnings_at_cliff == 300


class TestRefineCliff:
    def test_lo_must_be_less_than_hi(self) -> None:
        with pytest.raises(ValueError, match="lo must be < hi"):
            refine_cliff(lambda x: x, 100, 100)
        with pytest.raises(ValueError, match="lo must be < hi"):
            refine_cliff(lambda x: x, 200, 100)

    def test_no_cliff_raises(self) -> None:
        # Net income increases — no cliff to find.
        with pytest.raises(ValueError, match="no cliff"):
            refine_cliff(lambda x: x, 100, 200)

    def test_localizes_step_function_cliff(self) -> None:
        # Step-only cliff: net income drops by $150,000 at exactly $25M.
        # We use a step (no underlying earnings rate) so bisection's
        # preconditions hold across a wide search range.
        threshold = 25_000_000.0
        drop = 150_000.0

        def sweep(earnings: float) -> float:
            return 0.0 if earnings <= threshold else -drop

        cliff = refine_cliff(sweep, lo=0.0, hi=30_000_000.0, tol=10.0)
        assert cliff.earnings_step <= 10.0
        assert threshold < cliff.earnings_at_cliff <= threshold + 11
        assert cliff.net_income_drop == pytest.approx(drop, rel=1e-9)
        assert cliff.is_cliff is True

    def test_caller_must_bracket_a_real_cliff(self) -> None:
        # If the underlying earnings rate dominates the local cliff over
        # [lo, hi], the bisection precondition (ni_lo > ni_hi) fails and
        # the user should narrow the range. We document that here.
        threshold = 25_000_000.0
        drop = 150_000.0

        def sweep(earnings: float) -> float:
            return earnings * 0.5 - (drop if earnings > threshold else 0.0)

        with pytest.raises(ValueError, match="no cliff"):
            refine_cliff(sweep, lo=24_000_000.0, hi=26_000_000.0)

    def test_respects_max_iter(self) -> None:
        # An infinitely fine cliff: no matter how small the step, there's
        # still a discontinuity. Verify max_iter caps the loop.
        def sweep(x: float) -> float:
            return -1.0 if x > 0 else 1.0

        cliff = refine_cliff(sweep, lo=-1e6, hi=1e6, tol=0.0, max_iter=10)
        assert cliff.earnings_step > 0  # didn't fully converge
        assert cliff.net_income_drop == pytest.approx(2.0)
