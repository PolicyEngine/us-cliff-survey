"""Tests for analysis (post-sweep summarisation)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_cliff_survey.analysis import (
    cliff_prevalence,
    cliffs_by_state,
    render_findings,
    top_cliffs_by_size,
)


@pytest.fixture
def sample_cliffs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "household_index": 0,
                "household_weight": 100.0,
                "state": "NY",
                "cliff_earnings": 25_000_001,
                "cliff_step": 1,
                "cliff_drop": 149_953,
                "cliff_marginal_rate": 149953.0,
                "n_cliffs_detected": 1,
            },
            {
                "household_index": 1,
                "household_weight": 200.0,
                "state": "NJ",
                "cliff_earnings": 87_500,
                "cliff_step": 1_000,
                "cliff_drop": 6_500,
                "cliff_marginal_rate": 6.5,
                "n_cliffs_detected": 1,
            },
            {
                "household_index": 2,
                "household_weight": 50.0,
                "state": "CA",
                "cliff_earnings": 60_000,
                "cliff_step": 1_000,
                "cliff_drop": 800,
                "cliff_marginal_rate": 0.8,
                "n_cliffs_detected": 1,
            },
        ]
    )


class TestTopCliffsBySize:
    def test_orders_largest_first(self, sample_cliffs: pd.DataFrame) -> None:
        out = top_cliffs_by_size(sample_cliffs, n=3)
        assert list(out["state"]) == ["NY", "NJ", "CA"]
        assert list(out["cliff_drop"]) == [149_953, 6_500, 800]

    def test_respects_n_limit(self, sample_cliffs: pd.DataFrame) -> None:
        out = top_cliffs_by_size(sample_cliffs, n=2)
        assert len(out) == 2


class TestCliffPrevalence:
    def test_thresholds_apply(self, sample_cliffs: pd.DataFrame) -> None:
        prev = cliff_prevalence(sample_cliffs, thresholds=[100, 5_000, 100_000])
        assert list(prev["min_cliff_drop"]) == [100, 5_000, 100_000]
        assert list(prev["households"]) == [3, 2, 1]
        # Weighted counts: 100+200+50, 100+200, 100.
        assert list(prev["weighted_count"]) == [350.0, 300.0, 100.0]
        assert prev["weighted_share"].iloc[0] == pytest.approx(1.0)


class TestCliffsByState:
    def test_returns_largest_per_state_descending(
        self, sample_cliffs: pd.DataFrame
    ) -> None:
        out = cliffs_by_state(sample_cliffs, n_top=3)
        assert list(out["state"]) == ["NY", "NJ", "CA"]
        assert list(out["cliff_drop"]) == [149_953, 6_500, 800]

    def test_empty_input_returns_empty(self) -> None:
        empty = pd.DataFrame(
            columns=[
                "household_index",
                "household_weight",
                "state",
                "cliff_earnings",
                "cliff_step",
                "cliff_drop",
                "cliff_marginal_rate",
                "n_cliffs_detected",
            ]
        )
        assert cliffs_by_state(empty).empty


class TestRenderFindings:
    def test_writes_markdown_file(
        self, sample_cliffs: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = tmp_path / "findings.md"
        render_findings(sample_cliffs, out)
        text = out.read_text(encoding="utf-8")
        assert "Top 15 cliffs" in text
        assert "Largest cliff per state" in text
        assert "Population-weighted prevalence" in text
        # Sanity-check: the NY cliff drop appears.
        assert "149,953" in text
