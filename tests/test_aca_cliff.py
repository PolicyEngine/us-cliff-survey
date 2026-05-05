"""Tests for theoretical maximum ACA cliff calculation."""

from __future__ import annotations

from datetime import date

import pytest

from us_cliff_survey.aca_cliff import (
    DEFAULT_2026_FINAL_RATE_400_FPL,
    DEFAULT_PE_US_PARAMS,
    HouseholdComposition,
    _age_curve_multiplier,
    _value_at,
    all_cliffs,
    annual_slcsp,
    cliff_for,
    fpl_for,
    load_pe_us_params,
    max_base_monthly_per_state,
    maximum_cliff,
)

PE_US_AVAILABLE = DEFAULT_PE_US_PARAMS.exists()
needs_pe_us = pytest.mark.skipif(
    not PE_US_AVAILABLE,
    reason="PolicyEngine-US params not available locally",
)


class TestValueAt:
    def test_returns_latest_at_or_before(self) -> None:
        node = {date(2024, 1, 1): 1.0, date(2026, 1, 1): 3.0, date(2025, 1, 1): 2.0}
        assert _value_at(node, date(2026, 6, 1)) == 3.0
        assert _value_at(node, date(2025, 6, 1)) == 2.0
        assert _value_at(node, date(2024, 1, 1)) == 1.0

    def test_returns_none_for_too_early(self) -> None:
        node = {date(2026, 1, 1): 5.0}
        assert _value_at(node, date(2025, 1, 1)) is None


class TestAgeCurveMultiplier:
    def _curve(self) -> dict:
        # Two brackets: 0 -> 1.0, 64 -> 3.92
        return {
            "brackets": [
                {
                    "threshold": {date(2018, 1, 1): 0},
                    "amount": {date(2018, 1, 1): 1.0},
                },
                {
                    "threshold": {date(2018, 1, 1): 64},
                    "amount": {date(2018, 1, 1): 3.92},
                },
            ]
        }

    def test_below_first_bracket_returns_default_one(self) -> None:
        # No bracket below the lookup; the function returns 1.0.
        c = self._curve()
        assert _age_curve_multiplier(0, c, date(2026, 1, 1)) == 1.0

    def test_picks_largest_threshold_at_or_below_age(self) -> None:
        c = self._curve()
        assert _age_curve_multiplier(63, c, date(2026, 1, 1)) == 1.0
        assert _age_curve_multiplier(64, c, date(2026, 1, 1)) == 3.92
        assert _age_curve_multiplier(70, c, date(2026, 1, 1)) == 3.92


@needs_pe_us
class TestLoadParams:
    def test_loads_required_files(self) -> None:
        params = load_pe_us_params()
        assert "rating" in params
        assert "age_curves" in params
        assert "family_tier" in params
        assert "fpg" in params
        assert "default" in params["age_curves"]
        assert "AL" not in params["age_curves"]  # lowercase keys
        assert "al" in params["age_curves"]


@needs_pe_us
class TestFplFor:
    def test_2026_2person_contiguous(self) -> None:
        params = load_pe_us_params()
        # 2026 contiguous: $15,960 + 1 * $5,680 = $21,640
        assert fpl_for("CA", 2, params["fpg"], date(2026, 1, 1)) == 21_640

    def test_2026_5person_contiguous(self) -> None:
        params = load_pe_us_params()
        # $15,960 + 4 * $5,680 = $38,680
        assert fpl_for("IL", 5, params["fpg"], date(2026, 1, 1)) == 38_680

    def test_2026_2person_alaska_higher(self) -> None:
        params = load_pe_us_params()
        # AK: $19,950 + $7,100 = $27,050
        assert fpl_for("AK", 2, params["fpg"], date(2026, 1, 1)) == 27_050


@needs_pe_us
class TestMaxBaseMonthlyPerState:
    def test_returns_one_per_state(self) -> None:
        params = load_pe_us_params()
        out = max_base_monthly_per_state(params["rating"], date(2026, 1, 1))
        assert len(out) > 50  # all states + DC
        assert "IL" in out
        # The IL max base monthly in 2026 is $789 in rating area 13.
        ra, base = out["IL"]
        assert base == 789.0
        assert ra == 13


@needs_pe_us
class TestAnnualSlcsp:
    def test_2_adults_64_il(self) -> None:
        params = load_pe_us_params()
        comp = HouseholdComposition(n_adults_age_64=2, children_ages=())
        result = annual_slcsp(
            "IL",
            789.0,
            comp,
            params["age_curves"],
            params["family_tier"],
            date(2026, 1, 1),
        )
        # 2 * 789 * 3.9216 * 12 = 74,259.4...
        assert result == pytest.approx(74_259, abs=2)

    def test_2_adults_64_plus_3_kids_il(self) -> None:
        params = load_pe_us_params()
        # Children age 14 -> default age curve multiplier 1.0.
        comp = HouseholdComposition(n_adults_age_64=2, children_ages=(14, 14, 14))
        result = annual_slcsp(
            "IL",
            789.0,
            comp,
            params["age_curves"],
            params["family_tier"],
            date(2026, 1, 1),
        )
        # 12 * 789 * (2 * 3.9216 + 3 * 1.0) = 102,663.42
        assert result == pytest.approx(102_663, abs=2)

    def test_caps_children_at_3(self) -> None:
        params = load_pe_us_params()
        # Five children all age 14; only 3 should count.
        comp_3 = HouseholdComposition(n_adults_age_64=2, children_ages=(14, 14, 14))
        comp_5 = HouseholdComposition(
            n_adults_age_64=2, children_ages=(14, 14, 14, 14, 14)
        )
        a = annual_slcsp(
            "IL",
            789.0,
            comp_3,
            params["age_curves"],
            params["family_tier"],
            date(2026, 1, 1),
        )
        b = annual_slcsp(
            "IL",
            789.0,
            comp_5,
            params["age_curves"],
            params["family_tier"],
            date(2026, 1, 1),
        )
        assert a == b


@needs_pe_us
class TestMaximumCliff:
    def test_2026_max_is_il_with_2a_3kids(self) -> None:
        target = date(2026, 1, 1)
        result = maximum_cliff(target=target)
        assert result.state == "IL"
        assert result.rating_area == 13
        # 2 adults at 64 + 3 kids age 20 (1.268 mult) gives ~$94,866.
        # The default "standard" compositions include kid_age 14 and 20;
        # age 20 wins. Verify cliff is in the right neighbourhood.
        assert 87_000 <= result.cliff <= 95_000
        assert result.composition_label.startswith("2A64+3K")

    def test_il_2a_3kids_age14_matches_codex_verification(self) -> None:
        """Codex independently verified $87,253 for this exact config."""
        target = date(2026, 1, 1)
        comp = HouseholdComposition(n_adults_age_64=2, children_ages=(14, 14, 14))
        params = load_pe_us_params()
        result = cliff_for(
            state="IL",
            rating_area=13,
            base_monthly=789.0,
            composition=comp,
            composition_label="2A64+3K14",
            age_curves=params["age_curves"],
            family_tier=params["family_tier"],
            fpg=params["fpg"],
            final_rate_400_fpl=DEFAULT_2026_FINAL_RATE_400_FPL,
            target=target,
        )
        assert result.cliff == pytest.approx(87_253, abs=2)


@needs_pe_us
class TestAllCliffs:
    def test_returns_descending(self) -> None:
        results = all_cliffs(target=date(2026, 1, 1))
        cliffs = [r.cliff for r in results]
        assert cliffs == sorted(cliffs, reverse=True)

    def test_top_state_is_il(self) -> None:
        results = all_cliffs(target=date(2026, 1, 1))
        assert results[0].state == "IL"
