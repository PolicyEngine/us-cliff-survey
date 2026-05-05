"""Tests for the CLI entry points."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import pandas as pd

from us_cliff_survey import cli, ecps_sweep


class TestSweepPopulationCli:
    def test_quick_uses_short_grid(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured = {}

        def fake_run(year: int, earnings_levels: np.ndarray, **kwargs):
            captured["year"] = year
            captured["levels"] = earnings_levels
            return ecps_sweep.SweepOutput(
                earnings_levels=earnings_levels,
                net_income=np.zeros((1, len(earnings_levels)), dtype=np.float32),
                income_tax=np.zeros((1, len(earnings_levels)), dtype=np.float32),
                household_weight=np.array([1.0]),
                state_code=np.array(["NY"]),
            )

        monkeypatch.setattr(ecps_sweep, "run_ecps_sweep", fake_run)
        monkeypatch.setattr(cli, "run_ecps_sweep", fake_run)
        # save_outputs writes h5+parquet; bypass file IO in this test.
        monkeypatch.setattr(cli, "save_outputs", lambda *a, **kw: None)

        out_prefix = tmp_path / "smoke"
        argv = [
            "sweep-population",
            "--quick",
            "--output",
            str(out_prefix),
            "--year",
            "2026",
        ]
        with patch.object(sys, "argv", argv):
            cli.sweep_population_cli()

        assert captured["year"] == 2026
        # The quick grid is short — under 20 levels.
        assert len(captured["levels"]) < 20

    def test_full_uses_default_grid(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured = {}

        def fake_run(year: int, earnings_levels: np.ndarray, **kwargs):
            captured["levels"] = earnings_levels
            return ecps_sweep.SweepOutput(
                earnings_levels=earnings_levels,
                net_income=np.zeros((1, len(earnings_levels)), dtype=np.float32),
                income_tax=np.zeros((1, len(earnings_levels)), dtype=np.float32),
                household_weight=np.array([1.0]),
                state_code=np.array(["NY"]),
            )

        monkeypatch.setattr(ecps_sweep, "run_ecps_sweep", fake_run)
        monkeypatch.setattr(cli, "run_ecps_sweep", fake_run)
        monkeypatch.setattr(cli, "save_outputs", lambda *a, **kw: None)

        out_prefix = tmp_path / "full"
        argv = ["sweep-population", "--output", str(out_prefix)]
        with patch.object(sys, "argv", argv):
            cli.sweep_population_cli()

        assert np.array_equal(captured["levels"], ecps_sweep.DEFAULT_EARNINGS_LEVELS)


class TestAnalyzeCli:
    def test_writes_findings_markdown(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Stand-in cliffs parquet so analyze_cli can render findings.
        prefix = tmp_path / "smoke"
        cliffs = pd.DataFrame(
            [
                {
                    "household_index": 0,
                    "household_weight": 100.0,
                    "state": "NY",
                    "cliff_earnings": 25_000_001,
                    "cliff_step": 1,
                    "cliff_drop": 149_953,
                    "cliff_marginal_rate": 149_953.0,
                    "n_cliffs_detected": 1,
                },
            ]
        )
        cliffs.to_parquet(f"{prefix}_cliffs.parquet", index=False)

        out = tmp_path / "findings.md"
        argv = ["cliff-analyze", "--prefix", str(prefix), "--output", str(out)]
        with patch.object(sys, "argv", argv):
            cli.analyze_cli()

        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "149,953" in text
        assert "NY" in text
