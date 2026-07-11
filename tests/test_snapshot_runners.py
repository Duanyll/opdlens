"""Batch runners must execute the code recorded in each job's Git snapshot."""

from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("runner", ("run_round1.sh", "run_dapo_math.sh"))
def test_runner_prevents_checkout_from_shadowing_snapshot(runner: str) -> None:
    script = (_ROOT / "scripts" / runner).read_text(encoding="utf-8")

    assert 'PYTHONPATH="$SNAPSHOT"' in script
    assert "python -P -m opdlens.scripts.cli launch" in script
    assert "  opdlens launch" not in script
