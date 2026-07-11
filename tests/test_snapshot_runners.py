"""Batch runners must execute the code recorded in each job's Git snapshot."""

from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("runner", ("run_round1.sh", "run_dapo_math.sh"))
def test_runner_prevents_checkout_from_shadowing_snapshot(runner: str) -> None:
    script = (_ROOT / "scripts" / runner).read_text(encoding="utf-8")

    assert "PYTHONSAFEPATH=1" in script
    assert 'PYTHONPATH="$SNAPSHOT"' in script
    assert "python -P -m opdlens.scripts.cli launch" in script
    assert "  opdlens launch" not in script


def test_dapo_runner_uses_job_local_compiler_caches() -> None:
    script = (_ROOT / "scripts" / "run_dapo_math.sh").read_text(encoding="utf-8")

    assert "LOCAL_CACHE=/tmp/opdlens-cache-$SLURM_JOB_ID" in script
    assert 'VLLM_CACHE_ROOT="$LOCAL_CACHE/vllm"' in script
    assert 'TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE/torchinductor"' in script
