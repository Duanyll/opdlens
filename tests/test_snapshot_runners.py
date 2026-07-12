"""The batch runner must execute the code recorded in each job's Git snapshot."""

from pathlib import Path

from opdlens.utils import logging as logmod

_ROOT = Path(__file__).parents[1]
_RUNNER = _ROOT / "scripts" / "run.sbatch"


def test_runner_prevents_checkout_from_shadowing_snapshot() -> None:
    script = _RUNNER.read_text(encoding="utf-8")

    assert "PYTHONSAFEPATH=1" in script
    assert 'PYTHONPATH="$SNAPSHOT"' in script
    assert "python -P -m opdlens.scripts.cli launch" in script
    assert "  opdlens launch" not in script


def test_runner_exports_snapshot_commit_for_tracking() -> None:
    script = _RUNNER.read_text(encoding="utf-8")

    assert 'OPDLENS_EXPERIMENT_COMMIT="$COMMIT"' in script


def test_get_version_records_pinned_experiment_commit(monkeypatch) -> None:
    # get_version() feeds the run's recorded ``opdlens_version``; under the
    # snapshot launch workflow it must report the pinned commit, not repo HEAD.
    monkeypatch.setattr(logmod, "_version_cache", None)
    monkeypatch.setenv("OPDLENS_EXPERIMENT_COMMIT", "0123abc")
    version = logmod.get_version()

    assert version.endswith("+git.0123abc")
    assert ".wip" not in version


def test_runner_uses_persistent_gds_cache_with_tmp_fallback() -> None:
    script = _RUNNER.read_text(encoding="utf-8")

    # Persistent per-GPU-pair NVMe cache, ephemeral /tmp fallback, never NFS.
    assert "/gds/gpu" in script
    assert 'CACHE_ROOT="/tmp/opdlens-cache-$SLURM_JOB_ID"' in script
    assert 'VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"' in script
    assert 'TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"' in script
