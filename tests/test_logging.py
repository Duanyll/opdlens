from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from opdlens.training.mixins import logging as logging_module
from opdlens.training.mixins.logging import LoggingMixin


@pytest.mark.parametrize("enabled", [False, True])
def test_trackio_system_metrics_switch(tmp_path, monkeypatch, enabled: bool) -> None:
    init_kwargs: dict[str, object] = {}
    fake_trackio = SimpleNamespace(
        init=lambda **kwargs: init_kwargs.update(kwargs),
        finish=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "trackio", fake_trackio)
    monkeypatch.setattr(logging_module, "get_version", lambda: "test-version")
    monkeypatch.delenv("TRACKIO_DIR", raising=False)

    trainer = LoggingMixin(
        experiment_name="logging-test",
        run_id="run-1",
        runs_root=str(tmp_path),
        trackio_system_metrics=enabled,
    )
    trainer.init_tracker()
    trainer.finish_tracker()

    assert init_kwargs["auto_log_cpu"] is enabled
    assert init_kwargs["auto_log_gpu"] is enabled
