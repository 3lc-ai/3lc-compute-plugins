# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""A finished Image Metrics run stamps the preset it came from, so the preset list can say when it was used."""

from __future__ import annotations

from pathlib import Path

import pytest
from tlc_plugin_sdk.shared import config_store as sdk_config_store


def test_a_run_from_a_preset_stamps_its_last_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tlc_plugin_image_metrics import _stamp_last_run
    from tlc_plugin_image_metrics.config_store import ImageMetricsConfig, config_store

    monkeypatch.setattr(sdk_config_store, "CONFIG_ROOT", tmp_path)
    saved = config_store().save_config(ImageMetricsConfig(name="Quality check", metric_ids=["brightness"]))
    assert saved.last_run is None

    _stamp_last_run(saved.id)

    stamped = config_store().get_config(saved.id)
    assert stamped is not None and stamped.last_run


def test_no_preset_or_an_unknown_one_stamps_nothing_and_raises_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tlc_plugin_image_metrics import _stamp_last_run

    monkeypatch.setattr(sdk_config_store, "CONFIG_ROOT", tmp_path)
    _stamp_last_run("")
    _stamp_last_run("not-there")
    monkeypatch.setattr(sdk_config_store, "CONFIG_ROOT", None)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no home"))))
    _stamp_last_run("p1")  # a worker with no store: the run is not failed for it
