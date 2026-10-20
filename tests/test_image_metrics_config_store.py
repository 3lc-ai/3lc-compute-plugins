# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image Metrics saved configs: the home directory is read when a store is built, not at import."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from tlc_plugin_sdk.shared import config_store as sdk_config_store


def _no_home() -> Path:
    msg = "Could not determine home directory."
    raise RuntimeError(msg)


def test_the_module_imports_and_a_store_builds_without_a_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(_no_home))
    from tlc_plugin_image_metrics import config_store as module

    module = importlib.reload(module)  # import-time code runs again without a home
    monkeypatch.setattr(sdk_config_store, "CONFIG_ROOT", tmp_path / "configs")
    store = module.config_store()  # no legacy folder to look in; the SDK root is the override
    assert store.directory == tmp_path / "configs" / "image-metrics"
    monkeypatch.setattr(sdk_config_store, "CONFIG_ROOT", None)
    with pytest.raises(sdk_config_store.ConfigRootUnavailable):
        module.config_store()


def test_configs_in_the_legacy_folder_are_moved_on_first_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tlc_plugin_image_metrics import config_store as module

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    legacy = tmp_path / ".3lc-training" / "image-metrics-configs"
    legacy.mkdir(parents=True)
    (legacy / "c1.json").write_text(json.dumps({"id": "c1", "name": "old", "metric_ids": ["brightness"]}))
    monkeypatch.setattr(sdk_config_store, "CONFIG_ROOT", None)
    store = module.config_store()
    assert store.directory == tmp_path / ".3lc-plugin-configs" / "image-metrics"
    saved = store.get_config("c1")
    assert saved is not None and saved.name == "old" and saved.metric_ids == ["brightness"]
