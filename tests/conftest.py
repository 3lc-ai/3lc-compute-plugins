# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for compute-plugin tests."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

SRC = Path(__file__).resolve().parent.parent / "src"

PLUGIN_PACKAGES = [
    "tlc_plugin_importer",
    "tlc_plugin_exporter",
    "tlc_plugin_merger",
    "tlc_plugin_splitter",
    "tlc_plugin_table_statistics",
    "tlc_plugin_image_metrics",
]


@pytest.fixture(params=PLUGIN_PACKAGES)
def plugin_package(request: pytest.FixtureRequest) -> str:
    """Parametrize over every first-party plugin package name."""
    return request.param


@pytest.fixture()
def plugin_dir(plugin_package: str) -> Path:
    """Return the source directory for a plugin package."""
    return SRC / plugin_package


@pytest.fixture()
def manifest(plugin_dir: Path) -> dict[str, Any]:
    """Parse and return a plugin's ``plugin.toml`` manifest."""
    toml_path = plugin_dir / "plugin.toml"
    return tomllib.loads(toml_path.read_text(encoding="utf-8"))


@pytest.fixture()
def plugin_instance(plugin_package: str) -> Any:
    """Import the plugin module and instantiate its ComputePlugin subclass."""
    mod = importlib.import_module(plugin_package)
    manifest_path = SRC / plugin_package / "plugin.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    entrypoint = manifest["runtime"]["entrypoint"]
    _, class_name = entrypoint.split(":")
    cls = getattr(mod, class_name)
    return cls()
