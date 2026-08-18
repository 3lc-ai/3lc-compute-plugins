# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate that every plugin.toml manifest is well-formed and consistent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

SRC = Path(__file__).resolve().parent.parent / "src"

REQUIRED_TOP_LEVEL = {"id", "name", "description", "version", "min_service_version"}
REQUIRED_RUNTIME = {"isolation", "entrypoint"}


def test_every_plugin_has_manifest(plugin_package: str, plugin_dir: Path) -> None:
    assert (plugin_dir / "plugin.toml").is_file(), f"{plugin_package} is missing plugin.toml"


def test_required_fields(manifest: dict[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL - manifest.keys()
    assert not missing, f"Missing required fields: {missing}"


def test_runtime_section(manifest: dict[str, Any]) -> None:
    assert "runtime" in manifest, "Missing [runtime] section"
    missing = REQUIRED_RUNTIME - manifest["runtime"].keys()
    assert not missing, f"Missing runtime fields: {missing}"


def test_ui_section(manifest: dict[str, Any]) -> None:
    assert "ui" in manifest, "Missing [ui] section"
    assert "display_mode" in manifest["ui"]


def test_entrypoint_resolves(manifest: dict[str, Any]) -> None:
    entrypoint = manifest["runtime"]["entrypoint"]
    module_path, class_name = entrypoint.split(":")
    assert module_path, "Empty module path in entrypoint"
    assert class_name, "Empty class name in entrypoint"


def test_version_is_semver(manifest: dict[str, Any]) -> None:
    version = manifest["version"]
    assert re.match(r"^\d+\.\d+\.\d+$", version), f"Version {version!r} is not semver"


def test_ids_match_entry_points() -> None:
    """Manifest id fields must match the pyproject entry-point keys."""
    pyproject = SRC.parent / "pyproject.toml"
    pyproject_data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    entry_points = pyproject_data["project"]["entry-points"]["tlc_compute.plugins"]

    for ep_key, ep_package in entry_points.items():
        manifest_path = SRC / ep_package / "plugin.toml"
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        expected_id = ep_key.replace("_", "-")
        assert manifest["id"] == expected_id or manifest["id"] == ep_key, (
            f"Entry-point {ep_key!r} maps to package {ep_package!r} but manifest id is {manifest['id']!r}"
        )
