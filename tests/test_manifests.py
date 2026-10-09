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


# The run-body keys each plugin declares as the data it reads and writes. A table URL among the
# values is planned as a table (its aliases); anything else is a folder, file or URL reference.
DATA_KEYS: dict[str, tuple[list[str], list[str]]] = {
    "tlc_plugin_importer": (
        ["dataset_yaml", "annotations_file", "image_folder", "folder_path", "csv_path", "table_url"],
        [],
    ),
    "tlc_plugin_exporter": (["table_url"], ["output_path"]),
    "tlc_plugin_merger": (["table_urls"], []),
    "tlc_plugin_splitter": (["table_url"], []),
    "tlc_plugin_image_metrics": (["table_url"], []),
    # Statistics are served through /compute, not a run body: nothing to plan.
    "tlc_plugin_table_statistics": ([], []),
}


def test_data_keys_are_lists_of_dotted_keys(manifest: dict[str, Any]) -> None:
    for name in ("data_inputs", "data_outputs"):
        keys = manifest["runtime"].get(name, [])
        assert isinstance(keys, list), f"runtime.{name} must be a list"
        for key in keys:
            assert isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", key), (
                f"runtime.{name} entry {key!r} is not a dotted run-body key"
            )


def test_declared_data_keys(plugin_package: str, manifest: dict[str, Any]) -> None:
    inputs, outputs = DATA_KEYS[plugin_package]
    assert manifest["runtime"].get("data_inputs", []) == inputs
    assert manifest["runtime"].get("data_outputs", []) == outputs


def test_importer_form_fields_that_name_data_are_declared() -> None:
    """Every importer form field the person points at data is a declared input — a new one must be added."""
    import tlc_plugin_importer as imp

    declared = set(DATA_KEYS["tlc_plugin_importer"][0])
    for step in imp.IMPORT_STEPS.values():
        for field in step["form_fields"]:
            if field.get("type") == "data_source":
                assert field["id"] in declared, f"{step['name']}: {field['id']} reads data but is not declared"
