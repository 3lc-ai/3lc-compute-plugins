# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Importer references existing media; obsolete requests to relocate it fail explicitly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tlc_plugin_sdk.shared import aliases

import tlc_plugin_importer as imp


class _Ctx:
    def __init__(self) -> None:
        self.logs: list[str] = []
        self.progress_calls: list[dict[str, Any]] = []

    def log(self, msg: str) -> None:
        self.logs.append(msg)

    def progress(self, **kw: Any) -> None:
        self.progress_calls.append(kw)


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def copy(src: str, dst: str, *, progress: Any = None, workers: int = 8) -> dict[str, Any]:
        calls["copy"] = (src, dst)
        if progress:
            progress(1, 2, 5, 10)
            progress(2, 2, 10, 10)
        return {"files": 2, "bytes": 10, "skipped": 1, "url": dst}

    def register(**kw: Any) -> dict[str, Any]:
        calls["register"] = kw
        return {"token": kw["alias_token"], "path": kw["image_folder"], "primary_created": True}

    monkeypatch.setattr(aliases, "copy_folder_to_url", copy)
    monkeypatch.setattr(aliases, "register_alias", register)
    return calls


@pytest.mark.parametrize("enabled", [True, "true", "1"])
def test_legacy_copy_request_fails_before_alias_or_copy(fakes, enabled):
    with pytest.raises(imp.JobFailed, match="Import no longer copies source data"):
        imp._maybe_register_alias(
            {"project_name": "Fire", "alias_copy_to_root": enabled, "alias_copy_target": "s3://b/images"},
            "/data/fire",
            _Ctx(),
        )
    assert not fakes


def test_no_copy_without_the_checkbox(fakes: dict[str, Any]) -> None:
    base = {"project_name": "Fire", "alias_folder": "/data/fire", "alias_copy_target": "s3://b/p/Fire/data/fire"}
    imp._maybe_register_alias({**base, "alias_copy_to_root": "false"}, "/data/fire", _Ctx())
    assert "copy" not in fakes and fakes["register"]["remote_path"] is None


def test_alias_disabled_means_nothing_happens(fakes: dict[str, Any]) -> None:
    assert imp._maybe_register_alias({"project_name": "Fire", "alias_enabled": "false"}, "/data/fire", _Ctx()) is None
    assert not fakes


def test_regular_and_csv_forms_do_not_request_copies() -> None:
    ui = (Path(imp.__file__).parent / "ui.html").read_text()
    assert "alias_copy" not in ui and "copyOffer" not in ui
    assert "payload.alias_folder = csvAlias.alias_folder;" in ui


def test_the_project_root_choice_reaches_the_writers_and_the_forms() -> None:
    """ "Create project in" (this computer or the bucket root) — the one field that decides where the table lands."""
    assert imp._root({"project_root_url": " s3://b/projects/ "}) == "s3://b/projects"
    assert imp._root({}) is None and imp._root({"project_root_url": ""}) is None
    assert "project_root_url" in imp._PATH_FIELDS  # normalised like every other location (URL or absolute path)
    assert imp._normalize_path_fields({"project_root_url": "s3://b/projects/"})["project_root_url"] == "s3://b/projects"
    src = Path(imp.__file__).read_text(encoding="utf-8")
    assert src.count("root_url=_root(form_data),") == 6  # yolo, coco, folder, unlabeled, csv-detection, the alias
    assert "root_url=project_root_url or None," in src  # CSV create-new
    ui = (Path(imp.__file__).parent / "ui.html").read_text(encoding="utf-8")
    assert ui.count("_tlcProjectLocationHtml(") == 2 and ui.count("_tlcBindProjectLocation(") == 2
    assert "formData.project_root_url = _tlcGetProjectRoot('import');" in ui
    assert "payload.project_root_url = _tlcGetProjectRoot('csv');" in ui


def test_custom_alias_points_to_source_without_copy(fakes):
    imp._maybe_register_alias(
        {
            "project_name": "Fire",
            "alias_token": "MY_IMAGES",
            "alias_folder": "s3://b/images",
            "project_root_url": "s3://b/projects/",
        },
        "s3://b/images",
        _Ctx(),
    )
    assert "copy" not in fakes
    assert fakes["register"]["alias_token"] == "MY_IMAGES"
    assert fakes["register"]["image_folder"] == "s3://b/images"
    assert fakes["register"]["root_url"] == "s3://b/projects"
    assert fakes["register"]["remote_path"] is None


def test_csv_alias_is_checked_before_table_writer(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "tlc",
        SimpleNamespace(
            schemas=SimpleNamespace(ImageSchema=lambda **kw: object()),
            TableWriter=lambda **kw: calls.append("writer"),
        ),
    )
    monkeypatch.setattr(imp, "_read_spreadsheet", lambda *a: (["image"], [["/data/images/a.jpg"]]))

    def refuse(*args, **kwargs):
        msg = "alias would point to a local disk"
        raise imp.JobFailed(msg)

    monkeypatch.setattr(imp, "_maybe_register_alias", refuse)
    with pytest.raises(imp.JobFailed, match="local disk"):
        imp._execute_csv_new(
            b"",
            "images.csv",
            [{"name": "image", "index": 0, "type": "image_url"}],
            "Project",
            "Dataset",
            "initial",
            "",
            project_root_url="s3://b/projects",
        )
    assert not calls
