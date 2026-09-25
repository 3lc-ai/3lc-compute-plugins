# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Importer: when the table lands on a bucket and the data is local, copy the data first and alias the copy."""

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


def test_copy_then_alias_the_copy(fakes: dict[str, Any]) -> None:
    ctx = _Ctx()
    form = {
        "project_name": "Fire",
        "alias_token": "FIRE",
        "alias_folder": "/data/fire",
        "alias_copy_to_root": "true",
        "alias_copy_target": "s3://b/projects/Fire/data/fire",
        "project_root_url": "s3://b/projects/",
    }
    out = imp._maybe_register_alias(form, "/data/fire", ctx)
    assert fakes["copy"] == ("/data/fire", "s3://b/projects/Fire/data/fire")
    assert fakes["register"]["remote_path"] == "s3://b/projects/Fire/data/fire"
    assert fakes["register"]["root_url"] == "s3://b/projects"  # the alias lives in the project ON THE BUCKET
    assert fakes["register"]["image_folder"] == "/data/fire"  # this session still encodes from the local files
    assert out and out["token"] == "FIRE"
    # The job panel shows the copy as a real percentage, then the log says where the data went.
    assert [c["percent"] for c in ctx.progress_calls] == [50, 100]
    assert "Copying data to b…" in ctx.progress_calls[0]["label"]
    assert any("Copied 2 files" in m and "1 were already there" in m and "<FIRE> points there" in m for m in ctx.logs)


def test_no_copy_without_the_checkbox_or_with_a_local_target(fakes: dict[str, Any]) -> None:
    base = {"project_name": "Fire", "alias_folder": "/data/fire", "alias_copy_target": "s3://b/p/Fire/data/fire"}
    imp._maybe_register_alias({**base, "alias_copy_to_root": "false"}, "/data/fire", _Ctx())
    assert "copy" not in fakes and fakes["register"]["remote_path"] is None
    fakes.clear()
    imp._maybe_register_alias(
        {**base, "alias_copy_to_root": "true", "alias_copy_target": "/other/disk"}, "/data/fire", _Ctx()
    )
    assert "copy" not in fakes and fakes["register"]["remote_path"] is None  # not a bucket: ignored


def test_alias_disabled_means_nothing_happens(fakes: dict[str, Any]) -> None:
    assert imp._maybe_register_alias({"project_name": "Fire", "alias_enabled": "false"}, "/data/fire", _Ctx()) is None
    assert not fakes


def test_the_form_and_csv_bodies_carry_the_copy_fields() -> None:
    from pathlib import Path

    ui = (Path(imp.__file__).parent / "ui.html").read_text(encoding="utf-8")
    for name in ("alias_copy_to_root", "alias_copy_target"):  # the import form body and the CSV payload
        assert ui.count("formData." + name) == 1 and ui.count("payload." + name) == 1, name
    # The widget asks THIS plugin where its tables land (found live: the infra plugin's bucket root was
    # shown while the table went to the local projects folder).
    assert ui.count(", 'importer');") == 2


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
    assert "'import-project-root');" in ui and "'csv-project-root');" in ui
    assert "formData.project_root_url = _tlcGetProjectRoot('import');" in ui
    assert "payload.project_root_url = _tlcGetProjectRoot('csv');" in ui
