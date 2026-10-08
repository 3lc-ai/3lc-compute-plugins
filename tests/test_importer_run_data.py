# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Importer: sources are checked before anything acts on them, and a rewritten source keeps the alias durable."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from tlc_plugin_sdk import JobContext, JobFailed
from tlc_plugin_sdk.shared import aliases

import tlc_plugin_importer as imp

UI = (Path(imp.__file__).parent / "ui.html").read_text(encoding="utf-8")


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def copy(src: str, dst: str, *, progress: Any = None, workers: int = 8) -> dict[str, Any]:
        calls["copy"] = (src, dst)
        return {"files": 1, "bytes": 1, "skipped": 0, "url": dst}

    def register(**kw: Any) -> dict[str, Any]:
        calls["register"] = kw
        return {"token": kw["alias_token"], "path": kw["image_folder"], "primary_created": False}

    monkeypatch.setattr(aliases, "copy_folder_to_url", copy)
    monkeypatch.setattr(aliases, "register_alias", register)
    return calls


def _ctx(params: dict[str, Any], tmp_path: Path) -> JobContext:
    return JobContext("job-1", params, tmp_path, sink=lambda _e: None, cancel_event=threading.Event())


def _folder_form(folder: str, **extra: Any) -> dict[str, Any]:
    return {
        "format": "folder",
        "folder_path": folder,
        "project_name": "Fire",
        "dataset_name": "fire",
        "alias_token": "FIRE",
        "alias_folder": folder,
        **extra,
    }


def test_missing_source_fails_before_the_alias_or_the_copy(fakes: dict[str, Any], tmp_path: Path) -> None:
    missing = str(tmp_path / "nowhere")
    form = _folder_form(missing, alias_copy_to_root="true", alias_copy_target="s3://b/p/Fire/data/fire")
    with pytest.raises(JobFailed, match=r"Folder Path '.*nowhere' was not found, or cannot be read, on "):
        imp._run_format_import(_ctx(form, tmp_path), "folder")
    assert not fakes  # nothing copied, no alias persisted


def test_empty_required_field_fails_before_the_alias(fakes: dict[str, Any], tmp_path: Path) -> None:
    form = _folder_form(str(tmp_path), dataset_name="")
    with pytest.raises(JobFailed, match="Dataset Name is required"):
        imp._run_format_import(_ctx(form, tmp_path), "folder")
    assert not fakes


def test_existing_source_reaches_the_alias_and_the_executor(
    fakes: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(imp._EXECUTORS, "folder", lambda fd: {"success": True, "message": "ok", "details": {}})
    imp._run_format_import(_ctx(_folder_form(str(tmp_path)), tmp_path), "folder")
    assert fakes["register"]["image_folder"] == str(tmp_path)
    assert fakes["register"]["remote_path"] is None


def test_a_rewritten_source_keeps_the_alias_on_the_picked_folder(fakes: dict[str, Any]) -> None:
    """The host pointed this run at a copy on the node: rows are tokenised from the copy, the alias stays put."""
    form = _folder_form("/node/stage/fire-1a2b", alias_folder="s3://b/data/fire")
    out = imp._maybe_register_alias(form, "/node/stage/fire-1a2b", source_folder="s3://b/data/fire")
    assert fakes["register"]["image_folder"] == "/node/stage/fire-1a2b"  # the session alias: the files read
    assert fakes["register"]["remote_path"] == "s3://b/data/fire"  # the persisted alias: where they came from
    assert out and out["token"] == "FIRE"


def test_an_alias_above_the_source_moves_by_the_same_subfolders(fakes: dict[str, Any]) -> None:
    form = _folder_form("/node/coco/images/train", alias_folder="s3://b/coco")
    imp._maybe_register_alias(form, "/node/coco/images/train", source_folder="s3://b/coco/images/train")
    assert fakes["register"]["image_folder"] == "/node/coco"
    assert fakes["register"]["remote_path"] == "s3://b/coco"


def test_an_alias_above_a_flat_copy_is_refused(fakes: dict[str, Any]) -> None:
    form = _folder_form("/node/stage/train-1a2b", alias_folder="s3://b/coco")
    with pytest.raises(JobFailed, match="points at s3://b/coco, a folder above it"):
        imp._maybe_register_alias(form, "/node/stage/train-1a2b", source_folder="s3://b/coco/images/train")
    assert not fakes


def test_the_echo_names_the_picked_folder_only_when_the_host_rewrote_it(tmp_path: Path) -> None:
    picked = str(tmp_path / "picked")
    same = {"folder_path": picked, "submitted_sources": {"folder_path": picked}}
    assert imp._submitted_image_folder("folder", same) is None
    assert imp._submitted_image_folder("folder", {"folder_path": picked}) is None  # an older fragment
    rewritten = {"folder_path": "/node/stage/x", "submitted_sources": {"folder_path": picked}}
    assert imp._submitted_image_folder("folder", rewritten) == picked


def test_the_fragment_echoes_every_declared_source() -> None:
    assert "var _SOURCE_FIELDS = ['dataset_yaml', 'annotations_file', 'image_folder', 'folder_path', 'csv_path'];" in UI
    assert set(imp._SOURCE_FIELDS) == {"dataset_yaml", "annotations_file", "image_folder", "folder_path", "csv_path"}
    assert UI.count("PluginJobs.run('importer', _withSourceEcho(") == 2  # single and multi-split imports


def test_the_fragment_opts_into_the_copy_offer() -> None:
    assert UI.count("{ copyOffer: true }") == 2  # the import form and the CSV wizard both perform the copy


def test_csv_upload_not_held_here_says_why(tmp_path: Path) -> None:
    params = {"format": "csv", "session_id": "gone", "selected_columns": [{"index": 0}]}
    with pytest.raises(JobFailed, match="cannot run on a GPU node"):
        imp._run_csv_import(_ctx(params, tmp_path))
    assert "PLUGIN_API.getRunTarget()" in UI and "and cannot run on a" in UI
