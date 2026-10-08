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
