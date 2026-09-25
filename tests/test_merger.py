# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the Merge Tables plugin (vertical join / row concatenation).

The merger now presents a single operation — a vertical join that stacks rows — so
``run_job`` validates only the request shape (>= 2 tables, names present) and no longer
rejects a ``merge_type`` field. These tests drive ``run_job`` through the recording
``JobContext`` sink used elsewhere in the suite and assert the corrected behavior.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("tlc_plugin_sdk")
merger = pytest.importorskip("tlc_plugin_merger")

from tlc_plugin_sdk import JobContext, JobFailed  # noqa: E402  (after importorskip guard)

if TYPE_CHECKING:
    from pathlib import Path


def _ctx(events: list[dict[str, Any]], params: dict[str, Any], tmp_path: Path) -> JobContext:
    return JobContext("job-1", params, tmp_path, sink=events.append, cancel_event=threading.Event())


def test_run_job_requires_at_least_two_tables(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    with pytest.raises(JobFailed, match="at least 2 tables"):
        merger.MergePlugin().run_job(_ctx(events, {"table_urls": ["only-one"]}, tmp_path))
    assert not any(e["event"] in {"result", "custom"} for e in events)


def test_run_job_requires_output_names(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    params = {
        "table_urls": ["a", "b"],
        "project_name": "  ",
        "dataset_name": "",
        "table_name": "",
    }
    with pytest.raises(JobFailed, match="Project, dataset, and table name are required"):
        merger.MergePlugin().run_job(_ctx(events, params, tmp_path))
    assert not any(e["event"] in {"result", "custom"} for e in events)


def test_run_job_does_not_reject_merge_type_field(tmp_path: Path) -> None:
    """A stray ``merge_type`` must no longer trigger a 'union coming soon' ValueError.

    With valid names and unloadable table URLs, the job proceeds past validation into the
    merge attempt and fails there (load/merge/tlc-missing error) — never with the old
    not-implemented message and never as a ``ValueError``.
    """
    events: list[dict[str, Any]] = []
    params = {
        "table_urls": ["bogus://a", "bogus://b"],
        "project_name": "p",
        "dataset_name": "d",
        "table_name": "merged",
        "merge_type": "union",
    }
    with pytest.raises(Exception) as excinfo:
        merger.MergePlugin().run_job(_ctx(events, params, tmp_path))

    assert not isinstance(excinfo.value, ValueError), "merge_type must not be rejected as invalid input"
    message = str(excinfo.value).lower()
    assert "coming soon" not in message
    assert "not yet implemented" not in message
    assert "union" not in message


def test_run_job_reports_result_on_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A successful vertical join reports the merged table via ``ctx.result`` (no custom lifecycle event)."""
    table_url = "s3://bucket/projects/p/datasets/d/tables/merged"
    monkeypatch.setattr(
        merger,
        "_execute_merge",
        lambda data, root_url=None: {
            "success": True,
            "message": "Merged",
            "table_url": table_url,
            "details": {"input_count": 2},
        },
    )
    events: list[dict[str, Any]] = []
    params = {"table_urls": ["a", "b"], "project_name": "p", "dataset_name": "d", "table_name": "merged"}
    merger.MergePlugin().run_job(_ctx(events, params, tmp_path))

    results = [e for e in events if e["event"] == "result"]
    assert results == [{"event": "result", "run_url": table_url, "job_id": "job-1"}]
    assert not any(e["event"] == "custom" for e in events)


def test_execute_merge_reports_incompatible_schemas_friendlily(monkeypatch: pytest.MonkeyPatch) -> None:
    """A schema-incompatibility failure surfaces the vertical-join explanation, not a raw error."""
    fake_tlc = type("_T", (), {})()

    class _FakeTable:
        @staticmethod
        def from_url(_u: str) -> object:
            return object()

        @staticmethod
        def join_tables(*_a: Any, **_k: Any) -> object:
            msg = "Failed to join Tables due to incompatible schemas: col mismatch"
            raise ValueError(msg)

    fake_tlc.Table = _FakeTable  # type: ignore[attr-defined]

    import sys

    monkeypatch.setitem(sys.modules, "tlc", fake_tlc)
    result = merger._execute_merge({
        "table_urls": ["a", "b"],
        "project_name": "p",
        "dataset_name": "d",
        "table_name": "m",
    })
    assert result["success"] is False
    assert "columns don't match" in result["message"]
    assert "stacks rows" in result["message"]


def _capture_join(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``tlc`` with a fake whose ``join_tables`` records its kwargs."""
    captured: dict[str, Any] = {}

    class _FakeTable:
        @staticmethod
        def from_url(_u: str) -> object:
            return object()

        @staticmethod
        def join_tables(*_a: Any, **k: Any) -> object:
            captured.update(k)
            return type("_M", (), {"url": "s3://root/p/datasets/d/tables/m"})()

    fake_tlc = type("_T", (), {"Table": _FakeTable})()

    import sys

    monkeypatch.setitem(sys.modules, "tlc", fake_tlc)
    return captured


def test_run_job_writes_under_the_stamped_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The root the host stamps into the body reaches ``join_tables`` as ``root_url``."""
    captured = _capture_join(monkeypatch)
    params = {
        "table_urls": ["a", "b"],
        "project_name": "p",
        "dataset_name": "d",
        "table_name": "m",
        "project_root_url": "s3://root/",
    }
    merger.MergePlugin().run_job(_ctx([], params, tmp_path))
    assert captured["root_url"] == "s3://root"


def test_execute_merge_without_root_uses_the_configured_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """No root passed means ``root_url=None`` — tlc's configured root, as before."""
    captured = _capture_join(monkeypatch)
    merger._execute_merge({"table_urls": ["a", "b"], "project_name": "p", "dataset_name": "d", "table_name": "m"})
    assert captured["root_url"] is None
