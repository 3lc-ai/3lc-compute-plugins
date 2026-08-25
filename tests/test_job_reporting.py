# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job reporting goes through the generic ``JobContext`` surface, not hand-rolled events.

A plugin reports its result with ``ctx.result(url)`` (the Open link on the generic Queue
card) and fails by raising — the host records the failure message. Nothing here re-emits
the generic lifecycle as a custom event, and the long-running fragments seed themselves
from the host's job list on mount instead of relying on live events alone.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import pytest
from tlc_plugin_sdk import JobContext

SRC = Path(__file__).resolve().parent.parent / "src"
PLUGIN_PACKAGES = sorted(p.name for p in SRC.iterdir() if p.is_dir() and p.name.startswith("tlc_plugin_"))


def _ctx(events: list[dict[str, Any]], params: dict[str, Any], tmp_path: Path) -> JobContext:
    return JobContext("job-1", params, tmp_path, sink=events.append, cancel_event=threading.Event())


def _custom_names(events: list[dict[str, Any]]) -> set[str]:
    return {e["name"] for e in events if e["event"] == "custom"}


def test_merger_reports_table_via_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tlc_plugin_merger as merger

    table_url = "s3://bucket/projects/p/datasets/d/tables/merged"
    monkeypatch.setattr(
        merger,
        "_execute_merge",
        lambda data: {"success": True, "message": "Merged", "table_url": table_url, "details": {"input_count": 2}},
    )
    events: list[dict[str, Any]] = []
    params = {"table_urls": ["a", "b"], "project_name": "p", "dataset_name": "d", "table_name": "merged"}
    merger.MergePlugin().run_job(_ctx(events, params, tmp_path))

    results = [e for e in events if e["event"] == "result"]
    assert results == [{"event": "result", "run_url": table_url, "job_id": "job-1"}]
    assert not _custom_names(events) & {"job_completed", "job_failed"}


def test_merger_invalid_request_raises_without_custom_failure_event(tmp_path: Path) -> None:
    import tlc_plugin_merger as merger

    events: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="at least 2 tables"):
        merger.MergePlugin().run_job(_ctx(events, {"table_urls": ["only-one"]}, tmp_path))
    assert not any(e["event"] in {"result", "custom"} for e in events)


def test_importer_report_result_sets_run_url(tmp_path: Path) -> None:
    from tlc_plugin_importer import _report_result

    table_url = "s3://b/projects/p/datasets/d/tables/t"
    events: list[dict[str, Any]] = []
    _report_result(
        _ctx(events, {}, tmp_path),
        {"message": "Imported 3 rows", "table_url": table_url, "details": {"row_count": 3}},
    )
    results = [e for e in events if e["event"] == "result"]
    assert results == [{"event": "result", "run_url": table_url, "job_id": "job-1"}]
    assert not _custom_names(events)


@pytest.mark.parametrize("package", PLUGIN_PACKAGES)
def test_no_hand_rolled_lifecycle_in_source(package: str) -> None:
    for path in (SRC / package).rglob("*"):
        if path.suffix not in {".py", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "result(run_url=" not in text, f"{path}: ctx.result takes the URL positionally"
        assert not re.search(r"job_(completed|failed)", text), f"{path}: the generic job channel owns the lifecycle"


@pytest.mark.parametrize("package", PLUGIN_PACKAGES)
def test_fragment_accepts_sdk_job_tracker_injection(package: str) -> None:
    """The SDK's ``/ui`` handler injects ``window.PluginJobs`` itself — the fragment must offer it a slot."""
    from tlc_plugin_sdk.shared.job_tracker import JOB_TRACKER_JS
    from tlc_plugin_sdk.shared.ui_inject import inject_scripts

    ui_path = SRC / package / "ui.html"
    if not ui_path.exists():
        return  # inline-display plugins have no standalone page
    injected = inject_scripts(ui_path.read_text(encoding="utf-8"), JOB_TRACKER_JS)
    assert injected.index("window.PluginJobs") < injected.index("PLUGIN_API")


@pytest.mark.parametrize(
    ("package", "plugin_id"),
    [("tlc_plugin_image_metrics", "image-metrics"), ("tlc_plugin_importer", "importer")],
)
def test_long_running_fragments_seed_from_host_job_list(package: str, plugin_id: str) -> None:
    fragment = (SRC / package / "ui.html").read_text(encoding="utf-8")
    assert f"PluginJobs.list('{plugin_id}')" in fragment
    assert f"PluginJobs.track('/{plugin_id}'" in fragment
