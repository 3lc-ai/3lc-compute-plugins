# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Export's page: what it says when an export ends (Bugs 19271 and 19272).

The outcome of an export that ended without failing is a pure function in ``ui.html``, between two marker comments.
These tests run that block in node; the page's wiring is pinned from its text.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

exporter = pytest.importorskip("tlc_plugin_exporter")

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

UI = (Path(exporter.__file__).parent / "ui.html").read_text(encoding="utf-8")
OUTCOME = UI[UI.index("// ── Job outcome") : UI.index("// ── End of the job outcome ──")]
OUT = "s3://b/exports/run.json"


def _js(tmp_path: Path, expression: str) -> Any:
    """Evaluate ``expression`` in node after the page's job-outcome block, and return its value."""
    harness = tmp_path / "outcome.js"
    harness.write_text(OUTCOME + f"\nprocess.stdout.write(JSON.stringify({expression}));\n", encoding="utf-8")
    done = subprocess.run(["node", str(harness)], check=True, capture_output=True, text=True)
    return json.loads(done.stdout)


@needs_node
def test_the_page_script_parses(tmp_path: Path) -> None:
    script = tmp_path / "page.js"
    script.write_text(UI[UI.index("<script>") + len("<script>") : UI.index("</script>")], encoding="utf-8")
    subprocess.run(["node", "--check", str(script)], check=True)


@needs_node
def test_a_stopped_export_is_not_reported_as_completed(tmp_path: Path) -> None:
    """PluginJobs calls onDone for "cancelled" too. Stopped before it ran, nothing was exported; stopped and killed
    while it ran, part of it may have been; stopped too late, all of it was, and its own result says so."""
    exported = {"success": True, "message": "Exported 10 rows.", "output_path": OUT}
    cases = [
        ({"status": "completed", "progress": {"label": "Export complete"}}, None),
        ({"status": "completed", "progress": {"label": "Export complete"}}, exported),
        ({"status": "cancelled", "progress": {"label": ""}}, None),
        ({"status": "cancelled", "progress": {"label": "Exporting to COCO…"}}, None),
        ({"status": "cancelled", "progress": {"label": "Export complete"}}, exported),
    ]
    outcomes = _js(
        tmp_path,
        json.dumps(cases) + f".map(function (c) {{ return exportOutcome(c[0], c[1], {json.dumps(OUT)}); }})",
    )
    assert outcomes == [
        {"success": True, "message": f"Export completed. Output at: {OUT}"},
        exported,
        {"success": False, "message": "Export cancelled."},
        {"success": False, "message": f"Export stopped. {OUT} may hold part of the export."},
        {**exported, "message": "The export finished before it could be stopped. Exported 10 rows."},
    ]


@needs_node
def test_a_completed_exports_own_result_is_kept(tmp_path: Path) -> None:
    """onDone shows the outcome only when it differs from the result already on screen."""
    same = _js(
        tmp_path,
        "(function () { var r = {success: true, message: 'x'}; "
        "return exportOutcome({status: 'completed'}, r, 'out') === r; })()",
    )
    assert same is True


def test_the_page_decides_the_result_once_the_job_ends() -> None:
    assert "onDone: function(job) {" in UI
    assert "var outcome = exportOutcome(job, exported, outputPath);" in UI
    assert "showResult(ended ? exportOutcome(ended, payload, outputPath) : payload);" in UI


def test_every_export_starts_by_saying_it_exports() -> None:
    """cleanup() clears the status line when an export ends, so the next one sets it again (Bug 19271)."""
    assert "status.textContent = EXPORTING;" in UI
    assert "var EXPORTING = 'Exporting… this may take a moment.';" in UI
