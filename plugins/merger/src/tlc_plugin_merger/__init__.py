# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Merge plugin — join 3LC tables by row index."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tlc_plugin_sdk import ComputePlugin, JobContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution logic (ported from old 3LCDataTools merger)
# ---------------------------------------------------------------------------


def _execute_merge(data: dict[str, Any]) -> dict[str, Any]:
    """Execute a table merge (join).

    ``run_job`` has already validated ``table_urls``, project/dataset/table
    names, and merge_type by the time this runs.
    """
    table_urls = data["table_urls"]
    project_name = data["project_name"].strip()
    dataset_name = data["dataset_name"].strip()
    table_name = data["table_name"].strip()
    merge_type = data.get("merge_type", "join")

    try:
        import tlc
    except ImportError:
        return {"success": False, "message": "tlc package not installed.", "details": {}}

    # Validate input tables exist and are loadable
    try:
        input_tables = [tlc.Table.from_url(u) for u in table_urls]
    except Exception as e:
        logger.exception("Failed to load input tables")
        return {"success": False, "message": f"Could not load input tables: {e}", "details": {}}

    # Check row count compatibility
    row_counts = [(str(t.url), t.row_count) for t in input_tables]
    valid_counts = [rc for _, rc in row_counts if rc and rc > 0]
    if len(valid_counts) >= 2 and len(set(valid_counts)) > 1:
        summary = ", ".join(f"{u.split('/')[-1]} ({rc} rows)" for u, rc in row_counts)
        return {
            "success": False,
            "message": (
                f"Tables have different row counts and cannot be joined: {summary}. "
                "Join combines columns by row index, so all tables must have the same number of rows."
            ),
            "details": {},
        }

    try:
        merged = tlc.Table.join_tables(
            input_tables,
            project_name=project_name,
            dataset_name=dataset_name,
            table_name=table_name,
            if_exists="rename",
        )

        return {
            "success": True,
            "message": f"Merged {len(table_urls)} tables into '{table_name}'.",
            "table_url": str(merged.url),
            "project_name": project_name,
            "dataset_name": dataset_name,
            "details": {
                "merge_type": merge_type,
                "input_count": len(table_urls),
                "output_table": table_name,
            },
        }
    except Exception as e:
        logger.exception("Merge failed")
        return {"success": False, "message": f"Merge failed: {e}", "details": {}}


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class MergePlugin(ComputePlugin):
    """Sidebar plugin for merging two 3LC tables by column join."""

    _ui_cache: str | None = None

    def get_ui_fragment(self) -> str:
        """Return the self-contained merge wizard HTML+JS+CSS fragment."""
        if self._ui_cache is None:
            from tlc_plugin_sdk.shared.job_tracker import job_tracker_script

            ui_path = Path(__file__).resolve().parent / "ui.html"
            # Inject window.PluginJobs so the UI can drive the merge job over the
            # generic job_update channel (start + progress + result via run_job).
            self._ui_cache = "<script>\n" + job_tracker_script() + "\n</script>\n" + ui_path.read_text(encoding="utf-8")
        return self._ui_cache

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute merge via params (fallback for GET compute endpoint)."""
        return {"error": "Use POST /api/plugins/merger/run instead."}

    def run_job(self, ctx: JobContext) -> None:
        """Run a table merge as a host job; report the merged table via ``ctx.result``.

        Validates the request (moved here from the old ``/execute`` route),
        performs the join, and reports the merged table URL. Validation and
        execution failures are raised so the host marks the job failed and the
        message reaches the UI via the generic ``job_update`` channel.

        Args:
            ctx: Host-provided job context. ``ctx.params`` carries ``table_urls``,
                ``project_name``, ``dataset_name``, ``table_name``, ``merge_type``.

        Raises:
            ValueError: When the request is invalid.
            RuntimeError: When the merge itself fails.

        """
        data = ctx.params
        if len(data.get("table_urls", [])) < 2:
            msg = "Select at least 2 tables to merge."
            raise ValueError(msg)
        if not all((data.get(k) or "").strip() for k in ("project_name", "dataset_name", "table_name")):
            msg = "Project, dataset, and table name are required."
            raise ValueError(msg)
        if data.get("merge_type", "join") != "join":
            msg = "Union merge not yet implemented — coming soon."
            raise ValueError(msg)

        ctx.progress(percent=10, label="Merging tables")
        result = _execute_merge(data)
        if not result.get("success"):
            raise RuntimeError(result.get("message") or "Merge failed")

        details = result.get("details", {})
        if details.get("input_count"):
            ctx.metric("inputs", details["input_count"])
        ctx.progress(percent=100, label=result.get("message", "Merged"))
        ctx.result(run_url=result["table_url"])
