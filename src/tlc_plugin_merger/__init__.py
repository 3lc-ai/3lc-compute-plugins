# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Merge plugin — vertically join 3LC tables (row concatenation)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tlc_plugin_sdk import ComputePlugin, JobContext, JobFailed

if TYPE_CHECKING:
    from litestar.handlers import BaseRouteHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution logic
# ---------------------------------------------------------------------------


def _execute_merge(data: dict[str, Any]) -> dict[str, Any]:
    """Execute a vertical table merge (row concatenation).

    Stacks the rows of the input tables into a single new table, in the order
    given. The tables must have compatible schemas (the same columns); their
    row counts may differ. ``run_job`` has already validated ``table_urls`` and
    the project/dataset/table names by the time this runs.

    Args:
        data: The job params — ``table_urls``, ``project_name``, ``dataset_name``,
            ``table_name``.

    Returns:
        A result dict with ``success`` and a human-readable ``message``; on
        success it also carries ``table_url``, ``project_name``, ``dataset_name``,
        and ``details``.
    """
    table_urls = data["table_urls"]
    project_name = data["project_name"].strip()
    dataset_name = data["dataset_name"].strip()
    table_name = data["table_name"].strip()

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
                "input_count": len(table_urls),
                "output_table": table_name,
            },
        }
    except Exception as e:
        # Incompatible schemas is an expected, user-driven outcome — log it plainly
        # (no traceback) and return a concise message. The raw tlc schema diff is
        # noise for the user; the UI's pre-merge compatibility check already names
        # the differing columns.
        if "incompatible schemas" in str(e):
            logger.warning("Merge rejected: incompatible schemas (%s)", e)
            return {
                "success": False,
                "message": (
                    "Cannot vertically join these tables: their columns don't match. "
                    "Vertical join stacks rows, so every table must have the same columns."
                ),
                "details": {},
            }
        logger.exception("Merge failed")
        return {"success": False, "message": f"Merge failed: {e}", "details": {}}


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class MergePlugin(ComputePlugin):
    """Sidebar plugin for vertically joining two 3LC tables (row concatenation)."""

    _ui_cache: str | None = None

    def get_ui_fragment(self) -> str:
        """Return the self-contained merge wizard HTML+JS+CSS fragment."""
        if self._ui_cache is None:
            # window.PluginJobs (the generic job-channel client the UI drives the merge
            # job with) is injected by the SDK's /ui handler — nothing to prepend here.
            ui_path = Path(__file__).resolve().parent / "ui.html"
            self._ui_cache = ui_path.read_text(encoding="utf-8")
        return self._ui_cache

    def get_route_handlers(self) -> list[BaseRouteHandler]:
        """Serve the merger's custom routes as relative Litestar handlers (host + venv)."""
        from tlc_plugin_merger.routes import get_route_handlers

        return get_route_handlers()

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute merge via params (fallback for GET compute endpoint)."""
        return {"error": "Use POST /api/plugins/merger/run instead."}

    def run_job(self, ctx: JobContext) -> None:
        """Run a table merge as a host job; report the merged table via ``ctx.result``.

        Validates the request (moved here from the old ``/execute`` route),
        performs the vertical join (row concatenation), and reports the merged
        table URL. Validation and execution failures are raised so the host marks
        the job failed and the message reaches the UI via the generic
        ``job_update`` channel.

        Args:
            ctx: Host-provided job context. ``ctx.params`` carries ``table_urls``,
                ``project_name``, ``dataset_name``, ``table_name``.

        Raises:
            JobFailed: When the request is invalid or the merge itself fails. Raising
                the SDK's clean-failure type surfaces the message to the UI without a
                traceback, since these are expected, user-facing outcomes.

        """
        data = ctx.params
        if len(data.get("table_urls", [])) < 2:
            msg = "Select at least 2 tables to merge."
            raise JobFailed(msg)
        if not all((data.get(k) or "").strip() for k in ("project_name", "dataset_name", "table_name")):
            msg = "Project, dataset, and table name are required."
            raise JobFailed(msg)

        ctx.progress(percent=10, label="Merging tables")
        result = _execute_merge(data)
        if not result.get("success"):
            raise JobFailed(result.get("message") or "Merge failed")

        details = result.get("details", {})
        if details.get("input_count"):
            ctx.metric("inputs", details["input_count"])
        ctx.progress(percent=100, label=result.get("message", "Merged"))
        ctx.result(result["table_url"])
