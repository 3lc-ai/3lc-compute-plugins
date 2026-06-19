# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Custom routes for the Export plugin, as relative Litestar route handlers.

Returned by ``ExportPlugin.get_route_handlers()`` and served by the plugin's own
app (in-process for host mode, reverse-proxied for venv) under
``/api/plugins/exporter/`` — no static node on the main app, so nothing shadows
the generic ``/run`` route. Handlers are ``def`` (Litestar runs them in a
threadpool) because they touch the ``tlc`` SDK and the filesystem, which block.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from litestar import get, post

from tlc_plugin_exporter import (
    _EXECUTORS,
    EXPORT_FORMATS,
    _classify_column_type,
)

if TYPE_CHECKING:
    from litestar.handlers import BaseRouteHandler

logger = logging.getLogger(__name__)


def get_route_handlers() -> list[BaseRouteHandler]:
    """Build the Export plugin's custom route handlers (fresh per call)."""

    @get("/formats", sync_to_thread=True)
    def list_formats() -> list[dict[str, Any]]:
        """Return all export format definitions."""
        return list(EXPORT_FORMATS.values())

    @post("/columns", status_code=200, sync_to_thread=True)
    def list_columns(data: dict[str, Any]) -> dict[str, Any]:
        """Return column names and types for a table.

        Args:
            data: JSON body with ``table_url``.

        Returns:
            Dict with ``columns`` list of ``{"name": str, "type": str}`` entries.

        """
        import tlc

        table_url = (data.get("table_url") or "").strip()
        if not table_url:
            return {"error": "table_url is required"}

        try:
            from tlc_plugin_sdk.shared.url_utils import normalize_url

            table = tlc.Table.from_url(normalize_url(table_url))
            columns: list[dict[str, str]] = []
            schema_values = table.rows_schema.values if hasattr(table.rows_schema, "values") else {}
            first_row = table.table_rows[0] if table.row_count > 0 else {}

            for name in first_row.keys():
                col_schema = schema_values.get(name)
                col_type = _classify_column_type(col_schema, name) if col_schema else "other"
                columns.append({"name": name, "type": col_type})

            return {"columns": columns, "row_count": table.row_count}
        except Exception:
            logger.warning("Failed to list columns for %s", table_url, exc_info=True)
            return {"error": "Failed to load table columns"}

    @post("/execute", status_code=200, sync_to_thread=True)
    def execute_export(data: dict[str, Any]) -> dict[str, Any]:
        """Validate and execute an export.

        Args:
            data: JSON body with ``format``, ``table_url``, ``output_path``, and
                format-specific options.

        Returns:
            ``{"success": True, "message": str, "details": dict}`` on success, or
            ``{"success": False, "message": str}`` on failure.

        """
        format_name = data.get("format", "")
        table_url = (data.get("table_url") or "").strip()
        output_path = (data.get("output_path") or "").strip()

        if not format_name or format_name not in EXPORT_FORMATS:
            return {"success": False, "message": f"Unknown export format: {format_name}"}
        if not table_url:
            return {"success": False, "message": "Table URL is required."}
        if not output_path:
            return {"success": False, "message": "Output path is required."}

        executor = _EXECUTORS.get(format_name)
        if not executor:
            return {"success": False, "message": f"No executor for format: {format_name}"}

        # Apply alias overrides if requested
        originals: list[dict[str, str]] = []
        alias_overrides = data.get("alias_overrides") or {}
        if alias_overrides.get("enabled") and alias_overrides.get("overrides"):
            from tlc_plugin_sdk.shared.aliases import apply_alias_overrides

            originals = apply_alias_overrides(alias_overrides["overrides"])

        try:
            result: dict[str, Any] = executor(table_url, output_path, data)
            return result
        except Exception as exc:
            logger.exception("Export failed for format %s", format_name)
            return {"success": False, "message": f"Export failed: {exc}", "details": {}}
        finally:
            if originals:
                from tlc_plugin_sdk.shared.aliases import restore_aliases

                restore_aliases(originals)

    return [
        list_formats,
        list_columns,
        execute_export,
    ]
