# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Custom routes for the Merge Tables plugin, as relative Litestar route handlers.

Returned by ``MergePlugin.get_route_handlers()`` and served by the plugin's own app
(in-process for host mode, reverse-proxied for venv) under ``/api/plugins/merger/`` —
no static node on the main app, so nothing shadows the generic ``/run`` route. The
merge itself stays host-managed via ``/api/plugins/merger/run`` + ``/api/plugins/jobs``
and the unified ``run_job`` contract; the route here only serves a lightweight schema
pre-check so the UI can validate compatibility before enabling the merge button.
Handlers are ``def`` (Litestar runs them in a threadpool) because they touch the
``tlc`` SDK, which blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from litestar import Request, Response, get

if TYPE_CHECKING:
    from litestar.handlers import BaseRouteHandler


def get_route_handlers() -> list[BaseRouteHandler]:
    """Build the merger's custom route handlers (fresh per call, for per-app registration)."""

    @get("/columns", sync_to_thread=True)
    def columns(request: Request[Any, Any, Any]) -> Response[dict[str, Any]]:
        """Return the column names and row count for a table, for the compatibility pre-check.

        Args:
            request: The incoming request; ``url`` query param is the table URL.

        Returns:
            A response carrying ``{"columns": [...], "row_count": <int|null>}`` on
            success, ``{"error": ...}`` (status 400) when ``url`` is missing, or
            ``{"error": ...}`` (status 200) when the table cannot be loaded — so the
            UI can render the failure gracefully.
        """
        url = request.query_params.get("url", "")
        if not url:
            return Response({"error": "url is required"}, status_code=400)

        try:
            import tlc
            from tlc_plugin_sdk.shared.url_utils import normalize_url

            table = tlc.Table.from_url(normalize_url(url))
            column_names: list[str] = list(table.columns)
            row_count = table.row_count
        except Exception as e:
            # Surface any load failure to the UI as data so it can render gracefully.
            return Response({"error": str(e)})

        return Response({"columns": column_names, "row_count": row_count})

    return [columns]
