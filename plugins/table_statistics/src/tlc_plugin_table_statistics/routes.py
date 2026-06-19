# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Custom routes for the table_statistics plugin, as relative Litestar route handlers.

Returned by ``TableStatisticsPlugin.get_route_handlers()`` and served by the plugin's
own app under ``/api/plugins/table-statistics/``.

Handlers are ``def`` (Litestar runs them in a threadpool) because they touch the
``tlc`` SDK, PIL image I/O, and the stats cache, all of which block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from litestar import Request, Response, get

if TYPE_CHECKING:
    from litestar.handlers import BaseRouteHandler


def get_route_handlers() -> list[BaseRouteHandler]:
    """Build table_statistics' custom route handlers (fresh per call, for per-app registration)."""

    @get("/table-stats", sync_to_thread=True)
    def table_stats(request: Request[Any, Any, Any]) -> Response[Any]:
        """Progressive column statistics.

        Starts background computation on the first call and returns current progress
        on subsequent calls (the frontend polls until ``complete``).
        ``reset=1`` invalidates the cache and recomputes.
        """
        from tlc_plugin_table_statistics.table_stats import get_or_start_stats, invalidate_stats

        url = request.query_params.get("url", "")
        reset = request.query_params.get("reset", "0")
        if not url:
            return Response({"error": "url parameter is required"}, status_code=400)
        try:
            if reset and reset != "0":
                invalidate_stats(url)
            return Response(get_or_start_stats(url), status_code=200)
        except Exception as exc:
            return Response(
                {"error": str(exc), "columns": [], "row_count": 0, "column_count": 0, "complete": True},
                status_code=200,
            )

    @get("/thumbnail", sync_to_thread=True)
    def thumbnail(request: Request[Any, Any, Any]) -> Response[Any]:
        """Read a table row's image, resize, and return JPEG bytes.

        Reads the image via the tlc SDK (resolves aliases, S3, etc.), resizes with
        PIL, and returns JPEG bytes. Returns a JSON error body on failure.

        Query parameters:
            url: Table URL (required).
            index: Row index (default 0).
            size: Max thumbnail dimension in pixels (default 120).
            column: Image column name override (optional).
        """
        import io

        from tlc_plugin_sdk.shared.url_utils import normalize_url

        url = request.query_params.get("url", "")
        if not url:
            return Response({"error": "url parameter is required"}, status_code=400, media_type="application/json")
        index = int(request.query_params.get("index", "0") or 0)
        size = int(request.query_params.get("size", "120") or 120)
        column: str | None = request.query_params.get("column", None) or None

        try:
            import tlc

            from tlc_plugin_sdk.shared.images import get_image_column, read_image_from_table

            table = tlc.Table.from_url(normalize_url(url))
            try:
                img_col = get_image_column(table, override=column)
            except ValueError as exc:
                return Response({"error": str(exc)}, status_code=400, media_type="application/json")
            if index < 0 or index >= len(table):
                return Response({"error": "Index out of range"}, status_code=400, media_type="application/json")
            try:
                img = read_image_from_table(table, index, img_col)
            except ValueError as exc:
                return Response({"error": str(exc)}, status_code=400, media_type="application/json")
            img.thumbnail((size, size))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return Response(
                content=buf.getvalue(),
                media_type="image/jpeg",
                status_code=200,
                headers={"Cache-Control": "private, max-age=3600"},
            )
        except Exception as exc:
            return Response({"error": str(exc)}, status_code=500, media_type="application/json")

    return [table_stats, thumbnail]
