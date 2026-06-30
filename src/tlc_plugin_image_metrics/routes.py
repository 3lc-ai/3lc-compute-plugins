# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Custom routes for the Image Metrics plugin, as relative Litestar route handlers.

Returned by ``ImageMetricsPlugin.get_route_handlers()`` and served by the plugin's
own app (in-process for host mode, reverse-proxied for venv) under
``/api/plugins/image-metrics/`` — no static node on the main app, so nothing shadows
the generic ``/run`` route. Job submission / cancellation / queue state stay
host-managed via ``/api/plugins/<id>/run`` + ``/api/plugins/jobs`` and the unified
``run_job`` contract. Handlers are ``def`` (Litestar runs them in a threadpool)
because they touch the config store and the ``tlc`` SDK, which block.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from litestar import Request, Response, get, post
from litestar.params import FromPath

if TYPE_CHECKING:
    from litestar.handlers import BaseRouteHandler


def get_route_handlers() -> list[BaseRouteHandler]:
    """Build Image Metrics' custom route handlers (fresh per call, for per-app registration)."""

    @get("/metrics", sync_to_thread=False)
    def list_metrics() -> list[dict[str, Any]]:
        from tlc_plugin_image_metrics.metrics import METRICS

        return [
            {
                "id": m.id,
                "name": m.name,
                "description": m.description,
                "category": m.category,
                "icon": m.icon,
                "unit": m.unit,
            }
            for m in METRICS
        ]

    @get("/detect-columns", sync_to_thread=True)
    def detect_columns(request: Request[Any, Any, Any]) -> Response[dict[str, Any]]:
        from tlc_plugin_image_metrics import _detect_image_columns

        url = request.query_params.get("url", "")
        if not url:
            return Response({"error": "url is required"}, status_code=400)
        return Response({"image_columns": _detect_image_columns(url)})

    @get("/configs", sync_to_thread=True)
    def list_configs() -> list[dict[str, Any]]:
        from tlc_plugin_image_metrics.config_store import config_store

        store = config_store()
        return [asdict(c) for c in store.list_configs()]

    @post("/configs", status_code=200, sync_to_thread=True)
    def save_config(data: dict[str, Any]) -> dict[str, Any]:
        from tlc_plugin_image_metrics.config_store import ImageMetricsConfig, config_store

        store = config_store()
        config = store.save_config(
            ImageMetricsConfig(
                id=data.get("id", ""),
                name=data.get("name", "Untitled"),
                metric_ids=data.get("metric_ids", []),
                output_name_suffix=data.get("output_name_suffix", "metrics"),
                created=data.get("created", ""),
                last_run=data.get("last_run"),
            )
        )
        return {"id": config.id, "created": config.created}

    @get("/configs/{config_id:str}", sync_to_thread=True)
    def get_config(config_id: FromPath[str]) -> Response[dict[str, Any]]:
        from tlc_plugin_image_metrics.config_store import config_store

        store = config_store()
        existing = store.get_config(config_id)
        if not existing:
            return Response({"error": "Not found"}, status_code=404)
        return Response(asdict(existing))

    @post("/configs/{config_id:str}/delete", status_code=200, sync_to_thread=True)
    def delete_config(config_id: FromPath[str]) -> dict[str, Any]:
        from tlc_plugin_image_metrics.config_store import config_store

        store = config_store()
        if store.delete_config(config_id):
            return {"deleted": True}
        return {"error": "Not found"}

    return [
        list_metrics,
        detect_columns,
        list_configs,
        save_config,
        get_config,
        delete_config,
    ]
