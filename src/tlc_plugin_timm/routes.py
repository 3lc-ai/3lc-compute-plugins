# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Custom routes for the timm plugin, as relative Litestar route handlers.

Returned by ``TimmPlugin.get_route_handlers()`` and served by the plugin's own app
(in-process for host mode, reverse-proxied for venv) under ``/api/plugins/timm/`` —
no static node on the main app, so nothing shadows the generic ``/run`` route. Job
submission / cancellation / queue state stay host-managed via
``/api/plugins/<id>/run`` + ``/api/plugins/jobs`` and the unified ``run_job``
contract. Handlers are ``def`` (Litestar runs them in a threadpool) because they
touch the config store and the ``tlc`` SDK, which block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from litestar import Request, Response, get, post
from litestar.params import FromPath

from tlc_plugin_timm.config_store import TimmConfig, config_store
from tlc_plugin_timm.model_registry import get_model_info, list_models
from tlc_plugin_timm.task_detection import detect_task

if TYPE_CHECKING:
    from litestar.handlers import BaseRouteHandler


def _config_from_body(data: dict[str, Any]) -> TimmConfig:
    """Build a TimmConfig from a create/update request body."""
    return TimmConfig(
        id=data.get("id", ""),
        name=data.get("name", "Untitled"),
        run_name=data.get("run_name", ""),
        project_name=data.get("project_name", ""),
        model_name=data.get("model_name", ""),
        task_type=data.get("task_type", ""),
        train_table_url=data.get("train_table_url", ""),
        val_table_url=data.get("val_table_url", ""),
        use_latest=data.get("use_latest", False),
        mode=data.get("mode", "train"),
        image_column=data.get("image_column", ""),
        label_column=data.get("label_column", ""),
        params=data.get("params", {}),
        created=data.get("created", ""),
        last_run=data.get("last_run"),
    )


def _config_to_dict(c: TimmConfig) -> dict[str, Any]:
    """Serialise a TimmConfig for JSON responses."""
    return {
        "id": c.id,
        "name": c.name,
        "run_name": c.run_name,
        "project_name": c.project_name,
        "model_name": c.model_name,
        "task_type": c.task_type,
        "train_table_url": c.train_table_url,
        "val_table_url": c.val_table_url,
        "use_latest": c.use_latest,
        "mode": c.mode,
        "image_column": c.image_column,
        "label_column": c.label_column,
        "params": c.params,
        "created": c.created,
        "last_run": c.last_run,
    }


def get_route_handlers() -> list[BaseRouteHandler]:
    """Build timm's custom route handlers (fresh per call, for per-app registration)."""

    @get("/models", sync_to_thread=True)
    def list_models_route(request: Request[Any, Any, Any]) -> list[dict[str, Any]]:
        pretrained_param = request.query_params.get("pretrained", "true")
        pretrained = pretrained_param.lower() != "false"
        return list_models(pretrained_only=pretrained)

    @get("/models/{name:str}/info", sync_to_thread=True)
    def model_info(name: FromPath[str]) -> dict[str, Any]:
        return get_model_info(name)

    @post("/detect-task", status_code=200, sync_to_thread=True)
    def detect_task_route(data: dict[str, Any]) -> dict[str, Any]:
        url = str(data.get("url", "")).strip()
        if not url:
            return {"error": "url is required"}
        return detect_task(url)

    @get("/configs", sync_to_thread=True)
    def list_configs() -> list[dict[str, Any]]:
        return [_config_to_dict(c) for c in config_store().list_configs()]

    @post("/configs", status_code=200, sync_to_thread=True)
    def save_config(data: dict[str, Any]) -> dict[str, Any]:
        config = config_store().save_config(_config_from_body(data))
        return {"id": config.id, "created": config.created}

    @get("/configs/{config_id:str}", sync_to_thread=True)
    def get_config(config_id: FromPath[str]) -> Response[dict[str, Any]]:
        existing = config_store().get_config(config_id)
        if not existing:
            return Response({"error": "Not found"}, status_code=404)
        return Response(_config_to_dict(existing))

    @post("/configs/{config_id:str}/delete", status_code=200, sync_to_thread=True)
    def delete_config(config_id: FromPath[str]) -> dict[str, Any]:
        if config_store().delete_config(config_id):
            return {"deleted": True}
        return {"error": "Not found"}

    return [
        list_models_route,
        model_info,
        detect_task_route,
        list_configs,
        save_config,
        get_config,
        delete_config,
    ]
