# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Custom routes for the YOLO plugin, as relative Litestar route handlers.

Returned by ``YoloPlugin.get_route_handlers()`` and served by the plugin's own app
(in-process for host mode, reverse-proxied for venv) under ``/api/plugins/yolo/`` —
no static node on the main app, so nothing shadows the generic ``/run`` route. Job
submission / cancellation / queue state stay host-managed via
``/api/plugins/<id>/run`` + ``/api/plugins/jobs`` and the unified ``run_job``
contract. Handlers are ``def`` (Litestar runs them in a threadpool) because they
touch the ProjectStore and the ``tlc`` SDK, which block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from litestar import Response, get, post
from litestar.params import FromPath

from tlc_plugin_yolo.models import MODEL_REGISTRY
from tlc_plugin_yolo.project_store import TrainingProject
from tlc_plugin_yolo.runtime import get_store
from tlc_plugin_yolo.task_detection import detect_task

if TYPE_CHECKING:
    from litestar.handlers import BaseRouteHandler


def _project_to_dict(p: TrainingProject) -> dict[str, Any]:
    """Serialise a TrainingProject for JSON responses."""
    return {
        "id": p.id,
        "name": p.name,
        "run_name": p.run_name,
        "project_name": p.project_name,
        "model_name": p.model_name,
        "task_type": p.task_type,
        "train_table_url": p.train_table_url,
        "val_table_url": p.val_table_url,
        "use_latest": p.use_latest,
        "mode": p.mode,
        "params": p.params,
        "created": p.created,
        "last_run": p.last_run,
    }


def get_route_handlers() -> list[BaseRouteHandler]:
    """Build YOLO's custom route handlers (fresh per call, for per-app registration)."""

    @get("/models", sync_to_thread=True)
    def list_models() -> list[dict[str, Any]]:
        return [
            {"name": m.name, "display_name": m.display_name, "supported_tasks": m.supported_tasks}
            for m in MODEL_REGISTRY.values()
        ]

    # get_params() is model-defined (list or dict, heterogeneous by model) → Any body.
    @get("/models/{name:str}/params", sync_to_thread=True)
    def model_params(name: FromPath[str]) -> Response[Any]:
        model = MODEL_REGISTRY.get(name)
        if not model:
            return Response({"error": f"Model '{name}' not found"}, status_code=404)
        return Response(model.get_params())

    @post("/detect-task", status_code=200, sync_to_thread=True)
    def detect_task_route(data: dict[str, Any]) -> dict[str, Any]:
        url = str(data.get("url", "")).strip()
        if not url:
            return {"error": "url is required"}
        return detect_task(url)

    @get("/projects", sync_to_thread=True)
    def list_projects() -> list[dict[str, Any]]:
        store = get_store()
        # A GET listing tolerates an uninitialised store (empty list).
        return [_project_to_dict(p) for p in store.list_projects()] if store else []

    @post("/projects", status_code=200, sync_to_thread=True)
    def save_project(data: dict[str, Any]) -> Response[dict[str, Any]]:
        store = get_store()
        if not store:
            return Response({"error": "YOLO not initialized"})
        project = store.save_project(
            TrainingProject(
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
                params=data.get("params", {}),
                created=data.get("created", ""),
                last_run=data.get("last_run"),
            )
        )
        return Response({"id": project.id, "created": project.created})

    @get("/projects/{project_id:str}", sync_to_thread=True)
    def get_project(project_id: FromPath[str]) -> Response[dict[str, Any]]:
        store = get_store()
        if not store:
            return Response({"error": "YOLO not initialized"})
        existing = store.get_project(project_id)
        if not existing:
            return Response({"error": "Not found"}, status_code=404)
        return Response(_project_to_dict(existing))

    @post("/projects/{project_id:str}/delete", status_code=200, sync_to_thread=True)
    def delete_project(project_id: FromPath[str]) -> Response[dict[str, Any]]:
        store = get_store()
        if not store:
            return Response({"error": "YOLO not initialized"})
        if store.delete_project(project_id):
            return Response({"deleted": True})
        return Response({"error": "Not found"})

    @post("/projects/{project_id:str}/resolve-latest", status_code=200, sync_to_thread=True)
    def resolve_latest(project_id: FromPath[str]) -> dict[str, Any]:
        store = get_store()
        if not store:
            return {"error": "YOLO not initialized"}
        existing = store.get_project(project_id)
        if not existing:
            return {"error": "Not found"}
        try:
            import tlc

            train_url = str(tlc.Table.from_url(existing.train_table_url).latest().url)
            val_url = ""
            if existing.val_table_url:
                val_url = str(tlc.Table.from_url(existing.val_table_url).latest().url)
            return {"train_url": train_url, "val_url": val_url}
        except Exception as e:
            return {"error": str(e)}

    return [
        list_models,
        model_params,
        detect_task_route,
        list_projects,
        save_project,
        get_project,
        delete_project,
        resolve_latest,
    ]
