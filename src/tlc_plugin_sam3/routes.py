# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Custom routes for the SAM3 plugin, as relative Litestar route handlers.

Returned by ``SAM3Plugin.get_route_handlers()`` and served by the plugin's own
app (in-process for host mode, reverse-proxied for venv) under
``/api/plugins/sam3/`` — no static node on the main app, so nothing shadows the
generic ``/run`` route. Job submission / cancellation / queue state stay
host-managed via ``/api/plugins/<id>/run`` + ``/api/plugins/jobs`` and the
unified ``run_job`` contract. Handlers are ``def`` (Litestar runs them in a
threadpool) because they touch the tlc SDK, the file system, and run SAM3
inference — all blocking work.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from litestar import Response, get, post
from litestar.params import FromPath

if TYPE_CHECKING:
    from litestar.handlers import BaseRouteHandler

logger = logging.getLogger(__name__)


def get_route_handlers() -> list[BaseRouteHandler]:
    """Build SAM3's custom route handlers (fresh per call, for per-app registration)."""

    # ── HuggingFace token ──

    @post("/set-hf-token", status_code=200, sync_to_thread=True)
    def set_hf_token(data: dict[str, Any]) -> dict[str, Any]:
        token = str(data.get("token", "")).strip()
        if not token:
            return {"error": "token is required"}
        os.environ["HF_TOKEN"] = token
        return {"ok": True}

    @get("/hf-token-status", sync_to_thread=False)
    def hf_token_status() -> dict[str, Any]:
        return {"has_token": bool(os.environ.get("HF_TOKEN", ""))}

    # ── Preview (CPU-bound SAM3 inference — sync_to_thread keeps the event loop free) ──

    @post("/preview", status_code=200, sync_to_thread=True)
    def preview(data: dict[str, Any]) -> dict[str, Any]:
        import traceback

        try:
            return _run_preview(data)
        except Exception:
            logger.error("Preview failed:\n%s", traceback.format_exc())
            return {"error": traceback.format_exc()}

    # ── Image listing ──

    @post("/list-images", status_code=200, sync_to_thread=True)
    def list_images(data: dict[str, Any]) -> dict[str, Any]:
        from tlc_plugin_sam3.inference import list_images_in_folder

        folder = str(data.get("folder", "")).strip()
        if not folder:
            return {"error": "folder is required"}
        images = list_images_in_folder(folder)
        return {"count": len(images), "sample": images[:20]}

    # ── Read labels from table ──

    @post("/read-labels", status_code=200, sync_to_thread=True)
    def read_labels(data: dict[str, Any]) -> dict[str, Any]:
        table_url = str(data.get("table_url", "")).strip()
        if not table_url:
            return {"error": "table_url is required"}
        try:
            import tlc

            from tlc_plugin_sam3 import _read_labels_from_table, _read_modality_from_table

            table = tlc.Table.from_url(table_url)
            return {
                "labels": _read_labels_from_table(table),
                "modality": _read_modality_from_table(table),
                "table_length": len(table),
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Configs — list + create ──

    @get("/configs", sync_to_thread=True)
    def list_configs() -> list[dict[str, Any]]:
        from tlc_plugin_sam3.config_store import config_store

        return [asdict(c) for c in config_store().list_configs()]

    @post("/configs", status_code=200, sync_to_thread=True)
    def save_config(data: dict[str, Any]) -> dict[str, Any]:
        from tlc_plugin_sam3.config_store import SAM3Config, config_store

        config = SAM3Config(
            id=data.get("id", ""),
            name=data.get("name", ""),
            source_type=data.get("source_type", "folder"),
            folder=data.get("folder", ""),
            table_url=data.get("table_url", ""),
            labels=data.get("labels", []),
            modality=data.get("modality", "segmentation"),
            confidence=float(data.get("confidence", 0.2)),
            project_name=data.get("project_name", ""),
            table_name=data.get("table_name", ""),
            embedding_dim=int(data.get("embedding_dim", 2)),
            device=data.get("device", "cuda"),
            created=data.get("created", ""),
            last_run=data.get("last_run"),
        )
        return asdict(config_store().save_config(config))

    # ── Configs — single get + delete ──

    @get("/configs/{config_id:str}", sync_to_thread=True)
    def get_config(config_id: FromPath[str]) -> Response[dict[str, Any]]:
        from tlc_plugin_sam3.config_store import config_store

        existing = config_store().get_config(config_id)
        if not existing:
            return Response({"error": "Config not found"}, status_code=404)
        return Response(asdict(existing))

    @post("/configs/{config_id:str}/delete", status_code=200, sync_to_thread=True)
    def delete_config(config_id: FromPath[str]) -> dict[str, Any]:
        from tlc_plugin_sam3.config_store import config_store

        return {"deleted": config_store().delete_config(config_id)}

    return [
        set_hf_token,
        hf_token_status,
        preview,
        list_images,
        read_labels,
        list_configs,
        save_config,
        get_config,
        delete_config,
    ]


def _run_preview(data: dict[str, Any]) -> dict[str, Any]:
    """Run SAM3 on a single image and return an annotated preview.

    If ``image_path`` is omitted, a random image is chosen from ``table_url`` (or
    ``folder``). Returns image_path, preview (base64 PNG), num_detections, and the
    detections list.
    """
    from tlc_plugin_sam3.inference import (
        list_images_in_folder,
        predict_single_image,
        render_preview,
    )

    folder = data.get("folder", "").strip()
    table_url = data.get("table_url", "").strip()
    labels = data.get("labels", [])
    label_colors = data.get("label_colors", [])
    modality = data.get("modality", "segmentation")
    confidence = float(data.get("confidence", 0.2))
    device = data.get("device", "cuda")
    image_path = data.get("image_path", "").strip()

    if not labels:
        return {"error": "labels is required"}

    from tlc_plugin_sdk.shared.images import get_image_column, load_image, resolve_image_url

    if not image_path:
        if table_url:
            # Pick a random image from the table. The path is absolutized so
            # the response's image_path round-trips into later previews.
            import tlc

            from tlc_plugin_sdk.shared.url_utils import normalize_url

            table = tlc.Table.from_url(normalize_url(table_url))
            if len(table) == 0:
                return {"error": "Table is empty"}
            image_column = get_image_column(table)
            idx = random.randint(0, len(table) - 1)
            image_path = resolve_image_url(str(table.table_rows[idx][image_column]), table.url).to_str()
        elif folder:
            images = list_images_in_folder(folder)
            if not images:
                return {"error": f"No images found in {folder}"}
            image_path = random.choice(images)
        else:
            return {"error": "Either folder or table_url is required"}

    # load_image handles local paths (PIL directly), cloud/alias paths
    # (via tlc.Url), and paths relative to the table.
    image = load_image(image_path, table_url)
    predictions = predict_single_image(image, labels, confidence, device)

    preview_b64 = render_preview(image, predictions, modality, labels, label_colors=label_colors or None)

    return {
        "image_path": image_path,
        "preview": preview_b64,
        "num_detections": len(predictions["scores"]),
        "detections": [
            {
                "label": predictions["label_names"][i],
                "score": round(predictions["scores"][i], 3),
                "box": predictions["boxes"][i],
            }
            for i in range(len(predictions["scores"]))
        ],
    }
