# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Custom routes for the importer plugin, as relative Litestar route handlers.

Returned by ``ImportPlugin.get_route_handlers()`` and served by the plugin's own app
(in-process for host mode, reverse-proxied for venv) under ``/api/plugins/importer/``
— no static node on the main app, so nothing shadows the generic ``/run`` route. Job
submission / cancellation / queue state stay host-managed via
``/api/plugins/<id>/run`` + ``/api/plugins/jobs`` and the unified ``run_job``
contract, so this module no longer defines its own delegating ``/run``.

Most handlers are ``def`` (Litestar runs them in a threadpool) because they touch
the file-format parsers and the ``tlc`` SDK, which block. The two multipart upload
handlers (``/csv/parse`` and ``/upload-temp``) stay ``async def`` because they await
``UploadFile.read()``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from litestar import get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body

if TYPE_CHECKING:
    from litestar import Request
    from litestar.handlers import BaseRouteHandler

logger = logging.getLogger(__name__)


def get_route_handlers() -> list[BaseRouteHandler]:
    """Build the importer's custom route handlers (fresh per call, for per-app registration)."""
    # Imported lazily from the package __init__ (where the parsers / executors live)
    # to avoid a circular import: ImportPlugin lives there and imports this module.
    from tlc_plugin_sdk.shared.url_utils import normalize_local_path

    from tlc_plugin_importer import (
        _EXECUTORS,
        IMPORT_STEPS,
        _enhance_error_message,
        _get_image_folder,
        _maybe_register_alias,
        _normalize_path_fields,
        _parse_coco_folder,
        _parse_csv_file,
        _parse_yolo_splits,
        _parsed_csv_files,
        _validate,
    )

    @get("/formats", sync_to_thread=True)
    def list_formats() -> list[dict[str, Any]]:
        """Return all import format definitions (metadata + form fields)."""
        return list(IMPORT_STEPS.values())

    @get("/parse-yolo", sync_to_thread=True)
    def parse_yolo(request: Request[Any, Any, Any]) -> dict[str, Any]:
        """Parse a YOLO dataset.yaml and return available splits + metadata."""
        yaml_path = request.query_params.get("yaml_path", "")
        if not yaml_path.strip():
            return {"error": "yaml_path is required"}
        try:
            return _parse_yolo_splits(normalize_local_path(yaml_path))
        except Exception as exc:
            return {"error": str(exc)}

    @get("/parse-coco", sync_to_thread=True)
    def parse_coco(request: Request[Any, Any, Any]) -> dict[str, Any]:
        """Scan a COCO annotations directory and return available splits."""
        annotations_dir = request.query_params.get("annotations_dir", "")
        if not annotations_dir.strip():
            return {"error": "annotations_dir is required"}
        try:
            return _parse_coco_folder(normalize_local_path(annotations_dir))
        except Exception as exc:
            return {"error": str(exc)}

    @post("/execute", status_code=200, sync_to_thread=True)
    def execute_import(data: dict[str, Any]) -> dict[str, Any]:
        """Validate and execute an import."""
        format_name = data.get("format", "")
        step_def = IMPORT_STEPS.get(format_name)
        if step_def is None:
            return {"success": False, "message": f"Unknown import format: {format_name}", "table_url": None}

        valid, errors = _validate(step_def, data)
        if not valid:
            return {"success": False, "message": "Validation failed: " + "; ".join(errors), "table_url": None}

        try:
            data = _normalize_path_fields(data)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "table_url": None}

        executor = _EXECUTORS.get(format_name)
        if executor is None:
            return {"success": False, "message": f"No executor for format: {format_name}", "table_url": None}

        # Register the project's URL alias BEFORE the executor runs so the SDK
        # can use the token when encoding image paths. Mirrors the async path
        # in _submit_import_job — kept here so callers hitting /execute (tests,
        # scripts) get the same alias registration that interactive imports do.
        _maybe_register_alias(data, _get_image_folder(format_name, data))

        try:
            result: dict[str, Any] = executor(data)
            return result
        except Exception as exc:
            logger.exception("Import failed for format %s", format_name)
            msg = _enhance_error_message(str(exc))
            return {"success": False, "message": msg, "table_url": None, "details": {}}

    @post("/csv/parse", status_code=200)
    async def csv_parse(
        data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
    ) -> dict[str, Any]:
        """Upload and parse a CSV/Excel file, return column metadata + preview."""
        file_bytes = await data.read()
        filename = data.filename or "upload.csv"

        try:
            result = _parse_csv_file(file_bytes, filename)
        except Exception as exc:
            logger.exception("Failed to parse uploaded file: %s", filename)
            return {"error": str(exc), "columns": [], "preview": [], "total_rows": 0}

        # Store for later execution
        import uuid

        session_id = str(uuid.uuid4())
        _parsed_csv_files[session_id] = {"bytes": file_bytes, "filename": filename}

        # Limit stored files to 10
        if len(_parsed_csv_files) > 10:
            oldest = next(iter(_parsed_csv_files))
            del _parsed_csv_files[oldest]

        result["session_id"] = session_id
        return result

    @post("/upload-temp", status_code=200)
    async def upload_temp(
        data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
    ) -> dict[str, Any]:
        """Upload a file to a temp directory and return the server-side path.

        Used by drag-and-drop form fields that need a server-side file path.
        """
        import tempfile

        file_bytes = await data.read()
        filename = data.filename or "upload"

        # Write to a temp dir that persists until the process ends
        tmp_dir = Path(tempfile.gettempdir()) / "tlc-uploads"
        tmp_dir.mkdir(exist_ok=True)
        dest = tmp_dir / filename
        dest.write_bytes(file_bytes)

        return {"path": str(dest), "filename": filename}

    return [
        list_formats,
        parse_yolo,
        parse_coco,
        execute_import,
        csv_parse,
        upload_temp,
    ]
