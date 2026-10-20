# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Custom routes for the Export plugin, as relative Litestar route handlers.

Returned by ``ExportPlugin.get_route_handlers()`` and served by the plugin's own
app (in-process for host mode, reverse-proxied for venv) under
``/api/plugins/exporter/`` — no static node on the main app, so nothing shadows
the generic ``/run`` route. Handlers are ``def`` (Litestar runs them in a
threadpool) because they touch the ``tlc`` SDK and the filesystem, which block.
"""

from __future__ import annotations

import contextvars
import logging
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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


def run_export(data: dict[str, Any], *, progress: UploadProgress | None = None) -> dict[str, Any]:
    """Validate and execute one export request.

    Shared by ``ExportPlugin.run_job`` (the job channel the UI drives) and the
    legacy synchronous ``/execute`` route. The output may be a local path or a
    bucket URL (the picker offers both): a bucket export is written to a local
    scratch folder first and then uploaded, so every format's writer keeps
    writing files.

    Args:
        data: ``format``, ``table_url``, ``output_path`` and format-specific options.
        progress: Optional ``(files_done, files_total)`` callback for the upload of a
            bucket export.

    Returns:
        ``{"success": True, "message": str, "details": dict}`` on success, or
        ``{"success": False, "message": str}`` on failure. Never raises.

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

    from tlc_plugin_sdk.shared.url_utils import is_url, normalize_path_or_url

    try:
        output_path = normalize_path_or_url(output_path)
    except ValueError as exc:
        return {"success": False, "message": str(exc)}

    executor = _EXECUTORS.get(format_name)
    if not executor:
        return {"success": False, "message": f"No executor for format: {format_name}"}

    try:
        if is_url(output_path):
            return _export_to_url(executor, table_url, output_path, data, progress)
        result: dict[str, Any] = executor(table_url, output_path, data)
        return result
    except Exception as exc:
        logger.exception("Export failed for format %s", format_name)
        return {"success": False, "message": f"Export failed: {exc}", "details": {}}


UploadProgress = Callable[[int, int], None]
"""``(files_done, files_total)`` — called after every uploaded file of a bucket export."""


def _export_to_url(
    executor: Callable[[str, str, dict[str, Any]], dict[str, Any]],
    table_url: str,
    output_url: str,
    data: dict[str, Any],
    progress: UploadProgress | None,
) -> dict[str, Any]:
    """Run *executor* into a scratch folder, then upload what it wrote under *output_url*.

    An output with a file suffix (``…/export.csv``, ``…/dataset.yaml``) names the file in its
    parent prefix; anything else is the prefix itself. Whatever the writer puts beside the file
    (YOLO's labels and images) lands beside it at the destination too. Existing objects are
    overwritten, as a local export overwrites files.
    """
    from tlc_plugin_sdk.shared.url_utils import name_of, parent_of, suffix_of

    with tempfile.TemporaryDirectory(prefix="tlc-export-") as tmp:
        scratch = Path(tmp) / "out"
        scratch.mkdir()
        # A bucket root (``s3://my.bucket``) is a prefix even when its name has a dot in it.
        if output_url.rstrip("/").count("/") > 2 and suffix_of(output_url):
            local_target, prefix = scratch / name_of(output_url), parent_of(output_url)
        else:
            local_target, prefix = scratch, output_url
        result = executor(table_url, str(local_target), data)
        if result.get("success"):
            count = _upload_folder(scratch, prefix, progress)
            result = _rebase_paths(result, str(scratch), prefix.rstrip("/"))
            result.setdefault("details", {})["uploaded_files"] = count
        return result


def _upload_folder(root: Path, prefix: str, progress: UploadProgress | None, workers: int = 8) -> int:
    """Write every file under *root* to ``<prefix>/<relative path>``, overwriting; return the count.

    Raises:
        RuntimeError: A file could not be written (the first failure, after the uploads in flight).

    """
    import tlc

    base = prefix.rstrip("/")
    files = sorted(p for p in root.rglob("*") if p.is_file())

    def put(path: Path) -> None:
        tlc.Url(base + "/" + path.relative_to(root).as_posix()).write_bytes(path.read_bytes())

    done = 0
    first_error: Exception | None = None
    # Each upload runs in a copy of the caller's context, so the job's Connection (the SDK binds
    # its credential in a context variable) holds in the pool threads as it does here.
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for path, fut in [(p, pool.submit(context.copy().run, put, p)) for p in files]:
            try:
                fut.result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                    logger.warning("Upload failed for %s → %s: %s", path, base, exc)
                continue
            done += 1
            if progress is not None:
                progress(done, len(files))
    if first_error is not None:
        msg = f"Could not upload the export to {base}: {first_error}"
        raise RuntimeError(msg) from first_error
    return done


def _rebase_paths(value: Any, local: str, remote: str) -> Any:
    """Replace the scratch folder with the destination prefix in every string of an executor result."""
    if isinstance(value, str):
        return value.replace(local, remote)
    if isinstance(value, dict):
        return {k: _rebase_paths(v, local, remote) for k, v in value.items()}
    if isinstance(value, list):
        return [_rebase_paths(v, local, remote) for v in value]
    return value


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

            for name in first_row:
                col_schema = schema_values.get(name)
                col_type = _classify_column_type(col_schema, name) if col_schema else "other"
                columns.append({"name": name, "type": col_type})

            # project_name rides along so the UI can attribute the export job to the
            # table's own project — the launch context is empty for bare sidebar launches.
            return {"columns": columns, "row_count": table.row_count, "project_name": table.project_name or ""}
        except Exception:
            logger.warning("Failed to list columns for %s", table_url, exc_info=True)
            return {"error": "Failed to load table columns"}

    @post("/execute", status_code=200, sync_to_thread=True)
    def execute_export(data: dict[str, Any]) -> dict[str, Any]:
        """Validate and execute an export synchronously (legacy path).

        The UI drives exports through the generic job channel (``POST /run`` →
        ``ExportPlugin.run_job``) so long exports outlive any request timeout;
        this route remains for scripted/direct callers with the same body.

        Args:
            data: JSON body with ``format``, ``table_url``, ``output_path``, and
                format-specific options.

        Returns:
            ``{"success": True, "message": str, "details": dict}`` on success, or
            ``{"success": False, "message": str}`` on failure.

        """
        return run_export(data)

    return [
        list_formats,
        list_columns,
        execute_export,
    ]
