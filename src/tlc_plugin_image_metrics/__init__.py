# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image Metrics plugin — compute image quality metrics for 3LC tables.

Analyzes images across configurable metrics (brightness, sharpness, noise, etc.)
and creates an EditedTable with new metric columns preserving full lineage.

Job execution uses the unified ``run_job(ctx)`` contract: the host JobManager owns
the queue / cancel / generic progress, while this plugin re-emits its own
``/image-metrics`` SocketIO events (status / progress / complete / error) via
``ctx.emit`` for its embedded UI.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tlc_plugin_sdk import ComputePlugin

from tlc_plugin_image_metrics import routes as _routes

if TYPE_CHECKING:
    from tlc_plugin_sdk.job_context import JobContext

logger = logging.getLogger(__name__)

MAX_WORKERS = 8  # parallel image loaders


def _detect_image_columns(table_url: str) -> list[str]:
    """Detect image columns in a table using modality detection.

    Args:
        table_url: Table URL string.

    Returns:
        List of image column names.

    """
    import tlc
    from tlc_plugin_sdk.shared.modality import detect_modality_from_table
    from tlc_plugin_sdk.shared.url_utils import normalize_url

    table = tlc.Table.from_url(normalize_url(table_url))
    info = detect_modality_from_table(table)
    return info.image_columns


def _create_edited_table(
    table: Any,
    results_data: dict[str, dict[str, list[float]]],
    image_columns: list[str],
    metric_ids: list[str],
    multi_col: bool,
    output_name: str,
) -> str:
    """Create an EditedTable with metrics columns preserving lineage.

    Args:
        table: Source tlc.Table.
        results_data: Nested dict [col][metric_id] → list[float].
        image_columns: Image column names processed.
        metric_ids: Metric IDs computed.
        multi_col: Whether to prefix column names with image column name.
        output_name: Name for the output table.

    Returns:
        URL string of the created table.

    """
    import tlc
    from tlc._core.objects.tables.from_table.edited_table import EditedTable

    new_schemas: dict[str, Any] = {}
    edits: dict[str, Any] = {}

    for col in image_columns:
        col_data = results_data[col]
        for mid in metric_ids:
            col_name = f"{col}_{mid}" if multi_col else mid
            new_schemas[col_name] = tlc.schemas.Float32Schema()

            values = col_data[mid]
            runs_and_values: list[Any] = []
            for row_idx, val in enumerate(values):
                runs_and_values.append([row_idx])
                runs_and_values.append(val)
            edits[col_name] = {"runs_and_values": runs_and_values}

    base_url = str(table.url).rsplit("/", 1)[0]
    new_table = EditedTable(
        input_table_url=table,
        override_table_rows_schema={"values": new_schemas},
        edits=edits,
        description=f"Image metrics: {', '.join(metric_ids)}",
        url=tlc.Url(f"{base_url}/{output_name}"),
    )
    new_table.ensure_fully_defined()

    return str(new_table.url)


class ImageMetricsPlugin(ComputePlugin):
    """Compute image quality metrics for every image in a 3LC table."""

    # Display identity stamped onto the instance by the host from the manifest.
    id: str
    name: str
    icon: str

    _ui_cache: str | None = None

    def get_ui_fragment(self) -> str:
        """Return the self-contained Image Metrics UI HTML+JS+CSS fragment."""
        if self._ui_cache is None:
            from tlc_plugin_sdk.shared.config_ui import config_ui_script
            from tlc_plugin_sdk.shared.ui_inject import inject_scripts

            ui_path = Path(__file__).resolve().parent / "ui.html"
            raw = ui_path.read_text(encoding="utf-8")
            self._ui_cache = inject_scripts(raw, config_ui_script())
        return self._ui_cache

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Not used — Image Metrics runs jobs via run_job + dedicated REST endpoints."""
        return {"status": "Use POST /api/plugins/image-metrics/run"}

    def run_job(self, ctx: JobContext) -> None:
        """Compute image metrics for a table and write an EditedTable with lineage.

        Driven entirely by ``ctx``: ``ctx.progress`` / ``ctx.metric`` / ``ctx.log`` /
        ``ctx.result`` feed the generic Queue & Progress panel, while ``ctx.emit`` re-broadcasts the
        plugin's own ``/image-metrics`` events (``status`` / ``progress`` /
        ``complete`` / ``error``) for the embedded UI. Cancellation is cooperative
        via ``ctx.cancelled``.

        Args:
            ctx: Host-provided job context. ``ctx.params`` carries ``table_url``,
                ``metric_ids``, optional ``image_columns`` (auto-detected if empty),
                and optional ``output_name``.

        """

        def _status(message: str) -> None:
            ctx.emit("status", {"message": message})
            ctx.log(message)

        params = ctx.params
        table_url = str(params.get("table_url", "") or "")
        metric_ids = list(params.get("metric_ids", []) or [])
        image_columns = list(params.get("image_columns", []) or [])
        output_name = str(params.get("output_name", "") or "")

        try:
            # Validate request params before importing the (heavy) tlc SDK.
            if not table_url:
                msg = "table_url is required"
                raise ValueError(msg)
            if not metric_ids:
                msg = "metric_ids is required (non-empty list)"
                raise ValueError(msg)

            import tlc
            from tlc_plugin_sdk.shared.images import get_image_paths, load_image
            from tlc_plugin_sdk.shared.url_utils import normalize_url

            from tlc_plugin_image_metrics.metrics import METRIC_BY_ID, compute_metrics_for_image

            for mid in metric_ids:
                if mid not in METRIC_BY_ID:
                    msg = f"Unknown metric: {mid}"
                    raise ValueError(msg)
            if not image_columns:
                image_columns = _detect_image_columns(table_url)
                if not image_columns:
                    msg = "No image columns found in table"
                    raise ValueError(msg)

            table_name = table_url.rstrip("/").rsplit("/", 1)[-1] if "/" in table_url else table_url
            if not output_name:
                output_name = f"{table_name}-metrics"

            _status("Loading table...")
            table = tlc.Table.from_url(normalize_url(table_url))
            row_count = len(table)

            # Read all image paths up front (absolutized), validating the columns.
            paths_by_col: dict[str, list[str]] = {col: get_image_paths(table, col) for col in image_columns}

            multi_col = len(image_columns) > 1
            total_tasks = row_count * len(image_columns)
            _status(f"Processing {row_count} images...")

            results_data: dict[str, dict[str, list[float]]] = {}
            processed = 0

            for col in image_columns:
                col_paths = paths_by_col[col]
                col_results: dict[str, list[float]] = {mid: [0.0] * row_count for mid in metric_ids}
                failed_rows = 0
                first_error: str | None = None

                def _load_and_compute(idx: int, col_paths: list[str] = col_paths) -> tuple[int, dict[str, float]]:
                    """Load one image and compute all metrics."""
                    img_path = col_paths[idx]
                    if not img_path:
                        return idx, {mid: float("nan") for mid in metric_ids}
                    return idx, compute_metrics_for_image(load_image(img_path), metric_ids)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                    futures = {pool.submit(_load_and_compute, i): i for i in range(row_count)}

                    for future in as_completed(futures):
                        if ctx.cancelled:
                            pool.shutdown(wait=False, cancel_futures=True)
                            _status("Cancelled")
                            return

                        try:
                            idx, metric_vals = future.result()
                        except Exception as exc:
                            idx = futures[future]
                            failed_rows += 1
                            if first_error is None:
                                first_error = str(exc)
                            logger.warning("Failed to process row %d col %s: %s", idx, col, exc)
                            metric_vals = {mid: float("nan") for mid in metric_ids}

                        for mid, val in metric_vals.items():
                            col_results[mid][idx] = val

                        processed += 1
                        pct = processed / total_tasks * 100 if total_tasks else 100.0
                        label = f"Row {processed}/{total_tasks}"
                        ctx.progress(percent=round(pct, 1), label=label)
                        if processed % 10 == 0 or processed == total_tasks:
                            ctx.emit(
                                "progress",
                                {
                                    "percent": round(pct, 1),
                                    "processed": processed,
                                    "total": total_tasks,
                                    "label": label,
                                },
                            )

                # Every row failing means the table's image data is unreadable (bad
                # alias, missing files, no cloud credentials) — fail with the cause
                # instead of writing an all-NaN run.
                if row_count and failed_rows == row_count:
                    msg = f"All {row_count} images failed to load for column '{col}'. First error: {first_error}"
                    raise ValueError(msg)
                if failed_rows:
                    ctx.log(
                        f"{failed_rows}/{row_count} rows failed for column '{col}' (metrics set to NaN). "
                        f"First error: {first_error}"
                    )

                results_data[col] = col_results

            if ctx.cancelled:
                _status("Cancelled")
                return

            _status("Creating output table...")
            result_url = _create_edited_table(
                table=table,
                results_data=results_data,
                image_columns=image_columns,
                metric_ids=metric_ids,
                multi_col=multi_col,
                output_name=output_name,
            )

            ctx.progress(percent=100.0, label="Done")
            ctx.result(result_url)
            ctx.metric("rows", row_count)
            ctx.metric("metric values", len(metric_ids) * len(image_columns) * row_count)

            # Extract project/dataset for the "View in Project" link.
            project_name = ""
            dataset_name = ""
            try:
                project_name = str(getattr(table, "project_name", ""))
                dataset_name = str(getattr(table, "dataset_name", ""))
                if not project_name:
                    parts = str(table.url).replace("\\", "/").split("/")
                    for i, part in enumerate(parts):
                        if part == "projects" and i + 1 < len(parts):
                            project_name = parts[i + 1]
                        if part == "datasets" and i + 1 < len(parts):
                            dataset_name = parts[i + 1]
            except Exception:
                pass

            ctx.emit(
                "complete",
                {
                    "result_url": result_url,
                    "table_name": output_name,
                    "row_count": row_count,
                    "metrics_count": len(metric_ids) * len(image_columns),
                    "project_name": project_name,
                    "dataset_name": dataset_name,
                },
            )
            _status("Complete")

        except Exception as exc:
            logger.exception("image metrics run_job failed")
            ctx.emit("error", {"message": str(exc)})
            raise

    def get_route_handlers(self) -> list[Any]:
        """Serve Image Metrics' custom routes as relative Litestar handlers (host + venv)."""
        return _routes.get_route_handlers()
