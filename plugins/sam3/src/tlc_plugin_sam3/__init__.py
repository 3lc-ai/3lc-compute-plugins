# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""SAM3 plugin — sidebar plugin for auto-labeling with SAM3.

Job execution uses the unified ``run_job(ctx)`` contract: the host JobManager
owns the queue / cancel / generic progress, while this plugin re-emits its own
``/sam3`` SocketIO events (``sam3_log`` / ``sam3_progress``) via ``ctx.emit`` for
its embedded UI. ``run_job`` dispatches on a ``mode`` param:
``predict`` | ``create_table`` | ``create_and_predict``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tlc_plugin_sam3 import routes as _routes
from tlc_plugin_sdk import ComputePlugin

if TYPE_CHECKING:
    from tlc_plugin_sdk.job_context import JobContext

logger = logging.getLogger(__name__)


class SAM3Plugin(ComputePlugin):
    """Sidebar plugin for SAM3 auto-labeling: preview, table creation, and prediction."""

    # Display identity stamped onto the instance by the host from the manifest.
    id: str
    name: str
    icon: str

    _ui_cache: str | None = None

    def initialise_runtime(self) -> None:
        """No runtime resources to set up (job execution is host-managed via run_job)."""

    def get_ui_fragment(self) -> str:
        """Return the SAM3 UI HTML+JS+CSS fragment."""
        if self._ui_cache is None:
            from tlc_plugin_sdk.shared.alias_ui import alias_ui_script

            ui_path = Path(__file__).resolve().parent / "ui.html"
            raw = ui_path.read_text(encoding="utf-8")
            # Inject shared alias UI JS before the first <script> content
            self._ui_cache = raw.replace(
                "<script>",
                "<script>\n" + alias_ui_script(),
                1,  # only the first <script> tag
            )
        return self._ui_cache

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Not used — SAM3 uses dedicated REST endpoints."""
        return {"status": "Use /api/plugins/sam3/* endpoints"}

    def run_job(self, ctx: JobContext) -> None:
        """Run a SAM3 job (predict / create_table / create_and_predict).

        Driven entirely by ``ctx``: ``ctx.progress`` / ``ctx.metric`` / ``ctx.log``
        feed the generic Queue & Progress panel, while ``ctx.emit`` re-broadcasts the
        plugin's own ``/sam3`` events (``sam3_log`` / ``sam3_progress``) for the
        embedded UI. Cancellation is cooperative via ``ctx.cancelled``. The whole job
        is GPU-serialized by the host; for ``create_and_predict`` the (CPU) create
        step runs in the same GPU slot, immediately before the predict step.

        Args:
            ctx: Host-provided job context. ``ctx.params`` carries ``mode`` plus the
                mode-specific request fields.

        """
        params = ctx.params
        mode = str(params.get("mode", "predict") or "predict")
        config_id = (str(params.get("config_id", "") or "").strip()) or None

        try:
            if mode == "predict":
                _run_predict(ctx)
            elif mode == "create_table":
                _run_create_table(ctx)
            elif mode == "create_and_predict":
                table_url = _run_create_table(ctx)
                if ctx.cancelled:
                    return
                if table_url:
                    # Chain predict on the freshly created table, in the same job.
                    ctx.log(f"Table created at {table_url} — running predict")
                    _run_predict(ctx, table_url=table_url)
            else:
                msg = f"Unknown mode: {mode}"
                raise ValueError(msg)

            # Update last_run timestamp on the config (best-effort).
            if config_id and not ctx.cancelled:
                try:
                    from tlc_plugin_sam3.config_store import config_store

                    config_store().update_last_run(config_id)
                except Exception:
                    logger.debug("Could not update last_run for config %s", config_id, exc_info=True)

        except Exception as exc:
            logger.exception("sam3 run_job failed")
            ctx.emit("sam3_log", {"message": f"Error: {exc}"})
            raise

    def get_route_handlers(self) -> list[Any]:
        """Serve SAM3's custom routes as relative Litestar handlers (host + venv)."""
        return _routes.get_route_handlers()


def _log(ctx: JobContext, msg: str) -> None:
    """Emit a SAM3 log line to both the generic panel and the plugin's UI."""
    ctx.emit("sam3_log", {"message": msg})
    ctx.log(msg)


def _run_predict(ctx: JobContext, table_url: str = "") -> None:
    """Run SAM3 prediction on a 3LC table.

    Args:
        ctx: Job context. ``ctx.params`` carries ``confidence``, ``embedding_dim``,
            ``device``, ``run_name`` and (for standalone predict) ``table_url``.
        table_url: Overrides ``ctx.params["table_url"]`` — set by the
            ``create_and_predict`` chain to point at the freshly created table.

    """
    import numpy as np

    from tlc_plugin_sam3.inference import (
        get_box_embeddings,
        get_image_embedding,
        get_instance_embeddings,
        predict_single_image_with_state,
    )

    params = ctx.params
    confidence = float(params.get("confidence", 0.2))
    embedding_dim = int(params.get("embedding_dim", 2))
    device = str(params.get("device", "cuda") or "cuda")
    run_name = str(params.get("run_name", "") or "")
    table_url = table_url or str(params.get("table_url", "") or "")

    if not run_name:
        from tlc_plugin_sdk.shared.naming import generate_name

        run_name = generate_name()

    import tlc

    from tlc_plugin_sdk.shared.images import get_image_column, read_image_from_table

    _log(ctx, f"Loading table: {table_url}")
    table = tlc.Table.from_url(table_url)
    total = len(table)
    _log(ctx, f"Table has {total} images")

    image_column = get_image_column(table)
    _log(ctx, f"Using image column: {image_column}")

    # Always read labels and modality from the table schema
    labels = _read_labels_from_table(table)
    modality = _read_modality_from_table(table)
    _log(ctx, f"Read from table schema — labels: {labels}, modality: {modality}")
    if not labels:
        msg = "Could not read labels from table schema"
        raise ValueError(msg)

    _log(ctx, f"Labels: {labels}, modality: {modality}, confidence: {confidence}")
    _log(ctx, f"Run name: {run_name}")

    # Initialise 3LC run
    run = tlc.init(
        project_name=table.project_name,
        run_name=run_name,
        description=f"SAM3 predictions: {', '.join(labels)} ({modality})",
    )

    # Prediction loop
    all_predictions: list[Any] = []
    all_image_embeddings: list[np.ndarray] = []
    all_instance_embeddings: list[np.ndarray] = []

    job_start = time.monotonic()
    image_times: list[float] = []

    # Per-class running counts for stats
    class_counts: dict[str, int] = dict.fromkeys(labels, 0)

    for idx in range(total):
        if ctx.cancelled:
            _log(ctx, f"Cancelled at image {idx}/{total}")
            run.set_status_cancelled()
            return

        img_start = time.monotonic()
        image = read_image_from_table(table, idx, image_column)

        preds, state = predict_single_image_with_state(image, labels, confidence, device)

        # Image embedding
        img_emb = get_image_embedding(state)
        all_image_embeddings.append(img_emb)

        # Instance embeddings
        if modality == "bbox":
            inst_emb = get_box_embeddings(state, preds["boxes"], image.size)
        else:
            inst_emb = get_instance_embeddings(state, preds["masks_tensor"])
        all_instance_embeddings.append(inst_emb)

        # Build prediction data in 3LC format
        if modality == "bbox":
            n = len(preds["scores"])
            bboxes_arr = (
                np.asarray(preds["boxes"], dtype=np.float32).reshape(-1, 4) if n else np.empty((0, 4), dtype=np.float32)
            )
            all_predictions.append(
                tlc.data_types.BoundingBoxes2D(
                    bounding_boxes=bboxes_arr,
                    labels=np.asarray(preds["labels"], dtype=np.int32),
                    confidences=np.asarray(preds["scores"], dtype=np.float32),
                    x_max=float(image.width),
                    y_max=float(image.height),
                )
            )
        else:
            masks_np = preds["masks"]
            if masks_np:
                stacked = np.stack(masks_np, axis=-1).astype(np.uint8)  # (H, W, N)
            else:
                stacked = np.empty((image.height, image.width, 0), dtype=np.uint8)

            all_predictions.append(
                tlc.data_types.SegmentationMasks(
                    masks=stacked,
                    labels=np.asarray(preds["labels"], dtype=np.int32),
                    confidences=np.asarray(preds["scores"], dtype=np.float32),
                    image_width=image.width,
                    image_height=image.height,
                )
            )

        # Per-class counts
        for lbl_idx in preds["labels"]:
            lbl_name = labels[lbl_idx] if isinstance(lbl_idx, int) and lbl_idx < len(labels) else str(lbl_idx)
            class_counts[lbl_name] = class_counts.get(lbl_name, 0) + 1

        # Timing
        elapsed_img = time.monotonic() - img_start
        image_times.append(elapsed_img)
        elapsed = time.monotonic() - job_start
        avg = sum(image_times) / len(image_times)
        eta = avg * (total - idx - 1)
        pct = (idx + 1) / total * 100
        images_done = idx + 1

        # Per-class averages
        class_avg = {name: round(count / images_done, 1) for name, count in class_counts.items()}

        timing = {
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1),
            "avg_step_s": round(avg, 1),
            "step_label": "image",
        }
        label = f"Image {idx + 1}/{total}"
        ctx.progress(percent=round(pct, 1), label=label, timing=timing)
        ctx.emit(
            "sam3_progress",
            {
                "percent": round(pct, 1),
                "label": label,
                "timing": timing,
                "current_scores": len(preds["scores"]),
                "class_avg": class_avg,
            },
        )

    if ctx.cancelled:
        run.set_status_cancelled()
        return

    # Post-processing: UMAP reduction + metrics collection
    run.set_status_collecting()
    ctx.progress(percent=100.0, label="Reducing embeddings...")
    ctx.emit(
        "sam3_progress",
        {
            "percent": 100.0,
            "label": "Reducing embeddings...",
            "phase": "processing",
            "phase_label": "Reducing embeddings...",
        },
    )
    _log(ctx, f"Reducing embeddings to {embedding_dim}D with UMAP...")

    image_emb_array = np.array(all_image_embeddings)
    reduced_image_emb = _reduce_embeddings(image_emb_array, embedding_dim)

    # Instance embeddings
    flat_inst = []
    for emb in all_instance_embeddings:
        if emb.shape[0] > 0:
            for row in emb:
                flat_inst.append(row)

    # Reduce instance embeddings and redistribute per-image
    all_instance_emb_reduced: list[list[list[float]]] = []
    if flat_inst:
        flat_array = np.array(flat_inst)
        reduced_inst = _reduce_embeddings(flat_array, embedding_dim)

        ri = 0
        for idx, emb in enumerate(all_instance_embeddings):
            n = emb.shape[0]
            if n > 0:
                reduced_arr = np.asarray(reduced_inst[ri : ri + n], dtype=np.float32)
                ri += n
            else:
                reduced_arr = np.empty((0, embedding_dim), dtype=np.float32)
            all_instance_emb_reduced.append(reduced_arr.tolist())
            # For segmentation, embed inside the SegmentationMasks per_instance_extras
            if modality != "bbox":
                all_predictions[idx].per_instance_extras["instance_embedding"] = reduced_arr
    else:
        empty_arr = np.empty((0, embedding_dim), dtype=np.float32)
        for idx in range(len(all_predictions)):
            all_instance_emb_reduced.append([])
            if modality != "bbox":
                all_predictions[idx].per_instance_extras["instance_embedding"] = empty_arr

    # Write metrics to 3LC run
    _log(ctx, "Writing metrics to 3LC run...")

    image_emb_schema = tlc.Schema(
        value=tlc.schemas.values.Float32Value(number_role="xy_component" if embedding_dim == 2 else "xyz_component"),
        size0=tlc.schemas.values.DimensionNumericValue(embedding_dim, embedding_dim),
        display_name=f"Image Embedding ({embedding_dim}D)",
        display_importance=9,
    )

    pred_col = (
        f"predicted_{tlc.constants.BOUNDING_BOXES}"
        if modality == "bbox"
        else f"predicted_{tlc.constants.SEGMENTATIONS}"
    )
    inst_emb_schema = tlc.Schema(
        value=tlc.schemas.values.Float32Value(),
        size0=tlc.schemas.values.DimensionNumericValue(embedding_dim, embedding_dim),
        size1=tlc.schemas.values.DimensionNumericValue(0, 10000),
        display_name=f"Instance Embedding ({embedding_dim}D)",
        display_importance=8,
    )
    if modality == "bbox":
        pred_schema = tlc.data_types.BoundingBoxes2D.schema(classes=labels, include_per_instance_confidence=True)
    else:
        pred_schema = tlc.data_types.SegmentationMasks.schema(
            classes=labels,
            include_per_instance_confidence=True,
            per_instance_schemas={"instance_embedding": inst_emb_schema},
        )

    column_schemas = {
        pred_col: pred_schema,
        "image_embedding": image_emb_schema,
    }
    metrics_data: dict[str, Any] = {
        pred_col: all_predictions,
        "image_embedding": reduced_image_emb,
    }

    # For bbox mode, instance embeddings go as a separate column
    if modality == "bbox":
        column_schemas["instance_embeddings"] = inst_emb_schema
        metrics_data["instance_embeddings"] = all_instance_emb_reduced

    ctx.emit(
        "sam3_progress",
        {"percent": 100.0, "label": "Writing metrics...", "phase": "processing", "phase_label": "Writing metrics..."},
    )
    _log(ctx, "Writing metrics to run...")

    run.add_metrics(
        metrics=metrics_data,
        schema=column_schemas,
        foreign_table_url=table.url,
    )
    run.set_status_completed()
    ctx.metric("images", total)
    ctx.metric("run", str(run.url))
    _log(ctx, f"Done! Run URL: {run.url}")


def _run_create_table(ctx: JobContext) -> str:
    """Create a 3LC table from an image folder or an existing table.

    Args:
        ctx: Job context. ``ctx.params`` carries ``folder`` OR ``source_table_url``,
            plus ``labels``, ``modality``, ``project_name``, ``dataset_name``,
            ``table_name``, alias fields, and optional ``max_images``.

    Returns:
        The URL of the created table, or ``""`` if cancelled before finalize.

    """
    from tlc_plugin_sam3.inference import list_images_in_folder

    params = ctx.params

    # Apply alias overrides if requested (so image paths use the right token)
    alias_originals: list[dict[str, str]] = []
    alias_ov = params.get("_alias_overrides", None)
    if isinstance(alias_ov, dict) and alias_ov.get("enabled") and alias_ov.get("overrides"):
        from tlc_plugin_sdk.shared.aliases import apply_alias_overrides

        alias_originals = apply_alias_overrides(alias_ov["overrides"])
        if alias_originals:
            _log(ctx, f"Applied {len(alias_originals)} alias override(s)")

    folder = str(params.get("folder", "") or "")
    source_table_url = str(params.get("source_table_url", "") or "")
    labels = params["labels"]
    modality = params["modality"]
    project_name = params["project_name"]
    dataset_name = params.get("dataset_name", "train")
    table_name = params.get("table_name", "initial")

    import tlc

    try:
        # Resolve image paths from either an existing table or a folder
        if source_table_url:
            from tlc_plugin_sdk.shared.images import get_image_column, get_image_paths
            from tlc_plugin_sdk.shared.url_utils import normalize_url

            _log(ctx, f"Reading images from table: {source_table_url}")
            source_table = tlc.Table.from_url(normalize_url(source_table_url))
            image_column = get_image_column(source_table)
            _log(ctx, f"Reading image paths from column '{image_column}'")
            # Absolutized, since the paths are written into a new table at a
            # different location.
            image_paths = get_image_paths(source_table, image_column)
            if not image_paths:
                msg = f"No images found in table {source_table_url}"
                raise ValueError(msg)
        else:
            _log(ctx, f"Scanning images in: {folder}")
            image_paths = list_images_in_folder(folder)
            if not image_paths:
                msg = f"No images found in {folder}"
                raise ValueError(msg)

        # Limit to max_images if specified
        max_images = int(params.get("max_images", 0) or 0)
        if max_images and max_images > 0 and len(image_paths) > max_images:
            image_paths = image_paths[:max_images]

        total = len(image_paths)
        _log(ctx, f"Found {total} images" + (f" (limited to {max_images})" if max_images else ""))

        # Build schema based on modality — column names match tlc_ultralytics conventions
        if modality == "bbox":
            annotation_schema = tlc.data_types.BoundingBoxes2D.schema(classes=labels)
            annotation_column = tlc.constants.BOUNDING_BOXES
        else:
            annotation_schema = tlc.data_types.SegmentationPolygons.schema(classes=labels)
            annotation_column = tlc.constants.SEGMENTATIONS

        schemas: dict[str, Any] = {
            "id": tlc.schemas.Int32Schema(writable=False),
            "image": tlc.schemas.ImageSchema(sample_type="url"),
            annotation_column: annotation_schema,
            "review": tlc.schemas.CategoricalLabelSchema(classes=["Open", "Done"], display_name="review"),
            "weight": tlc.schemas.SampleWeightSchema(),
        }

        _log(ctx, f"Creating table: {project_name}/{dataset_name}/{table_name}")
        writer = tlc.TableWriter(
            table_name=table_name,
            dataset_name=dataset_name,
            project_name=project_name,
            description=f"SAM3 dataset: {', '.join(labels)} ({modality})",
            schema=schemas,
            if_exists="overwrite",
        )

        # Read each image's real dimensions so the empty annotation carries the
        # correct image_width/image_height (bbox x_max/y_max) up front. The
        # input may be images alone — with no table to read sizes from — so we
        # must open every image. ``read_image_size`` reads only the header, not
        # the full image, and works on any storage backend.
        from tlc_plugin_sdk.shared.images import read_image_size

        _log(ctx, f"Reading dimensions for {total} images")
        image_dims: list[tuple[int, int]] = []
        for idx, image_path in enumerate(image_paths):
            if ctx.cancelled:
                _log(ctx, f"Cancelled while reading dimensions at image {idx}/{total}")
                return ""
            try:
                image_dims.append(read_image_size(image_path))
            except Exception:
                logger.warning("Could not read dimensions for %s; storing 0x0", image_path, exc_info=True)
                image_dims.append((0, 0))
            if (idx + 1) % 500 == 0:
                _log(ctx, f"Read dimensions for {idx + 1}/{total} images")

        empty_annotations: list[Any]
        if modality == "bbox":
            empty_annotations = [
                tlc.data_types.BoundingBoxes2D.create_empty(image_width=w, image_height=h) for (w, h) in image_dims
            ]
        else:
            empty_annotations = [
                tlc.data_types.SegmentationPolygons.create_empty(image_width=w, image_height=h) for (w, h) in image_dims
            ]

        writer.add_batch({
            "image": image_paths,
            annotation_column: empty_annotations,
            "review": [0] * total,
            "weight": [0.0] * total,
        })
        table = writer.finalize()
        table_url = str(table.url)
        ctx.metric("images", total)
        ctx.metric("table", table_url)
        _log(ctx, f"Created table: {table.url} ({total} images)")

        # Register URL alias if requested (only for folder-based sources)
        if folder and params.get("alias_enabled", True):
            from tlc_plugin_sdk.shared.aliases import default_alias_token, register_alias

            token = str(params.get("alias_token", "") or "").strip() or default_alias_token(project_name)
            alias_folder = str(params.get("alias_folder", "") or "").strip() or folder
            register_alias(project_name=project_name, image_folder=alias_folder, alias_token=token)
            _log(ctx, f"Registered alias <{token}> → {alias_folder}")

        return table_url
    finally:
        # Restore alias overrides
        if alias_originals:
            from tlc_plugin_sdk.shared.aliases import restore_aliases

            restore_aliases(alias_originals)


def _reduce_embeddings(embeddings: Any, n_components: int) -> Any:
    """Reduce high-dimensional embeddings with UMAP, with fallback for small datasets."""
    import numpy as np

    n_samples = embeddings.shape[0]
    if n_samples <= n_components:
        # Too few samples for UMAP — just truncate/pad to target dimensions
        return (
            embeddings[:, :n_components] if embeddings.shape[1] >= n_components else np.zeros((n_samples, n_components))
        )

    try:
        import umap

        n_neighbors = min(15, max(2, n_samples - 1))
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors)
        return reducer.fit_transform(embeddings)
    except Exception:
        logger.warning("UMAP failed, using PCA fallback", exc_info=True)
        # PCA fallback
        centered = embeddings - embeddings.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            return centered @ vt[:n_components].T
        except Exception:
            return np.zeros((n_samples, n_components))


def _read_labels_from_table(table: Any) -> list[str]:
    """Read class labels from a table's schema, ordered by class index."""
    from tlc_plugin_sdk.shared.labels import get_label_names

    return get_label_names(table)


def _read_modality_from_table(table: Any) -> str:
    """Detect modality from table schema: 'segmentation' or 'bbox'."""
    from tlc_plugin_sdk.shared.modality import detect_modality_from_table

    try:
        info = detect_modality_from_table(table)
        if info.modality == "detection":
            return "bbox"
        return "segmentation"
    except Exception:
        logger.debug("Could not read modality from table schema", exc_info=True)
        return "segmentation"
