# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""YOLO model via 3lc-ultralytics package."""

from __future__ import annotations

import contextlib
import traceback
from pathlib import Path
from typing import Any, ClassVar

import tlc
from tlc_ultralytics import YOLO
from tlc_ultralytics.settings import Settings

from tlc_plugin_yolo.models import register_model
from tlc_plugin_yolo.models.base import BaseTrainingModel

# Map our task names to YOLO task names
_TASK_MAP = {
    "detection": "detect",
    "segmentation": "segment",
    "classification": "classify",
    "pose": "pose",
    "obb": "obb",
}


class YOLOModel(BaseTrainingModel):
    """YOLO training model integration."""

    name = "yolov8"
    display_name = "YOLO"
    supported_tasks: ClassVar[list[str]] = ["detection", "segmentation", "pose", "obb", "classification"]

    def get_params(self) -> list[dict[str, Any]]:
        """Return YOLO parameter field definitions."""
        return [
            # ── YOLO Settings ──
            {
                "id": "model",
                "label": "Checkpoint",
                "type": "select",
                "default": "yolov8n.pt",
                "options": [
                    # Detection
                    {"value": "yolov5nu.pt", "label": "YOLOv5n (nano)", "task": "detection"},
                    {"value": "yolov5su.pt", "label": "YOLOv5s (small)", "task": "detection"},
                    {"value": "yolov5mu.pt", "label": "YOLOv5m (medium)", "task": "detection"},
                    {"value": "yolov5lu.pt", "label": "YOLOv5l (large)", "task": "detection"},
                    {"value": "yolov5xu.pt", "label": "YOLOv5x (extra-large)", "task": "detection"},
                    {"value": "yolov8n.pt", "label": "YOLOv8n (nano)", "task": "detection"},
                    {"value": "yolov8s.pt", "label": "YOLOv8s (small)", "task": "detection"},
                    {"value": "yolov8m.pt", "label": "YOLOv8m (medium)", "task": "detection"},
                    {"value": "yolov8l.pt", "label": "YOLOv8l (large)", "task": "detection"},
                    {"value": "yolov8x.pt", "label": "YOLOv8x (extra-large)", "task": "detection"},
                    {"value": "yolo11n.pt", "label": "YOLO11n (nano)", "task": "detection"},
                    {"value": "yolo11s.pt", "label": "YOLO11s (small)", "task": "detection"},
                    {"value": "yolo11m.pt", "label": "YOLO11m (medium)", "task": "detection"},
                    {"value": "yolo11l.pt", "label": "YOLO11l (large)", "task": "detection"},
                    {"value": "yolo11x.pt", "label": "YOLO11x (extra-large)", "task": "detection"},
                    # Segmentation
                    {"value": "yolov8n-seg.pt", "label": "YOLOv8n-seg (nano)", "task": "segmentation"},
                    {"value": "yolov8s-seg.pt", "label": "YOLOv8s-seg (small)", "task": "segmentation"},
                    {"value": "yolov8m-seg.pt", "label": "YOLOv8m-seg (medium)", "task": "segmentation"},
                    {"value": "yolov8l-seg.pt", "label": "YOLOv8l-seg (large)", "task": "segmentation"},
                    {"value": "yolov8x-seg.pt", "label": "YOLOv8x-seg (extra-large)", "task": "segmentation"},
                    {"value": "yolo11n-seg.pt", "label": "YOLO11n-seg (nano)", "task": "segmentation"},
                    {"value": "yolo11s-seg.pt", "label": "YOLO11s-seg (small)", "task": "segmentation"},
                    {"value": "yolo11m-seg.pt", "label": "YOLO11m-seg (medium)", "task": "segmentation"},
                    {"value": "yolo11l-seg.pt", "label": "YOLO11l-seg (large)", "task": "segmentation"},
                    {"value": "yolo11x-seg.pt", "label": "YOLO11x-seg (extra-large)", "task": "segmentation"},
                    # Classification
                    {"value": "yolov8n-cls.pt", "label": "YOLOv8n-cls (nano)", "task": "classification"},
                    {"value": "yolov8s-cls.pt", "label": "YOLOv8s-cls (small)", "task": "classification"},
                    {"value": "yolov8m-cls.pt", "label": "YOLOv8m-cls (medium)", "task": "classification"},
                    {"value": "yolov8l-cls.pt", "label": "YOLOv8l-cls (large)", "task": "classification"},
                    {"value": "yolov8x-cls.pt", "label": "YOLOv8x-cls (extra-large)", "task": "classification"},
                    {"value": "yolo11n-cls.pt", "label": "YOLO11n-cls (nano)", "task": "classification"},
                    {"value": "yolo11s-cls.pt", "label": "YOLO11s-cls (small)", "task": "classification"},
                    {"value": "yolo11m-cls.pt", "label": "YOLO11m-cls (medium)", "task": "classification"},
                    {"value": "yolo11l-cls.pt", "label": "YOLO11l-cls (large)", "task": "classification"},
                    {"value": "yolo11x-cls.pt", "label": "YOLO11x-cls (extra-large)", "task": "classification"},
                ],
                "help": "Pre-trained model checkpoint.",
                "group": "YOLO Settings",
            },
            {
                "id": "epochs",
                "label": "Epochs",
                "type": "number",
                "default": 50,
                "min": 1,
                "max": 1000,
                "step": 1,
                "group": "YOLO Settings",
            },
            {
                "id": "batch",
                "label": "Batch Size",
                "type": "number",
                "default": 16,
                "min": 1,
                "max": 512,
                "step": 1,
                "group": "YOLO Settings",
            },
            {
                "id": "imgsz",
                "label": "Image Size",
                "type": "number",
                "default": 640,
                "min": 32,
                "max": 1280,
                "step": 32,
                "group": "YOLO Settings",
            },
            {
                "id": "lr0",
                "label": "Initial LR",
                "type": "number",
                "default": 0.01,
                "min": 0.0001,
                "max": 1.0,
                "step": 0.001,
                "group": "YOLO Settings",
            },
            {
                "id": "lrf",
                "label": "Final LR Factor",
                "type": "number",
                "default": 0.01,
                "min": 0.0001,
                "max": 1.0,
                "step": 0.001,
                "help": "Final LR = lr0 * lrf.",
                "group": "YOLO Settings",
            },
            {
                "id": "optimizer",
                "label": "Optimizer",
                "type": "select",
                "default": "auto",
                "options": [
                    {"value": "auto", "label": "Auto"},
                    {"value": "SGD", "label": "SGD"},
                    {"value": "Adam", "label": "Adam"},
                    {"value": "AdamW", "label": "AdamW"},
                ],
                "group": "YOLO Settings",
            },
            {
                "id": "patience",
                "label": "Patience",
                "type": "number",
                "default": 50,
                "min": 0,
                "max": 500,
                "step": 1,
                "help": "Early stopping patience (0 = disabled).",
                "group": "YOLO Settings",
            },
            {
                "id": "device",
                "label": "Device",
                "type": "text",
                "default": "",
                "help": "'' (auto), 'cpu', '0', '0,1', etc.",
                "group": "YOLO Settings",
            },
            {
                "id": "extra_args",
                "label": "Extra Arguments",
                "type": "text",
                "default": "",
                "help": "Additional YOLO args as key=value pairs, e.g. cos_lr=True weight_decay=0.001 warmup_epochs=5",
                "group": "YOLO Settings",
                "wide": True,
            },
            # ── 3LC Settings ──
            {
                "id": "image_embeddings_dim",
                "label": "Embeddings Dim",
                "type": "select",
                "default": "2",
                "options": [
                    {"value": "0", "label": "Disabled"},
                    {"value": "2", "label": "2D"},
                    {"value": "3", "label": "3D"},
                ],
                "help": "Image embedding dimensionality.",
                "group": "3LC Settings",
            },
            {
                "id": "image_embeddings_reducer",
                "label": "Embeddings Reducer",
                "type": "select",
                "default": "umap",
                "options": [
                    {"value": "umap", "label": "UMAP"},
                    {"value": "pacmap", "label": "PaCMAP"},
                ],
                "help": "Reduction algorithm for image embeddings.",
                "group": "3LC Settings",
            },
            {
                "id": "conf_thres",
                "label": "Conf Threshold",
                "type": "number",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
                "step": 0.01,
                "help": "Confidence threshold for detections.",
                "group": "3LC Settings",
            },
            {
                "id": "max_det",
                "label": "Max Detections",
                "type": "number",
                "default": 300,
                "min": 1,
                "max": 2000,
                "step": 1,
                "help": "Max detections collected per image.",
                "group": "3LC Settings",
            },
            {
                "id": "sampling_weights",
                "label": "Sampling Weights",
                "type": "checkbox",
                "default": False,
                "help": "Use 3LC sampling weights.",
                "group": "3LC Settings",
            },
            {
                "id": "exclude_zero_weight_training",
                "label": "Exclude Zero (Train)",
                "type": "checkbox",
                "default": False,
                "help": "Exclude zero-weighted samples in training.",
                "group": "3LC Settings",
            },
            {
                "id": "exclude_zero_weight_collection",
                "label": "Exclude Zero (Collect)",
                "type": "checkbox",
                "default": False,
                "help": "Exclude zero-weighted samples in collection.",
                "group": "3LC Settings",
            },
            {
                "id": "collection_disable",
                "label": "Disable Collection",
                "type": "checkbox",
                "default": False,
                "help": "Disable 3LC metrics collection entirely.",
                "group": "3LC Settings",
            },
            {
                "id": "collection_val_only",
                "label": "Collect Val Only",
                "type": "checkbox",
                "default": False,
                "help": "Collect metrics only on the validation set (skip training set collection).",
                "group": "3LC Settings",
            },
            {
                "id": "collection_epoch_start",
                "label": "Collection Start Epoch",
                "type": "number",
                "default": "",
                "min": 1,
                "max": 1000,
                "step": 1,
                "help": "Start epoch for collection (empty = best epoch only).",
                "group": "3LC Settings",
            },
            {
                "id": "collection_epoch_interval",
                "label": "Collection Interval",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 100,
                "step": 1,
                "help": "Epoch interval for metrics collection.",
                "group": "3LC Settings",
            },
            {
                "id": "instance_embeddings_dim",
                "label": "Instance Embeddings Dim",
                "type": "select",
                "default": "2",
                "options": [
                    {"value": "0", "label": "Disabled"},
                    {"value": "2", "label": "2D"},
                    {"value": "3", "label": "3D"},
                ],
                "help": "Per-instance embedding dimensionality (per bounding box).",
                "group": "3LC Settings",
            },
            {
                "id": "instance_embeddings_reducer",
                "label": "Instance Embeddings Reducer",
                "type": "select",
                "default": "pacmap",
                "options": [
                    {"value": "pacmap", "label": "PaCMAP"},
                    {"value": "umap", "label": "UMAP"},
                    {"value": "pca", "label": "PCA"},
                ],
                "help": "Reduction algorithm for per-instance embeddings. PCA allows arbitrary output dimensions.",
                "group": "3LC Settings",
            },
            {
                "id": "ground_truth_instance_embeddings",
                "label": "GT Instance Embeddings",
                "type": "checkbox",
                "default": True,
                "help": "Collect instance embeddings for ground-truth annotations.",
                "group": "3LC Settings",
            },
        ]

    def train(self, tables: dict[str, Any], params: dict[str, Any], callbacks: dict[str, Any]) -> dict[str, Any]:
        """Run YOLO training with 3LC integration."""
        on_epoch = callbacks.get("on_epoch", lambda *a: None)
        on_status = callbacks.get("on_status", lambda m: None)
        is_cancelled = callbacks.get("is_cancelled", lambda: False)

        # ── YOLO args ──
        model_name = params.get("model", "yolov8n.pt")
        epochs = int(params.get("epochs", 50))
        batch = int(params.get("batch", 16))
        imgsz = int(params.get("imgsz", 640))
        lr0 = float(params.get("lr0", 0.01))
        lrf = float(params.get("lrf", 0.01))
        optimizer = params.get("optimizer", "auto")
        patience = int(params.get("patience", 50))
        workers = int(params.get("workers", 8))
        device = params.get("device", "").strip() or None

        # ── Extra YOLO args (key=value pairs) ──
        extra_kwargs: dict[str, Any] = {}
        raw_extra = params.get("extra_args", "").strip()
        if raw_extra:
            for token in raw_extra.split():
                if "=" not in token:
                    continue
                k, v = token.split("=", 1)
                k = k.strip()
                if not k:
                    continue
                if v.lower() in ("true", "false"):
                    extra_kwargs[k] = v.lower() == "true"
                else:
                    try:
                        extra_kwargs[k] = int(v)
                    except ValueError:
                        try:
                            extra_kwargs[k] = float(v)
                        except ValueError:
                            extra_kwargs[k] = v
            if extra_kwargs:
                on_status(f"Extra YOLO args: {extra_kwargs}")

        # ── Resolve task ──
        task_type = params.get("_task_type", "")
        yolo_task = _TASK_MAP.get(task_type)

        # ── 3LC Settings ──
        project_name = params.get("_project_name", "").strip() or None
        run_name = params.get("_run_name", "").strip() or None

        # Auto-generate run name if not provided
        if not run_name:
            from tlc_plugin_sdk.shared.naming import generate_name

            run_name = generate_name()

        on_status(f"Run name: {run_name}")

        settings_kwargs: dict[str, Any] = {}
        if project_name:
            settings_kwargs["project_name"] = project_name
        settings_kwargs["run_name"] = run_name

        settings_kwargs["image_embeddings_dim"] = int(params.get("image_embeddings_dim", 0))
        settings_kwargs["image_embeddings_reducer"] = params.get("image_embeddings_reducer", "umap")
        settings_kwargs["conf_thres"] = float(params.get("conf_thres", 0.1))
        settings_kwargs["max_det"] = int(params.get("max_det", 300))
        settings_kwargs["collect_loss"] = _to_bool(params.get("collect_loss", False))
        settings_kwargs["sampling_weights"] = _to_bool(params.get("sampling_weights", False))
        settings_kwargs["exclude_zero_weight_training"] = _to_bool(params.get("exclude_zero_weight_training", False))
        settings_kwargs["exclude_zero_weight_collection"] = _to_bool(
            params.get("exclude_zero_weight_collection", False)
        )
        settings_kwargs["collection_val_only"] = _to_bool(params.get("collection_val_only", False))
        settings_kwargs["collection_disable"] = _to_bool(params.get("collection_disable", False))
        settings_kwargs["collection_epoch_interval"] = int(params.get("collection_epoch_interval", 1))
        settings_kwargs["instance_embeddings_dim"] = int(params.get("instance_embeddings_dim", 0))
        settings_kwargs["instance_embeddings_reducer"] = params.get("instance_embeddings_reducer", "pacmap")
        settings_kwargs["ground_truth_instance_embeddings"] = _to_bool(
            params.get("ground_truth_instance_embeddings", False)
        )
        if settings_kwargs["instance_embeddings_dim"] == 0:
            settings_kwargs["ground_truth_instance_embeddings"] = False

        instance_layer = params.get("instance_embeddings_layer", "")
        if instance_layer and str(instance_layer).strip():
            settings_kwargs["instance_embeddings_layer"] = int(instance_layer)

        epoch_start = params.get("collection_epoch_start", "")
        if epoch_start and str(epoch_start).strip():
            settings_kwargs["collection_epoch_start"] = int(epoch_start)

        settings = Settings(**settings_kwargs)
        on_status(f"3LC Settings: {settings_kwargs}")

        # Use pretrained model URL if provided (fine-tuning from existing checkpoint)
        pretrained_url = params.get("pretrained_model_url", "").strip()
        source_model_name = model_name  # Remember the base model for metadata
        if pretrained_url:
            on_status(f"Loading pretrained model from: {pretrained_url}")
            try:
                resolved_path = str(tlc.Url(pretrained_url).to_absolute())
                if Path(resolved_path).exists():
                    model_name = resolved_path
                    on_status(f"Resolved model path: {model_name}")
                else:
                    on_status(f"Pretrained path not found locally, using as-is: {pretrained_url}")
                    model_name = pretrained_url
            except Exception:
                model_name = pretrained_url
        copy_model_to_run = _to_bool(params.get("copy_model_to_run", True))

        on_status(f"Loading YOLO model: {model_name}")
        model = YOLO(model_name, task=yolo_task)

        # Load 3LC tables
        on_status("Loading training table...")
        train_table = tlc.Table.from_url(tables["train"])
        val_table = None
        skip_val = False
        if tables.get("val"):
            on_status("Loading validation table...")
            val_table = tlc.Table.from_url(tables["val"])
        else:
            on_status("No validation table — training without validation (val=False).")
            skip_val = True
            val_table = train_table  # tlc_ultralytics requires val key, but val=False skips it

        data: dict[str, Any] = {"train": train_table, "val": val_table}

        # Epoch and batch callbacks for progress tracking and cancellation
        _state: dict[str, Any] = {"epoch": 0, "stop_logged": False, "tlc_run": None}

        def _on_train_start(trainer: Any) -> None:
            # Capture the 3LC run reference as soon as training starts
            if hasattr(trainer, "_run"):
                _state["tlc_run"] = trainer._run

        def _on_train_epoch_end(trainer: Any) -> None:
            if is_cancelled():
                trainer.stop = True
                return
            _state["epoch"] += 1

        def _on_train_batch_end(trainer: Any) -> None:
            if is_cancelled():
                trainer.stop = True
                if not _state["stop_logged"]:
                    _state["stop_logged"] = True
                    on_status("Stop requested — aborting after current batch...")
                return
            batch_i = trainer.batch_i if hasattr(trainer, "batch_i") else 0
            nb = trainer.nb if hasattr(trainer, "nb") else 0
            epoch = _state["epoch"] + 1
            batch_frac = (batch_i + 1) / nb if nb > 0 else 0
            metrics = {"phase": "train", "batch": batch_i + 1, "total_batches": nb, "batch_frac": batch_frac}
            on_epoch(epoch, epochs, metrics)

        def _on_fit_epoch_end(trainer: Any) -> None:
            if is_cancelled():
                trainer.stop = True
                # Disable 3LC collection before final_eval runs (no callback between
                # loop exit and final_eval, so this is the last chance).
                if hasattr(trainer, "_settings"):
                    trainer._settings.collection_disable = True
                    trainer._settings.image_embeddings_dim = 0
                    trainer._settings.instance_embeddings_dim = 0
                return
            # Report metrics — use val metrics if available, else training loss
            metrics: dict[str, float] = {}
            if not skip_val and hasattr(trainer, "metrics") and trainer.metrics:
                for k, v in trainer.metrics.items():
                    with contextlib.suppress(TypeError, ValueError):
                        metrics[k] = float(v)
            # When val=False, report training losses instead of zero-filled val metrics
            if skip_val and hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                loss_names = ["box_loss", "cls_loss", "dfl_loss", "seg_loss"]
                try:
                    items = (
                        trainer.loss_items.cpu().tolist()
                        if hasattr(trainer.loss_items, "cpu")
                        else list(trainer.loss_items)
                    )
                    for i, val in enumerate(items):
                        name = loss_names[i] if i < len(loss_names) else f"loss_{i}"
                        metrics[f"train/{name}"] = float(val)
                except Exception:
                    pass
            on_epoch(_state["epoch"], epochs, metrics)

        model.add_callback("on_train_start", _on_train_start)
        model.add_callback("on_train_epoch_end", _on_train_epoch_end)
        model.add_callback("on_train_batch_end", _on_train_batch_end)
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

        try:
            on_status("Starting YOLO training...")
            train_kwargs: dict[str, Any] = {
                "tables": data,
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "lr0": lr0,
                "lrf": lrf,
                "optimizer": optimizer,
                "patience": patience,
                "workers": workers,
                "device": device,
                "settings": settings,
                **extra_kwargs,
            }
            if skip_val:
                train_kwargs["val"] = False
            results = model.train(**train_kwargs)

            run_url = None
            tlc_run = _state.get("tlc_run")
            if tlc_run and hasattr(tlc_run, "url"):
                run_url = str(tlc_run.url)
            elif hasattr(model, "tlc_run") and hasattr(model.tlc_run, "url"):
                run_url = str(model.tlc_run.url)
                tlc_run = model.tlc_run

            final_metrics: dict[str, float] = {}
            if results and hasattr(results, "results_dict"):
                for k, v in results.results_dict.items():
                    with contextlib.suppress(TypeError, ValueError):
                        final_metrics[k] = float(v)

            # ── Save model info to Run ──
            best_model_src = None

            # Find best.pt from ultralytics trainer
            if hasattr(model, "trainer") and hasattr(model.trainer, "best"):
                best_model_src = Path(str(model.trainer.best))
                if not best_model_src.exists() and hasattr(model.trainer, "save_dir"):
                    fallback = Path(str(model.trainer.save_dir)) / "weights" / "best.pt"
                    if fallback.exists():
                        best_model_src = fallback

            if best_model_src and best_model_src.exists() and tlc_run and run_url:
                from tlc_plugin_sdk.shared.model_storage import save_model_to_run, store_model_info_in_run

                if copy_model_to_run:
                    try:
                        model_path = save_model_to_run(
                            run_url=run_url,
                            model_data=None,
                            filename=best_model_src.name,
                            source_file=best_model_src,
                            on_status=on_status,
                        )
                    except Exception as e:
                        on_status(f"Warning: could not copy model to run folder: {e}")
                        model_path = str(best_model_src)
                else:
                    model_path = str(best_model_src)

                store_model_info_in_run(
                    run=tlc_run,
                    model_name=source_model_name,
                    model_path=model_path,
                    source_url=pretrained_url,
                    on_status=on_status,
                )

            return {"run_url": run_url, "tlc_run": tlc_run, "final_metrics": final_metrics}

        except (Exception, KeyboardInterrupt):
            if is_cancelled():
                on_status(f"Training stopped after epoch {_state['epoch']}/{epochs}")
                run_url = None
                tlc_run = _state.get("tlc_run")
                if tlc_run and hasattr(tlc_run, "url"):
                    run_url = str(tlc_run.url)
                return {"run_url": run_url, "tlc_run": tlc_run, "final_metrics": {}}
            traceback.print_exc()
            raise

    def collect(self, tables: dict[str, Any], params: dict[str, Any], callbacks: dict[str, Any]) -> dict[str, Any]:
        """Run metrics collection only (no training)."""
        on_status = callbacks.get("on_status", lambda m: None)
        on_epoch = callbacks.get("on_epoch", lambda *a: None)
        is_cancelled = callbacks.get("is_cancelled", lambda: False)

        model_name = params.get("model", "yolov8n.pt")
        batch = int(params.get("batch", 16))
        imgsz = int(params.get("imgsz", 640))
        device = params.get("device", "").strip() or None

        task_type = params.get("_task_type", "")
        yolo_task = _TASK_MAP.get(task_type)

        project_name = params.get("_project_name", "").strip() or None
        run_name = params.get("_run_name", "").strip() or None

        settings_kwargs: dict[str, Any] = {}
        if project_name:
            settings_kwargs["project_name"] = project_name
        if run_name:
            settings_kwargs["run_name"] = run_name

        settings_kwargs["image_embeddings_dim"] = int(params.get("image_embeddings_dim", 0))
        settings_kwargs["image_embeddings_reducer"] = params.get("image_embeddings_reducer", "umap")
        settings_kwargs["conf_thres"] = float(params.get("conf_thres", 0.1))
        settings_kwargs["max_det"] = int(params.get("max_det", 300))
        settings_kwargs["collect_loss"] = _to_bool(params.get("collect_loss", False))
        settings_kwargs["sampling_weights"] = _to_bool(params.get("sampling_weights", False))
        settings_kwargs["exclude_zero_weight_collection"] = _to_bool(
            params.get("exclude_zero_weight_collection", False)
        )
        settings_kwargs["instance_embeddings_dim"] = int(params.get("instance_embeddings_dim", 0))
        settings_kwargs["instance_embeddings_reducer"] = params.get("instance_embeddings_reducer", "pacmap")
        settings_kwargs["ground_truth_instance_embeddings"] = _to_bool(
            params.get("ground_truth_instance_embeddings", False)
        )
        if settings_kwargs["instance_embeddings_dim"] == 0:
            settings_kwargs["ground_truth_instance_embeddings"] = False

        instance_layer = params.get("instance_embeddings_layer", "")
        if instance_layer and str(instance_layer).strip():
            settings_kwargs["instance_embeddings_layer"] = int(instance_layer)

        settings = Settings(**settings_kwargs)
        on_status(f"3LC Settings: {settings_kwargs}")

        # Clear any leftover active run from a previous job
        try:
            if tlc.active_run() is not None:
                tlc.close()
        except Exception:
            pass

        # Use pretrained model URL if provided
        pretrained_url = params.get("pretrained_model_url", "").strip()
        if pretrained_url:
            on_status(f"Loading model from: {pretrained_url}")
            try:
                resolved_path = str(tlc.Url(pretrained_url).to_absolute())
                if Path(resolved_path).exists():
                    model_name = resolved_path
                else:
                    model_name = pretrained_url
            except Exception:
                model_name = pretrained_url

        on_status(f"Loading YOLO model: {model_name}")
        model = YOLO(model_name, task=yolo_task)

        on_status("Loading training table...")
        train_table = tlc.Table.from_url(tables["train"])
        val_table = None
        skip_val = False
        if tables.get("val"):
            on_status("Loading validation table...")
            val_table = tlc.Table.from_url(tables["val"])
        else:
            on_status("No validation table — collecting on train set only.")
            skip_val = True

        data: dict[str, Any]
        if skip_val:
            # Only collect on train — don't duplicate by passing train as val
            data = {"train": train_table}
        else:
            data = {"train": train_table, "val": val_table}

        # Progress callback for embedding reduction phases
        def _reduction_progress(phase: str, current: int, total: int) -> None:
            if phase == "split_start":
                pass
            elif phase == "fit":
                if current == 0 and total > 0:
                    on_status(f"Fitting instance embedding reducer on {total} vectors...")
                elif current == total and total > 0:
                    on_status("Reducer fitted.")
            elif phase == "transform":
                if total > 0:
                    on_status(f"Transforming embeddings {current}/{total}...")
                    on_epoch(
                        0,
                        1,
                        {
                            "phase": "collect",
                            "batch": current,
                            "total_batches": total,
                            "batch_frac": current / total,
                        },
                    )
            elif phase == "image_embeddings":
                on_status("Reducing image embeddings (server-side)...")
            elif phase == "image_embeddings_done":
                on_status("Image embeddings reduced.")

        _collect_state: dict[str, Any] = {"split": "", "nb": 0}

        def _on_val_start(validator: Any) -> None:
            split = getattr(validator, "_current_split", None) or ""
            _collect_state["split"] = split
            dl = getattr(validator, "dataloader", None)
            nb = len(dl) if dl is not None else 0
            _collect_state["nb"] = nb
            on_status(f"Collecting {split} split ({nb} batches)...")

        def _on_val_batch_end_collect(validator: Any) -> None:
            if is_cancelled():
                msg = "Collection cancelled by user"
                raise KeyboardInterrupt(msg)
            batch_i = getattr(validator, "batch_i", None)
            if batch_i is None:
                return
            nb = _collect_state.get("nb", 0) or 0
            batch_frac = (batch_i + 1) / nb if nb > 0 else 0
            on_epoch(
                0,
                1,
                {
                    "phase": "collect",
                    "batch": batch_i + 1,
                    "total_batches": nb,
                    "batch_frac": batch_frac,
                },
            )

        model.add_callback("on_val_start", _on_val_start)
        model.add_callback("on_val_batch_end", _on_val_batch_end_collect)

        try:
            on_status("Collecting metrics...")
            model.collect(
                tables=data,
                settings=settings,
                batch=batch,
                imgsz=imgsz,
                device=device,
                progress_callback=_reduction_progress,
            )

            run_url = None
            if hasattr(model, "tlc_run") and hasattr(model.tlc_run, "url"):
                run_url = str(model.tlc_run.url)

            return {"run_url": run_url, "final_metrics": {}}

        except (KeyboardInterrupt, Exception):
            if is_cancelled():
                on_status("Collection stopped by user")
                run_url = None
                if hasattr(model, "tlc_run") and hasattr(model.tlc_run, "url"):
                    run_url = str(model.tlc_run.url)
                return {"run_url": run_url, "final_metrics": {}}
            traceback.print_exc()
            raise


def _to_bool(val: Any) -> bool:
    """Convert param value to bool (handles string 'true'/'false' from form)."""
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


# Auto-register on import
register_model(YOLOModel())
