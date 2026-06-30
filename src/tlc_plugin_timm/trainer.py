# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""timm training and collection logic — PyTorch training loop with 3LC integration."""

from __future__ import annotations

import gc
import logging
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  (PyTorch's conventional alias)
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _to_bool(val: Any) -> bool:
    """Convert param value to bool (handles string 'true'/'false' from form)."""
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


def _parse_int(val: Any, *, default: int = 0) -> int:
    """Parse ``val`` to int, returning ``default`` for empty or invalid input."""
    try:
        text = str(val).strip()
        return int(text) if text else default
    except (TypeError, ValueError):
        return default


class _SampleTransform:
    """Picklable transform mapping table samples to (tensor, label) tuples.

    Used with `Table.with_transform()` to build a `TableView` for DataLoaders.
    Must be a top-level class (not a closure) so it can be pickled by
    DataLoader workers when `num_workers > 0`.

    Args:
        transform: torchvision/timm transform to apply to images.
        image_column: Column name for image paths.
        label_column: Column name for labels (empty string for feature extraction).

    """

    def __init__(self, transform: Any, image_column: str, label_column: str) -> None:
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column

    def __call__(self, sample: Any) -> Any:
        if isinstance(sample, dict):
            image = _as_rgb_image(sample[self.image_column])
            label = sample.get(self.label_column, 0) if self.label_column else 0
        else:
            image = _as_rgb_image(sample[0])
            label = sample[1] if len(sample) > 1 else 0
        return self.transform(image), int(label)


def _as_rgb_image(value: Any) -> Any:
    """Convert a sample-view image value to an RGB PIL image.

    The sample view yields a decoded PIL image for pil-backed image columns
    and an (absolutized) path string for url-backed ones — handle both. Paths
    are opened via the shared helper so cloud and aliased paths work.

    Top-level function (not a method or closure) so ``_SampleTransform`` stays
    picklable for DataLoader workers.
    """
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    from tlc_plugin_sdk.shared.images import load_image

    return load_image(str(value))


def train(tables: dict[str, str], params: dict[str, Any], callbacks: dict[str, Any]) -> dict[str, Any]:
    """Run timm training with 3LC integration.

    Args:
        tables: Dict with 'train' and optional 'val' table URLs.
        params: Training parameters (from config.params + internal fields).
        callbacks: Dict with 'on_epoch', 'on_status', 'is_cancelled' callables.

    Returns:
        Dict with 'run_url' and 'final_metrics'.

    """
    import timm
    import timm.data
    import tlc

    on_epoch = callbacks.get("on_epoch", lambda *a: None)
    on_status = callbacks.get("on_status", lambda m: None)
    is_cancelled = callbacks.get("is_cancelled", lambda: False)

    # ── Extract params ──
    model_name = params.get("_model_name", "resnet50")
    epochs = int(params.get("epochs", 10))
    batch_size = int(params.get("batch_size", 32))
    image_size = int(params.get("image_size", 224))
    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 1e-4))
    optimizer_name = params.get("optimizer", "adamw")
    scheduler_name = params.get("scheduler", "cosine")
    warmup_epochs = int(params.get("warmup_epochs", 3))

    # Augmentation params
    auto_augment = params.get("auto_augment", "")
    reprob = float(params.get("reprob", 0.0))
    mixup_alpha = float(params.get("mixup_alpha", 0.0))
    cutmix_alpha = float(params.get("cutmix_alpha", 0.0))
    hflip = _to_bool(params.get("hflip", True))
    color_jitter = _to_bool(params.get("color_jitter", True))

    # 3LC params
    project_name = params.get("_project_name", "").strip() or None
    run_name = params.get("_run_name", "").strip() or None
    image_column = params.get("_image_column", "image")
    label_column = params.get("_label_column", "label")
    task_type = params.get("_task_type", "classification")

    embeddings_dim = int(params.get("image_embeddings_dim", params.get("embeddings_dim", 2)))
    embeddings_reducer = params.get("image_embeddings_reducer", params.get("embeddings_reducer", "umap"))
    write_full_embeddings = _to_bool(params.get("write_full_embeddings", False))
    collect_loss = _to_bool(params.get("collect_loss", True))
    sampling_weights = _to_bool(params.get("sampling_weights", False))
    exclude_zero_training = _to_bool(params.get("exclude_zero_weight_training", False))
    collection_val_only = _to_bool(params.get("collection_val_only", False))
    collection_disable = _to_bool(params.get("collection_disable", False))
    # Periodic per-sample metrics collection. Setting a start epoch opts in:
    # collection then runs every ``collection_interval`` epochs from that epoch
    # onward (anchored at the start), in addition to the final best-model pass.
    # An empty start ("best only" in the UI) keeps the historical behaviour of a
    # single final-pass collection.
    collection_epoch_start_int = _parse_int(params.get("collection_epoch_start", ""), default=0)
    collection_interval = _parse_int(params.get("collection_interval", "1"), default=1)
    periodic_collection = collection_epoch_start_int > 0 and collection_interval > 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_status(f"Device: {device}")

    # ── Load tables ──
    on_status("Loading training table...")
    train_table = tlc.Table.from_url(tables["train"])
    num_classes = 0

    if label_column:
        try:
            value_map = train_table.get_value_map(label_column)
            if value_map:
                num_classes = len(value_map)
                on_status(f"Detected {num_classes} classes from '{label_column}'")
        except Exception:
            pass

    if num_classes == 0 and task_type == "classification":
        # Fallback: scan label column for max value
        try:
            labels_arr = train_table.get_column_as_pyarrow_array(label_column)
            num_classes = int(max(labels_arr.to_pylist())) + 1
            on_status(f"Inferred {num_classes} classes from label values")
        except Exception:
            num_classes = 2
            on_status(f"Could not detect num_classes, defaulting to {num_classes}")

    val_table = None
    if tables.get("val"):
        on_status("Loading validation table...")
        val_table = tlc.Table.from_url(tables["val"])
    else:
        on_status("No validation table — using train table for validation.")
        val_table = train_table

    # ── Create timm model ──
    pretrained_url = params.get("pretrained_model_url", "").strip()
    copy_model_to_run = _to_bool(params.get("copy_model_to_run", True))

    on_status(f"Creating model: {model_name} (num_classes={num_classes})")
    is_feature_extraction = task_type == "feature_extraction" or num_classes == 0

    if pretrained_url:
        # Fine-tune from user-provided checkpoint
        on_status(f"Loading pretrained weights from: {pretrained_url}")
        resolved_path = pretrained_url
        try:
            resolved = str(tlc.Url(pretrained_url).to_absolute())
            if Path(resolved).exists():
                resolved_path = resolved
        except Exception:
            pass

        # Create model architecture, then load state dict
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes if not is_feature_extraction else 0,
        )
        state = torch.load(resolved_path, map_location="cpu", weights_only=True)
        # Handle both raw state_dict and wrapped formats
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        on_status("Loaded pretrained weights successfully")
    else:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes if not is_feature_extraction else 0,
        )
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    on_status(f"Model loaded: {num_params / 1e6:.1f}M parameters")

    # ── Create transforms ──
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    data_config["input_size"] = (3, image_size, image_size)

    train_transform = timm.data.create_transform(
        **data_config,
        is_training=True,
        auto_augment=auto_augment or None,
        re_prob=reprob,
        hflip=0.5 if hflip else 0.0,
        color_jitter=0.4 if color_jitter else 0.0,
    )
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    on_status(f"Transforms: train={train_transform.__class__.__name__}, image_size={image_size}")

    # ── Build transformed views over the tables ──
    # TableView is non-mutating, so train and val views can share the same underlying table.
    train_view = train_table.with_transform(_SampleTransform(train_transform, image_column, label_column))
    val_view = val_table.with_transform(_SampleTransform(val_transform, image_column, label_column))

    # ── DataLoaders ──
    sampler = None
    shuffle = True
    if sampling_weights or exclude_zero_training:
        try:
            from tlc.integration.torch.samplers import create_sampler

            sampler = create_sampler(
                train_table,
                exclude_zero_weights=exclude_zero_training,
                weighted=sampling_weights,
            )
            shuffle = False
        except Exception as e:
            on_status(f"Warning: could not create sampler: {e}")

    num_workers = int(params.get("num_workers", 8))
    train_loader: DataLoader[Any] = DataLoader(
        cast("Dataset[Any]", train_view),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader: DataLoader[Any] = DataLoader(
        cast("Dataset[Any]", val_view),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    on_status(f"Train: {len(train_table)} samples, {len(train_loader)} batches")
    on_status(f"Val: {len(val_view)} samples, {len(val_loader)} batches")

    # ── Optimizer ──
    optimizer = _create_optimizer(model, optimizer_name, lr, weight_decay)
    on_status(f"Optimizer: {optimizer_name}, lr={lr}, weight_decay={weight_decay}")

    # ── Scheduler ──
    scheduler = _create_scheduler(optimizer, scheduler_name, epochs, warmup_epochs, len(train_loader))
    on_status(f"Scheduler: {scheduler_name}, warmup={warmup_epochs}")

    # ── Loss ──
    criterion = nn.CrossEntropyLoss()

    # ── Mixup / CutMix ──
    mixup_fn = None
    if mixup_alpha > 0 or cutmix_alpha > 0:
        try:
            from timm.data.mixup import Mixup

            mixup_fn = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                num_classes=num_classes,
            )
            criterion = nn.CrossEntropyLoss() if mixup_fn is None else nn.CrossEntropyLoss(label_smoothing=0.1)
            on_status(f"Mixup: alpha={mixup_alpha}, CutMix: alpha={cutmix_alpha}")
        except Exception as e:
            on_status(f"Warning: could not create Mixup: {e}")

    # ── 3LC Run ──
    run = None
    if not collection_disable:
        try:
            run = tlc.init(project_name=project_name, run_name=run_name, description=f"timm {model_name} training")
            on_status(f"3LC Run created: {run_name}")
            on_status(f"Run URL: {run.url}")
        except Exception as e:
            on_status(f"Warning: could not create 3LC Run: {e}")

    def _collect_at(collect_model: nn.Module, epoch_tag: int, *, update_status: bool) -> None:
        """Run a per-sample metrics collection pass on the given model.

        Each pass writes its own metrics table tagged with ``epoch_tag`` (via
        ``add_metrics`` constants) so periodic collections stay distinguishable
        by epoch. ``update_status`` flips the run into the "collecting" state —
        used for the final pass, but skipped for in-training periodic passes so
        the run stays in its running state.
        """
        if update_status and run is not None:
            run.set_status_collecting()
        on_status(f"Collecting per-sample metrics (epoch {epoch_tag})...")
        # View over the train table with val_transform (no augmentation) for
        # deterministic metrics.
        train_view_for_collect = train_table.with_transform(_SampleTransform(val_transform, image_column, label_column))
        _collect_metrics(
            model=collect_model,
            train_table=train_view_for_collect,
            val_table_mapped=val_view,
            tables=tables,
            run=run,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            image_column=image_column,
            label_column=label_column,
            val_transform=val_transform,
            embeddings_dim=embeddings_dim,
            embeddings_reducer=embeddings_reducer,
            write_full_embeddings=write_full_embeddings,
            collect_loss=collect_loss,
            collection_val_only=collection_val_only,
            num_classes=num_classes,
            epoch=epoch_tag,
            on_status=on_status,
            is_cancelled=is_cancelled,
        )

    # ── Training loop ──
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0

    on_status("Starting training...")

    for epoch in range(1, epochs + 1):
        if is_cancelled():
            on_status(f"Training cancelled at epoch {epoch}")
            break

        # ── Train phase ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if is_cancelled():
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if mixup_fn is not None:
                images, labels_mixed = mixup_fn(images, labels)
                outputs = model(images)
                loss = criterion(outputs, labels_mixed)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            if mixup_fn is None:
                train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

            # Batch progress
            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0 or batch_idx == len(train_loader) - 1:
                batch_frac = (batch_idx + 1) / len(train_loader)
                on_epoch(
                    epoch,
                    epochs,
                    {
                        "phase": "train",
                        "batch": batch_idx + 1,
                        "total_batches": len(train_loader),
                        "batch_frac": batch_frac,
                    },
                )

        if is_cancelled():
            break

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = 100.0 * train_correct / max(train_total, 1) if mixup_fn is None else 0.0

        # ── Val phase ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                if is_cancelled():
                    break

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

                if (batch_idx + 1) % max(1, len(val_loader) // 5) == 0 or batch_idx == len(val_loader) - 1:
                    batch_frac = (batch_idx + 1) / len(val_loader)
                    on_epoch(
                        epoch,
                        epochs,
                        {
                            "phase": "val",
                            "batch": batch_idx + 1,
                            "total_batches": len(val_loader),
                            "batch_frac": batch_frac,
                        },
                    )

        if is_cancelled():
            break

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)

        # Epoch metrics
        metrics: dict[str, Any] = {
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_acc": round(val_acc, 2),
        }
        if mixup_fn is None:
            metrics["train_acc"] = round(train_acc, 2)
        metrics["lr"] = round(optimizer.param_groups[0]["lr"], 6)

        on_epoch(epoch, epochs, metrics)
        on_status(f"Epoch {epoch}/{epochs}: val_acc={val_acc:.2f}%, val_loss={avg_val_loss:.4f}")

        # Log to 3LC Run
        if run is not None:
            try:
                tlc.log({
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "val_loss": avg_val_loss,
                    "train_loss": avg_train_loss,
                })
            except Exception:
                pass

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        # Periodic per-sample metrics collection on the current model. Skip the
        # final epoch: the post-training pass below always collects the
        # best/final model, so collecting here too would write a duplicate
        # metrics table for that same state.
        if (
            run is not None
            and not collection_disable
            and periodic_collection
            and epoch >= collection_epoch_start_int
            and epoch < epochs
            and (epoch - collection_epoch_start_int) % collection_interval == 0
            and not is_cancelled()
        ):
            try:
                _collect_at(model, epoch, update_status=False)
            except Exception as e:
                on_status(f"Warning: periodic metrics collection failed at epoch {epoch}: {e}")
                traceback.print_exc()

    cancelled = is_cancelled()
    if cancelled:
        on_status(f"Training stopped at epoch {epoch}/{epochs}")

    # ── Set run status based on outcome ──
    run_url = None
    if run is not None:
        run_url = str(run.url)

    if cancelled:
        # Cancelled — mark run and skip collection
        if run is not None:
            try:
                run.set_status_cancelled()
            except Exception:
                pass
    else:
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model = model.to(device)
            on_status(f"Restored best model from epoch {best_epoch} (val_acc={best_val_acc:.2f}%)")

        # ── Metrics collection ──
        # Final pass: collect on the restored best model, tagged with
        # ``best_epoch`` so it lines up with any periodic passes. Skip it only
        # when a periodic pass already collected the best epoch — that pass ran
        # on the same model state with the same epoch tag, so re-collecting here
        # would write a duplicate metrics table.
        best_already_collected = (
            periodic_collection
            and collection_epoch_start_int <= best_epoch < epochs
            and (best_epoch - collection_epoch_start_int) % collection_interval == 0
        )
        if run is not None and not collection_disable and not best_already_collected:
            try:
                _collect_at(model, best_epoch, update_status=True)
            except Exception as e:
                on_status(f"Warning: metrics collection failed: {e}")
                traceback.print_exc()

            try:
                run.set_status_completed()
                on_status(f"Run completed: {run_url}")
            except Exception as e:
                on_status(f"Warning: could not finalize run: {e}")

    # ── Save model checkpoint and store model info in Run ──
    if best_model_state is not None and run_url and not cancelled and copy_model_to_run:
        from tlc_plugin_sdk.shared.model_storage import save_model_to_run, store_model_info_in_run

        try:
            model_path = save_model_to_run(
                run_url=run_url,
                model_data=best_model_state,
                filename="best_model.pt",
                on_status=on_status,
            )
            if run is not None:
                store_model_info_in_run(
                    run=run,
                    model_name=model_name,
                    model_path=model_path,
                    source_url=pretrained_url,
                    on_status=on_status,
                )
        except Exception as e:
            on_status(f"Warning: could not save model checkpoint: {e}")

    # Clean up
    del model, train_loader, val_loader
    if best_model_state is not None:
        del best_model_state
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "run_url": run_url,
        "final_metrics": {
            "best_val_acc": round(best_val_acc, 2),
            "best_epoch": best_epoch,
        },
    }


def collect(tables: dict[str, str], params: dict[str, Any], callbacks: dict[str, Any]) -> dict[str, Any]:
    """Run metrics collection only (no training) with a pretrained timm model.

    Args:
        tables: Dict with 'train' and optional 'val' table URLs.
        params: Collection parameters.
        callbacks: Dict with 'on_status', 'is_cancelled' callables.

    Returns:
        Dict with 'run_url'.

    """
    import timm
    import timm.data
    import tlc

    on_status = callbacks.get("on_status", lambda m: None)
    is_cancelled = callbacks.get("is_cancelled", lambda: False)

    model_name = params.get("_model_name", "resnet50")
    batch_size = int(params.get("batch_size", 32))
    image_size = int(params.get("image_size", 224))
    image_column = params.get("_image_column", "image")
    label_column = params.get("_label_column", "label")
    project_name = params.get("_project_name", "").strip() or None
    run_name = params.get("_run_name", "").strip() or None
    embeddings_dim = int(params.get("image_embeddings_dim", params.get("embeddings_dim", 2)))
    embeddings_reducer = params.get("image_embeddings_reducer", params.get("embeddings_reducer", "umap"))
    write_full_embeddings = _to_bool(params.get("write_full_embeddings", False))
    collect_loss = _to_bool(params.get("collect_loss", True))
    collection_val_only = _to_bool(params.get("collection_val_only", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_status(f"Device: {device}")

    # Load tables
    on_status("Loading training table...")
    train_table = tlc.Table.from_url(tables["train"])
    num_classes = 0
    if label_column:
        try:
            value_map = train_table.get_value_map(label_column)
            if value_map:
                num_classes = len(value_map)
        except Exception:
            try:
                labels_arr = train_table.get_column_as_pyarrow_array(label_column)
                num_classes = int(max(labels_arr.to_pylist())) + 1
            except Exception:
                pass

    val_table = None
    if tables.get("val"):
        on_status("Loading validation table...")
        val_table = tlc.Table.from_url(tables["val"])
    else:
        val_table = train_table

    # Create model (pretrained, no training)
    pretrained_url = params.get("pretrained_model_url", "").strip()
    if pretrained_url:
        on_status(f"Loading model from: {pretrained_url}")
        resolved_path = pretrained_url
        try:
            resolved = str(tlc.Url(pretrained_url).to_absolute())
            if Path(resolved).exists():
                resolved_path = resolved
        except Exception:
            pass
        model = timm.create_model(model_name, pretrained=False, num_classes=max(0, num_classes))
        state = torch.load(resolved_path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        on_status("Loaded pretrained weights successfully")
    else:
        on_status(f"Loading pretrained model: {model_name}")
        model = timm.create_model(model_name, pretrained=True, num_classes=max(0, num_classes))
    model = model.to(device)
    model.eval()

    # Create val transform
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    data_config["input_size"] = (3, image_size, image_size)
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    # Build transformed views over the tables
    sample_transform = _SampleTransform(val_transform, image_column, label_column)
    train_view = train_table.with_transform(sample_transform)
    val_view = val_table.with_transform(sample_transform) if tables.get("val") else train_view

    num_workers = int(params.get("num_workers", 8))

    # Create 3LC Run
    run = tlc.init(project_name=project_name, run_name=run_name, description=f"timm {model_name} collection")
    on_status(f"3LC Run created: {run_name}")
    on_status(f"Run URL: {run.url}")

    # Collect metrics
    run.set_status_collecting()
    on_status("Collecting per-sample metrics...")
    _collect_metrics(
        model=model,
        train_table=train_view,
        val_table_mapped=val_view,
        tables=tables,
        run=run,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        image_column=image_column,
        label_column=label_column,
        val_transform=val_transform,
        embeddings_dim=embeddings_dim,
        embeddings_reducer=embeddings_reducer,
        write_full_embeddings=write_full_embeddings,
        collect_loss=collect_loss,
        collection_val_only=collection_val_only,
        num_classes=num_classes,
        on_status=on_status,
        is_cancelled=is_cancelled,
    )

    run_url = str(run.url)
    if is_cancelled():
        run.set_status_cancelled()
        on_status(f"Collection cancelled: {run_url}")
    else:
        run.set_status_completed()
        on_status(f"Collection completed: {run_url}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {"run_url": run_url, "final_metrics": {}}


def _collect_metrics(
    *,
    model: nn.Module,
    train_table: Any,
    val_table_mapped: Any,
    tables: dict[str, str],
    run: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_column: str,
    label_column: str,
    val_transform: Any,
    embeddings_dim: int,
    embeddings_reducer: str,
    write_full_embeddings: bool,
    collect_loss: bool,
    collection_val_only: bool,
    num_classes: int,
    on_status: Any,
    is_cancelled: Any,
    epoch: int | None = None,
) -> None:
    """Extract embeddings and per-sample metrics, reduce, and store via run.add_metrics().

    Args:
        model: Trained or pretrained timm model.
        train_table: TableView of the train table (val transform applied).
        val_table_mapped: TableView of the val table.
        tables: Raw table URLs dict.
        run: 3LC Run object.
        device: Torch device.
        batch_size: Batch size for inference.
        num_workers: DataLoader workers.
        image_column: Image column name.
        label_column: Label column name.
        val_transform: Validation transform (for re-mapping if needed).
        embeddings_dim: Target embedding dimensionality (0=disabled, 2, 3).
        embeddings_reducer: Reducer algorithm ('umap' or 'pacmap').
        write_full_embeddings: If True, also write the full (unreduced) embeddings as a metric.
        collect_loss: Whether to collect per-sample loss.
        collection_val_only: Only collect on val set.
        num_classes: Number of classes.
        on_status: Status callback.
        is_cancelled: Cancellation check callback.
        epoch: When set, tag the written metrics table with this epoch (added
            as a constant column) so collections from different epochs stay
            distinguishable.

    """
    import tlc

    model.eval()

    # Check if model supports forward_features + forward_head (all timm models do)
    use_forward_head = hasattr(model, "forward_features") and hasattr(model, "forward_head")

    # Determine which tables to collect on
    collect_targets: list[tuple[str, Any, str]] = []
    if not collection_val_only:
        collect_targets.append(("train", train_table, tables["train"]))
    if tables.get("val"):
        collect_targets.append(("val", val_table_mapped, tables["val"]))
    elif collection_val_only:
        # Val-only but no val table — use train
        collect_targets.append(("train", train_table, tables["train"]))

    criterion = nn.CrossEntropyLoss(reduction="none") if collect_loss and num_classes > 0 else None

    # ── Pass 1: Inference — collect raw embeddings + predictions per split ──
    split_results: list[dict[str, Any]] = []
    for split_name, table, table_url in collect_targets:
        if is_cancelled():
            break

        on_status(f"Collecting metrics on {split_name} ({len(table)} samples)...")

        loader = DataLoader(
            table,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        all_embeddings: list[np.ndarray] = []
        all_predicted: list[np.ndarray] = []
        all_confidence: list[np.ndarray] = []
        all_loss: list[np.ndarray] = []

        with torch.no_grad():
            for images, labels in loader:
                if is_cancelled():
                    break

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Extract embeddings and logits using timm's forward_features/forward_head
                if use_forward_head:
                    # timm models expose forward_features/forward_head dynamically;
                    # nn.Module's typed __getattr__ surfaces them as Tensor, so bind
                    # through a Callable cast to call them.
                    forward_features = cast("Callable[..., torch.Tensor]", model.forward_features)
                    forward_head = cast("Callable[..., torch.Tensor]", model.forward_head)
                    features = forward_features(images)
                    embeddings_batch = forward_head(features, pre_logits=True)
                    outputs = forward_head(features)
                    all_embeddings.append(embeddings_batch.cpu().numpy())
                else:
                    outputs = model(images)

                # Classification metrics
                if num_classes > 0 and outputs.dim() == 2 and outputs.size(1) > 1:
                    probs = F.softmax(outputs, dim=1)
                    predicted = outputs.argmax(dim=1)
                    confidence = torch.gather(probs, 1, predicted.unsqueeze(1)).squeeze(1)
                    all_predicted.append(predicted.cpu().numpy())
                    all_confidence.append(confidence.cpu().numpy())

                    if criterion is not None:
                        loss_per_sample = criterion(outputs, labels)
                        all_loss.append(loss_per_sample.cpu().numpy())

        if is_cancelled():
            break

        split_results.append({
            "split_name": split_name,
            "table_url": table_url,
            "embeddings": np.vstack(all_embeddings) if all_embeddings else None,
            "predicted": np.concatenate(all_predicted) if all_predicted else None,
            "confidence": np.concatenate(all_confidence) if all_confidence else None,
            "loss": np.concatenate(all_loss) if all_loss else None,
        })

    del all_embeddings, all_predicted, all_confidence, all_loss

    # ── Pass 2: Reduce embeddings — fit on train, transform all splits ──
    # This ensures train and val embeddings share the same coordinate space.
    if embeddings_dim > 0 and not is_cancelled():
        train_emb = next(
            (s["embeddings"] for s in split_results if s["split_name"] == "train" and s["embeddings"] is not None), None
        )
        if train_emb is not None:
            on_status(f"Fitting {embeddings_reducer} on train embeddings: {train_emb.shape} → {embeddings_dim}D")
            reducer = _fit_reducer(train_emb, embeddings_dim, embeddings_reducer)
            if reducer is not None:
                for sr in split_results:
                    if sr["embeddings"] is not None:
                        on_status(f"Transforming {sr['split_name']} embeddings ({sr['embeddings'].shape[0]} samples)")
                        sr["reduced"] = _transform_embeddings(reducer, sr["embeddings"])
        else:
            # No train embeddings (val-only mode) — fit on whatever we have
            for sr in split_results:
                if sr["embeddings"] is not None:
                    on_status(
                        f"Reducing {sr['split_name']} embeddings: {sr['embeddings'].shape} → "
                        f"{embeddings_dim}D ({embeddings_reducer})"
                    )
                    sr["reduced"] = _reduce_embeddings(sr["embeddings"], embeddings_dim, embeddings_reducer)

    # ── Pass 3: Store metrics per split ──
    for sr in split_results:
        if is_cancelled():
            break

        split_name = sr["split_name"]
        table_url = sr["table_url"]
        run_metrics: dict[str, Any] = {}
        column_schemas: dict[str, Any] = {}

        reduced = sr.get("reduced")
        if reduced is not None:
            run_metrics["embeddings"] = list(reduced)
            # Declare the reduced embedding as a fixed-size vector (e.g. 2D → min=max=2),
            # otherwise add_metrics infers a variable-sized list and the frontend can't
            # treat it as plottable coordinates.
            column_schemas["embeddings"] = tlc.schemas.Float32Schema(
                shape=(embeddings_dim,),
                display_name=f"Embedding ({embeddings_dim}D)",
            )
            on_status(f"Reduced {split_name} embeddings: {reduced.shape}")

        if write_full_embeddings and sr["embeddings"] is not None:
            run_metrics["full_embeddings"] = list(sr["embeddings"])
            on_status(f"Full {split_name} embeddings: {sr['embeddings'].shape}")

        if sr["predicted"] is not None:
            run_metrics["predicted"] = sr["predicted"]

            # Copy the label column's schema so predicted has the same categorical map
            try:
                import copy

                source_table = tlc.Table.from_url(table_url)
                label_schema = source_table.rows_schema.values.get(label_column)
                if label_schema is not None:
                    column_schemas["predicted"] = copy.deepcopy(label_schema)
            except Exception:
                pass

        if sr["confidence"] is not None:
            run_metrics["confidence"] = sr["confidence"]

        if sr["loss"] is not None:
            run_metrics["loss"] = sr["loss"]

        # Store metrics
        if run_metrics:
            on_status(f"Storing {split_name} metrics ({len(run_metrics)} columns)...")
            try:
                run.add_metrics(
                    run_metrics,
                    foreign_table_url=table_url,
                    schema=column_schemas or None,
                    constants={"epoch": epoch} if epoch is not None else None,
                )
            except Exception as e:
                on_status(f"Warning: failed to store {split_name} metrics: {e}")
                traceback.print_exc()

    del split_results
    gc.collect()


def _fit_reducer(embeddings: np.ndarray, target_dim: int, reducer_name: str) -> Any:
    """Fit a dimensionality reducer on embeddings (typically train set).

    Returns the fitted reducer object, or None on failure.
    """
    n = len(embeddings)
    if n < 3:
        return None

    n_neighbors = min(15, max(2, n - 1))

    try:
        if reducer_name == "pacmap":
            import pacmap

            reducer = pacmap.PaCMAP(n_components=target_dim, n_neighbors=n_neighbors)
            reducer.fit(embeddings)
            return reducer
        else:
            import umap

            reducer = umap.UMAP(n_components=target_dim, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
            reducer.fit(embeddings)
            return reducer
    except Exception as e:
        logger.warning("Fitting reducer failed: %s", e)
        return None


def _transform_embeddings(reducer: Any, embeddings: np.ndarray) -> np.ndarray | None:
    """Transform embeddings using an already-fitted reducer."""
    try:
        transformed: np.ndarray = reducer.transform(embeddings)
        return transformed
    except Exception as e:
        logger.warning("Embedding transform failed: %s", e)
        return None


def _reduce_embeddings(embeddings: np.ndarray, target_dim: int, reducer_name: str) -> np.ndarray | None:
    """Reduce high-dimensional embeddings using UMAP or PaCMAP (fit+transform in one step).

    Used as fallback when there's no separate train set to fit on.
    """
    n = len(embeddings)
    if n < 3:
        return None

    n_neighbors = min(15, max(2, n - 1))

    try:
        if reducer_name == "pacmap":
            import pacmap

            reducer = pacmap.PaCMAP(n_components=target_dim, n_neighbors=n_neighbors)
            pacmap_reduced: np.ndarray = reducer.fit_transform(embeddings)
            return pacmap_reduced
        else:
            import umap

            reducer = umap.UMAP(n_components=target_dim, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
            umap_reduced: np.ndarray = reducer.fit_transform(embeddings)
            return umap_reduced
    except Exception as e:
        logger.warning("Embedding reduction failed: %s", e)
        return None


def _create_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Create an optimizer by name."""
    name = name.lower()
    params = model.parameters()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "lamb":
        try:
            from timm.optim import Lamb

            optimizer: torch.optim.Optimizer = Lamb(params, lr=lr, weight_decay=weight_decay)
            return optimizer
        except ImportError:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:  # adamw (default)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    epochs: int,
    warmup_epochs: int,
    steps_per_epoch: int,
) -> Any:
    """Create a learning rate scheduler with optional linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        name: Scheduler name — "cosine", "step", or "plateau".
        epochs: Total training epochs.
        warmup_epochs: Number of warmup epochs (linear ramp from ~0 to base LR).
        steps_per_epoch: Not used currently (schedulers step per-epoch).

    Returns:
        A scheduler, or None for plateau.

    """
    name = name.lower()

    if name == "plateau":
        return None  # ReduceLROnPlateau needs val loss — skip for now

    # Build the main scheduler over the post-warmup epochs
    main_epochs = max(1, epochs - warmup_epochs)
    if name == "step":
        main_scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, main_epochs // 3), gamma=0.1
        )
    else:  # cosine (default)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_epochs)

    # If no warmup requested, just return the main scheduler
    if warmup_epochs <= 0:
        return main_scheduler

    # Linear warmup: ramp from 1/10th of base LR to full LR over warmup_epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
    )
