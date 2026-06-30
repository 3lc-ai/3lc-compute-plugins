# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Detect task type and auto-detect image/label columns from a 3LC table for timm."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def detect_task(table_url: str) -> dict[str, Any]:
    """Inspect a table's schema and detect task type + column mapping.

    Delegates schema inspection to the shared modality module, then
    extracts classification-specific fields (image/label columns, class names).

    Args:
        table_url: URL of the 3LC table to inspect.

    Returns:
        Dict with task, image_column, label_column, num_classes, class_names, project_name.

    """
    try:
        import tlc
        from tlc_plugin_sdk.shared.modality import detect_modality_from_table

        table = tlc.Table.from_url(table_url)
        info = detect_modality_from_table(table)
        project_name = getattr(table, "project_name", "") or ""
    except Exception as e:
        return {
            "task": "unknown",
            "image_column": "",
            "label_column": "",
            "num_classes": 0,
            "class_names": [],
            "project_name": "",
            "details": f"Could not load table: {e}",
        }

    image_column = info.image_column or ""
    label_column = info.label_column or ""
    num_classes = info.num_classes
    class_names = list(info.class_names.values()) if info.class_names else []

    # Determine task from timm's perspective
    if info.modality == "classification" and image_column and label_column:
        task = "classification"
    elif image_column:
        task = "feature_extraction"
    else:
        task = "unknown"

    return {
        "task": task,
        "image_column": image_column,
        "label_column": label_column,
        "num_classes": num_classes,
        "class_names": class_names,
        "project_name": project_name,
        "details": f"Detected: {task}" + (f" ({num_classes} classes)" if num_classes else ""),
    }
