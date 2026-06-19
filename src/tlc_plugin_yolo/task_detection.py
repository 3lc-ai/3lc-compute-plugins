# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Detect the ML task type from a 3LC table's schema for YOLO training."""

from __future__ import annotations

from typing import Any

from tlc_plugin_yolo.models import MODEL_REGISTRY


def detect_task(table_url: str) -> dict[str, Any]:
    """Read a table's schema and detect the task type.

    Delegates schema inspection to the shared modality module, then maps
    the result to YOLO-specific task names and compatible models.

    Args:
        table_url: URL of the 3LC table to inspect.

    Returns:
        Dict with task, models, details, and project_name.

    """
    try:
        import tlc

        from tlc_plugin_sdk.shared.modality import detect_modality_from_table

        table = tlc.Table.from_url(table_url)
        info = detect_modality_from_table(table)
        task = _modality_to_yolo_task(info.modality)
        project_name = getattr(table, "project_name", "") or ""
    except Exception as e:
        return {
            "task": "unknown",
            "models": _models_for_task("unknown"),
            "details": f"Could not load table: {e}",
            "project_name": "",
        }

    return {
        "task": task,
        "models": _models_for_task(task),
        "details": f"Detected task: {task}",
        "project_name": project_name,
    }


# Map shared modality names to YOLO task names.
_MODALITY_TO_YOLO: dict[str, str] = {
    "detection": "detection",
    "segmentation": "segmentation",
    "classification": "classification",
    "pose": "pose",
    "obb": "obb",
}


def _modality_to_yolo_task(modality: str) -> str:
    """Convert a shared modality name to a YOLO task name."""
    return _MODALITY_TO_YOLO.get(modality, "unknown")


def _models_for_task(task: str) -> list[dict[str, Any]]:
    """Return models compatible with the given task."""
    results = []
    for model in MODEL_REGISTRY.values():
        if task == "unknown" or task in model.supported_tasks:
            results.append({
                "name": model.name,
                "display_name": model.display_name,
                "supported_tasks": model.supported_tasks,
            })
    return results
