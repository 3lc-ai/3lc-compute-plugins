# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Model registry — auto-discovers training model implementations."""

from __future__ import annotations

import logging
from typing import Any

from tlc_plugin_yolo.models.base import BaseTrainingModel

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, BaseTrainingModel] = {}


def register_model(model: BaseTrainingModel) -> None:
    """Register a model instance."""
    MODEL_REGISTRY[model.name] = model
    logger.info("Registered training model: %s (%s)", model.name, model.display_name)


def discover_models() -> None:
    """Import model modules to trigger registration."""
    try:
        from tlc_plugin_yolo.models import yolo  # noqa: F401
    except ImportError:
        logger.warning("YOLO model not available (3lc-ultralytics not installed)")
    except Exception:
        logger.exception("Failed to load YOLO model")

    if MODEL_REGISTRY:
        logger.info("Discovered %d training model(s): %s", len(MODEL_REGISTRY), list(MODEL_REGISTRY.keys()))
    else:
        logger.warning("No training models discovered — model dropdown will be empty")


def get_models_for_task(task: str) -> list[dict[str, Any]]:
    """Return models compatible with a task type."""
    results = []
    for model in MODEL_REGISTRY.values():
        if task == "unknown" or task in model.supported_tasks:
            results.append({
                "name": model.name,
                "display_name": model.display_name,
                "supported_tasks": model.supported_tasks,
            })
    return results
