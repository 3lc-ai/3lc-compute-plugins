# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Abstract base class for training models."""

from __future__ import annotations

import abc
from typing import Any, ClassVar


class BaseTrainingModel(abc.ABC):
    """Abstract base for all training model integrations.

    Subclasses define class-level metadata and implement train() and collect().
    """

    name: str = ""
    """Unique model identifier (e.g. ``yolov8``)."""

    display_name: str = ""
    """Human-readable name shown in the UI."""

    supported_tasks: ClassVar[list[str]] = []
    """Task types this model supports (e.g. ``["detection", "classification"]``)."""

    @abc.abstractmethod
    def get_params(self) -> list[dict[str, Any]]:
        """Return parameter field definitions for the UI form.

        Each dict has: id, label, type, default, group, and optional min/max/step/options/help.
        """

    @abc.abstractmethod
    def train(self, tables: dict[str, Any], params: dict[str, Any], callbacks: dict[str, Any]) -> dict[str, Any]:
        """Run training.

        Args:
            tables: ``{"train": url, "val": url | None}``.
            params: Merged user parameters + internal ``_project_name``, ``_run_name``, ``_task_type``.
            callbacks: ``{"on_epoch": fn, "on_status": fn, "is_cancelled": fn}``.

        Returns:
            ``{"run_url": str | None, "final_metrics": dict}``.

        """

    @abc.abstractmethod
    def collect(self, tables: dict[str, Any], params: dict[str, Any], callbacks: dict[str, Any]) -> dict[str, Any]:
        """Run metrics collection only (no training).

        Same signature and return type as ``train()``.
        """

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate parameters. Returns list of error messages (empty = valid)."""
        return []
