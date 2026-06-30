# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""timm saved job configs — schema + store factory.

The JSON-on-disk CRUD lives in the shared
:class:`tlc_plugin_sdk.shared.config_store.PluginConfigStore`; this
module only declares the plugin's config schema and a store factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tlc_plugin_sdk.shared.config_store import PluginConfigStore


@dataclass
class TimmConfig:
    """A saved timm training configuration."""

    id: str = ""
    name: str = ""
    run_name: str = ""  # 3LC Run name (empty = random)
    project_name: str = ""  # 3LC project for Run association
    model_name: str = ""  # timm model name, e.g. "resnet50", "vit_base_patch16_224"
    task_type: str = ""  # "classification" or "feature_extraction"
    train_table_url: str = ""
    val_table_url: str = ""
    use_latest: bool = False  # Resolve .latest() at train time
    mode: str = "train"  # "train" or "collect"
    image_column: str = ""  # Auto-detected or user-specified
    label_column: str = ""  # Auto-detected or user-specified
    params: dict[str, Any] = field(default_factory=dict)
    created: str = ""
    last_run: str | None = None


# Pre-standardization location, migrated into ~/.3lc-plugin-configs/timm/ on
# first store construction. Remove once the cutover is complete.
_LEGACY_DIR = Path.home() / ".3lc-training" / "timm-configs"


def config_store() -> PluginConfigStore[TimmConfig]:
    """Return a store for timm saved configs (cheap; not cached)."""
    return PluginConfigStore(TimmConfig, "timm", legacy_dir=_LEGACY_DIR)
