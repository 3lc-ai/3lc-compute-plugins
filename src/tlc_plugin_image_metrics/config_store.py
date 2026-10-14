# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image Metrics saved job configs — schema + store factory.

The JSON-on-disk CRUD lives in the shared
:class:`tlc_plugin_sdk.shared.config_store.PluginConfigStore`; this
module only declares the plugin's config schema and a store factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tlc_plugin_sdk.shared.config_store import PluginConfigStore


@dataclass
class ImageMetricsConfig:
    """A saved Image Metrics configuration."""

    id: str = ""
    name: str = ""
    metric_ids: list[str] = field(default_factory=list)
    output_name_suffix: str = "metrics"
    created: str = ""
    last_run: str | None = None


def _legacy_dir() -> Path | None:
    """Where older versions saved configs (``~/.3lc-training/image-metrics-configs``), or ``None`` without a home.

    Resolved when a store is built, not at import, as the SDK resolves its own config root: a worker
    whose environment carries no home directory still imports the plugin.
    """
    try:
        return Path.home() / ".3lc-training" / "image-metrics-configs"
    except RuntimeError:
        return None


def config_store() -> PluginConfigStore[ImageMetricsConfig]:
    """Return a store for Image Metrics saved configs (cheap; not cached).

    Configs saved under the older location are moved into the SDK's config root
    (``~/.3lc-plugin-configs/image-metrics/``) on first store construction.

    Raises:
        ConfigRootUnavailable: When the SDK cannot place its config root (no home directory, no override).
    """
    return PluginConfigStore(ImageMetricsConfig, "image-metrics", legacy_dir=_legacy_dir())
