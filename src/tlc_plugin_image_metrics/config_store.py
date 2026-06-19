# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
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


# Pre-standardization location, migrated into ~/.3lc-plugin-configs/image-metrics/
# on first store construction. Remove once the cutover is complete.
_LEGACY_DIR = Path.home() / ".3lc-training" / "image-metrics-configs"


def config_store() -> PluginConfigStore[ImageMetricsConfig]:
    """Return a store for Image Metrics saved configs (cheap; not cached)."""
    return PluginConfigStore(ImageMetricsConfig, "image-metrics", legacy_dir=_LEGACY_DIR)
