# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""SAM3 saved job configs — schema + store factory.

The JSON-on-disk CRUD lives in the shared
:class:`tlc_plugin_sdk.shared.config_store.PluginConfigStore`; this
module only declares the plugin's config schema and a store factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tlc_plugin_sdk.shared.config_store import PluginConfigStore


@dataclass
class SAM3Config:
    """A saved SAM3 auto-labeling configuration."""

    id: str = ""
    name: str = ""
    source_type: str = "folder"  # folder | table
    folder: str = ""
    table_url: str = ""
    labels: list[dict[str, str]] = field(default_factory=list)  # [{name, color}]
    modality: str = "segmentation"
    confidence: float = 0.2
    project_name: str = ""
    dataset_name: str = "train"
    table_name: str = "initial"
    embedding_dim: int = 2
    device: str = "cuda"
    created: str = ""
    last_run: str | None = None


# Pre-standardization location, migrated into ~/.3lc-plugin-configs/sam3/ on
# first store construction. Remove once the cutover is complete.
_LEGACY_DIR = Path.home() / ".3lc-sam3" / "configs"


def config_store() -> PluginConfigStore[SAM3Config]:
    """Return a store for SAM3 saved configs (cheap; not cached)."""
    return PluginConfigStore(SAM3Config, "sam3", legacy_dir=_LEGACY_DIR)
