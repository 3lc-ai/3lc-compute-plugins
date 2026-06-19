# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Table Statistics plugin — column stats and image thumbnails for tables."""

from __future__ import annotations

from typing import Any

from tlc_plugin_table_statistics import routes as _routes
from tlc_plugin_sdk import ComputePlugin


class TableStatisticsPlugin(ComputePlugin):
    """Per-column statistics and image thumbnails for 3LC tables."""

    def get_ui_fragment(self) -> str:
        """No standalone UI — statistics are rendered inline by the frontend."""
        return ""

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Compute table statistics via the generic compute endpoint."""
        from tlc_plugin_table_statistics.table_stats import get_or_start_stats
        from tlc_plugin_sdk.shared.url_utils import normalize_url

        url = params.get("url", "")
        if not url:
            return {"error": "url parameter is required"}

        try:
            return get_or_start_stats(normalize_url(url))
        except Exception as exc:
            return {"error": str(exc)}

    def get_route_handlers(self) -> list[Any]:
        """Serve table_statistics' custom routes as relative Litestar handlers."""
        return _routes.get_route_handlers()
