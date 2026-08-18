# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify that every plugin class instantiates and exposes the expected interface."""

from __future__ import annotations

from typing import Any

import pytest
from tlc_plugin_sdk import ComputePlugin

_has_data_source = True
try:
    import tlc_plugin_sdk.shared.data_source_routes  # noqa: F401
except ModuleNotFoundError:
    _has_data_source = False

needs_data_source = pytest.mark.skipif(not _has_data_source, reason="SDK data_source module not yet published")


def test_is_compute_plugin_subclass(plugin_instance: Any) -> None:
    assert isinstance(plugin_instance, ComputePlugin)


@needs_data_source
def test_get_ui_fragment_returns_html(plugin_instance: Any) -> None:
    fragment = plugin_instance.get_ui_fragment()
    assert isinstance(fragment, str)
    if not fragment:
        # An empty fragment is a legitimate shape: inline-display plugins
        # (table_statistics) render in the frontend and have no standalone page.
        return
    assert "<" in fragment, "UI fragment should contain HTML"


@needs_data_source
def test_get_route_handlers_returns_list(plugin_instance: Any) -> None:
    if not hasattr(plugin_instance, "get_route_handlers"):
        return
    handlers = plugin_instance.get_route_handlers()
    assert isinstance(handlers, list)
