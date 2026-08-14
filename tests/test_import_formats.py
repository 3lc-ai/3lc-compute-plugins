# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate the importer's format definitions are well-formed."""

from __future__ import annotations

import pytest

from tlc_plugin_importer import IMPORT_STEPS

VALID_FIELD_TYPES = {"text", "select", "textarea", "file_upload", "table_picker", "data_source"}


@pytest.fixture(params=list(IMPORT_STEPS.keys()), ids=list(IMPORT_STEPS.keys()))
def format_def(request: pytest.FixtureRequest) -> dict:
    return IMPORT_STEPS[request.param]


def test_format_has_required_keys(format_def: dict) -> None:
    for key in ("name", "display_name", "form_fields"):
        assert key in format_def, f"Format {format_def.get('name', '?')!r} missing {key!r}"


def test_fields_have_valid_types(format_def: dict) -> None:
    for field in format_def["form_fields"]:
        assert "id" in field, f"Field missing 'id' in format {format_def['name']!r}"
        assert "type" in field, f"Field {field['id']!r} missing 'type'"
        assert field["type"] in VALID_FIELD_TYPES, f"Field {field['id']!r} has unknown type {field['type']!r}"


def test_field_ids_unique(format_def: dict) -> None:
    ids = [f["id"] for f in format_def["form_fields"]]
    assert len(ids) == len(set(ids)), f"Duplicate field ids in format {format_def['name']!r}: {ids}"
