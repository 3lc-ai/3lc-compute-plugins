# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The exporter's column classifier reads the live schema object, not just its JSON.

``tlc.Schema.to_json()`` omits fields equal to their defaults, and float32 is tlc's
default scalar type — so a plain float column serializes as ``{"value": {}}`` with no
``type`` key at all. Classifying from the serialized JSON alone therefore labeled every
float32 column "other" in the export column list, while int32/string/bool columns (whose
types survive serialization) were labeled correctly.

The fakes below reproduce the exact ``to_json()`` shapes captured from real ``tlc``
schemas (a real table's ``rows_schema.values`` entries). Real ``tlc`` is not imported
here: importing it requires credential activation, which CI does not have.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from tlc_plugin_exporter import _classify_column_type


class _FakeSchema:
    """Structural stand-in for ``tlc.Schema`` as the classifier consumes it.

    ``json_dict`` is what ``to_json()`` yields (already default-elided, as real tlc
    serializes); ``value`` is the live scalar-value object (``None`` for composite
    schemas — verified against real tlc, where a composite schema's ``value`` is None).
    """

    def __init__(self, json_dict: dict[str, Any], value: Any = None) -> None:
        self._json = json_dict
        self.value = value

    def to_json(self) -> str:
        return json.dumps(self._json)


def _scalar(value_type: str, string_role: str = "") -> SimpleNamespace:
    return SimpleNamespace(type=value_type, string_role=string_role)


# Captured shapes: (serialized JSON, live value object) → expected classification.
CASES = [
    # THE regression: a default float32 scalar serializes as {"value": {}} — the JSON
    # carries no type, only the live object knows it. Previously classified "other".
    pytest.param(_FakeSchema({"value": {}}, _scalar("float32")), "col", "number", id="float32-default"),
    # Weight is a plain Float32Value whose serialized form has no type either; it used
    # to be rescued by a col_name == "weight" special case, now classified naturally.
    pytest.param(
        _FakeSchema({"default_value": 1.0, "value": {"number_role": "sample_weight"}}, _scalar("float32")),
        "weight",
        "number",
        id="sample-weight",
    ),
    # Non-default scalar types survive serialization and agree with the live object.
    pytest.param(
        _FakeSchema({"value": {"type": "int32", "number_role": "label"}}, _scalar("int32")),
        "label",
        "number",
        id="int32-label",
    ),
    pytest.param(_FakeSchema({"value": {"type": "int64"}}, _scalar("int64")), "count", "number", id="int64"),
    pytest.param(_FakeSchema({"value": {"type": "float64"}}, _scalar("float64")), "score", "number", id="float64"),
    pytest.param(_FakeSchema({"value": {"type": "string"}}, _scalar("string")), "caption", "string", id="string"),
    pytest.param(_FakeSchema({"value": {"type": "bool"}}, _scalar("bool")), "flag", "boolean", id="bool"),
    # Image detection is by string_role (real role is "URL/Image").
    pytest.param(
        _FakeSchema({"value": {"type": "string", "string_role": "URL/Image"}}, _scalar("string", "URL/Image")),
        "image",
        "image",
        id="image-role",
    ),
    # Embeddings are detected by column-name convention.
    pytest.param(_FakeSchema({"value": {}}, _scalar("float32")), "clip_embedding", "embedding", id="embedding-name"),
    # Composite schemas (live value is None) classify from their nested JSON — and a
    # composite whose value object DID exist would still hit bbox/segmentation first,
    # because those branches precede the scalar-type ladder.
    pytest.param(
        _FakeSchema({"values": {"bb_list": {"values": {"x0": {}, "y0": {}}}}}),
        "bbs",
        "bbox",
        id="bbox-nested",
    ),
    pytest.param(
        _FakeSchema({"values": {"bb_list": {"values": {"x0": {}}}}}, _scalar("float32")),
        "bbs",
        "bbox",
        id="bbox-wins-over-scalar-value",
    ),
    pytest.param(
        _FakeSchema({"values": {"instance_properties": {}, "rles": {}}}),
        "segments",
        "segmentation",
        id="segmentation-nested",
    ),
    # Unmatched nested schema is "complex"; a bare unknown scalar is "other".
    pytest.param(_FakeSchema({"values": {"a": {}, "b": {}}}), "pair", "complex", id="complex-nested"),
    pytest.param(_FakeSchema({"value": {}}), "mystery", "other", id="no-type-anywhere"),
]


@pytest.mark.parametrize(("schema", "col_name", "expected"), CASES)
def test_classify_column_type(schema: _FakeSchema, col_name: str, expected: str) -> None:
    assert _classify_column_type(schema, col_name) == expected


def test_json_only_fallback_still_classifies() -> None:
    """A schema whose live value is unreadable still classifies from the JSON."""

    class _Hostile:
        @property
        def value(self) -> Any:
            msg = "no live value"
            raise RuntimeError(msg)

        def to_json(self) -> str:
            return json.dumps({"value": {"type": "int32"}})

    assert _classify_column_type(_Hostile(), "label") == "number"


def test_never_raises_on_junk_input() -> None:
    assert _classify_column_type(object(), "anything") == "other"
    assert _classify_column_type(None, "") == "other"

    class _BrokenToJson:
        def to_json(self) -> str:
            msg = "boom"
            raise RuntimeError(msg)

    assert _classify_column_type(_BrokenToJson(), "col") == "other"
