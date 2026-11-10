# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Import: a YOLO table can be read where its data lives, including in a bucket.

tlc refuses a YOLO table whose source is remote unless ``allow_fetch_remote_data`` is set, and it
checks this on every load, not only while building rows. A local import whose alias points at a
bucket was written, then refused by every reader: the Dashboard showed "malformed data". An import
straight from a bucket failed outright. The importer sets the flag, which is saved with the table.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

importer = pytest.importorskip("tlc_plugin_importer")


def test_a_yolo_import_allows_its_data_to_be_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def from_yolo_url(images_url: str, **kwargs: Any) -> Any:
        calls.append({"images_url": images_url, **kwargs})
        return types.SimpleNamespace(url="s3://bucket/p/datasets/d/tables/initial", row_count=3)

    fake_tlc = types.SimpleNamespace(Table=types.SimpleNamespace(from_yolo_url=from_yolo_url))
    monkeypatch.setitem(sys.modules, "tlc", fake_tlc)
    monkeypatch.setattr(
        importer, "_parse_yolo_yaml_for_split", lambda _yaml, _split: ("s3://bucket/data/test/images", None)
    )

    result = importer._execute_yolo({
        "dataset_yaml": "s3://bucket/data/dataset.yaml",
        "split": "test",
        "task": "detect",
        "project_name": "p",
        "dataset_name": "d",
        "table_name": "",
        "description": "",
        "project_root_url": "s3://bucket",
    })

    assert result["success"] is True
    assert len(calls) == 1
    assert calls[0]["images_url"] == "s3://bucket/data/test/images"
    assert calls[0]["allow_fetch_remote_data"] is True
