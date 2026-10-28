# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Split: the Random Seed, 0 included, is the seed the split uses (Bug 19084).

A seed of 0 was replaced by 42 in the page and in the job, so a seed-0 split was the seed-42 split. ``import tlc``
asks for an API key, which this suite has none of, so the job runs against a stand-in ``tlc`` with an in-memory table.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("tlc_plugin_sdk")
splitter = pytest.importorskip("tlc_plugin_splitter")

UI = (Path(splitter.__file__).parent / "ui.html").read_text(encoding="utf-8")


class _Table:
    """``n`` rows; records the rows each split keeps."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.kept: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return self.n

    def delete_rows(self, remove: list[int], table_name: str, description: str) -> Any:
        self.kept[table_name] = sorted(set(range(self.n)) - set(remove))
        return types.SimpleNamespace(url=f"s3://bucket/{table_name}")


def _split(monkeypatch: pytest.MonkeyPatch, **form: Any) -> tuple[dict[str, Any], _Table]:
    table = _Table(20)
    fake_tlc = types.ModuleType("tlc")
    fake_tlc.Table = types.SimpleNamespace(from_url=lambda url: table)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tlc", fake_tlc)
    return splitter._execute_split({"table_url": "s3://bucket/t", "train_pct": 70, "val_pct": 30, **form}), table


@pytest.mark.parametrize(("value", "seed"), [(0, 0), ("0", 0), (7, 7), ("7", 7), (None, 42), ("", 42), ("  ", 42)])
def test_seed_is_42_only_when_blank(value: Any, seed: int) -> None:
    assert splitter._seed(value) == seed


def test_seed_0_splits_with_0(monkeypatch: pytest.MonkeyPatch) -> None:
    result, zero = _split(monkeypatch, seed=0)
    assert result["success"]
    assert result["details"]["seed"] == 0
    _, forty_two = _split(monkeypatch, seed=42)
    assert zero.kept["train"] != forty_two.kept["train"]


def test_no_seed_splits_with_42(monkeypatch: pytest.MonkeyPatch) -> None:
    result, blank = _split(monkeypatch)
    assert result["details"]["seed"] == 42
    _, forty_two = _split(monkeypatch, seed=42)
    assert blank.kept == forty_two.kept


def test_page_sends_seed_0() -> None:
    assert "var seed = document.getElementById('split-seed').valueAsNumber;" in UI
    assert "seed: Number.isInteger(seed) ? seed : 42," in UI
    assert "|| 42" not in UI
