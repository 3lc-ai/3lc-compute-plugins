# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Progressive statistics always finish, including failed SDK initialization."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from tlc_plugin_sdk.shared import modality

from tlc_plugin_table_statistics import table_stats

URL = "s3://test-project/datasets/test/tables/initial"


@pytest.fixture(autouse=True)
def isolated_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(table_stats, "_sessions", {})


def _wait_for_stats() -> dict[str, Any]:
    table_stats.get_or_start_stats(URL)
    thread = table_stats._sessions[URL]._thread
    assert thread is not None
    thread.join(timeout=5)
    assert not thread.is_alive(), "Statistics thread did not finish"
    return table_stats.get_or_start_stats(URL)


@pytest.mark.parametrize("error_type", [RuntimeError, ModuleNotFoundError])
def test_failed_sdk_import_finishes_cached_session(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    original_import = builtins.__import__
    imports = []

    def fail_tlc_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tlc":
            imports.append(name)
            msg = "SDK activation failed"
            raise error_type(msg)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_tlc_import)

    result = _wait_for_stats()

    assert result["complete"] is True
    assert result["error"] == "SDK activation failed"
    assert result["rows_processed"] == 0
    assert result["columns"] == []
    # Subsequent polls keep the terminal error instead of launching another worker.
    assert table_stats.get_or_start_stats(URL) == result
    assert imports == ["tlc"]


def test_successful_stats_still_process_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    table = MagicMock()
    table.__len__.return_value = 3
    table.rows_schema = SimpleNamespace(values={"score": object()})
    table.get_column_as_pyarrow_array.return_value = pa.array([1.0, 2.0, 3.0])
    fake_tlc = SimpleNamespace(Table=SimpleNamespace(from_url=lambda url: table))
    monkeypatch.setitem(sys.modules, "tlc", fake_tlc)
    monkeypatch.setattr(modality, "detect_modality_from_table", lambda table: modality.ModalityInfo())
    monkeypatch.setattr(table_stats, "_get_value_map", lambda table, name: None)
    monkeypatch.setattr(table_stats, "BATCH_SIZE", 2)

    result = _wait_for_stats()

    assert result["complete"] is True
    assert result["error"] is None
    assert result["row_count"] == result["rows_processed"] == 3
    assert result["column_count"] == 1
    column = result["columns"][0]
    assert column["kind"] == "numeric"
    assert (column["min"], column["max"], column["mean"]) == (1.0, 3.0, 2.0)
    assert sum(column["histogram"]["counts"]) == 3
