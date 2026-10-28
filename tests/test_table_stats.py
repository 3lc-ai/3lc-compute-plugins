# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Progressive statistics always finish, and a failed session is retried on the next request."""

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


def _fake_tlc(monkeypatch: pytest.MonkeyPatch, from_url: Any) -> None:
    monkeypatch.setitem(sys.modules, "tlc", SimpleNamespace(Table=SimpleNamespace(from_url=from_url)))


def _numeric_table() -> MagicMock:
    table = MagicMock()
    table.__len__.return_value = 3
    table.rows_schema = SimpleNamespace(values={"score": object()})
    table.get_column_as_pyarrow_array.return_value = pa.array([1.0, 2.0, 3.0])
    return table


@pytest.fixture
def plain_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modality, "detect_modality_from_table", lambda table: modality.ModalityInfo())
    monkeypatch.setattr(table_stats, "_get_value_map", lambda table, name: None)


@pytest.mark.parametrize("error_type", [RuntimeError, ModuleNotFoundError])
def test_failed_sdk_import_finishes_session_and_next_request_retries(
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
    # The poll that saw the error dropped the session, so the next request starts a new attempt.
    assert URL not in table_stats._sessions
    assert _wait_for_stats()["error"] == "SDK activation failed"
    assert imports == ["tlc", "tlc"]


def test_failed_read_is_not_served_from_cache_after_it_is_fixed(
    monkeypatch: pytest.MonkeyPatch, plain_columns: None
) -> None:
    attempts = []

    def from_url(url: str) -> MagicMock:
        attempts.append(url)
        if len(attempts) == 1:
            msg = "s3 read_string failed: connection reset"
            raise RuntimeError(msg)
        return _numeric_table()

    _fake_tlc(monkeypatch, from_url)

    assert _wait_for_stats()["error"] == "s3 read_string failed: connection reset"
    retried = _wait_for_stats()

    assert retried["error"] is None
    assert retried["complete"] is True
    assert retried["rows_processed"] == 3
    assert len(attempts) == 2


def test_successful_stats_stay_cached(monkeypatch: pytest.MonkeyPatch, plain_columns: None) -> None:
    attempts = []

    def from_url(url: str) -> MagicMock:
        attempts.append(url)
        return _numeric_table()

    _fake_tlc(monkeypatch, from_url)

    result = _wait_for_stats()

    assert result["error"] is None
    assert table_stats.get_or_start_stats(URL) == result
    assert len(attempts) == 1


def test_missing_aws_credentials_say_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = "s3 read_string failed: object_store error: Generic S3 error: the credential provider was not enabled"

    def from_url(url: str) -> MagicMock:
        raise RuntimeError(raw)

    _fake_tlc(monkeypatch, from_url)

    error = _wait_for_stats()["error"]

    assert error.startswith("This compute service found no AWS credentials for bucket test-project.")
    assert "AWS_PROFILE" in error
    assert "restart it" in error
    assert error.endswith(raw)


def test_successful_stats_still_process_batches(monkeypatch: pytest.MonkeyPatch, plain_columns: None) -> None:
    table = _numeric_table()
    _fake_tlc(monkeypatch, lambda url: table)
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


def _table_with(columns: dict[str, pa.Array]) -> MagicMock:
    table = MagicMock()
    table.__len__.return_value = len(next(iter(columns.values())))
    table.rows_schema = SimpleNamespace(values=dict.fromkeys(columns, object()))
    table.get_column_as_pyarrow_array.side_effect = lambda name, combine_chunks=True: columns[name]
    return table


def test_list_valued_label_column_counts_every_label(monkeypatch: pytest.MonkeyPatch) -> None:
    # A multi-label column: classified as a label column by its value map, but each row holds a list.
    labels = pa.array([[0, 1], [1], [], None, [2, 1]], type=pa.list_(pa.int32()))
    table = _table_with({"labels": labels})
    _fake_tlc(monkeypatch, lambda url: table)
    monkeypatch.setattr(
        modality, "detect_modality_from_table", lambda table: modality.ModalityInfo(classification_columns=["labels"])
    )
    monkeypatch.setattr(table_stats, "BATCH_SIZE", 2)

    result = _wait_for_stats()

    assert result["error"] is None
    assert result["complete"] is True
    column = result["columns"][0]
    assert column["kind"] == "label"
    assert column["total_count"] == 5
    assert column["null_count"] == 1
    assert column["value_counts"] == {"1": 3, "0": 1, "2": 1}
    assert column["unique_count"] == 3


def test_nested_list_labels_are_flattened_and_mapped() -> None:
    # Per-instance labels inside a per-row list are flattened all the way down, then named by the value map.
    labels = pa.array([[[0.0], [1.0]], [[1.0]], None], type=pa.list_(pa.list_(pa.float32())))
    accum = table_stats._ColAccum(name="labels", kind="categorical", _value_map={0.0: "cat", 1.0: "dog"})

    accum.update(labels)

    assert dict(accum._value_counts) == {"cat": 1, "dog": 2}
    assert (accum.count, accum.null_count) == (3, 1)


def test_column_that_cannot_be_counted_does_not_fail_the_table(
    monkeypatch: pytest.MonkeyPatch, plain_columns: None
) -> None:
    table = _table_with({"score": pa.array([1.0, 2.0, 3.0]), "odd": pa.array([1, 2, 3])})
    _fake_tlc(monkeypatch, lambda url: table)
    original_update = table_stats._ColAccum.update

    def update(self: table_stats._ColAccum, arr: pa.Array) -> None:
        if self.name == "odd" and self.kind != "unknown":
            self.count += len(arr)
            msg = "Function 'value_counts' has no kernel matching input types"
            raise pa.ArrowNotImplementedError(msg)
        original_update(self, arr)

    monkeypatch.setattr(table_stats._ColAccum, "update", update)

    result = _wait_for_stats()

    assert result["error"] is None
    assert result["complete"] is True
    columns = {c["name"]: c for c in result["columns"]}
    assert columns["score"]["kind"] == "numeric"
    assert columns["score"]["mean"] == 2.0
    assert columns["odd"]["kind"] == "unknown"
    assert columns["odd"]["total_count"] == 3
