# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from tlc_plugin_splitter.weighted_subset import run_weighted_subset


def run(monkeypatch, count=3000, seed=42, cancelled=False):
    weights = np.concatenate([np.full(5000, 2.0), np.zeros(24537)])
    writes = []

    class URL:
        def create_sibling(self, name):
            assert name == "subset"
            return self

        def create_unique(self):
            return "unique-child"

        def __str__(self):
            return "exact-source"

    class Table:
        url = URL()
        weights_column_name = "selection"

        def __len__(self):
            return len(weights)

        def get_column_as_pyarrow_array(self, column):
            assert column == "selection"
            return pa.array(weights)

    table = Table()

    class Edited:
        indices_to_run = staticmethod(lambda rows: rows)

        def __init__(self, **kw):
            self.kw = kw
            self.url = kw["url"]

        def ensure_fully_defined(self):
            pass

        def write_to_url(self):
            writes.append(self.kw)

    monkeypatch.setitem(
        sys.modules, "tlc", SimpleNamespace(Table=SimpleNamespace(from_url=lambda url: table), EditedTable=Edited)
    )
    monkeypatch.setitem(
        sys.modules, "tlc._core.objects.tables.from_table.edited_table", SimpleNamespace(EditedTable=Edited)
    )
    outputs = {}
    ctx = SimpleNamespace(
        params={"table_url": "exact-source", "keep_count": count, "seed": seed, "table_name": "subset"},
        cancelled=cancelled,
        progress=lambda **kw: None,
        result=lambda url: None,
        output=lambda key, value: outputs.update({key: value}),
    )
    run_weighted_subset(ctx)
    return table, weights, writes, outputs


def test_3k_subset_zeroes_only_2k_and_retains_history_and_nonunit_weights(monkeypatch):
    source, weights, writes, outputs = run(monkeypatch)
    assert len(writes) == 1
    assert writes[0]["input_table_url"] is source
    assert writes[0]["url"] == "unique-child"
    excluded, value = writes[0]["edits"]["selection"]["runs_and_values"]
    assert len(excluded) == len(set(excluded)) == 2000
    assert max(excluded) < 5000 and value == 0.0
    assert np.count_nonzero(weights) == 5000  # source untouched
    modified = weights.copy()
    modified[excluded] = value
    assert np.count_nonzero(modified) == 3000 and len(modified) == 29537
    assert set(modified) == {0.0, 2.0}
    assert outputs["subset"]["newly_zeroed_rows"] == 2000
    assert run(monkeypatch)[2][0]["edits"] == writes[0]["edits"]
    assert run(monkeypatch, seed=0)[2][0]["edits"] != writes[0]["edits"]


@pytest.mark.parametrize("count", [0, -1, True, 5001, "3000"])
def test_bad_count_does_not_publish(monkeypatch, count):
    with pytest.raises(ValueError):
        run(monkeypatch, count=count)


def test_cancelled_job_does_not_publish(monkeypatch):
    assert run(monkeypatch, cancelled=True)[2] == []
