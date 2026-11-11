# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
import itertools
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tlc_plugin_splitter.binary_sweep import run_binary_sweep


def execute(monkeypatch, counts, *, column="selection", cancel_after=None, seed=42):
    weights = np.array([2.0] * 5000 + [0.0] * 24537) if column else np.ones(29537)
    writes, snapshots = [], []

    class URL:
        def create_sibling(self, name):
            return self

        def create_unique(self):
            return f"child-{len(writes)}"

        def __str__(self):
            return "exact-source"

    table = SimpleNamespace(url=URL(), rows_schema=SimpleNamespace(values={}))

    class Edited:
        indices_to_run = staticmethod(lambda rows: rows)

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.url = kwargs["url"]

        def ensure_fully_defined(self):
            pass

        def write_to_url(self):
            writes.append(self.kwargs)

    monkeypatch.setitem(
        sys.modules,
        "tlc",
        SimpleNamespace(
            Table=SimpleNamespace(from_url=lambda url: table),
            schemas=SimpleNamespace(SampleWeightSchema=lambda **kw: kw),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "tlc._core.objects.tables.from_table.edited_table", SimpleNamespace(EditedTable=Edited)
    )
    monkeypatch.setattr("tlc_plugin_sdk.shared.weights.sample_weights", lambda t: (column, weights))

    class Context:
        def __init__(self):
            self.params = {"table_url": "exact-source", "counts": counts, "seed": seed}

        @property
        def cancelled(self):
            return cancel_after is not None and len(writes) >= cancel_after

        def output(self, name, value):
            snapshots.append(value)

        def progress(self, **kwargs):
            pass

        def result(self, value):
            pass

    run_binary_sweep(Context())
    values = []
    for write in writes:
        assert write["input_table_url"] is table
        edited = weights.copy()
        runs = write["edits"].get(column or "weight", {}).get("runs_and_values", [])
        for indices, value in zip(runs[::2], runs[1::2]):
            edited[indices] = value
        values.append(edited)
    return writes, values, snapshots


def test_ten_nested_binary_revisions_expand_and_shrink_without_changing_source(monkeypatch):
    counts = [3000, 3444, 3889, 4333, 4778, 5222, 5667, 6111, 6556, 7000]
    writes, values, snapshots = execute(monkeypatch, counts)
    assert len(writes) == 10
    for count, value in zip(counts, values):
        assert len(value) == 29537 and np.count_nonzero(value) == count
        assert set(value) == {0, 1}
    assert all(np.all(a <= b) for a, b in itertools.pairwise(values))
    assert np.all(values[-1][:5000] == 1)
    assert snapshots[-1][0]["newly_zeroed_rows"] == 2000
    assert snapshots[-1][-1]["newly_enabled_rows"] == 2000
    assert np.array_equal(execute(monkeypatch, counts)[1][-1], values[-1])
    assert not np.array_equal(execute(monkeypatch, counts, seed=7)[1][-1], values[-1])


@pytest.mark.parametrize("counts", [[], [0], [True], [3000, 3000], [3000, 29538], [1.5], "3000"])
def test_invalid_batch_fails_before_any_publication(monkeypatch, counts):
    with pytest.raises(ValueError):
        execute(monkeypatch, counts)


def test_implicit_weights_and_cancellation_preserve_published_receipts(monkeypatch):
    writes, values, snapshots = execute(monkeypatch, [3000, 5000], column=None, cancel_after=1)
    assert len(writes) == len(snapshots[-1]) == 1
    assert np.count_nonzero(values[0]) == 3000
    assert writes[0]["override_table_rows_schema"] is not None


def test_3000_to_7500_preserves_existing_5000_first(monkeypatch):
    counts = list(range(3000, 7501, 500))
    writes, values, snapshots = execute(monkeypatch, counts)
    assert len(writes) == 10
    for count, value, receipt in zip(counts, values, snapshots[-1]):
        assert len(value) == 29537
        assert np.count_nonzero(value[:5000]) == min(count, 5000)
        assert np.count_nonzero(value[5000:]) == max(count - 5000, 0)
        assert receipt["newly_zeroed_rows"] == max(5000 - count, 0)
        assert receipt["newly_enabled_rows"] == max(count - 5000, 0)
    assert all(np.all(a <= b) for a, b in itertools.pairwise(values))
