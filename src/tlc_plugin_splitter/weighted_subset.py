# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep a reproducible subset of eligible samples as a sparse EditedTable revision."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tlc_plugin_sdk import JobContext


def run_weighted_subset(ctx: JobContext) -> None:
    import numpy as np
    import tlc
    from tlc._core.objects.tables.from_table.edited_table import EditedTable
    from tlc_plugin_sdk.shared.url_utils import normalize_url
    from tlc_plugin_sdk.shared.weights import sample_weights

    data = ctx.params
    count = data.get("keep_count")
    seed = data.get("seed", 42)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        msg = "keep_count must be a positive integer"
        raise ValueError(msg)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        msg = "seed must be a nonnegative integer"
        raise ValueError(msg)
    name = str(data.get("table_name") or f"weighted_subset_{count}").strip()
    if not name or name in {".", ".."} or any(c in name for c in "/\\%"):
        msg = "table_name must be a single table name, not a path"
        raise ValueError(msg)
    table = tlc.Table.from_url(normalize_url(data["table_url"]))
    ctx.progress(percent=10, label="Counting eligible samples")
    column, weights = sample_weights(table)
    eligible = np.flatnonzero(weights > 0)
    if count > len(eligible):
        msg = f"Requested {count} samples, but this revision has only {len(eligible)} positive-weight rows"
        raise ValueError(msg)
    # Choose exclusions directly; kept weights and already-zero rows stay untouched.
    excluded = np.sort(np.random.default_rng(seed).choice(eligible, len(eligible) - count, replace=False)).tolist()
    override = None
    runs_and_values: list[object] = []
    if column is None:
        if "weight" in table.rows_schema.values:
            msg = "The existing weight column is not declared as sample weights"
            raise ValueError(msg)
        column = "weight"
        override = {"values": {column: tlc.schemas.SampleWeightSchema(default_value=1.0)}}
        runs_and_values.extend([([0] if len(table) == 1 else [0, 1 - len(table)]), 1.0])
    if excluded:
        runs_and_values.extend([EditedTable.indices_to_run(excluded), 0.0])
    if ctx.cancelled:
        return
    revised = EditedTable(
        input_table_url=table,
        url=table.url.create_sibling(name).create_unique(),
        edits={column: {"runs_and_values": runs_and_values}} if runs_and_values else {},
        override_table_rows_schema=override,
        description=f"Keep {count} of {len(eligible)} positive-weight samples (seed {seed}); other weights set to zero",
    )
    revised.ensure_fully_defined()
    revised.write_to_url()
    result = {
        "table_url": str(revised.url),
        "source_table_url": str(table.url),
        "total_rows": len(table),
        "previous_positive_weight_rows": len(eligible),
        "positive_weight_rows": count,
        "newly_zeroed_rows": len(excluded),
        "zero_weight_rows": len(table) - count,
        "seed": seed,
    }
    ctx.result(str(revised.url))
    ctx.output("table_url", str(revised.url))
    ctx.output("subset", result)
    ctx.progress(percent=100, label=f"Created revision with {count} eligible samples")
