# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Publish nested binary-weight selections without copying sample data."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tlc_plugin_sdk import JobContext


def run_binary_sweep(ctx: JobContext) -> None:
    """Create sibling EditedTables, keeping current positives before enabling zeros."""
    import numpy as np
    import tlc
    from tlc._core.objects.tables.from_table.edited_table import EditedTable
    from tlc_plugin_sdk.shared.url_utils import normalize_url
    from tlc_plugin_sdk.shared.weights import sample_weights

    data = ctx.params
    counts = data.get("counts")
    if not isinstance(counts, list) or not 1 <= len(counts) <= 50:
        msg = "counts must contain between 1 and 50 sample counts"
        raise ValueError(msg)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
        msg = "Every count must be a positive integer"
        raise ValueError(msg)
    if len(set(counts)) != len(counts):
        msg = "counts must be unique"
        raise ValueError(msg)
    seed = data.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        msg = "seed must be a nonnegative integer"
        raise ValueError(msg)
    prefix = str(data.get("name_prefix") or "binary_selection").strip()
    if not prefix or prefix in {".", ".."} or any(c in prefix for c in "/\\%"):
        msg = "name_prefix must be a single name, not a path"
        raise ValueError(msg)
    ctx.progress(percent=0, label="Creating weighted table revisions")
    table = tlc.Table.from_url(normalize_url(data["table_url"]))
    column, weights = sample_weights(table)
    if max(counts) > len(weights):
        msg = f"Requested {max(counts)} samples, but the source has only {len(weights)} physical rows"
        raise ValueError(msg)
    positive = np.flatnonzero(weights > 0)
    zero = np.flatnonzero(weights == 0)
    rng = np.random.default_rng(seed)
    # The same ordering makes every smaller selection a subset of the next.
    # Expansion preserves current positives before re-enabling excluded rows.
    order = np.concatenate((rng.permutation(positive), rng.permutation(zero)))
    override = None
    if column is None:
        if "weight" in table.rows_schema.values:
            msg = "The existing weight column is not declared as sample weights"
            raise ValueError(msg)
        column = "weight"
        override = {"values": {column: tlc.schemas.SampleWeightSchema(default_value=1.0)}}
    results = []
    ctx.output("revisions", results.copy())
    for index, count in enumerate(sorted(counts)):
        if ctx.cancelled:
            return
        desired = np.zeros(len(weights))
        desired[order[:count]] = 1.0
        edits: list[object] = []
        if override is not None:
            edits.extend([EditedTable.indices_to_run(list(range(len(weights)))), 1.0])
        for value in (0.0, 1.0):
            changed = np.flatnonzero((desired == value) & (weights != value)).tolist()
            if changed:
                edits.extend([EditedTable.indices_to_run(changed), value])
        revised = EditedTable(
            input_table_url=table,
            url=table.url.create_sibling(f"{prefix}_{count}").create_unique(),
            edits={column: {"runs_and_values": edits}} if edits else {},
            override_table_rows_schema=override,
            description=f"Binary selection: {count} rows at weight 1, others 0; seed {seed}; current positives first",
        )
        revised.ensure_fully_defined()
        revised.write_to_url()
        results.append({
            "table_url": str(revised.url),
            "source_table_url": str(table.url),
            "total_rows": len(weights),
            "positive_weight_rows": count,
            "zero_weight_rows": len(weights) - count,
            "previous_positive_weight_rows": len(positive),
            "newly_zeroed_rows": int(np.count_nonzero((weights > 0) & (desired == 0))),
            "newly_enabled_rows": int(np.count_nonzero((weights == 0) & (desired == 1))),
            "seed": seed,
        })
        # Preserve each published receipt if cancellation or a later write fails.
        ctx.output("revisions", results.copy())
        ctx.result(str(revised.url))
        ctx.progress(percent=100 * (index + 1) / len(counts), label=f"Created {count}-sample binary revision")
