# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproducible first-dataset split, keeping a capture/video in one partition."""

from __future__ import annotations

import hashlib
import json


def run_grouped_split(ctx):
    import tlc
    from tlc._core.objects.tables.from_table.edited_table import EditedTable
    from tlc.constants import SHOULD_DELETE

    data = ctx.params
    source = tlc.Table.from_url(data["table_url"])
    group_column = data.get("group_column", "")
    percentages = [data.get("train_pct", 80), data.get("val_pct", 20), data.get("test_pct", 0)]
    if any(type(n) is not int or n < 0 for n in percentages) or sum(percentages) != 100 or min(percentages[:2]) < 1:
        msg = "Train and validation must be positive and all percentages must total 100."
        raise ValueError(msg)
    if group_column and group_column not in source.rows_schema.values:
        msg = "The selected grouping column is missing."
        raise ValueError(msg)
    groups = {}
    for index, row in enumerate(source.table_rows):
        value = row[group_column] if group_column else index
        if value is None or isinstance(value, (dict, list)) or str(value).strip() == "":
            msg = "Every sample must have a group identifier."
            raise ValueError(msg)
        groups.setdefault(json.dumps(value, sort_keys=True), []).append(index)
    parts = [(name, percentage) for name, percentage in zip(("train", "val", "test"), percentages) if percentage]
    if len(groups) < len(parts):
        msg = "There are too few independent groups for these splits. Add data or choose fewer splits."
        raise ValueError(msg)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{data.get('seed', 42)}:{key}".encode()).digest())
    counts = [max(1, int(len(groups) * percentage / 100)) for _, percentage in parts]
    while sum(counts) > len(groups):
        counts[max(range(len(counts)), key=lambda i: counts[i])] -= 1
    counts[0] += len(groups) - sum(counts)
    receipt = hashlib.sha256(
        json.dumps([str(source.url), group_column, percentages, data.get("seed", 42)], sort_keys=True).encode()
    ).hexdigest()
    outputs = {}
    cursor = 0
    for (name, _), count in zip(parts, counts):
        keep = {index for group in ordered[cursor : cursor + count] for index in groups[group]}
        cursor += count
        url = source.url.create_sibling(f"{name}-{receipt[:24]}")
        if ctx.cancelled:
            return
        description = f"Grouped split {receipt} {name}"
        if url.exists():
            table = tlc.Table.from_url(url)
            if table.description != description:
                msg = "A different table uses this split name. Nothing was overwritten."
                raise ValueError(msg)
        else:
            removed = sorted(set(range(len(source))) - keep)
            table = EditedTable(
                url=url,
                input_table_url=source,
                input_tables=[source.url],
                edits={SHOULD_DELETE: {"runs_and_values": [EditedTable.indices_to_run(removed), True]}},
                description=description,
            )
            table.ensure_fully_defined()
            table.write_to_url()
        outputs[name] = {"table_url": str(table.url), "rows": len(table), "groups": count}
        ctx.result(str(table.url))
    ctx.output("splits", outputs)
    ctx.progress(percent=100, label="Train and evaluation sets ready")
