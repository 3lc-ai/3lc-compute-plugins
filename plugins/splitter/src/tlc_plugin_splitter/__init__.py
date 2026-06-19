# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Split plugin — split a 3LC table into train/val/test sets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tlc_plugin_sdk import ComputePlugin, JobContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution logic
# ---------------------------------------------------------------------------


def _execute_split(form_data: dict[str, Any]) -> dict[str, Any]:
    """Split a table into train/val/test sets using delete_rows to create EditedTable revisions."""
    import numpy as np
    import tlc

    from tlc_plugin_sdk.shared.url_utils import normalize_url

    table_url = form_data["table_url"].strip()
    strategy = form_data.get("strategy", "random").strip()
    train_pct = int(form_data.get("train_pct", 70))
    val_pct = int(form_data.get("val_pct", 20))
    test_pct = int(form_data.get("test_pct", 0) or 0)
    seed = int(form_data.get("seed", 42) or 42)

    # Validate percentages
    total_pct = train_pct + val_pct + test_pct
    if total_pct != 100:
        return {"success": False, "message": f"Train + Val + Test must equal 100%, got {total_pct}%."}
    if train_pct < 1 or val_pct < 1:
        return {"success": False, "message": "Train and Val must each be at least 1%."}

    # Load source table
    table = tlc.Table.from_url(normalize_url(table_url))
    n = len(table)
    if n < 3:
        return {"success": False, "message": f"Table has only {n} rows — too few to split."}

    rng = np.random.default_rng(seed)
    indices = np.arange(n)

    if strategy == "stratified":
        label_col = _find_label_column(table)
        if label_col is not None:
            # Read the raw label column (row view). The sample view
            # (table[i][label_col]) would decode every bulk-data column in the
            # row — including images — making the split needlessly slow and
            # failing entirely when image data is unavailable (e.g. an
            # unregistered alias).
            labels_arr = table.get_column_as_pyarrow_array(label_col, combine_chunks=True)
            labels = labels_arr.to_numpy(zero_copy_only=False)
            indices = _stratified_shuffle(indices, labels, rng)
        else:
            rng.shuffle(indices)
    else:
        rng.shuffle(indices)

    # Compute split sizes
    n_train = max(1, int(n * train_pct / 100))
    n_test = max(0, int(n * test_pct / 100)) if test_pct > 0 else 0
    n_val = n - n_train - n_test
    if n_val < 1:
        n_val = 1
        n_train = n - n_val - n_test

    train_indices = sorted(indices[:n_train].tolist())
    val_indices = sorted(indices[n_train : n_train + n_val].tolist())
    test_indices = sorted(indices[n_train + n_val :].tolist()) if n_test > 0 else []

    # Build splits — each is an EditedTable that deletes the rows NOT in the split
    all_indices = set(range(n))
    results: list[dict[str, Any]] = []

    splits = [("train", train_indices), ("val", val_indices)]
    if test_indices:
        splits.append(("test", test_indices))

    for split_name, keep in splits:
        remove = sorted(all_indices - set(keep))
        split_table = table.delete_rows(
            remove, table_name=split_name, description=f"{split_name} split ({len(keep)} rows)"
        )
        results.append({"name": split_name, "url": str(split_table.url), "rows": len(keep)})
        logger.info("Created %s split: %d rows → %s", split_name, len(keep), split_table.url)

    summary_parts = [f"{r['name']}: {r['rows']} rows" for r in results]
    return {
        "success": True,
        "message": f"Split {n} rows into {', '.join(summary_parts)}.",
        "splits": results,
        "details": {
            "source_rows": n,
            "strategy": strategy,
            "seed": seed,
            **{r["name"]: r["rows"] for r in results},
        },
    }


def _find_label_column(table: Any) -> str | None:
    """Find a categorical label column in the table schema, if any."""
    from tlc_plugin_sdk.shared.labels import find_label_column

    return find_label_column(table)


def _stratified_shuffle(indices: Any, labels: Any, rng: Any) -> Any:
    """Shuffle indices while grouping by class for proportional splits."""
    import numpy as np

    unique_labels = np.unique(labels)
    shuffled: list[Any] = []
    per_class: dict[int, list[int]] = {}
    for label in unique_labels:
        mask = labels == label
        class_indices = indices[mask].copy()
        rng.shuffle(class_indices)
        per_class[int(label)] = class_indices.tolist()

    # Interleave classes proportionally
    max_len = max(len(v) for v in per_class.values())
    for i in range(max_len):
        for label in unique_labels:
            cls_list = per_class[int(label)]
            if i < len(cls_list):
                shuffled.append(cls_list[i])

    return np.array(shuffled)


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class SplitterPlugin(ComputePlugin):
    """Sidebar plugin for splitting 3LC tables into train/val/test sets."""

    _ui_cache: str | None = None

    def get_ui_fragment(self) -> str:
        """Return the self-contained split wizard HTML+JS+CSS fragment."""
        if self._ui_cache is None:
            from tlc_plugin_sdk.shared.job_tracker import job_tracker_script

            ui_path = Path(__file__).resolve().parent / "ui.html"
            # Inject window.PluginJobs so the UI can drive the split job over the
            # generic job_update channel and receive the structured split_result event.
            self._ui_cache = "<script>\n" + job_tracker_script() + "\n</script>\n" + ui_path.read_text(encoding="utf-8")
        return self._ui_cache

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute split via compute endpoint (not used — prefer POST /run)."""
        return {"error": "Use POST /api/plugins/splitter/run instead."}

    def run_job(self, ctx: JobContext) -> None:
        """Run a train/val/test split as a host job.

        Validates the request (moved here from the old ``/execute`` route), runs
        the split, reports per-split row counts as generic metrics, and emits the
        structured ``split_result`` event the plugin UI renders (the flat generic
        schema can't carry the per-split breakdown). Validation and execution
        failures are raised so the job is marked failed and the message reaches
        the UI via the generic channel.

        Args:
            ctx: Host-provided job context. ``ctx.params`` carries ``table_url``,
                ``strategy``, ``train_pct``, ``val_pct``, ``test_pct``, ``seed``.

        Raises:
            ValueError: When the request is invalid.
            RuntimeError: When the split itself fails.

        """
        data = ctx.params
        if not (data.get("table_url") or "").strip():
            msg = "Table URL is required."
            raise ValueError(msg)
        train_pct = int(data.get("train_pct", 70))
        val_pct = int(data.get("val_pct", 20))
        test_pct = int(data.get("test_pct", 0) or 0)
        total = train_pct + val_pct + test_pct
        if total != 100:
            msg = f"Train + Val + Test must equal 100%, got {total}%."
            raise ValueError(msg)
        if train_pct < 1 or val_pct < 1:
            msg = "Train and Val must each be at least 1%."
            raise ValueError(msg)

        ctx.progress(percent=10, label="Splitting table")
        result = _execute_split(data)
        if not result.get("success"):
            raise RuntimeError(result.get("message") or "Split failed")

        for split in result.get("splits", []):
            ctx.metric(split["name"], split["rows"])
        ctx.progress(percent=100, label=result.get("message", "Split complete"))
        ctx.emit("split_result", result)
