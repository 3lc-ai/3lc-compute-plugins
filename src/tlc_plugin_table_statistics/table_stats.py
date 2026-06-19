# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Table statistics computation — progressive, async.

Computes per-column statistics (min, max, mean, histogram for numeric;
value counts for categorical) using the ``tlc`` SDK and PyArrow.

Statistics are computed in batches of BATCH_SIZE rows in a background thread.
The frontend polls and gets progressively refined results.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

logger = logging.getLogger(__name__)

BATCH_SIZE = 5000

# ---------------------------------------------------------------------------
# Progressive stats session cache
# ---------------------------------------------------------------------------

_sessions: dict[str, StatsSession] = {}
_lock = threading.Lock()


@dataclass
class _ColAccum:
    """Running accumulator for a single column."""

    name: str
    kind: str  # numeric, categorical, bool, image, bounding_boxes, segmentation, label, unknown
    dtype: str = ""
    count: int = 0
    null_count: int = 0
    # Numeric running stats
    _min: float | None = None
    _max: float | None = None
    _sum: float = 0.0
    _sum_sq: float = 0.0
    _valid_count: int = 0
    # Categorical / bool
    _value_counts: Counter[str] = field(default_factory=Counter)
    _unique: set[Any] = field(default_factory=set)
    # Value map (for mapped numeric -> categorical)
    _value_map: dict[float, str] | None = None

    def update(self, arr: pa.Array) -> None:
        """Incorporate a batch of values."""
        self.count += len(arr)
        self.null_count += arr.null_count

        if self.kind in ("image", "bounding_boxes", "segmentation", "unknown"):
            return  # nothing to accumulate

        if self.kind == "bool":
            for v in pc.value_counts(arr):
                self._value_counts[str(v["values"].as_py())] += v["counts"].as_py()
            return

        if self.kind == "numeric":
            valid = pc.drop_null(arr)
            n = len(valid)
            if n == 0:
                return
            batch_min = _safe_scalar(pc.min(valid))
            batch_max = _safe_scalar(pc.max(valid))
            if batch_min is not None:
                self._min = batch_min if self._min is None else min(self._min, batch_min)
            if batch_max is not None:
                self._max = batch_max if self._max is None else max(self._max, batch_max)
            self._sum += _safe_scalar(pc.sum(valid)) or 0.0
            # sum of squares for std
            try:
                import numpy as np

                np_arr = valid.to_numpy(zero_copy_only=False).astype(float)
                self._sum_sq += float(np.nansum(np_arr**2))
            except Exception:
                pass
            self._valid_count += n
            return

        if self.kind in ("categorical", "label"):
            for v in pc.value_counts(arr):
                raw = v["values"].as_py()
                if self._value_map and raw is not None:
                    label = self._value_map.get(float(raw), str(raw))
                else:
                    label = str(raw)
                self._value_counts[label] += v["counts"].as_py()
                self._unique.add(raw)
            return

    def to_dict(self, is_complete: bool) -> dict[str, Any]:
        """Snapshot current stats as a JSON-safe dict."""
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
            "null_count": self.null_count,
            "total_count": self.count,
        }
        if self.kind == "numeric":
            result["min"] = self._min
            result["max"] = self._max
            if self._valid_count > 0:
                mean = self._sum / self._valid_count
                result["mean"] = round(mean, 8)
                variance = (self._sum_sq / self._valid_count) - mean**2
                result["std"] = round(math.sqrt(max(0.0, variance)), 8)
            # Histogram is only available when complete (needs full data range)
            # — added by the session after final pass
        elif self.kind in ("categorical", "label"):
            sorted_vc = self._value_counts.most_common(20)
            result["value_counts"] = dict(sorted_vc)
            result["unique_count"] = len(self._unique) if self._unique else len(self._value_counts)
            if self._value_map:
                result["value_map"] = {str(k): v for k, v in self._value_map.items()}
        elif self.kind == "bool":
            result["value_counts"] = dict(self._value_counts)
        return result


@dataclass
class StatsSession:
    """Tracks a progressive stats computation for one table URL."""

    url: str
    total_rows: int = 0
    rows_processed: int = 0
    complete: bool = False
    error: str | None = None
    image_columns: list[str] = field(default_factory=list)
    sample_image_indices: list[int] = field(default_factory=list)
    _accums: dict[str, _ColAccum] = field(default_factory=dict)
    _col_order: list[str] = field(default_factory=list)
    _histograms: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        """Return current stats as a JSON-serializable dict."""
        columns = [self._accums[c].to_dict(self.complete) for c in self._col_order if c in self._accums]
        # Attach histograms if complete
        if self.complete:
            for col in columns:
                if col["name"] in self._histograms:
                    col["histogram"] = self._histograms[col["name"]]
        return {
            "row_count": self.total_rows,
            "rows_processed": self.rows_processed,
            "sampled_rows": self.rows_processed,
            "complete": self.complete,
            "column_count": len(columns),
            "columns": columns,
            "image_columns": self.image_columns,
            "sample_image_indices": self.sample_image_indices,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_or_start_stats(url: str) -> dict[str, Any]:
    """Get current stats for a URL, starting background computation if needed."""
    from tlc_plugin_sdk.shared.url_utils import normalize_url

    norm = normalize_url(url)

    with _lock:
        if norm in _sessions:
            return _sessions[norm].snapshot()

        session = StatsSession(url=norm)
        _sessions[norm] = session

    t = threading.Thread(target=_compute_progressive, args=(norm,), daemon=True)
    session._thread = t
    t.start()

    return session.snapshot()


def invalidate_stats(url: str) -> None:
    """Remove cached stats for a URL so they are recomputed on next request."""
    from tlc_plugin_sdk.shared.url_utils import normalize_url

    norm = normalize_url(url)
    with _lock:
        _sessions.pop(norm, None)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _compute_progressive(url: str) -> None:
    """Background thread: compute stats in BATCH_SIZE-row increments."""
    import tlc

    session = _sessions[url]
    try:
        table = tlc.Table.from_url(url)
        total = len(table)
        session.total_rows = total

        schema = table.rows_schema
        if not schema or not hasattr(schema, "values") or not schema.values:
            session.complete = True
            return

        # Modality detection
        from tlc_plugin_sdk.shared.modality import ModalityInfo, detect_modality_from_table

        try:
            modality_info = detect_modality_from_table(table)
        except Exception:
            modality_info = ModalityInfo()

        col_kinds: dict[str, str] = {}
        for c in modality_info.image_columns:
            col_kinds[c] = "image"
        for c in modality_info.detection_columns:
            col_kinds[c] = "bounding_boxes"
        for c in modality_info.segmentation_columns:
            col_kinds[c] = "segmentation"
        for c in modality_info.classification_columns:
            col_kinds[c] = "label"

        # Image info
        session.image_columns = modality_info.image_columns or []
        if total > 0 and session.image_columns:
            step = max(1, total // 12)
            session.sample_image_indices = list(range(0, total, step))[:12]

        # Initialize accumulators
        col_names = list(schema.values.keys())
        session._col_order = col_names

        # Load full column arrays once (SDK caches internally)
        arrays: dict[str, pa.Array] = {}
        for name in col_names:
            try:
                arr = table.get_column_as_pyarrow_array(name, combine_chunks=True)
                if isinstance(arr, pa.ChunkedArray):
                    arr = arr.combine_chunks()
                arrays[name] = arr

                # Determine kind
                dtype = arr.type
                kind = col_kinds.get(name)
                if not kind:
                    value_map = _get_value_map(table, name)
                    if pa.types.is_boolean(dtype):
                        kind = "bool"
                    elif (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)) and value_map:
                        kind = "categorical"
                    elif pa.types.is_integer(dtype) or pa.types.is_floating(dtype):
                        kind = "numeric"
                    elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
                        kind = "categorical"
                    else:
                        kind = "unknown"

                accum = _ColAccum(name=name, kind=kind, dtype=str(dtype))
                if kind == "categorical" and value_map:
                    accum._value_map = value_map
                session._accums[name] = accum

            except Exception as exc:
                session._accums[name] = _ColAccum(name=name, kind=col_kinds.get(name, "unknown"), dtype="unknown")
                logger.debug("Failed to load column %s: %s", name, exc)

        # Process in batches
        processed = 0
        while processed < total:
            end = min(processed + BATCH_SIZE, total)
            batch_len = end - processed

            for name, arr in arrays.items():
                if name not in session._accums:
                    continue
                batch = arr.slice(processed, batch_len)
                session._accums[name].update(batch)

            processed = end
            session.rows_processed = processed

        # Final pass: compute histograms for numeric columns
        for name, accum in session._accums.items():
            if accum.kind == "numeric" and name in arrays:
                try:
                    valid = pc.drop_null(arrays[name])
                    if len(valid) > 0:
                        session._histograms[name] = _compute_histogram(valid, bins=20)
                except Exception:
                    pass

        session.complete = True

    except Exception as exc:
        logger.exception("Progressive stats failed for %s", url)
        session.error = str(exc)
        session.complete = True


# ---------------------------------------------------------------------------
# Helpers (kept from original for backward compat)
# ---------------------------------------------------------------------------


def _get_value_map(table: Any, col_name: str) -> dict[float, str] | None:
    """Extract a display-name value map from a column's schema, if present."""
    from tlc_plugin_sdk.shared.labels import get_display_value_map

    return get_display_value_map(table, col_name)


def _safe_scalar(scalar: Any) -> float | None:
    """Convert a PyArrow scalar to a Python float, handling NaN/None."""
    if scalar is None:
        return None
    val = scalar.as_py() if hasattr(scalar, "as_py") else scalar
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return float(val)


def _compute_histogram(arr: pa.Array, bins: int = 20) -> dict[str, list[float]]:
    """Compute a simple histogram for a numeric array."""
    import numpy as np

    np_arr = arr.to_numpy(zero_copy_only=False)
    np_arr = np_arr[~np.isnan(np_arr)]

    if len(np_arr) == 0:
        return {"counts": [], "bins": []}

    counts, bin_edges = np.histogram(np_arr, bins=bins)
    return {
        "counts": counts.tolist(),
        "bins": bin_edges.tolist(),
    }
