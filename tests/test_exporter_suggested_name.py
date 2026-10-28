# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The name an export suggests for its output, from the table's project, dataset and name (PBI 19097).

``/columns`` sends it as ``suggested_stem``; the page adds the format's extension, or ``-yolo`` for a YOLO
folder. It is built here, without ``tlc``, so the rule is testable where CI has no credentials.
"""

from __future__ import annotations

import pytest

from tlc_plugin_exporter import suggested_stem


@pytest.mark.parametrize(
    ("project", "dataset", "table", "expected"),
    [
        ("TestCoco", "TestCoco", "train", "TestCoco-train"),  # a dataset named as its project: once
        ("boulder_detection", "train", "initial", "boulder_detection-train"),  # "initial" says nothing
        ("cats", "cats", "initial", "cats"),  # both rules at once
        ("", "", "initial", "initial"),  # the only part known is kept, whatever it is
        ("My Project", "val set", "v2 (final)", "My_Project-val_set-v2_final_"),  # one file-safe name
        ("", "", "", ""),
    ],
)
def test_suggested_stem(project: str, dataset: str, table: str, expected: str) -> None:
    assert suggested_stem(project, dataset, table) == expected
