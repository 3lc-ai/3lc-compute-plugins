# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Merge's page: two or more tables, as a list (PBI 19068).

The list's rules (the automatic name, the order, the column check and its message, and why Merge Tables waits) are
pure functions in ``ui.html``, between two marker comments. These tests run that block in node; the page's wiring and
the manifest's counts are pinned from their text.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

merger = pytest.importorskip("tlc_plugin_merger")

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

PACKAGE = Path(merger.__file__).parent
UI = (PACKAGE / "ui.html").read_text(encoding="utf-8")
RULES = UI[UI.index("// ── List rules") : UI.index("// ── End of the list rules ──")]
URLS = [f"s3://b/projects/p/datasets/d/tables/{name}" for name in "abcde"]
READY = {"merging": False, "count": 3, "check": "ok", "index": None, "named": True}


def _js(tmp_path: Path, expression: str) -> Any:
    """Evaluate ``expression`` in node after the page's list rules, with ``URLS`` as ``urls``, and return its value."""
    harness = tmp_path / "rules.js"
    harness.write_text(
        RULES + f"\nvar urls = {json.dumps(URLS)};\nprocess.stdout.write(JSON.stringify({expression}));\n",
        encoding="utf-8",
    )
    done = subprocess.run(["node", str(harness)], check=True, capture_output=True, text=True)
    return json.loads(done.stdout)


@needs_node
def test_the_page_script_parses(tmp_path: Path) -> None:
    script = tmp_path / "page.js"
    script.write_text(UI[UI.index("<script>") + len("<script>") : UI.index("</script>")], encoding="utf-8")
    subprocess.run(["node", "--check", str(script)], check=True)


@needs_node
def test_a_table_is_named_by_its_urls_last_segment(tmp_path: Path) -> None:
    windows = json.dumps("C:\\data\\t1")
    names = _js(tmp_path, f"[tableNameOf('s3://b/p/tables/initial/'), tableNameOf({windows})]")
    assert names == ["initial", "t1"]


@needs_node
def test_the_automatic_name_joins_up_to_three_tables(tmp_path: Path) -> None:
    names = _js(tmp_path, "[2, 3, 4, 5].map(function (n) { return autoName(urls.slice(0, n)); })")
    assert names == ["a_b_merged", "a_b_c_merged", "a_and_3_more_merged", "a_and_4_more_merged"]


@needs_node
def test_the_name_field_shows_the_name_the_merge_gets(tmp_path: Path) -> None:
    shown = _js(tmp_path, "[0, 1, 2, 5].map(function (n) { return namePlaceholder(urls.slice(0, n)); })")
    assert shown == ["(Named from input tables)", "(Named from input tables)", "a_b_merged", "a_and_4_more_merged"]
    # The run uses the same name: the placeholder follows every change to the list.
    assert "document.getElementById('merge-table-name').placeholder = namePlaceholder(tables);" in UI


@needs_node
def test_a_move_swaps_with_the_neighbour_and_stops_at_the_ends(tmp_path: Path) -> None:
    moves = _js(
        tmp_path, "[[1, -1], [1, 1], [0, -1], [2, 1]].map(function (m) { return moved(['a', 'b', 'c'], m[0], m[1]); })"
    )
    assert moves == [["b", "a", "c"], ["a", "c", "b"], ["a", "b", "c"], ["a", "b", "c"]]


@needs_node
def test_tables_with_the_same_columns_in_any_order_match(tmp_path: Path) -> None:
    answers = [{"columns": ["x", "y", "z"]}, {"columns": ["z", "x", "y"]}, {"columns": ["y", "z", "x"]}]
    check = _js(tmp_path, f"compareColumns({json.dumps(answers)})")
    assert check == {"state": "ok", "conflicts": [], "has": [[], [], []], "columns": 3}


@needs_node
def test_the_columns_some_table_lacks_are_tabled_for_every_table(tmp_path: Path) -> None:
    answers = [
        {"columns": ["x", "y", "z"]},
        {"columns": ["x", "y", "z"]},
        {"columns": ["x", "w", "y"]},
        {"columns": ["y", "x"]},
    ]
    check = _js(tmp_path, f"compareColumns({json.dumps(answers)})")
    assert check["state"] == "differ"
    # Only z and w are not in every table; x and y are left out of the grid.
    assert check["conflicts"] == ["z", "w"]
    assert check["has"] == [[True, False], [True, False], [False, True], [False, False]]


@needs_node
def test_when_the_first_table_is_the_odd_one_the_grid_shows_it(tmp_path: Path) -> None:
    answers = [{"columns": ["image"]}, {"columns": ["image", "brightness"]}, {"columns": ["brightness", "image"]}]
    check = _js(tmp_path, f"compareColumns({json.dumps(answers)})")
    assert check["conflicts"] == ["brightness"]
    assert check["has"] == [[False], [True], [True]]


@needs_node
def test_a_table_that_cannot_be_read_is_named_and_nothing_is_compared(tmp_path: Path) -> None:
    answers = [{"columns": ["x"]}, {"columns": ["y"]}, {"error": "no such table"}, {"error": "also unreadable"}]
    check = _js(tmp_path, f"compareColumns({json.dumps(answers)})")
    assert check == {
        "state": "unreadable",
        "index": 2,
        "error": "no such table",
    }


@needs_node
def test_the_messages_name_each_table_by_its_position(tmp_path: Path) -> None:
    messages = _js(
        tmp_path,
        "["
        "compatMessage(compareColumns([{columns: ['x', 'y']}, {columns: ['y', 'x']}, {columns: ['x', 'y']}]),"
        " urls.slice(0, 3)),"
        "compatMessage(compareColumns([{columns: ['x', 'y', 'z']}, {columns: ['x', 'y', 'z']},"
        " {columns: ['x', 'y', 'w', 'v']}, {columns: ['x', 'y']}]), urls.slice(0, 4)),"
        "compatMessage(compareColumns([{columns: ['x']}, {columns: ['x']}, {error: 'Not found'}]), urls.slice(0, 3)),"
        "compatMessage({state: 'failed', error: 'TypeError: Failed to fetch'}, urls),"
        "compatMessage({state: 'checking'}, urls),"
        "compatMessage({state: 'none'}, urls)"
        "]",
    )
    assert messages == [
        {"cls": "merger-compat-ok", "text": "✓ Schemas match — 2 columns in all 3 tables"},
        {
            "cls": "merger-compat-err",
            "text": "✗ Schemas don't match — a vertical join needs the same columns in every table. These differ:",
            "grid": {
                "columns": ["z", "w", "v"],
                "rows": [
                    {"label": "1 a", "has": [True, False, False]},
                    {"label": "2 b", "has": [True, False, False]},
                    {"label": "3 c", "has": [False, True, True]},
                    {"label": "4 d", "has": [False, False, False]},
                ],
            },
        },
        {"cls": "merger-compat-err", "text": "✗ Table 3 (c) could not be loaded: Not found"},
        {"cls": "merger-compat-err", "text": "✗ Could not check table compatibility: TypeError: Failed to fetch"},
        {"cls": "merger-compat-checking", "text": "Checking table compatibility…"},
        None,
    ]


@needs_node
def test_merge_tables_says_what_it_waits_for(tmp_path: Path) -> None:
    cases: list[tuple[dict[str, Any], str]] = [
        ({"merging": True}, "Wait for the merge to finish to enable Merge Tables."),
        ({"count": 0, "check": "none"}, "Add two or more tables to enable Merge Tables."),
        ({"count": 1, "check": "none"}, "Add another table to enable Merge Tables."),
        ({"check": "checking"}, "Wait while the tables' columns are compared to enable Merge Tables."),
        ({"check": "unreadable", "index": 2}, "Remove table 3, which cannot be read, to enable Merge Tables."),
        ({"check": "failed"}, "Choose tables whose columns can be compared to enable Merge Tables."),
        ({"check": "differ"}, "Choose tables with the same columns to enable Merge Tables."),
        ({"named": False}, "Choose a first table that sits in a project's dataset to enable Merge Tables."),
        ({}, ""),
    ]
    reasons = _js(
        tmp_path, json.dumps([{**READY, **case} for case, _ in cases]) + ".map(function (s) { return blocker(s); })"
    )
    assert reasons == [reason for _, reason in cases]


def test_the_page_runs_the_list_in_its_order() -> None:
    assert "merge-table-a" not in UI and "merge-table-b" not in UI
    assert "table_urls: tables.slice()," in UI
    assert "document.getElementById('merge-table-name').value.trim() || autoName(tables);" in UI


def test_browse_tables_marks_the_listed_tables_added() -> None:
    assert "addField.tagged = tables.map(function(url) { return { url: url, label: 'Added' }; });" in UI


def test_the_actions_menu_offers_merge_for_2_to_10_tables() -> None:
    manifest = tomllib.loads((PACKAGE / "plugin.toml").read_text(encoding="utf-8"))
    assert manifest["ui"]["input_types"] == ["table"] * 10
    assert manifest["ui"]["min_input_count"] == 2
    assert manifest["description"] == "Combine two or more tables into one by stacking their rows."
    assert manifest["ui"]["quick_action_description"] == "Stack the rows of two or more tables into one"
