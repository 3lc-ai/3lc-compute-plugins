# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Format choices retain native keyboard activation and an announced selection state."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("plugin,grid_id", [("importer", "format-grid"), ("exporter", "export-format-grid")])
def test_formats_are_native_buttons_with_a_selection_state(tmp_path: Path, plugin: str, grid_id: str) -> None:
    ui = (Path(__file__).parents[1] / "src" / f"tlc_plugin_{plugin}" / "ui.html").read_text()
    match = re.search(r"  function renderFormatGrid\(\) \{.*?\n  \}", ui, re.S)
    assert match is not None
    script = tmp_path / "format-choices.cjs"
    script.write_text(
        "const assert = require('node:assert/strict');\n"
        "const formats = [{name:'csv',display_name:'CSV',description:'Rows and columns'},"
        "{name:'coco',display_name:'COCO',description:'Annotations'}];\n"
        "let selectedFormat = 'csv'; let selected = '';\n"
        "const grid = {children:[], appendChild(c) {this.children.push(c)}};\n"
        "const document = {getElementById(id) {assert.equal(id, '" + grid_id + "'); return grid},"
        "createElement(tag) { return {tag, dataset:{}, attrs:{}, setAttribute(k,v) {this.attrs[k]=v}}}};\n"
        "const _esc = s => s; const selectFormat = name => {selected=name};\n"
        + match.group()
        + "\nrenderFormatGrid();\n"
        "assert.equal(grid.children.length,2);\n"
        "for (const c of grid.children) {assert.equal(c.tag,'button'); assert.equal(c.type,'button')}\n"
        "assert.equal(grid.children[0].attrs['aria-pressed'],'true');\n"
        "assert.equal(grid.children[1].attrs['aria-pressed'],'false');\n"
        "grid.children[1].onclick(); assert.equal(selected,'coco');\n"
        "assert.match(grid.children[1].innerHTML,/COCO/);\n"
    )
    subprocess.run(["node", str(script)], check=True, capture_output=True, text=True)


def test_metric_buttons_announce_individual_and_bulk_selection(tmp_path: Path) -> None:
    ui = (Path(__file__).parents[1] / "src/tlc_plugin_image_metrics/ui.html").read_text()
    functions = []
    for name in ("_renderMetricGrid", "_toggleMetric", "_refreshCards"):
        match = re.search(r"  function " + name + r"\([^)]*\) \{.*?\n  \}", ui, re.S)
        assert match is not None
        functions.append(match.group())
    script = tmp_path / "metric-choices.cjs"
    script.write_text(
        "const assert = require('node:assert/strict');\n"
        "const elements=[];\n"
        "function element(tag) {const e={tag,children:[],dataset:{},attrs:{},"
        "classList:{add(){},remove(){}},appendChild(c){this.children.push(c)},"
        "setAttribute(k,v){this.attrs[k]=v}}; elements.push(e);return e;}\n"
        "const container=element('div');\n"
        "const document={getElementById(){return container},createElement:element,"
        "querySelectorAll(){return elements.filter(e=>e.dataset.metricId)}};\n"
        "const selectedMetrics={sharpness:true}; const _esc=s=>s;\n"
        "const imPresets={changed(){}}; function _updateState(){}\n"
        + "\n".join(functions)
        + "\n_renderMetricGrid([{id:'sharpness',name:'Sharpness',category:'Quality',description:'Focus'}]);\n"
        "const card=document.querySelectorAll()[0];\n"
        "assert.equal(card.tag,'button'); assert.equal(card.type,'button');\n"
        "assert.equal(card.attrs['aria-pressed'],'true');\n"
        "card.onclick(); assert.equal(card.attrs['aria-pressed'],'false');\n"
        "assert.equal(selectedMetrics.sharpness,undefined);\n"
        "selectedMetrics.sharpness=true; _refreshCards();\n"
        "assert.equal(card.attrs['aria-pressed'],'true');\n"
    )
    subprocess.run(["node", str(script)], check=True, capture_output=True, text=True)
