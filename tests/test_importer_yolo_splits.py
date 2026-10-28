# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Import: the splits a YOLO or COCO import uses are the ones read from the given path (Bug 19083).

A YAML that could not be read left the page's hidden Split select at ``train`` and Import enabled, so one split was
imported as the dataset; a changed path kept the previous path's splits until the new one was read. The page now
waits for the path's answer and blocks Import on a failure. It tells a YAML that cannot be read (``/parse-yolo``
answers ``error``) from one that lists no split (``splits`` empty), which these tests pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

importer = pytest.importorskip("tlc_plugin_importer")

UI = (Path(importer.__file__).parent / "ui.html").read_text(encoding="utf-8")


def _yaml(tmp_path: Path, text: str, name: str = "coco8.yaml") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_splits_are_the_ones_the_yaml_lists(tmp_path: Path) -> None:
    result = importer._parse_yolo_splits(_yaml(tmp_path, "train: images/train\nval: images/val\nnames: [a, b]\n"))
    assert result["splits"] == ["train", "val"]
    assert result["name"] == "coco8"


def test_a_yaml_without_a_split_answers_no_splits(tmp_path: Path) -> None:
    result = importer._parse_yolo_splits(_yaml(tmp_path, "names: [a, b]\n"))
    assert result["splits"] == []


def test_a_missing_yaml_raises(tmp_path: Path) -> None:
    # The route answers {"error": …} for it, which the page shows under the path.
    with pytest.raises(FileNotFoundError):
        importer._parse_yolo_splits(str(tmp_path / "missing.yaml"))


def test_page_does_not_import_until_the_path_is_read() -> None:
    assert "if (data.error) return;" not in UI
    assert "'Wait while the YAML is read' + label" in UI
    assert "'Choose a YAML file that can be read' + label" in UI
    assert "'Choose a YAML that lists a train, val or test split' + label" in UI
    assert "'Wait while the annotations are read' + label" in UI


def test_page_drops_an_answer_for_an_earlier_path() -> None:
    assert "if (seq !== _yoloDetectSeq) return;" in UI
    assert "if (seq !== _cocoDetectSeq) return;" in UI
