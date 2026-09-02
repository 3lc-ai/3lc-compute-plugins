# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for COCO image_folder / file_name path reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

from tlc_plugin_importer import _reconcile_coco_image_folder


def _write_coco_json(path: Path, file_names: list[str]) -> None:
    """Write a minimal COCO annotations JSON with the given file_name values."""
    coco = {
        "images": [{"id": i, "file_name": fn, "width": 64, "height": 64} for i, fn in enumerate(file_names)],
        "annotations": [],
        "categories": [{"id": 1, "name": "obj"}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco))


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class TestReconcileCocoImageFolder:
    """_reconcile_coco_image_folder should strip overlapping path segments."""

    def test_bare_filenames_no_adjustment(self, tmp_path: Path) -> None:
        """file_name='001.jpg' + image_folder='.../train' → no change."""
        ann = tmp_path / "annotations" / "instances.json"
        _write_coco_json(ann, ["001.jpg", "002.jpg"])
        img_dir = tmp_path / "train"
        _touch(img_dir / "001.jpg")

        result = _reconcile_coco_image_folder(str(ann), str(img_dir))
        assert result == str(img_dir)

    def test_naive_join_works_no_adjustment(self, tmp_path: Path) -> None:
        """file_name='sub/001.jpg' but image_folder/sub/001.jpg exists → no change."""
        ann = tmp_path / "annotations" / "instances.json"
        _write_coco_json(ann, ["sub/001.jpg"])
        img_dir = tmp_path / "images"
        _touch(img_dir / "sub" / "001.jpg")

        result = _reconcile_coco_image_folder(str(ann), str(img_dir))
        assert result == str(img_dir)

    def test_full_overlap_stripped(self, tmp_path: Path) -> None:
        """file_name='train/images/001.jpg', image_folder='.../train/images' → strip 'train/images'."""
        ann = tmp_path / "annotations" / "instances.json"
        _write_coco_json(ann, ["train/images/001.jpg", "train/images/002.jpg"])
        _touch(tmp_path / "train" / "images" / "001.jpg")

        result = _reconcile_coco_image_folder(str(ann), str(tmp_path / "train" / "images"))
        assert result == str(tmp_path)

    def test_partial_overlap_stripped(self, tmp_path: Path) -> None:
        """file_name='images/train/001.jpg', image_folder='.../data/images' → strip 'images'."""
        ann = tmp_path / "annotations" / "instances.json"
        _write_coco_json(ann, ["images/train/001.jpg"])
        _touch(tmp_path / "data" / "images" / "train" / "001.jpg")

        result = _reconcile_coco_image_folder(str(ann), str(tmp_path / "data" / "images"))
        assert result == str(tmp_path / "data")

    def test_single_segment_overlap(self, tmp_path: Path) -> None:
        """file_name='train2017/001.jpg', image_folder='.../coco/train2017' → strip 'train2017'."""
        ann = tmp_path / "annotations" / "instances.json"
        _write_coco_json(ann, ["train2017/001.jpg"])
        _touch(tmp_path / "coco" / "train2017" / "001.jpg")

        result = _reconcile_coco_image_folder(str(ann), str(tmp_path / "coco" / "train2017"))
        assert result == str(tmp_path / "coco")

    def test_backslashes_in_file_name(self, tmp_path: Path) -> None:
        r"""file_name='train\images\001.jpg' (Windows) → normalized and overlap stripped."""
        ann = tmp_path / "annotations" / "instances.json"
        _write_coco_json(ann, ["train\\images\\001.jpg"])
        _touch(tmp_path / "train" / "images" / "001.jpg")

        result = _reconcile_coco_image_folder(str(ann), str(tmp_path / "train" / "images"))
        assert result == str(tmp_path)

    def test_no_overlap_no_adjustment(self, tmp_path: Path) -> None:
        """file_name='subset_a/001.jpg', image_folder='.../train' → no matching suffix, no change."""
        ann = tmp_path / "annotations" / "instances.json"
        _write_coco_json(ann, ["subset_a/001.jpg"])
        img_dir = tmp_path / "train"
        img_dir.mkdir(parents=True, exist_ok=True)

        result = _reconcile_coco_image_folder(str(ann), str(img_dir))
        assert result == str(img_dir)

    def test_empty_images_no_adjustment(self, tmp_path: Path) -> None:
        """Empty images array → no change."""
        ann = tmp_path / "instances.json"
        _write_coco_json(ann, [])
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        result = _reconcile_coco_image_folder(str(ann), str(img_dir))
        assert result == str(img_dir)

    def test_invalid_json_no_adjustment(self, tmp_path: Path) -> None:
        """Malformed JSON → no change (graceful fallback)."""
        ann = tmp_path / "bad.json"
        ann.write_text("{invalid json")
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        result = _reconcile_coco_image_folder(str(ann), str(img_dir))
        assert result == str(img_dir)

    def test_overlap_but_adjusted_file_missing(self, tmp_path: Path) -> None:
        """Overlap detected but the adjusted path doesn't resolve → no change."""
        ann = tmp_path / "annotations" / "instances.json"
        _write_coco_json(ann, ["train/images/001.jpg"])
        img_dir = tmp_path / "wrong" / "train" / "images"
        img_dir.mkdir(parents=True)
        # Don't create the file at the adjusted path

        result = _reconcile_coco_image_folder(str(ann), str(img_dir))
        assert result == str(img_dir)
