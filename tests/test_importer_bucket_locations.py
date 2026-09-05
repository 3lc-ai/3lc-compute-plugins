# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Importer: every location field takes a bucket URL as readily as a local path (found live: the
unlabeled import rejected ``s3://…/images/val`` as "not absolute")."""

from __future__ import annotations

from pathlib import Path

import pytest
from tlc_plugin_sdk.shared import url_utils as pu

import tlc_plugin_importer as imp


def test_path_fields_accept_urls_and_still_require_absolute_local_paths() -> None:
    out = imp._normalize_path_fields({
        "folder_path": " s3://b/data/fire/images/val/ ",
        "alias_folder": "~/x",
        "csv_path": "",
    })
    assert out["folder_path"] == "s3://b/data/fire/images/val" and out["alias_folder"].startswith("/")
    with pytest.raises(ValueError, match="absolute"):
        imp._normalize_path_fields({"folder_path": "relative/images"})


YAML = "path: s3://b/datasets/coco128\ntrain: images/train\nval: images/val\nnames: [person, bike]\n"


def test_yolo_yaml_on_a_bucket_resolves_to_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pu, "read_text", lambda v, **kw: YAML)
    yaml_url = "s3://b/datasets/coco128/data.yaml"
    images, categories = imp._parse_yolo_yaml_for_split(yaml_url, "val")
    assert images == "s3://b/datasets/coco128/images/val" and categories == {0: "person", 1: "bike"}
    assert imp._parse_yolo_dataset_root(yaml_url) == "s3://b/datasets/coco128"
    assert imp._parse_yolo_image_root(yaml_url, "train") == "s3://b/datasets/coco128/images/train"
    splits = imp._parse_yolo_splits(yaml_url)
    assert splits["splits"] == ["train", "val"] and splits["name"] == "coco128"


def test_yolo_yaml_without_path_is_relative_to_where_it_sits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pu, "read_text", lambda v, **kw: "train: images/train\nval: ../shared/val\nnc: 2\n")
    images, categories = imp._parse_yolo_yaml_for_split("s3://b/ds/data.yaml", "val")
    assert images == "s3://b/ds/../shared/val" and categories is None  # a bucket path is not normalised: it is a key
    assert imp._parse_yolo_dataset_root("s3://b/ds/data.yaml") == "s3://b/ds"


def test_yolo_local_yaml_still_resolves_on_disk(tmp_path: Path) -> None:
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "data.yaml").write_text("train: images/train\nval: images/val\nnames:\n  0: cat\n")
    images, categories = imp._parse_yolo_yaml_for_split(str(tmp_path / "data.yaml"), "val")
    assert images == str((tmp_path / "images" / "val").resolve()) and categories == {0: "cat"}
    assert imp._parse_yolo_image_root(str(tmp_path / "data.yaml"), "val") == str((tmp_path / "images" / "val").resolve())


def test_coco_images_folder_is_inferred_next_to_bucket_annotations(monkeypatch: pytest.MonkeyPatch) -> None:
    folders = {"s3://b/coco/images/val2017"}
    monkeypatch.setattr(pu, "is_folder", lambda v: v in folders)
    assert (
        imp._infer_coco_images_folder("s3://b/coco/annotations/instances_val2017.json") == "s3://b/coco/images/val2017"
    )
    assert imp._infer_coco_images_folder("s3://b/coco/annotations/captions_train2017.json") == ""


def test_coco_folder_scan_on_a_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pu, "is_file", lambda v: False)
    monkeypatch.setattr(pu, "is_folder", lambda v: v in {"s3://b/coco/annotations", "s3://b/coco/images/val2017"})
    monkeypatch.setattr(
        pu,
        "iter_files",
        lambda v, **kw: [
            "s3://b/coco/annotations/instances_val2017.json",
            "s3://b/coco/annotations/captions_val2017.json",
        ],
    )
    out = imp._parse_coco_folder("s3://b/coco/annotations/")
    assert out["name"] == "coco" and out["root"] == "s3://b/coco" and out["default_type"] == "instances"
    assert out["splits"] == [
        {
            "split": "val",
            "file": "s3://b/coco/annotations/instances_val2017.json",
            "images_hint": "s3://b/coco/images/val2017",
        }
    ]


def test_image_dimensions_read_a_url_through_tlc(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (7, 3)).save(buf, format="PNG")
    monkeypatch.setattr(pu, "read_bytes", lambda v: buf.getvalue())
    assert imp._get_image_dimensions("s3://b/x.png") == (7, 3)
