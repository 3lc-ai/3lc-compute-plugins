# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Alias failures must not silently create tables with absolute media paths.

The real-writer cases are opt-in, since importing 3LC activates its license:
TLC_RUN_IMPORTER_INTEGRATION_TESTS=1 pytest tests/test_importer_alias_persistence.py
They write only temporary local projects; S3 image reads are simulated.
"""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path
from uuid import uuid4

import pytest
from tlc_plugin_sdk import JobContext, JobFailed
from tlc_plugin_sdk.shared import aliases

import tlc_plugin_importer as imp


def test_unlabeled_alias_failure_stops_before_the_executor(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(aliases, "register_source_alias", lambda *a, **kw: {"error": "Alias registration failed"})
    monkeypatch.setitem(imp._EXECUTORS, "unlabeled", lambda form: calls.append(form))
    params = {
        "folder_path": str(tmp_path),
        "modality": "detection",
        "project_name": "Images",
        "dataset_name": "images",
        "alias_token": "CARS_IMAGES",
    }
    ctx = JobContext("alias-test", params, tmp_path, sink=lambda e: None, cancel_event=threading.Event())
    with pytest.raises(JobFailed, match=r"Alias registration failed.*before creating a table"):
        imp._run_format_import(ctx, "unlabeled")
    assert not calls


def test_csv_alias_failure_stops_before_the_writer(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "tlc",
        SimpleNamespace(
            schemas=SimpleNamespace(ImageSchema=lambda **kw: object()),
            TableWriter=lambda **kw: calls.append(kw),
        ),
    )
    monkeypatch.setattr(aliases, "register_source_alias", lambda *a, **kw: {"error": "Alias registration failed"})
    monkeypatch.setattr(imp, "_read_spreadsheet", lambda *a: (["image"], [["/data/images/a.jpg"]]))
    with pytest.raises(JobFailed, match=r"Alias registration failed.*before creating a table"):
        imp._execute_csv_new(
            b"", "images.csv", [{"name": "image", "index": 0, "type": "image_url"}], "Images", "images", "initial", ""
        )
    assert not calls


_real_writer = pytest.mark.skipif(
    os.environ.get("TLC_RUN_IMPORTER_INTEGRATION_TESTS") != "1",
    reason="Opt-in integration tests require a configured 3LC license",
)


@_real_writer
@pytest.mark.parametrize("modality", ["classification", "detection", "instance_segmentation"])
@pytest.mark.parametrize("location", ["local", "s3", "project"])
def test_unlabeled_tablewriter_persists_project_alias(tmp_path: Path, monkeypatch, modality, location):
    import pyarrow.parquet as pq
    import tlc
    import yaml
    from PIL import Image

    token = "IMPORT_TEST_" + uuid4().hex.upper()
    project_root = tmp_path / "projects"
    alias_root = str(project_root / "Images" / "data" if location == "project" else tmp_path / "data")
    picture = io.BytesIO()
    Image.new("RGB", (7, 3)).save(picture, format="PNG")
    if location == "s3":
        alias_root = f"s3://import-test-{uuid4().hex}/data"
        source = alias_root + "/val"
        image_path = source + "/nested/one.png"
        monkeypatch.setattr(imp.pu, "is_folder", lambda path: path == source)
        monkeypatch.setattr(imp.pu, "iter_files", lambda *a, **kw: [image_path])
        monkeypatch.setattr(imp.pu, "read_bytes", lambda path: picture.getvalue())
    else:
        source = alias_root + "/val"
        image = Path(source) / "nested" / "one.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(picture.getvalue())

    params = {
        "folder_path": source,
        "modality": modality,
        "project_name": "Images",
        "dataset_name": "images",
        "project_root_url": str(project_root),
        "alias_token": token,
        "alias_folder": alias_root,
    }
    ctx = JobContext("alias-test", params, tmp_path / "job", sink=lambda e: None, cancel_event=threading.Event())
    imp._run_format_import(ctx, "unlabeled")

    # Inspect the persisted column, not a mocked writer or a resolved sample view.
    parquet = next(project_root.rglob("*.parquet"))
    rows = pq.read_table(parquet).to_pylist()
    assert len(rows) == 1
    assert rows[0]["image"] == f"<{token}>/val/nested/one.png"
    assert rows[0]["weight"] == 0.0
    config = yaml.safe_load(next(project_root.rglob("default_aliases.3lc.yaml")).read_text())
    expected_root = "../../../../data" if location == "project" else alias_root
    assert config["aliases"][token] == expected_root
    # Source values are left to the library's own URL conversion on table write.
    assert tlc.Table.from_url(str(parquet.parent)).row_count == 1


@_real_writer
@pytest.mark.parametrize(
    "task,label", [("detect", "0 0.5 0.5 0.5 0.5"), ("segment", "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9")]
)
def test_yolo_inside_project_keeps_portable_alias_after_job(tmp_path, task, label):
    import pyarrow.parquet as pq
    import tlc
    import yaml
    from PIL import Image

    root = tmp_path / "projects"
    source = root / "Images" / "source"
    (source / "images/train").mkdir(parents=True)
    (source / "labels/train").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(source / "images/train/one.png")
    (source / "labels/train/one.txt").write_text(label + "\n")
    (source / "data.yaml").write_text("train: images/train\nnames: [cat]\n")
    token = "YOLO_IMPORT_" + uuid4().hex.upper()
    params = {
        "dataset_yaml": str(source / "data.yaml"),
        "split": "train",
        "task": task,
        "project_name": "Images",
        "dataset_name": "train",
        "project_root_url": str(root),
        "alias_token": token,
        "alias_folder": str(source),
    }
    ctx = JobContext("alias-test", params, tmp_path / "job", sink=lambda e: None, cancel_event=threading.Event())
    imp._run_format_import(ctx, "yolo")
    parquet = next(root.rglob("row_cache.parquet"))
    assert pq.read_table(parquet).to_pylist()[0]["image"] == f"<{token}>/images/train/one.png"
    config = yaml.safe_load(next(root.rglob("default_aliases.3lc.yaml")).read_text())
    assert config["aliases"][token] == "../../../../source"
    # Resolve the stored token using the durable project mapping, after job cleanup.
    table = tlc.Table.from_url(str(parquet.parent))
    assert str(tlc.Url(table.table_rows[0]["image"]).to_absolute(table.url)) == str(source / "images/train/one.png")


@_real_writer
def test_real_alias_conflict_does_not_create_a_table(tmp_path: Path):
    import tlc
    from PIL import Image

    token = "IMPORT_TEST_" + uuid4().hex.upper()
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (7, 3)).save(source / "one.png")
    project_root = tmp_path / "projects"
    tlc.helpers.ProjectHelper.register_project_url_alias(
        token, str(tmp_path / "other-images"), project_name="Other", root_url=str(project_root)
    )
    params = {
        "folder_path": str(source),
        "modality": "detection",
        "project_name": "Images",
        "dataset_name": "images",
        "project_root_url": str(project_root),
        "alias_token": token,
        "alias_folder": str(source),
    }
    ctx = JobContext("alias-test", params, tmp_path / "job", sink=lambda e: None, cancel_event=threading.Event())
    with pytest.raises(JobFailed, match="before creating a table"):
        imp._run_format_import(ctx, "unlabeled")
    assert not list(project_root.rglob("*.parquet"))
    assert tlc.url.get_registered_url_aliases()[f"<{token}>"] == str(tmp_path / "other-images")
