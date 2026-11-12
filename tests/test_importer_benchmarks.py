# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pinned benchmark sources cannot escape the selected project or overwrite edits."""

import io
import json
import stat
import sys
import zipfile
from types import SimpleNamespace

import pytest

from tlc_plugin_importer import benchmarks as b


def archive(extra=None, name="coco8"):
    splits = set(b.SPLITS[name].values())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for split, count in splits:
            for index in range(count - 2 if name == "coco128" else count):
                z.writestr(f"{name}/images/{split}/{index}.jpg", b"image")
                z.writestr(f"{name}/labels/{split}/{index}.txt", b"0 .5 .5 .2 .2")
            if name == "coco128":
                for stem in ("000000000250", "000000000508"):
                    z.writestr(f"{name}/images/{split}/{stem}.jpg", b"background")
                for stem in ("000000000656", "000000000659"):
                    z.writestr(f"{name}/labels/{split}/{stem}.txt", b"0 .5 .5 .2 .2")
        if extra:
            z.writestr(*extra)
    return buf.getvalue()


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/x",
        "https://localhost/x",
        "https://github.com.evil/x",
        "https://user:pass@github.com/x",
        "https://github.com:8080/x",
    ],
)
def test_download_redirects_stay_with_the_public_vendor(url):
    with pytest.raises(ValueError):
        b.checked_download_url(url)


@pytest.mark.parametrize("path", ["coco8/../../escape", "/absolute", "coco8/labels/../escape", "coco8/\\escape"])
def test_archive_paths_never_escape(path):
    with pytest.raises(ValueError, match="Unsafe"):
        b.archive_files(archive((path, b"bad")), "coco8")


def test_archive_rejects_links_and_missing_pairs():
    link = zipfile.ZipInfo("coco8/images/train/link.jpg")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match="links"):
        b.archive_files(archive((link, b"target")), "coco8")
    with pytest.raises(ValueError, match="4 images"):
        b.archive_files(archive(("coco8/images/train/extra.jpg", b"image")), "coco8")


def test_download_refuses_changed_archive_before_any_publication(monkeypatch):
    response = io.BytesIO(b"changed")
    monkeypatch.setattr(b.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=lambda *a, **k: response))
    with pytest.raises(ValueError, match="changed"):
        b.download("coco8")


def test_prepare_keeps_cloud_paths_reuses_exact_files_and_refuses_changed_files(monkeypatch):
    objects, outputs, writes = {}, {}, []

    class Url:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return self.path in objects

        def read_bytes_bounded(self, limit):
            assert len(objects[self.path]) <= limit
            return objects[self.path]

        def write_bytes(self, data, *, if_exists):
            assert if_exists == "raise" and self.path not in objects
            objects[self.path] = data
            writes.append(self.path)

    monkeypatch.setitem(sys.modules, "tlc", SimpleNamespace(Url=Url))
    monkeypatch.setattr(b, "download", lambda name: archive())
    ctx = SimpleNamespace(
        params={"benchmark": "coco8", "project_name": "New project"},
        project_root_url="s3://bucket/projects",
        cancelled=False,
        progress=lambda **kw: None,
        output=lambda key, value: outputs.update({key: value}),
    )
    b.prepare(ctx)
    assert writes[-1].endswith("/dataset.yaml")
    config = json.loads(objects[outputs["dataset_yaml"]])
    assert config["path"].startswith("s3://bucket/projects/New project/source/")
    assert outputs["train_images"] == outputs["validation_images"] == 4
    assert len(config["names"]) == 80
    count = len(writes)
    b.prepare(ctx)
    assert len(writes) == count
    objects[writes[0]] = b"person edited this"
    with pytest.raises(ValueError, match="changed files"):
        b.prepare(ctx)
    assert objects[writes[0]] == b"person edited this"
    for invalid in ("../outside", "x/y", "..", "%2e%2e"):
        ctx.params["project_name"] = invalid
        with pytest.raises(ValueError, match="project name"):
            b.prepare(ctx)


@pytest.mark.parametrize("name,count", [("coco8", 8), ("coco8-seg", 8), ("coco128", 128)])
def test_all_benchmark_layouts_have_exact_unique_images(name, count):
    files = b.archive_files(archive(name=name), name)
    assert len([p for p in files if p.startswith("images/")]) == count
    assert len([p for p in files if p.startswith("labels/")]) == count


def test_coco128_shared_split_has_no_duplicate_uploads(monkeypatch):
    objects, outputs = {}, {}

    class Url:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return False

        def write_bytes(self, data, *, if_exists):
            assert if_exists == "raise"
            assert self.path not in objects
            objects[self.path] = data

    monkeypatch.setitem(sys.modules, "tlc", SimpleNamespace(Url=Url))
    monkeypatch.setattr(b, "download", lambda name: archive(name=name))
    ctx = SimpleNamespace(
        params={"benchmark": "coco128", "project_name": "COCO128"},
        project_root_url="s3://bucket/projects",
        cancelled=False,
        progress=lambda **kw: None,
        output=lambda key, value: outputs.update({key: value}),
    )
    b.prepare(ctx)
    config = json.loads(objects[outputs["dataset_yaml"]])
    assert config["train"] == config["val"] == "images/train2017"
    assert outputs["task"] == "detection"
    assert outputs["train_images"] == outputs["validation_images"] == outputs["unique_images"] == 128
    assert outputs["shared_train_validation"] is True
    assert len([p for p in objects if "/images/" in p]) == 128
    assert len(config["names"]) == 80
    with pytest.raises(ValueError, match="128 images"):
        b.archive_files(archive(("coco128/images/train2017/extra.jpg", b"image"), name="coco128"), "coco128")
