# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prepare small, checksum-pinned public fixtures in the selected project.

No arbitrary download URLs, archive extraction or executable dataset scripts.
The ordinary importer creates the tables afterwards, using the returned YAML.
"""

from __future__ import annotations

import hashlib
import io
import json
import stat
import urllib.request
import zipfile
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from tlc_plugin_sdk.job_context import JobContext

BENCHMARKS = {
    "coco8": "54c67fe9ef88313e021ec0e92b73c200167bb0a86633e8df8658d832cca828c9",
    "coco8-seg": "82c651abd01d556c77769ea834ebff1e77f76a291463e987fb69b63a87e8eb80",
}
MAX_BYTES = 16 * 1024 * 1024
NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def checked_download_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"github.com", "release-assets.githubusercontent.com"}
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
    ):
        msg = "The benchmark download redirected outside its approved source"
        raise ValueError(msg)
    return url


class _Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, checked_download_url(newurl))


def download(name: str) -> bytes:
    url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{name}.zip"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _Redirect())
    with opener.open(checked_download_url(url), timeout=30) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != BENCHMARKS[name]:
        msg = "The public benchmark changed or exceeded its limit; no source files were written"
        raise ValueError(msg)
    return data


def archive_files(data: bytes, name: str) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 100 or sum(entry.file_size for entry in entries) > MAX_BYTES:
            msg = "Benchmark archive exceeds its file or size limit"
            raise ValueError(msg)
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                msg = "Unsafe benchmark archive path"
                raise ValueError(msg)
            if not path.parts or path.parts[0] != name:
                msg = "Unexpected benchmark archive directory"
                raise ValueError(msg)
            if stat.S_ISLNK(entry.external_attr >> 16):
                msg = "Benchmark archive links are not supported"
                raise ValueError(msg)
            if entry.is_dir():
                continue
            relative = str(PurePosixPath(*path.parts[1:]))
            if relative in files:
                msg = "Duplicate benchmark archive path"
                raise ValueError(msg)
            # Retain the original license/readme; no downloaded code is executed.
            if relative not in {"LICENSE", "README.md", "README.txt"} and not (
                len(path.parts) == 4
                and path.parts[1] in {"images", "labels"}
                and path.parts[2] in {"train", "val"}
                and path.suffix in {".jpg", ".txt"}
            ):
                continue
            files[relative] = archive.read(entry)
    for split in ("train", "val"):
        images = {PurePosixPath(p).stem for p in files if p.startswith(f"images/{split}/")}
        labels = {PurePosixPath(p).stem for p in files if p.startswith(f"labels/{split}/")}
        if len(images) != 4 or images != labels:
            msg = "Expected exactly four labeled images in each benchmark split"
            raise ValueError(msg)
    return files


def prepare(ctx: JobContext) -> None:
    import tlc

    name = str(ctx.params.get("benchmark") or "")
    if name not in BENCHMARKS:
        msg = "Choose coco8 for bounding boxes or coco8-seg for instance segmentation"
        raise ValueError(msg)
    project = str(ctx.params.get("project_name") or "").strip()
    if not project or project in {".", ".."} or any(c in project for c in "/\\%?#\x00"):
        msg = "A valid destination project name is required"
        raise ValueError(msg)
    root = ctx.project_root_url
    if not root:
        msg = "Choose the project storage location first"
        raise ValueError(msg)
    ctx.progress(percent=0, label="Downloading eight public benchmark images")
    files = archive_files(download(name), name)
    destination = f"{root}/{project}/source/benchmarks/{name}-{BENCHMARKS[name][:12]}"
    # JSON is YAML-compatible. Absolute cloud paths keep tables usable from another node.
    config = {"path": destination, "train": "images/train", "val": "images/val", "names": NAMES}
    files["provenance.json"] = json.dumps(
        {
            "source": f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{name}.zip",
            "sha256": BENCHMARKS[name],
            "images": 8,
            "purpose": "workflow smoke test",
        },
        indent=2,
    ).encode()
    # Publish the YAML last, after every input exists. Existing files must match.
    files["dataset.yaml"] = json.dumps(config, indent=2).encode()
    for index, (relative, data) in enumerate(files.items()):
        if ctx.cancelled:
            return
        target = tlc.Url(destination + "/" + relative)
        if target.exists():
            read_bounded = getattr(target, "read_bytes_bounded", None)
            if not callable(read_bounded):
                msg = "Update 3LC to verify an existing benchmark safely, or use a new project"
                raise ValueError(msg)
            if read_bounded(MAX_BYTES) != data:
                msg = "The benchmark destination contains changed files; nothing was overwritten"
                raise ValueError(msg)
        else:
            target.write_bytes(data, if_exists="raise")
        ctx.progress(percent=round(100 * (index + 1) / len(files)), label="Preparing project source")
    ctx.output("dataset_yaml", destination + "/dataset.yaml")
    ctx.output("task", "detection" if name == "coco8" else "segmentation")
    ctx.output("train_images", 4)
    ctx.output("validation_images", 4)
    ctx.output("sha256", BENCHMARKS[name])
