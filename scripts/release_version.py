# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep distribution, plugin manifests, and built wheels on one release version."""

from __future__ import annotations

import argparse
import re
import sys
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def check_sources(root: Path) -> str:
    """Return the distribution version after checking every advertised plugin."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = str(project["version"])
    packages = project["entry-points"]["tlc_compute.plugins"].values()
    for package in packages:
        path = root / "src" / package / "plugin.toml"
        manifest = tomllib.loads(path.read_text(encoding="utf-8"))
        if manifest.get("version") != version:
            msg = f"{path}: expected version {version}, got {manifest.get('version')}"
            raise ValueError(msg)
    return version


def stamp(root: Path, version: str) -> None:
    """Stamp the project and its plugin manifests after validating the source tree."""
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:\.\d+)*", version):
        msg = f"Expected a numeric release or POC build version, got {version!r}"
        raise ValueError(msg)
    check_sources(root)
    project_path = root / "pyproject.toml"
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))["project"]
    paths = [
        project_path,
        *(root / "src" / p / "plugin.toml" for p in project["entry-points"]["tlc_compute.plugins"].values()),
    ]
    replacements: list[tuple[Path, str]] = []
    for path in paths:
        updated, count = re.subn(
            r'^version\s*=\s*"[^"]+"',
            f'version = "{version}"',
            path.read_text(encoding="utf-8"),
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            msg = f"{path}: expected one version assignment"
            raise ValueError(msg)
        replacements.append((path, updated))
    for path, updated in replacements:
        path.write_text(updated, encoding="utf-8")
    check_sources(root)


def check_wheel(root: Path, wheel: Path) -> None:
    """Require the wheel metadata and every advertised manifest to match the sources."""
    version = check_sources(root)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    with ZipFile(wheel) as archive:
        (metadata_path,) = [p for p in archive.namelist() if p.endswith(".dist-info/METADATA")]
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        if metadata["Name"] != project["name"] or metadata["Version"] != version:
            msg = f"{wheel}: distribution metadata does not match the source project"
            raise ValueError(msg)
        for package in project["entry-points"]["tlc_compute.plugins"].values():
            manifest = tomllib.loads(archive.read(f"{package}/plugin.toml").decode())
            if manifest.get("version") != version:
                msg = f"{wheel}: {package} manifest does not match distribution version {version}"
                raise ValueError(msg)


def main() -> None:
    """Check sources, optionally stamp a build version, and validate built wheels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stamp")
    parser.add_argument("--wheel", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.stamp:
        stamp(args.root, args.stamp)
    version = check_sources(args.root)
    for wheel in args.wheel:
        check_wheel(args.root, wheel)
    print(f"Distribution and plugin manifests agree: {version}")


if __name__ == "__main__":
    main()
