# Changelog

All notable changes to `3lc-compute-plugins` (the umbrella distribution of open, first-party
plugins for the 3LC compute service) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Importer and exporter adopt the SDK's shared data-source picker (7 importer fields across
  5 formats, plus the exporter output path): one consistent browse/upload widget instead of
  bespoke `file_upload`/`text` fields, with the importer's ad-hoc upload route promoted to
  the SDK's shared `/browse` + `/upload-temp` handlers. Requires SDK 0.2.2 (#9).
- CI: ruff lint/format gate, lockfile freshness check (`uv lock --check`), and a pytest suite
  covering manifests, import formats, and plugin-class instantiation; `uv.lock` is now
  committed (#9).

### Changed
- Exports run as host-managed jobs instead of one held request, so long exports (a YOLO
  export copies every image) survive request timeouts and report progress through the
  generic queue panel (#10).
- `3lc` resolves from public PyPI, its home since the 3.2 rust release; the SDK keeps its
  prereleases-index pin until its own PyPI move (#11).

### Fixed
- User-typed paths are normalized at every ingress (`~` expands, bare-relative paths are
  rejected), and CSV/XLSX exports expand URL aliases to absolute paths — exported files no
  longer carry 3LC-internal alias tokens or user-relative paths (#9).
- Export jobs are attributed to the exported table's own project (falling back to the
  launch context), so they appear in that project's Queue & Progress stacks — previously
  a bare sidebar launch ran the job unattributed and the panel's project filter hid it
  from every view (#9).

## [0.1.2] - 2026-07-03

### Fixed
- Per-plugin manifest versions and the distribution version are bumped together, so the version
  a plugin card reports matches the installed distribution.

## [0.1.1] - 2026-07-03

### Fixed
- Corrected the `requires_gpu` flag on the image-metrics plugin, so it is classified and queued
  as a CPU workload.

### Changed
- The plugin SDK dependency is resolved from the public package index under its final name
  `3lc-compute-plugin-sdk` (was a git pin).

## [0.1.0] - 2026-06-30

First release: the open, first-party plugins for the 3LC compute service as a single umbrella
distribution built against the public `3lc-compute-plugin-sdk`.

### Added
- Six venv-isolated plugins, each behind its own extra:
  - `tlc_plugin_importer` — import CSV/Parquet/COCO/… (`[importer]`)
  - `tlc_plugin_exporter` — export CSV/XLSX/YOLO/COCO/… (`[exporter]`)
  - `tlc_plugin_merger` — merge two tables (`[merger]`)
  - `tlc_plugin_splitter` — train/val/test splits (`[splitter]`)
  - `tlc_plugin_table_statistics` — per-column stats + thumbnails (`[table_statistics]`)
  - `tlc_plugin_image_metrics` — image-quality metric columns (`[image_metrics]`)

### Changed
- The heavy GPU plugins (timm, SAM3, YOLO) moved out to their own dedicated repositories
  (`3lc-compute-plugin-timm`, `3lc-compute-plugin-sam3`, `3lc-compute-plugin-yolo`) and are no
  longer part of this distribution.
