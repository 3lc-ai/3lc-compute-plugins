# Changelog

All notable changes to `3lc-compute-plugins` (the umbrella distribution of open, first-party
plugins for the 3LC compute service) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

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
