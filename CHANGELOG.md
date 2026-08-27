# Changelog

All notable changes to `3lc-compute-plugins` (the umbrella distribution of open, first-party
plugins for the 3LC compute service) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- The Merge Tables page now checks schema compatibility before you can start a merge: it
  loads both tables' columns (via a new `/api/plugins/merger/columns` route) and keeps the merge
  button disabled until two distinct tables are chosen, the output project/dataset resolve, and
  the columns match. When they don't, it blocks the merge and names the differing columns inline
  (e.g. "only in A: segmentations; only in B: bbs") instead of letting a doomed merge start.
- The Import and Image Metrics pages now pick up a job that is already queued or running
  when you navigate back to them, instead of showing the empty form as if nothing were
  happening: a compact "job running — watch it in the Queue" notice with the live status,
  and the result (or failure message) when it finishes. Previously a long import or metrics
  run was only visible from the page you started it on, until you found it in the Queue.
- Image Metrics reports its output table as the job's result, so the generic Queue card now
  has an Open link for it like the importer, merger, and splitter already had.

### Changed
- Built against plugin SDK 0.3 (`3lc-compute-plugin-sdk>=0.3.0,<0.4.0`): `ctx.result(url)` is
  positional, and the shared `PluginJobs` client is injected by the SDK's `/ui` handler, so
  the plugins no longer prepend it themselves. Hosts and frontends on the 0.3 contract show a
  failed job's message on the generic Queue card; the plugin pages read it from the same
  field (falling back to the older status-line placement on older hosts).
- The exporter's destination-folder picker now uses the shared browse widget's output mode
  (`purpose: 'output'`): it lists all folders and flags non-writable ones, instead of hiding
  inaccessible folders the way input pickers now do. Input pickers keep the default behaviour.

### Fixed
- Merge Tables no longer leaves the **Merge Tables** button disabled when two tables with
  matching schemas are selected. The output project/dataset are now read from the loaded
  table object on the backend (`/api/plugins/merger/columns` returns `project_name` and
  `dataset_name`) instead of string-parsing the table URL for a `projects/` segment that real
  table URLs don't contain — so the output naming resolves and the button enables. URL parsing
  remains only as an instant pre-fetch display fallback.
- A failed merge now surfaces a concise, plain-language message (e.g. the tables' columns don't
  match) instead of a raw exception carrying the underlying `tlc` schema-key dump. Expected
  failures use the SDK's clean job-failure path, so they no longer appear as unhandled worker
  tracebacks in the logs either.
- The exporter's column list (CSV/Excel formats) now labels float columns as **number**
  instead of **other**. The classifier read the column type out of the schema's serialized
  JSON, but `tlc` omits default-valued fields there and float32 is the default scalar type —
  so every plain float column (image metrics, width/height, and the like) fell through to
  "other". It now reads the live schema object first, which also makes the old
  `col_name == "weight"` special case unnecessary. (#17)
- Merge Tables now describes and performs its operation truthfully as a **vertical join
  (row concatenation)** — stacking the selected tables' rows into one table — instead of the
  previous, incorrect "column join by row index" labelling (the underlying operation was always
  vertical). Removed the bogus equal-row-count constraint (vertical join does not require it) and
  the disabled "Union (Row Concatenation) — coming soon" option (that behavior *is* what the
  plugin does). Schema-incompatibility failures now surface a clear, actionable message.
- The COCO importer's Annotations Path field now lets you select a folder from the browse
  dialog, not just individual JSON files — the field already accepted a folder (auto-detecting
  splits/types), but the widget's `mode` was locked to `"file"`, which never offered a "Select
  This Folder" option. Individual `.json` file selection still works the same as before.
- The COCO importer no longer crashes with a raw `OSError: Is a directory` traceback when a
  folder doesn't auto-detect (client-side detection failed silently, leaving an unresolved
  folder path submittable). The backend now re-resolves the folder itself — importing directly
  when there's exactly one split, or raising a clear, actionable error (no JSON files found;
  multiple types/splits need the picker) instead of ever handing a directory to `tlc`. The UI
  also now surfaces detection failures inline instead of silently hiding the splits panel, and
  blocks "Ready" until the path actually resolves.

### Changed
- The importer's Table Name field is greyed out during a multi-split YOLO or COCO import
  (each split already gets its own dataset name, so a shared table name isn't disambiguating
  anything) instead of looking like it needs a distinct value per split.
- The splitter now shows a "View in Project" link on a successful split, matching the importer,
  merger, and image-metrics plugins — it previously had no way to navigate to the resulting
  tables from the result panel.

## [0.2.1] - 2026-08-21

### Changed
- Packaging: added a PyPI project README (`README-wheel.md`) and tightened the distribution
  description. No functional or contract change.

## [0.2.0] - 2026-08-18

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
- **Distribution moved to PyPI**: `3lc-compute-plugins` is now published to public
  [PyPI](https://pypi.org/project/3lc-compute-plugins/) via Trusted Publishing; the private
  CloudRepo index (pypi.3lc.ai) is no longer needed to install the plugins. Manual prerelease
  builds keep publishing to CloudRepo for a grace period (#9).
- All dependencies resolve from public PyPI: `3lc` (its home since the 3.2 rust release, #11)
  and the plugin SDK (`3lc-compute-plugin-sdk[shared]>=0.2.2,<0.3.0`, on PyPI since 0.2.2) —
  no custom package indexes remain (#9).

### Fixed
- User-typed paths are normalized at every ingress (`~` expands, bare-relative paths are
  rejected), and CSV/XLSX exports expand URL aliases to absolute paths — exported files no
  longer carry 3LC-internal alias tokens or user-relative paths (#9).
- Export jobs are attributed to the exported table's own project (falling back to the
  launch context), so they appear in that project's Queue & Progress stacks — previously
  a bare sidebar launch ran the job unattributed and the panel's project filter hid it
  from every view (#9).
- Splitter and image-metrics jobs get the same table-project attribution: the splitter
  relied on the launch-context default (empty on a bare sidebar launch), and image-metrics
  bypassed the SDK job tracker entirely so every one of its jobs ran unattributed (#9).

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
