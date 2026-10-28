# Changelog

All notable changes to `3lc-compute-plugins` (the umbrella distribution of open, first-party
plugins for the 3LC compute service) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

#### Actions that cannot run yet

- **Export waits for what an export needs, and says so.** It stays disabled until there is a table, an
  output path and, for CSV and Excel, a ticked column; the line under it and its tooltip name the first
  thing missing ("Tick at least one column to enable Export."), and say to wait while the table's
  columns are listed or while an export runs. A click no longer reports a missing field afterwards.
- **Export's Select all and Deselect all** are disabled until the table's columns are listed, and
  while they would change nothing ("Nothing to deselect: no column is ticked."), saying so in their
  tooltips.
- **Merge Tables says what it waits for.** It was already disabled until two distinct tables with the
  same columns were chosen; the line under it and its tooltip now say which ("Choose Table B to enable
  Merge Tables.", "Choose tables with the same columns to enable Merge Tables.", or to wait while the
  columns are compared or the merge runs).
- **Split Table is disabled until a table is chosen**, and while a split runs, saying why on the line
  under it and in its tooltip, instead of answering a click with "Select a table to split."
- **Compute Metrics says what it waits for.** It stays disabled until a table is added, the metric list
  has loaded and a metric is selected, and while metrics are computed; the line under it and its
  tooltip name the first ("Select at least one metric, or click Recommended, to enable Compute
  Metrics."), also after a preset fills the form. The alerts a click raised are gone.
- **Image Metrics' Select All, Clear and Recommended** are disabled while the metric list loads and
  while they would change nothing ("Nothing to clear: no metric is selected."), saying so in their
  tooltips.
- **Import says what it waits for.** The format form's Import names the first required field left
  empty ("Fill in Project Name to enable Import YOLO Dataset."), an annotations path with no COCO
  annotations, detected splits all unticked, a cloud project whose alias would point at a local disk,
  and a running import, on the line under it and in its tooltip; also after split detection or
  "Import Next Split" fill the form.
- **The CSV / Excel wizard's Next and Import say what they wait for.** Next (step 2) is disabled until a
  column is ticked, instead of answering with a toast. Import (step 3) waits for a project and a dataset
  name, or the table to extend, and is disabled while "Run on" is a GPU node ("Set Run on to this
  computer to enable Import: …"), from the moment Run on changes rather than after a click. Select All
  and Deselect All are disabled while they would change nothing.

#### Image Metrics: presets

- **The Config bar is the SDK's preset bar.** One line at the top of the page: a select with Default
  settings (the recommended metrics) and the saved presets (metric count and age in the label), Save
  as… that opens a prefilled name field in the bar, and a trash icon that asks in the bar. The entry
  the form came from gains "· modified" once the form differs; picking it again reverts. A preset is
  a template of the whole form, **tables included**: picking one fills the form, only Save writes one.
  The "Config Name" field and "+ New", "Save New" and "Delete" are gone with the bar. Saved configs
  appear as presets as they are (with no tables); the config the old bar remembered opens once more,
  then the page opens as it was last left in this browser.
- **Opened from a table**, that table is the list (the preset's tables are set aside) and stays
  while the page is open, also when another preset is picked.
- A saved config now holds `table_urls`. Requires SDK 0.5.5.

#### Data movement

- **Importer: aliases follow the SDK's one rule for picked folders (`register_source_alias`).** The
  behaviour for a source root above the picked folder is unchanged; an alias folder that is not
  above the picked folder (a durable copy elsewhere) now folds the rows from the folder read and
  saves the alias on the copy, like every other plugin using the alias card. Requires an SDK with
  `register_source_alias`.

- **Importer references existing data without relocating it.** Regular and CSV imports no longer offer or perform permanent copies. Put data where the project can read it first (Storage copies between buckets; a local folder is uploaded with a tool such as `aws s3 sync`), then import from that source. Old saved requests that explicitly ask for copying fail with an explanation that says so, before writing a table or alias.
- **Exporter: a bucket URL from its own picker is a valid destination.** The output picker offered
  buckets, but the export refused any URL ("Path must be absolute"). A bucket export is now written
  to a local scratch folder and uploaded under the URL — a file URL (`…/export.csv`) names the
  file, anything else is the prefix — with upload progress on the job, existing objects
  overwritten as local files are. The never-sent `alias_overrides` body key is no longer read; the
  SDK worker applies the host's `_alias_overrides` around the job.
- **Importer: a CSV import aimed at a GPU node says why it cannot run there.** The uploaded file is
  held by the importer's worker on the computer it was uploaded to, so a node run failed with "File
  session expired". The fragment now refuses up front when "Run on" is a node, and the job's
  message explains where the upload lives and what to do.
- **Importer: the alias stays on the folder you picked when a run reads a copy.** When the Hub points
  an import at a copy of its data on a GPU node, or at a path named there, the persisted alias keeps
  pointing at the picked folder and only the run's session alias follows the copy, so the table's
  paths resolve to the source everywhere. The fragment echoes its source fields
  (`submitted_sources`) so the import can tell. An alias set above the source folder (a dataset
  root) moves by the same subfolders; when the copy does not keep that layout, the import refuses
  rather than writing paths that resolve to the wrong place.
- **Importer: a missing source fails the import before anything acts on it.** Required fields and
  every data-source path are checked on the machine the import runs on before the alias is
  registered in the project, so a mistyped path, or one that exists
  only on another machine, no longer leaves a persisted alias behind. The
  message names the field, the path and the machine.
- **The manifests declare the data each run reads and writes.** `[runtime] data_inputs` /
  `data_outputs` name the run-body keys that hold data: the importer's source fields
  (`dataset_yaml`, `annotations_file`, `image_folder`, `folder_path`, `csv_path`, and `table_url`
  for a CSV extend), the exporter's `table_url` and `output_path`, the merger's `table_urls`, and
  the splitter's and Image Metrics' `table_url`. A Hub that plans data movement asks where that
  data is for the chosen machine before the run starts; older hosts ignore the keys. Table
  Statistics has no run body and declares none.


#### Other changes

- **Merge, Split and Export use the Hub's table field (`<tlc-table-field>`).** Browse…, or a
  double-click in an empty table field, opens Browse Tables, which opens at the table the field holds.
  In Merge it also tags the table the other field holds ("Table A"), so picking B starts where A is.
  These three now need Hub 1.2.0 (`min_frontend_version`); an older Hub shows them disabled with an
  update badge. Import renders its `table_picker` fields the same way, though no format has one yet.
- **Image Metrics adds tables with a table field instead of "+ Add Table".** The field sits under the
  list of chosen tables: a table picked there, by Browse…, a double-click or a pasted URL and Enter,
  joins the list and the field empties for the next. Browse Tables marks the tables already in the
  list as this field's selection ("Added"). A URL can now be pasted; before, the picker was the only way in. Image Metrics now
  needs Hub 1.2.0.

- **Run results can be closed, and look the same in every plugin.** Import, Export, Merge, Split and
  Image Metrics draw their result with the Hub's shared result box (`PLUGIN_API.showResult`), which
  has a close button (✕). Image Metrics' own banner is gone, and so is the second copy of it that a
  run could leave behind. On a frontend that predates the shared box, the result shows its message
  and links without the details or the close button.
- **Importer: no Hugging Face dependencies.** Its extra no longer installs `datasets` and `transformers`,
  which nothing in the importer used. Its descriptions name the formats it imports (CSV, Excel, COCO, YOLO,
  image folders) instead of Parquet and Hugging Face, which it never read; Hugging Face imports are the
  huggingface plugin's.
- **Every plugin in this distribution can run on a node.** Importer, Exporter, Merger, Splitter,
  Table Statistics and Image Metrics declare `remote_capable`, so a host accepts a node target for
  them as well as itself: the work runs beside its data and on a larger machine. The host still
  refuses a node run whose project root would land on that node's own disk. An import's missing-path
  and permission messages now name the machine the import ran on instead of saying imports always
  run on the host.
- Lock the staged 3lc 3.5.0.dev149866 and SDK 0.5.0.20261007121635.34.1; require the staged core
  directly (`3lc>=3.5.0.dev149866,<4.0.0`) and resolve it from the `staging` index alongside the SDK.
- CI also runs on pull requests into, and pushes to, `config-service-poc`.
- **Merger: writes under the job's project root.** The merged table goes under the root the host
  stamps into the job (`ctx.project_root_url`) instead of the worker's configured root, so a merge
  on a remote node lands where the deployment writes projects.
- Require SDK 0.5 and use the verified private CloudRepo snapshot for POC development and CI.
- Manual release workflows default to build-only. Publishing uses private CloudRepo;
  distribution and all six plugin manifest versions advance together, including on reruns.
  Built wheels are checked before publication so the Hub can identify staged plugin updates.
- Resolve the plugin SDK from the explicit `staging` index declared in `pyproject.toml`; developers
  and CI set only `UV_INDEX_STAGING_USERNAME` / `UV_INDEX_STAGING_PASSWORD`.

### Fixed

- **Export: Deselect all no longer exports every column.** An empty column list reached the server as
  no `columns`, which it reads as all of them; Export now waits for a ticked column. A column list read
  for an earlier table can no longer go out with the table that replaced it.
- **Merge: a table changed after a successful column check no longer leaves Merge Tables enabled**
  until the new check answers, and a change while a merge runs no longer re-enables it.
- **Split: a split cancelled from the Queue no longer leaves Split Table disabled** until the page is
  reloaded; the page says "Split cancelled."
- **Image Metrics: Compute Metrics stays disabled while metrics are computed.** Changing the metric or
  table selection during a run re-enabled it. When no job could be started at all, it stayed disabled
  until a reload; the page now says "Not started" and enables it again.
- **Import: the CSV wizard no longer starts an import without its names.** A missing project or
  dataset name, or table to extend, reached the job, which refused it. With a COCO annotations file
  named directly, Import stayed disabled after detection filled the images folder, until another field
  changed.
- Import success messages show `initial` when the table name is left blank.
- **Table Statistics stops polling when SDK initialization fails.** An activation or import
  failure now completes the statistics request with an error instead of leaving its spinner running.
- **Table Statistics retries a failed table on the next request.** A statistics session that ended
  in an error is returned to the poll that finishes it and then dropped, so the next request reads
  the table again instead of serving the cached error until `reset=1` or a worker restart. When the
  S3 read failed because the compute service has no AWS credentials (the AWS SDK's "the credential
  provider was not enabled"), the error starts with a sentence saying so and naming the fix: set
  `AWS_PROFILE` or access keys in the compute service's environment and restart it.
- **Image Metrics imports on a worker without a home directory.** Its legacy config folder is resolved
  when a config store is built, not at import, as the SDK resolves its own config root; without a home
  the store raises the SDK's `ConfigRootUnavailable` instead of the plugin failing to load.
- Import failures (a bad path, an existing table, a failed executor) now surface a clean,
  one-line message with no worker traceback attached. A missing or unreadable input path names
  the path.

### Added

- **Image Metrics: Mean Red, Mean Green and Mean Blue.** Three metrics in the Color category, each
  the mean of one channel over the image's pixels (0-255), so an image's color cast, a source with
  its own white balance, or an image saved in BGR order stands out when they are charted side by
  side. A grayscale image gets three equal means.
- **Importer: locations may be bucket URLs.** A project can be created "in this computer" or at the
  bucket root the deployment names. A cloud project requires a durable cloud alias; a local default
  alias is refused with a reason. Import never relocates the source. Every
  table an import wrote is reported below the form that wrote it, with the shared ending in all
  three places the importer reports a table.

## [0.2.4] - 2026-09-07

### Changed
- Requires plugin SDK `>=0.3.3`, which raises the `3lc` floor to 3.3.0 so every plugin venv
  resolves the current 3lc release. The umbrella declares no `3lc` dependency of its own; the
  data plane arrives through the SDK.

### Fixed
- The COCO importer's Table Name field no longer becomes stuck on "initial" once a multi-split
  dataset is detected. It stays editable, and the value you enter is applied to every split's
  table (only the dataset name differs per split, e.g. `coco_train`, `coco_val`).
- The COCO importer no longer duplicates path segments when a dataset's `file_name` values
  already include subdirectories (e.g. `train/images/001.jpg`) that overlap with the supplied
  Images Folder. It now checks the annotations against the filesystem and strips the overlapping
  segment before import.

## [0.2.3] - 2026-08-31

### Changed
- Required text inputs on the Import, Export, Merge and Split pages now carry the `required`
  attribute alongside a placeholder, so the Hub's shared required-field cue (the muted wash that
  clears once a value is entered) applies to them the same way it does everywhere else in the Hub.
  The importer's own amber "guide pulse" on the next empty field is gone in favour of that shared
  cue; the form still auto-focuses the first missing field when it opens.
- The Merge page no longer marks Output Table Name as required — leaving it blank auto-names the
  table, and the label now says so consistently.
- The hero "what this does" badges use the host's shared `.plugin-hero-badge` styling instead of
  inline styles, so they render as flat, non-interactive tiles in both themes (the inline
  background and shadow made them look like clickable cards in dark mode). Requires a Hub
  frontend that styles `.plugin-hero-badge` itself.
- Image Metrics launches its jobs through the SDK's `PluginJobs.start` instead of a hand-rolled
  `POST /run`. Each job is still attributed to its own table's project; when the table's project
  is unknown the job now falls back to the project page it was launched from, so it appears in
  that project's Queue & Progress panel. When the host answers that the plugin environment is
  still being built (or its build failed), the log says so instead of failing on a missing job id.

## [0.2.2] - 2026-08-27

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
- Built against plugin SDK 0.3 (`3lc-compute-plugin-sdk>=0.3.1,<0.4.0`, pinned without the
  deprecated `[shared]` extra — since 0.3.1 the SDK brings the `3lc` data plane itself):
  `ctx.result(url)` is positional, and the shared `PluginJobs` client is injected by the SDK's `/ui` handler, so
  the plugins no longer prepend it themselves. Hosts and frontends on the 0.3 contract show a
  failed job's message on the generic Queue card; the plugin pages read it from the same
  field (falling back to the older status-line placement on older hosts).
- The exporter's destination-folder picker now uses the shared browse widget's output mode
  (`purpose: 'output'`): it lists all folders and flags non-writable ones, instead of hiding
  inaccessible folders the way input pickers now do. Input pickers keep the default behaviour.
- The importer's Table Name field is greyed out during a multi-split YOLO or COCO import
  (each split already gets its own dataset name, so a shared table name isn't disambiguating
  anything) instead of looking like it needs a distinct value per split.
- The splitter now shows a "View in Project" link on a successful split, matching the importer,
  merger, and image-metrics plugins — it previously had no way to navigate to the resulting
  tables from the result panel.

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
