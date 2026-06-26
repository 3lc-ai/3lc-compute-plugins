# 3lc-compute-plugins

The open, first-party plugins for the [3LC](https://3lc.ai) compute service, packaged as a single
distribution `3lc-compute-plugins` built against the public
[`3lc-plugin-sdk`](https://github.com/3lc-ai/3lc-plugin-sdk).

```
pyproject.toml         # the one distribution (SDK floor + one extra per plugin + entry-points)
src/
  # every plugin is venv-isolated: its own provisioned venv, out-of-process; deps behind its extra
  tlc_plugin_timm/              # fine-tune timm image classifiers     ([timm] extra)
  tlc_plugin_sam3/              # auto-label with SAM3                  ([sam3] extra)
  tlc_plugin_yolo/              # fine-tune YOLO models                 ([yolo] extra)
  tlc_plugin_importer/          # import CSV/Parquet/COCO/…            ([importer] extra)
  tlc_plugin_exporter/          # export CSV/XLSX/YOLO/COCO/…          ([exporter] extra)
  tlc_plugin_merger/            # merge two tables                     ([merger] extra — empty)
  tlc_plugin_splitter/          # train/val/test splits                ([splitter] extra)
  tlc_plugin_table_statistics/  # per-column stats + thumbnails        ([table_statistics] extra)
  tlc_plugin_image_metrics/     # image-quality metric columns         ([image_metrics] extra)
```

Each `src/tlc_plugin_<name>/` bundles its own `plugin.toml` manifest (the host reads it **without
importing**). The host discovers these plugins via a **folder Source** pointed at `src/` — it scans
each subdirectory's manifest; it does **not** pip-install this distribution. (An entry-point group is
still declared for the optional installed-package discovery path, but that is not how the host
consumes these today.)

## Isolation tiers

Every open plugin here is **`venv`-isolated** (`runtime.isolation = "venv"` in its manifest): the
host provisions a dedicated venv per plugin — `uv sync --extra <id>` against this distribution,
installing only that plugin's extra — and runs it out-of-process. **No plugin dep ever touches the
host venv.** This is uniform across the light CPU plugins (importer, merger, …) and the heavy GPU
ones (timm, sam3, yolo): it's the always-`venv` shape the static-index / shop deliverability model
expects. The base distribution carries only the SDK floor (`3lc-plugin-sdk[shared]`).

`host`-mode (in-process) is reserved for the in-tree plugins (`run_insights`, `table_insights`)
that ship inside the compute service itself — not for anything in this repo.

## Develop

```bash
uv sync                       # SDK floor only
uv sync --extra importer      # one plugin's deps (exactly what the host provisions into its venv)
uv sync --extra timm          # a heavy GPU plugin's stack
uv run ruff check .
```

During development, `3lc-plugin-sdk` is resolved from a sibling checkout via a **dev-only**
`[tool.uv.sources]` path — see `CLAUDE.md`. That reverts to an index/git pin before publish.

### Consumed by the host

The compute service can load these plugins three ways (all end at a per-plugin provisioned venv):
a **folder Source** at `src/`, an **editable install**, or the **install API** which materializes a
venv per plugin via `uv pip install "3lc-compute-plugins[<id>]"` (index/git). The host never installs
this distribution into its own venv. (The install API resolves the umbrella's own SDK dep from an
index, so it lands fully once `3lc-plugin-sdk` is published.)

## Status

Pre-1.0. License pending (see `CLAUDE.md`). The proprietary `run_insights` / `table_insights`
plugins are maintained in the compute service itself and are **not** part of this open distribution.
