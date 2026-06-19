# 3lc-compute-plugins

The open, first-party plugins for the [3LC](https://3lc.ai) compute service, packaged as a single
distribution `3lc-compute-plugins` built against the public
[`3lc-plugin-sdk`](https://github.com/3lc-ai/3lc-plugin-sdk).

```
pyproject.toml         # the one distribution (base deps + per-venv-plugin extras + entry-points)
src/
  # venv (own venv, provisioned out-of-process; heavy/GPU deps behind an extra)
  tlc_plugin_timm/              # fine-tune timm image classifiers     ([timm] extra)
  tlc_plugin_sam3/              # auto-label with SAM3                  ([sam3] extra)
  tlc_plugin_yolo/              # fine-tune YOLO models                 ([yolo] extra)
  # host-mode (light; installed into the host venv via 3lc-compute[plugins], runs in-process)
  tlc_plugin_importer/          # import CSV/Parquet/COCO/…
  tlc_plugin_exporter/          # export CSV/XLSX/YOLO/COCO/…
  tlc_plugin_merger/            # merge two tables
  tlc_plugin_splitter/          # train/val/test splits
  tlc_plugin_table_statistics/  # per-column stats + thumbnails
  tlc_plugin_image_metrics/     # image-quality metric columns
```

Each `src/tlc_plugin_<name>/` bundles its own `plugin.toml` manifest (the host reads it **without
importing**) and advertises a `tlc_compute.plugins` entry point. One umbrella `pyproject.toml`
declares them all; single-plugin or multi-dist repos are equally valid — the host discovers by
entry point, not by repo shape.

## Isolation tiers

A plugin declares `runtime.isolation` in its manifest:

- **`venv`** (e.g. timm, sam3, yolo) — heavy/conflicting deps; the host provisions a dedicated venv
  installing the plugin's heavy extra (`3lc-compute-plugins[timm]`) and runs it out-of-process. Its
  deps never touch the host venv. Only the plugin's light code + manifest sit in the base.
- **`host`** (e.g. importer, exporter, the light ones) — deps are light and compatible with the host
  venv; runs in-process. Delivered as part of the host's `[plugins]` extra (base deps).

The rule: **anything installed into the host venv must be light.** A venv plugin's heavy stack lives
only behind its extra, installed into its own venv.

## Develop

```bash
uv sync                  # base: host-mode plugins' light deps + the SDK
uv sync --extra timm     # add a venv plugin's heavy stack (what the host provisions into its venv)
uv run ruff check .
```

During the build-out, `3lc-plugin-sdk` is resolved from a sibling checkout via a **dev-only**
`[tool.uv.sources]` path — see `CLAUDE.md`. That reverts to an index/git pin before publish.

## Status

Pre-1.0. License pending (see `CLAUDE.md`). The proprietary `run_insights` / `table_insights`
plugins deliberately stay in the private `3lc-insights` monorepo and are **not** here.
