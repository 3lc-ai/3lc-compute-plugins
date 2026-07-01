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

### Editor autocomplete for `ui.html` (the JS bridge)

Each `src/tlc_plugin_*/ui.html` drives the host through `window.PLUGIN_API` /
`window.PluginJobs`. Those are typed by a declaration that ships inside the installed
`3lc-plugin-sdk` (`<site-packages>/tlc_plugin_sdk/contract/plugin-api.d.ts`, JS_CONTRACT
0.2). The repo-root **`jsconfig.json`** wires it up so VS Code gives autocomplete in every
`ui.html` — **no node build, no `node_modules`**:

```jsonc
{ "compilerOptions": {
    "allowJs": true,
    "typeRoots": [".venv/lib/python3.12/site-packages"],
    "types": ["tlc_plugin_sdk/contract/plugin-api"] },
  "include": ["src/**/ui.html"] }
```

Run `uv sync` once (it installs the SDK into `.venv`) and the types resolve. Resolution
gotchas: reference the **import name `tlc_plugin_sdk`**, not the dist name `3lc-plugin-sdk`;
it must go through `typeRoots` (a `/// <reference types>` ignores `paths`); bump the
`python3.NN` segment to match your venv. No per-file edit is needed — `types` loads the
declaration into every `ui.html` globally; you may optionally add
`/// <reference types="tlc_plugin_sdk/contract/plugin-api" />` at the top of a `<script>`
to self-document.

## Status

Pre-1.0. License pending (see `CLAUDE.md`). The proprietary `run_insights` / `table_insights`
plugins deliberately stay in the private `3lc-compute` host repo and are **not** here.
