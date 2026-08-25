# 3lc-compute-plugins

The open, first-party plugins for the [3LC](https://3lc.ai) compute service, packaged as a single
distribution `3lc-compute-plugins` built against the public
[`3lc-compute-plugin-sdk`](https://github.com/3lc-ai/3lc-compute-plugin-sdk).

```
pyproject.toml         # the one distribution (SDK floor + per-plugin extras + entry-points)
src/
  tlc_plugin_importer/          # import CSV/Parquet/COCO/…      ([importer] extra)
  tlc_plugin_exporter/          # export CSV/XLSX/YOLO/COCO/…    ([exporter] extra)
  tlc_plugin_merger/            # merge two tables               ([merger] extra, empty)
  tlc_plugin_splitter/          # train/val/test splits          ([splitter] extra)
  tlc_plugin_table_statistics/  # per-column stats + thumbnails  ([table_statistics] extra)
  tlc_plugin_image_metrics/     # image-quality metric columns   ([image_metrics] extra)
```

Each `src/tlc_plugin_<name>/` bundles its own `plugin.toml` manifest (the host reads it **without
importing**) and advertises a `tlc_compute.plugins` entry point. One umbrella `pyproject.toml`
declares them all; single-plugin or multi-dist repos are equally valid — the host discovers by
entry point (or by scanning a folder Source), not by repo shape.

### Heavy ML plugins now live in their own repos

The GPU/heavy-stack plugins moved out to dedicated single-plugin repos and are **no longer here**:

- [`3lc-compute-plugin-timm`](https://github.com/3lc-ai/3lc-compute-plugin-timm) — fine-tune timm image classifiers
- [`3lc-compute-plugin-sam3`](https://github.com/3lc-ai/3lc-compute-plugin-sam3) — auto-label with SAM3
- [`3lc-compute-plugin-yolo`](https://github.com/3lc-ai/3lc-compute-plugin-yolo) — fine-tune YOLO models

They build against the same `3lc-compute-plugin-sdk` contract; the host discovers them the same way.

## Isolation

Every plugin in this repo is **`venv`-isolated** (`runtime.isolation = "venv"` in its manifest): the
host provisions a dedicated venv, installs the plugin's extra (e.g. `3lc-compute-plugins[importer]`),
and runs the plugin out-of-process. Its deps never touch the host venv.

That's why the base `dependencies` is the **SDK floor only** (`3lc-compute-plugin-sdk[shared]`) — this
distribution is never installed into the host venv. Each plugin's real deps ride its own extra and
land only in that plugin's provisioned venv.

(`host`-mode — light deps, in-process in the host venv — is reserved for plugins that ship
with the service itself, and is **not** used here.)

## Develop

```bash
uv sync                      # SDK floor only
uv sync --extra importer     # one plugin's deps (exactly what the host provisions into its venv)
uv run ruff check .
```

The committed source resolves `3lc-compute-plugin-sdk` from the 3LC package index. For local
SDK development, override it (uncommitted) with an editable path source — see `CLAUDE.md`.

### Editor autocomplete for `ui.html` (the JS bridge)

Each `src/tlc_plugin_*/ui.html` drives the host through `window.PLUGIN_API` /
`window.PluginJobs`. Those are typed by a declaration that ships inside the installed
SDK (`<site-packages>/tlc_plugin_sdk/contract/plugin-api.d.ts`, versioned with the SDK). The
repo-root **`jsconfig.json`** wires it up so VS Code gives autocomplete in every
`ui.html` — **no node build, no `node_modules`**:

```jsonc
{ "compilerOptions": {
    "allowJs": true,
    "typeRoots": [".venv/lib/python3.12/site-packages"],
    "types": ["tlc_plugin_sdk/contract/plugin-api"] },
  "include": ["src/**/ui.html"] }
```

Run `uv sync` once (it installs the SDK into `.venv`) and the types resolve. Resolution
gotchas: reference the **import name `tlc_plugin_sdk`**, not the dist name `3lc-compute-plugin-sdk`;
it must go through `typeRoots` (a `/// <reference types>` ignores `paths`); bump the
`python3.NN` segment to match your venv. No per-file edit is needed — `types` loads the
declaration into every `ui.html` globally; you may optionally add
`/// <reference types="tlc_plugin_sdk/contract/plugin-api" />` at the top of a `<script>`
to self-document.

## Status

Pre-1.0; the SDK contract's 0.x line is additive-only (see the `3lc-compute-plugin-sdk`
README → Status). Apache-2.0 (see `LICENSE`), matching the SDK. The service's built-in insights
plugins ship with the service itself and are not part of this collection.
