# 3lc-compute-plugins — agent & contributor orientation

This repo is the first-party, Apache-2.0 plugin collection for the 3LC compute service, packaged as
a single **umbrella distribution** `3lc-compute-plugins`, built against the public
[`3lc-compute-plugin-sdk`](https://github.com/3lc-ai/3lc-compute-plugin-sdk). The umbrella is a
packaging choice —
the host discovers plugins by scanning a folder Source (`src/`), not by repo shape.

## Layout

```
pyproject.toml             # the one distribution: 3lc-compute-plugins
src/
  tlc_plugin_<name>/       # one package per plugin (import name tlc_plugin_<name>)
    __init__.py            # the plugin class — behavior only
    plugin.toml            # manifest — host reads it WITHOUT importing (id, ui, runtime.*)
    ui.html, ...           # data files live inside the package (bundled in the wheel)
```

One `pyproject.toml` declares **all** plugins:
- **base `dependencies`** = the SDK floor ONLY (`3lc-compute-plugin-sdk[shared]`). Every plugin here is
  `venv`-isolated, so NO plugin deps live in the base — this distribution is never installed into
  the host venv.
- **per-plugin extras** (`[importer]`/`[exporter]`/`[merger]`/`[splitter]`/`[table_statistics]`/
  `[image_metrics]`) = each plugin's deps, installed ONLY into that plugin's provisioned
  venv. `merger` is intentionally empty (SDK floor suffices).
- **`[project.entry-points."tlc_compute.plugins"]`** = one entry per plugin, supporting the optional
  installed-package discovery path. The primary path is a **folder Source** pointed at `src/`: the
  host reads each bundled `plugin.toml` without importing and provisions a venv installing the
  `provision_extra` named in the manifest.

These are real installable packages (resolved from site-packages).

## The rules — do not break these

1. **Touch the host only through the SDK.** A plugin imports `tlc_plugin_sdk` and nothing
   from `tlc_compute` (the host). The mental test: *could this build against just the SDK wheel, with
   the host source deleted?* It must.
2. **Every plugin's deps are its own venv, never the host venv.** Each plugin here is `venv`-isolated
   and provisioned into its own environment from its extra; nothing in this repo installs into the
   host venv. (`host`-mode is reserved for plugins that ship with the service itself.)
3. **Metadata lives in `plugin.toml`, not on the class.** No `register()` at import; the class is
   behavior-only. The host stamps display identity from the manifest.
4. **Custom events / routes are plugin-private.** Use the generic job channel
   (`progress`/`metric`/`result`) for anything the shell renders; `ctx.emit` + your own `ui.html`
   for plugin-specific UI.

## Versioning

The **distribution** versions as a whole (`[project] version` in the umbrella `pyproject.toml`).
Each plugin still declares its own `plugin.toml` `version` + `min_service_version` /
`min_frontend_version` floors — that's the compatibility contract the host reads. The dist pins the
SDK via `3lc-compute-plugin-sdk>=X,<Y`. SemVer throughout (the SDK's 0.x contract line is
additive-only — see its README → Status).

## Dev setup

The committed `[tool.uv.sources]` resolve the SDK from the 3LC prereleases index and `3lc`
from the releases index. For local SDK development, override (uncommitted) with an editable
path source pointing at your SDK checkout.

```bash
uv sync                       # SDK floor only
uv sync --extra importer      # one plugin's deps (exactly what the host provisions into its venv)
uv run ruff check .
```

## Where the rest of the context lives

The plugin contract and the author guide live in `3lc-compute-plugin-sdk` (`docs/plugin-guide.md`,
published at <https://3lc-ai.github.io/3lc-compute-plugin-sdk/>). Read those for the "why" and for
how to build a plugin against the contract.

## Conventions

Python 3.10+, uv, Hatchling, Litestar route handlers (`get_route_handlers()`), Ruff (line-length
120), mypy `--strict` where deps allow. Google-style docstrings. SPDX header on new files
(`# Copyright <year> 3LC Inc.` / `# SPDX-License-Identifier: Apache-2.0`).
