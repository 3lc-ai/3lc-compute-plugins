# 3lc-compute-plugins — agent & contributor orientation

This repo is the **public, first-party plugin collection** for the 3LC compute service, packaged as
a single **umbrella distribution** `3lc-compute-plugins` built against the public
[`3lc-plugin-sdk`](https://github.com/3lc-ai/3lc-plugin-sdk). Extracted from the private
`3lc-insights` monorepo (where the host runtime + the design docs live). (The umbrella is this repo's
packaging choice — the host discovers plugins by entry point, so single-plugin and multi-dist repos
are equally valid elsewhere.)

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
- **base `dependencies`** = `3lc-plugin-sdk` + the light deps the in-process (**host-mode**) plugins
  need; these install into the host venv via `3lc-compute[plugins]`.
- **per-venv-plugin extras** (`[timm]`/`[sam3]`/`[yolo]`) = each one's heavy stack (torch, …),
  installed ONLY into that plugin's provisioned venv.
- **`[project.entry-points."tlc_compute.plugins"]`** = one entry per plugin (value = its import
  package). The host iterates this group to discover every plugin, then reads each bundled
  `plugin.toml`. For host-mode it imports the entrypoint in-process; for venv it registers the
  plugin and provisions a venv installing the `provision_extra` named in the manifest.

These are **real installable packages** (resolved from site-packages), not the old flat
`package = false` + cwd-on-sys.path form.

## The rules — do not break these

1. **Touch the host only through `3lc-plugin-sdk`.** A plugin imports `tlc_plugin_sdk` and nothing
   from `tlc_compute` (the host). The mental test: *could this build against just the SDK wheel, with
   the host source deleted?* It must.
2. **Heavy deps are this plugin's own venv, never the host venv.** A `venv`-mode plugin (torch,
   ultralytics, …) is provisioned into its own environment; its heavy deps must never become a host
   dependency. Only light `host`-mode plugins install into the host venv.
3. **Metadata lives in `plugin.toml`, not on the class.** No `register()` at import; the class is
   behavior-only. The host stamps display identity from the manifest.
4. **Custom events / routes are plugin-private.** Use the generic job channel
   (`progress`/`metric`/`result`) for anything the shell renders; `ctx.emit` + your own `ui.html`
   for plugin-specific UI.

## Versioning

The **distribution** versions as a whole (`[project] version` in the umbrella `pyproject.toml`).
Each plugin still declares its own `plugin.toml` `version` + `min_service_version` /
`min_frontend_version` floors — that's the compatibility contract the host reads. The dist pins the
SDK via `3lc-plugin-sdk>=X,<Y`. SemVer throughout.

## Dev setup (build-out)

`3lc-plugin-sdk` is unpublished during the build-out, so the umbrella resolves it from a sibling
checkout via a **dev-only** `[tool.uv.sources] 3lc-plugin-sdk = { path = "../3lc-plugin-sdk" }`.
torch comes from the cu126 index; 3lc from the 3lc-releases index. These dev sources revert to
index/git pins before any real publish.

```bash
uv sync                       # base: host-mode plugins' light deps + SDK
uv sync --extra timm          # add a venv plugin's heavy stack (what provisioning installs into its venv)
uv run ruff check .
```

## Where the rest of the context lives

The plugin contract, the author guide, and the architecture/distribution design docs live in
`3lc-plugin-sdk` (`docs/plugin-guide.md`, `CLAUDE.md`) and the private `3lc-insights` monorepo
(`docs/plugin-*.md`, the deployment doc §14). Read those for the "why."

## Conventions

Python 3.10+, uv, Hatchling, Litestar route handlers (`get_route_handlers()`), Ruff (line-length
120), mypy `--strict` where deps allow. Google-style docstrings. Copyright header on new files.
