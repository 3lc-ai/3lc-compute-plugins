# 3lc-compute-plugins — agent & contributor orientation

This repo is the **public, first-party plugin collection** for the 3LC compute service. Each
`plugins/<name>/` is a standalone package built against the public
[`3lc-plugin-sdk`](https://github.com/3lc-ai/3lc-plugin-sdk). Extracted from the private
`3lc-insights` monorepo (where the host runtime + the design docs live).

## Layout (per plugin)

```
plugins/<name>/
  plugin.toml              # manifest — host reads it WITHOUT importing (id, ui, runtime.*)
  pyproject.toml           # distribution 3lc-plugin-<name>; deps = 3lc-plugin-sdk + this plugin's own
  src/tlc_plugin_<name>/   # the package (import name tlc_plugin_<name>); data files (ui.html) live here
```

`plugin.toml` `runtime.entrypoint = "tlc_plugin_<name>:SomePlugin"` is resolved by the worker from
the provisioned venv's site-packages — these are **real installable packages**, not the old flat
`package = false` + cwd-on-sys.path form (that constraint is gone once the code is a proper package).

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

Each plugin versions independently (its `pyproject.toml` / `plugin.toml` `version`). It pins the
contract via `3lc-plugin-sdk>=X,<Y` and declares `min_service_version` / `min_frontend_version`
floors in its manifest. SemVer.

## Dev setup (build-out)

`3lc-plugin-sdk` is unpublished during the build-out, so each plugin resolves it from a sibling
checkout via a **dev-only** `[tool.uv.sources] 3lc-plugin-sdk = { path = "../../../3lc-plugin-sdk" }`.
torch comes from the cu126 index; 3lc from the 3lc-releases index. These dev sources revert to
index/git pins before any real publish.

```bash
cd plugins/<name>
uv sync                 # provision this plugin's venv (SDK + its deps)
uv run ruff check .
```

## Where the rest of the context lives

The plugin contract, the author guide, and the architecture/distribution design docs live in
`3lc-plugin-sdk` (`docs/plugin-guide.md`, `CLAUDE.md`) and the private `3lc-insights` monorepo
(`docs/plugin-*.md`, the deployment doc §14). Read those for the "why."

## Conventions

Python 3.10+, uv, Hatchling, Litestar route handlers (`get_route_handlers()`), Ruff (line-length
120), mypy `--strict` where deps allow. Google-style docstrings. Copyright header on new files.
