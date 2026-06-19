# 3lc-compute-plugins

The open, first-party plugins for the [3LC](https://3lc.ai) compute service. A monorepo
of standalone plugin packages, each built against the public
[`3lc-plugin-sdk`](https://github.com/3lc-ai/3lc-plugin-sdk).

```
plugins/
  timm/          # 3lc-plugin-timm — fine-tune timm image classifiers (venv, GPU)
  …              # (sam3, yolo, importer, exporter, merger, splitter,
                 #  table_statistics, image_metrics migrate here next)
```

Each `plugins/<name>/` is its own package: a `pyproject.toml` (distribution `3lc-plugin-<name>`,
import `tlc_plugin_<name>`), a `plugin.toml` manifest the host reads **without importing**, and the
code under `src/tlc_plugin_<name>/`.

## Isolation tiers

A plugin declares `runtime.isolation` in its manifest:

- **`venv`** (e.g. timm, sam3, yolo) — heavy/conflicting deps; the host provisions a dedicated venv
  from the package and runs it out-of-process. Its deps never touch the host venv.
- **`host`** (e.g. importer, exporter, the light ones) — deps are light and compatible with the host
  venv; runs in-process. Delivered as a light dependency of the host's `[plugins]` extra.

The rule: **anything installed into the host venv must be light.** Heavy plugins are always `venv`.

## Develop

Each plugin is provisioned/run by the compute service. To work on one directly:

```bash
cd plugins/timm
uv sync          # builds this plugin's venv (SDK + its own deps)
uv run ruff check .
```

During the build-out, `3lc-plugin-sdk` is resolved from a sibling checkout via a **dev-only**
`[tool.uv.sources]` path — see `CLAUDE.md`. That reverts to an index/git pin before publish.

## Status

Pre-1.0. License pending (see `CLAUDE.md`). The proprietary `run_insights` / `table_insights`
plugins deliberately stay in the private `3lc-insights` monorepo and are **not** here.
