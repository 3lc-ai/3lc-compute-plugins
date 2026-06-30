# 3lc-compute-plugins

The open, first-party plugins for the [3LC](https://3lc.ai) compute service — one distribution,
`3lc-compute-plugins`, built against the public
[`3lc-plugin-sdk`](https://github.com/3lc-ai/3lc-plugin-sdk).

```
pyproject.toml         # one distribution: the SDK floor + one extra per plugin + entry-points
src/
  tlc_plugin_importer/          # import CSV/Parquet/COCO/…            ([importer] extra)
  tlc_plugin_exporter/          # export CSV/XLSX/YOLO/COCO/…          ([exporter] extra)
  tlc_plugin_merger/            # merge two tables                     ([merger] extra — empty)
  tlc_plugin_splitter/          # train/val/test splits                ([splitter] extra)
  tlc_plugin_table_statistics/  # per-column stats + thumbnails        ([table_statistics] extra)
  tlc_plugin_image_metrics/     # image-quality metric columns         ([image_metrics] extra)
```

Each `src/tlc_plugin_<name>/` bundles a `plugin.toml` manifest that the host reads **without
importing** the plugin, plus the plugin's UI fragment and code.

## Isolation

Every plugin is **`venv`-isolated** (`runtime.isolation = "venv"`): the host provisions a dedicated
venv per plugin — installing only that plugin's extra (`uv sync --extra <id>`) — and runs it
out-of-process. No plugin dependency is installed into the host venv; the base distribution carries
only the SDK floor (`3lc-plugin-sdk[shared]`).

## Consumed by the compute service

The host loads these plugins via a **folder Source** (a directory it scans — `src/`), an **editable
install**, or the **install API** (which materializes a per-plugin venv from a wheel or git ref). It
never installs this distribution into its own venv.

## Develop

```bash
uv sync                       # SDK floor only
uv sync --extra importer      # one plugin's deps (what the host provisions into its venv)
uv run ruff check .
```

`3lc-plugin-sdk` is resolved from a sibling checkout via a dev-only `[tool.uv.sources]` path — see
`CLAUDE.md`.

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
