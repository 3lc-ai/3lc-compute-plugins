# 3lc-compute-plugins

The first-party plugins for the [3LC Hub](https://docs.3lc.ai) compute service — the everyday
table tools, bundled as one distribution:

- **Importer** — import CSV, Parquet, COCO, and Hugging Face datasets as tables
- **Exporter** — export tables to CSV, XLSX, YOLO, or COCO
- **Merger** — merge tables
- **Splitter** — create train / validation / test splits
- **Table statistics** — per-column statistics and thumbnails
- **Image metrics** — add image-quality metric columns

## How it's used

You don't install this yourself. The 3LC Hub installs each plugin into its own isolated
environment and runs it for you; the tools then appear in the Hub next to the built-ins.

## License

Apache-2.0. See `LICENSE`.

## Links

- 3LC Hub documentation: <https://docs.3lc.ai>
- Plugin SDK & author guide: <https://3lc-ai.github.io/3lc-compute-plugin-sdk/>
