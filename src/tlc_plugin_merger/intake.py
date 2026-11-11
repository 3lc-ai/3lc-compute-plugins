# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, immutable dataset intake. Shared by MCP jobs and the module UI.

The destination journal lives on the deployment, never on an agent. A transaction
serializes releases across users; deterministic table names make a retry after a
crash recover the same output. Source tables and held-out tables are never edited.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def encoded(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda x: (
            x.decode("ascii") if isinstance(x, bytes) else x.tolist() if hasattr(x, "tolist") else x.item()
        ),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def label_path(schema: dict, path: tuple[str, ...] = ()) -> tuple[str, ...] | None:
    value = schema.get("value", {})
    if value.get("number_role") == "label" and value.get("map"):
        return path
    for key, child in schema.get("values", {}).items():
        found = label_path(child, (*path, key))
        if found is not None:
            return found
    return None


def at(value: dict, path: tuple[str, ...], *, schema: bool = False) -> Any:
    for key in path:
        value = value["values"][key] if schema else value[key]
    return value


def annotation(table: Any, preferred: str = "") -> tuple[str, str, dict, tuple[str, ...]]:
    kinds = {
        "bounding_boxes_2d": "bounding_boxes",
        "segmentation_masks": "instance_masks",
        "semantic_segmentation": "semantic_map",
    }
    found = []
    for name, column in table.rows_schema.values.items():
        schema = column.to_minimal_dict(include_all=False)
        path = label_path(schema)
        kind = kinds.get(getattr(column, "sample_type", ""))
        if path is not None and (kind or path in ((), ("label",))):
            found.append((name, kind or "classification", schema, path))
    if preferred:
        found = [item for item in found if item[0] == preferred]
    if len(found) != 1:
        msg = "Choose a table with one supported annotation column, or select its annotation column explicitly."
        raise ValueError(msg)
    return found[0]


def class_map(schema: dict, path: tuple[str, ...]) -> dict[int, str]:
    result = {int(float(k)): v["internal_name"] for k, v in at(schema, path, schema=True)["value"]["map"].items()}
    if len(set(result.values())) != len(result):
        msg = "Class names are ambiguous. Give each class a unique internal name first."
        raise ValueError(msg)
    return result


def inspect(table_url: str) -> dict[str, Any]:
    import tlc
    from tlc_plugin_sdk.shared.images import get_image_column

    table = tlc.Table.from_url(table_url)
    result = {
        "table_url": str(table.url),
        "rows": len(table),
        "project": table.project_name,
        "dataset": table.dataset_name,
        "image_column": get_image_column(table),
        "columns": list(table.rows_schema.values),
    }
    try:
        column, modality, schema, path = annotation(table)
    except ValueError as error:
        result.update(modality=None, classes=[], annotation_column=None, annotation_note=str(error))
    else:
        result.update(
            modality=modality,
            annotation_column=column,
            classes=[{"id": key, "name": name} for key, name in class_map(schema, path).items()],
        )
        if modality == "semantic_map":
            result["background_id"] = schema.get("metadata", {}).get("semantic_segmentation", {}).get("background", 0)
    return result


def identities(table: Any, identity_column: str = "", group_column: str = "") -> list[tuple[str, str]]:
    """Read encoded images, never decode them. Equal bytes at a new URL remain duplicates."""
    from tlc_plugin_sdk.shared.images import get_image_column, resolve_image_url
    from tlcurl.url_adapters._registry import UrlAdapterRegistry

    image_column = get_image_column(table)
    for name in (identity_column, group_column):
        if name and name not in table.rows_schema.values:
            msg = f"Column {name!r} is missing from {table.name}."
            raise ValueError(msg)
    rows = table.table_rows
    hashes = {}
    if not identity_column:
        addresses = dict.fromkeys(str(resolve_image_url(row[image_column], table.url)) for row in rows)

        def content_hash(address):
            from tlcurl import Url

            return hashlib.sha256(UrlAdapterRegistry.read_binary_content_from_url(Url(address))).hexdigest()

        # Fetch only encoded bytes; overlap bounded storage reads instead of
        # serializing thousands of network round trips or decoding images.
        with ThreadPoolExecutor(max_workers=16) as pool:
            pending = [
                (address, pool.submit(contextvars.copy_context().run, content_hash, address)) for address in addresses
            ]
            hashes = {address: task.result() for address, task in pending}
    result = []
    for row in rows:
        if identity_column:
            value = row[identity_column]
            if value is None or isinstance(value, (list, dict)) or str(value).strip() == "":
                msg = "Sample identifiers must be non-empty scalar values."
                raise ValueError(msg)
            key = digest(["sample", value])
        else:
            url = resolve_image_url(row[image_column], table.url)
            address = str(url)
            key = hashes[address]
        group = row[group_column] if group_column else key
        if group is None or isinstance(group, (list, dict)) or str(group).strip() == "":
            msg = "Every sample needs a non-empty group, such as its video or capture session."
            raise ValueError(msg)
        result.append((key, digest(group)))
    return result


def _name(value: str) -> str:
    if not value.strip() or value in {".", ".."} or any(c in value for c in "/\\%"):
        msg = "Choose a dataset name without a path."
        raise ValueError(msg)
    return value.strip()


def revision(
    source: Any, url: Any, keep: list[int], *, overrides: dict | None = None, edits: dict | None = None, receipt: str
) -> Any:
    import tlc
    from tlc._core.objects.tables.from_table.edited_table import EditedTable
    from tlc.constants import SHOULD_DELETE

    if url.exists():
        existing = tlc.Table.from_url(url)
        if existing.description != receipt or [
            str(parent.to_absolute(existing.url)) for parent in existing.input_tables
        ] != [str(source.url)]:
            msg = "The output name already belongs to another operation. Nothing was overwritten."
            raise ValueError(msg)
        return existing
    edits = dict(edits or {})
    removed = sorted(set(range(len(source))) - set(keep))
    if removed:
        edits[SHOULD_DELETE] = {"runs_and_values": [removed, True]}
    table = EditedTable(
        url=url,
        input_table_url=source,
        input_tables=[source.url],
        edits=edits,
        override_table_rows_schema={"values": overrides} if overrides else None,
        description=receipt,
    )
    table.ensure_fully_defined()
    table.write_to_url()
    return table


def _output(root: str, project: str, dataset: str, name: str) -> Any:
    from tlc.helpers.project_layout import ProjectLayout

    return ProjectLayout.table_url(root_url=root, project_name=project, dataset_name=_name(dataset), table_name=name)


def normalize(source: Any, reference: Any | None, modality: str, url: Any, receipt: str) -> Any:
    """Map verified class names to the reference's exact IDs, never invent a class.

    Refine classification proposals contain label/confidence/description. Their
    reviewed label becomes a native categorical column for training. Geometry
    stays native; only its label IDs change when the reference uses other IDs.
    """
    from tlc_plugin_sdk.shared.images import get_image_column, resolve_image_url

    name, kind, schema, path = annotation(source)
    if kind != modality:
        msg = f"Expected {modality}, but this table contains {kind}."
        raise ValueError(msg)
    target_name, target_schema, target_path = name, schema, path
    if reference is not None:
        target_name, target_kind, target_schema, target_path = annotation(reference)
        if kind != target_kind:
            msg = "The incoming annotations and reference have different modalities."
            raise ValueError(msg)
        if kind == "classification" and target_path:
            msg = "Prepare the reference as a first dataset before extending its compound classification labels."
            raise ValueError(msg)
    elif kind == "classification" and path:
        target_schema, target_path = at(schema, path, schema=True), ()
        # Older native cores cannot replace a struct column with a scalar in
        # place. Add a categorical column and remove the proposal struct.
        target_name = "label"
        while target_name in source.rows_schema.values:
            target_name = "reviewed_" + target_name
    old, new = class_map(schema, path), class_map(target_schema, target_path)
    inverse = {name: key for key, name in new.items()}
    if set(old.values()) != set(new.values()):
        msg = "Class names do not match the reference. Review the class mapping before adding this batch."
        raise ValueError(msg)
    mapping = {key: inverse[name] for key, name in old.items()}
    values = []
    for row in source.table_rows:
        value = copy.deepcopy(row[name])
        labels = at(value, path) if path else value
        try:
            translated = [mapping[int(x)] for x in labels] if isinstance(labels, list) else mapping[int(labels)]
        except (KeyError, TypeError, ValueError):
            msg = "Some rows have no valid reviewed label. Finish reviewing this batch first."
            raise ValueError(msg) from None
        if kind == "classification":
            value = translated
        else:
            parent = at(value, path[:-1])
            parent[path[-1]] = translated
        values.append(json.loads(encoded(value)))
    # Refine marks partially published outputs. Empty placeholders must not
    # silently become negative training samples.
    coverage_columns = []
    for column, column_schema in source.rows_schema.values.items():
        metadata = getattr(column_schema, "metadata", None) or {}
        if metadata.get("refine_output_coverage", {}).get("column") == name:
            coverage_columns.append(column)
            if not all(row[column] is True for row in source.table_rows):
                msg = "This batch contains unreviewed or unpublished rows. Complete them before adding it."
                raise ValueError(msg)
    image = get_image_column(source)
    target_image = get_image_column(reference) if reference is not None else image
    overrides = {
        target_name: target_schema,
        target_image: source.rows_schema.values[image].to_minimal_dict(include_all=False),
        # These receipts belong to the reviewed source, not to the training
        # schema required of every subsequent incoming batch.
        **dict.fromkeys(coverage_columns),
    }
    edits = {
        target_name: {"runs_and_values": [part for i, value in enumerate(values) for part in ([i], value)]},
        target_image: {
            "runs_and_values": [
                part
                for i, row in enumerate(source.table_rows)
                for part in ([i], str(resolve_image_url(row[image], source.url)))
            ]
        },
    }
    if reference is not None:
        # Match the destination's columns; reject missing metadata rather than
        # filling it with arbitrary defaults. The source revision is preserved.
        required = set(reference.rows_schema.values) - {target_name, target_image}
        missing = required - set(source.rows_schema.values)
        weight = reference.weights_column_name
        if weight in missing:
            overrides[weight] = reference.rows_schema.values[weight].to_minimal_dict(include_all=False)
            edits[weight] = (
                {"runs_and_values": [list(range(len(source))), 1.0]} if len(source) else {"runs_and_values": []}
            )
            missing.remove(weight)
        if missing:
            raise ValueError("Incoming data is missing reference columns: " + ", ".join(sorted(missing)))
        for column in required:
            if column not in source.rows_schema.values:
                continue
            if source.rows_schema.values[column].to_minimal_dict(include_all=False) != reference.rows_schema.values[
                column
            ].to_minimal_dict(include_all=False):
                msg = f"Column {column!r} is incompatible with the reference."
                raise ValueError(msg)
        for column in set(source.rows_schema.values) - set(reference.rows_schema.values):
            overrides[column] = None
    elif target_name != name:
        overrides[name] = None
    if kind == "classification" and path and target_name == name:
        source = revision(
            source,
            url.create_sibling(url.parts[-1] + "-input"),
            list(range(len(source))),
            overrides={name: None},
            receipt=receipt + " scalar label input",
        )
    return revision(source, url, list(range(len(source))), overrides=overrides, edits=edits, receipt=receipt)


@contextmanager
def journal(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=120)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS destinations (key TEXT PRIMARY KEY, base TEXT, head TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS receipts (key TEXT PRIMARY KEY, result TEXT)")
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    finally:
        db.close()


def execute(data: dict, *, root: str, state: Path, tenant: str, cancelled=lambda: False) -> dict:
    import tlc
    from tlc_plugin_sdk.job_contract import normalize_configuration

    from .mcp_tools import tools

    mode = data.get("operation")
    specification = next((tool for tool in tools() if tool["name"] == mode and "job" in tool), None)
    if specification is None:
        msg = "Unknown dataset intake operation."
        raise ValueError(msg)
    schema = specification["input_schema"]
    # Host-owned context accompanies native job parameters. Validate the same
    # public fields whether this job came from MCP or the native Run endpoint.
    parameters = {key: value for key, value in data.items() if key in schema["properties"]}
    normalize_configuration(parameters, schema)

    source = tlc.Table.from_url(data["table_url"])
    project = data.get("project_name") or source.project_name
    if project != source.project_name:
        msg = "Incoming data must belong to the selected project."
        raise ValueError(msg)
    dataset = _name(data["dataset_name"])
    base_url = data.get("destination_table") or ""
    reference_url = data.get("reference_table") or base_url
    heldout_urls = data.get("heldout_tables") or []
    identity_column, group_column = data.get("identity_column", ""), data.get("group_column", "")
    key = digest([
        tenant,
        root,
        project,
        dataset,
        (data.get("batch_url") or str(source.url)) if data.get("keep_separate") else "",
    ])
    policy = {
        k: data.get(k)
        for k in (
            "destination_table",
            "reference_table",
            "modality",
            "identity_column",
            "group_column",
            "heldout_tables",
            "expected_classes",
            "keep_separate",
        )
    }
    batch = data.get("batch_url") or str(source.url)
    receipt_key = digest([key, batch, policy])
    # Separate destinations can prepare concurrently; only writers extending
    # the same dataset share a transaction and its current head.
    with journal(state.with_name("intake-" + key + ".sqlite3")) as db:
        saved = db.execute("SELECT result FROM receipts WHERE key=?", (receipt_key,)).fetchone()
        if saved and mode == "intake_publish":
            return {**json.loads(saved[0]), "reused": True}
        destination = db.execute("SELECT base, head FROM destinations WHERE key=?", (key,)).fetchone()
        if destination and destination[0] != base_url:
            msg = (
                "This output dataset already has an intake using another starting table. "
                "Choose a separate output dataset."
            )
            raise ValueError(msg)
        head_url = destination[1] if destination else base_url
        head = tlc.Table.from_url(head_url) if head_url else None
        reference = tlc.Table.from_url(reference_url) if reference_url else head
        heldout = [tlc.Table.from_url(url) for url in heldout_urls]
        for table in [head, reference, *heldout]:
            if table is not None and table.project_name != project:
                msg = "Reference, destination and held-out tables must be in this project."
                raise ValueError(msg)
        expected = data.get("expected_classes")
        if expected and (reference is not None or mode == "intake_publish" or data.get("labeled")):
            _, kind, schema, path = annotation(reference if reference is not None else source)
            names = class_map(schema, path)
            if kind == "semantic_map":
                names.pop(schema.get("metadata", {}).get("semantic_segmentation", {}).get("background", 0), None)
            if kind != data["modality"] or set(names.values()) != set(expected):
                msg = "The modality or class names differ from the approved setup. Review the schema before continuing."
                raise ValueError(msg)
        working = source
        annotated = mode == "intake_publish" or bool(data.get("labeled"))
        if annotated:
            working = normalize(
                source,
                head if head is not None else reference,
                data["modality"],
                _output(root, project, dataset, "normalized-" + digest([receipt_key, str(source.url)])[:24]),
                "Intake normalization " + receipt_key,
            )
        identities_in = identities(working, identity_column, group_column)
        protected = [pair for table in heldout for pair in identities(table, identity_column, group_column)]
        excluded = {key for key, _ in protected}
        excluded_groups = {group for _, group in protected} if group_column else set()
        if any(key in excluded or group in excluded_groups for key, group in identities_in):
            msg = (
                "This batch overlaps a protected validation/test sample or group. "
                "Separate that overlap before adding it."
            )
            raise ValueError(msg)
        previous = identities(head, identity_column, group_column) if head is not None else []
        known = {key: index for index, (key, _) in enumerate(previous)}
        annotations = {}
        source_column = annotation(working)[0] if annotated else None
        head_column = annotation(head)[0] if head is not None and annotated else None
        keep = []
        for index, (identity, _) in enumerate(identities_in):
            value = encoded(working.table_rows[index][source_column]) if source_column else None
            if identity in known:
                if head_column and value != encoded(head.table_rows[known[identity]][head_column]):
                    msg = "A duplicate image has different annotations. Resolve that conflict in Refine before merging."
                    raise ValueError(msg)
                continue
            if identity in annotations:
                if value != annotations[identity]:
                    msg = "This batch repeats an image with conflicting annotations."
                    raise ValueError(msg)
                continue
            annotations[identity] = value
            keep.append(index)
        if len(keep) > int(data.get("max_rows", 10000)):
            msg = "This batch exceeds the approved row limit. Increase the limit or send a smaller batch."
            raise ValueError(msg)
        if cancelled():
            msg = "Dataset intake was cancelled before publication."
            raise ValueError(msg)
        stage = "prepared" if mode == "intake_prepare" else "batch"
        filtered = revision(
            working,
            _output(root, project, dataset, stage + "-" + digest([receipt_key, str(working.url), keep])[:24]),
            keep,
            receipt=f"Intake {stage} {receipt_key}",
        )
        result = {
            "table_url": str(filtered.url),
            "rows": len(keep),
            "duplicates": len(source) - len(keep),
            "batch_url": batch,
            "destination_table": head_url,
            "reused": False,
        }
        if mode == "intake_prepare":
            return result
        if head is not None and keep:
            output = _output(root, project, dataset, "release-" + digest([receipt_key, head_url])[:24])
            if output.exists():
                merged = tlc.Table.from_url(output)
                if merged.description != "Intake release " + receipt_key:
                    msg = "An unrelated table uses this release name. Nothing was overwritten."
                    raise ValueError(msg)
            else:
                merged = tlc.Table.join_tables(
                    [head, filtered],
                    table_url=output,
                    if_exists="raise",
                    description="Intake release " + receipt_key,
                    add_weight_column=False,
                )
            result["table_url"] = str(merged.url)
        elif head is not None:
            result["table_url"] = str(head.url)
        db.execute("INSERT OR REPLACE INTO destinations VALUES (?,?,?)", (key, base_url, result["table_url"]))
        db.execute("INSERT INTO receipts VALUES (?,?)", (receipt_key, encoded(result)))
        return result
