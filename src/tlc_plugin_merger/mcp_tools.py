# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Typed native Merge operations; no generated scripts or UI automation."""


def tools():
    text = {"type": "string", "minLength": 1, "maxLength": 4096}
    fields = {
        "keep_separate": {"type": "boolean", "default": False},
        "labeled": {"type": "boolean", "default": False},
        "node_id": {"type": "string", "const": "local", "default": "local"},
        "expected_classes": {"type": "array", "items": {"type": "string"}, "maxItems": 1000},
        "table_url": text,
        "dataset_name": {"type": "string", "minLength": 1, "maxLength": 200},
        "destination_table": text,
        "reference_table": text,
        "batch_url": text,
        "modality": {"enum": ["classification", "bounding_boxes", "instance_masks", "semantic_map"]},
        "identity_column": {"type": "string", "maxLength": 200},
        "group_column": {"type": "string", "maxLength": 200},
        "heldout_tables": {"type": "array", "maxItems": 20, "items": text},
        "max_rows": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 10000},
        "project_name": {"type": "string", "maxLength": 200},
    }
    result = [
        {
            "name": "inspect_incoming",
            "title": "Inspect incoming data",
            "description": (
                "Read a table's modality, exact class IDs, image and annotation columns, "
                "and row count. No AI or writes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"table_url": text},
                "required": ["table_url"],
                "additionalProperties": False,
            },
            "output_schema": {"type": "object"},
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": True,
            },
            "route": {"method": "GET", "path": "/intake/inspect", "query": ["table_url"]},
        }
    ]
    for name, title, description in [
        (
            "intake_prepare",
            "Check and prepare an incoming batch",
            (
                "Check sample/group overlap with held-out data and skip images already in the destination. "
                "Publish a new filtered native table for labeling; originals unchanged."
            ),
        ),
        (
            "intake_publish",
            "Add reviewed data to a dataset",
            (
                "Validate native annotations and exact reference classes; preserve reference class IDs, "
                "deduplicate and publish an immutable dataset release. Existing validation/test tables "
                "are never changed. The deployment serializes concurrent batches and reuses completed receipts."
            ),
        ),
    ]:
        result.append({
            "name": name,
            "title": title,
            "description": description
            + (
                " Run on node_id=local: the destination journal belongs to this deployment. "
                "For retries keep batch_url as the original incoming table. "
                "Do not share an intake destination between deployments."
            ),
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": fields,
                "required": ["table_url", "dataset_name", "modality"],
            },
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": True,
            },
            "job": {
                "body": {"operation": name, **{key: "{{" + key + "}}" for key in fields if key != "node_id"}},
                "outputs": ["table_url", "intake"],
                "wait": True,
            },
        })
    return result


if __name__ == "__main__":
    import json
    from pathlib import Path

    Path(__file__).with_name("mcp-tools.json").write_text(json.dumps(tools(), indent=2) + "\n")
