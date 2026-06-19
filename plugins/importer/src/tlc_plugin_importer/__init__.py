# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Import plugin — create 3LC tables from YOLO, COCO, Image Folder, Unlabeled, and CSV/Excel formats."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tlc_plugin_importer import routes as _routes
from tlc_plugin_sdk import ComputePlugin

if TYPE_CHECKING:
    from tlc_plugin_sdk.job_context import JobContext

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

MAX_PREVIEW_ROWS = 50
MAX_UNIQUE_VALUES = 200

# Characters forbidden in tlc MapElement names
_INVALID_LABEL_CHARS = str.maketrans(dict.fromkeys(r"""<>\|.:"'?*&""", "_"))


def _sanitize_label(s: str) -> str:
    """Strip characters that tlc MapElement names do not allow."""
    return s.translate(_INVALID_LABEL_CHARS)


# ---------------------------------------------------------------------------
# CSV / Excel parsing utilities
# ---------------------------------------------------------------------------


def _read_spreadsheet(file_bytes: bytes, filename: str) -> tuple[list[str], list[list[Any]]]:
    """Read a CSV or Excel file and return (headers, rows).

    Each row is a list of raw string/numeric values.
    """
    ext = Path(filename).suffix.lower()
    if ext in (".xlsx", ".xls"):
        try:
            import openpyxl

            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
            ws = wb.active
            if ws is None:
                return [], []
            rows_iter = ws.iter_rows(values_only=True)
            header_row = next(rows_iter, None)
            if header_row is None:
                return [], []
            headers = [str(h or f"col_{i}") for i, h in enumerate(header_row)]
            data_rows = []
            for row in rows_iter:
                data_rows.append(list(row))
            wb.close()
            return headers, data_rows
        except ImportError:
            msg = "openpyxl is required to read Excel files. Install with: pip install openpyxl"
            raise ValueError(msg)
    else:
        # CSV
        text = file_bytes.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        header_row = next(reader, None)
        if header_row is None:
            return [], []
        headers = [h.strip() or f"col_{i}" for i, h in enumerate(header_row)]
        data_rows = []
        for row in reader:
            data_rows.append(list(row))
        return headers, data_rows


def _infer_column_type(values: list[Any]) -> str:
    """Infer column type from values: 'float', 'int', or 'string'."""
    has_float = False
    has_int = False
    for v in values:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue
        if isinstance(v, float):
            has_float = True
        elif isinstance(v, int) and not isinstance(v, bool):
            has_int = True
        elif isinstance(v, str):
            s = v.strip()
            try:
                float(s)
                if "." in s:
                    has_float = True
                else:
                    has_int = True
            except ValueError:
                return "string"
        else:
            return "string"
    if has_float:
        return "float"
    if has_int:
        return "int"
    return "string"


def _unique_values(values: list[Any], max_count: int = MAX_UNIQUE_VALUES) -> list[Any]:
    """Get sorted unique non-empty values, capped at max_count."""
    seen: set[str] = set()
    uniques: list[Any] = []
    for v in values:
        s = str(v).strip() if v is not None else ""
        if s == "" or s in seen:
            continue
        seen.add(s)
        uniques.append(s)
        if len(uniques) >= max_count:
            break
    return sorted(uniques)


def _parse_csv_file(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """Parse a CSV/Excel file and return column metadata + preview data."""
    headers, rows = _read_spreadsheet(file_bytes, filename)
    if not headers:
        return {"error": "No data found in file", "columns": [], "preview": [], "total_rows": 0}

    # Build column-wise value lists
    col_values: dict[int, list[Any]] = {i: [] for i in range(len(headers))}
    for row in rows:
        for i in range(len(headers)):
            val = row[i] if i < len(row) else None
            col_values[i].append(val)

    columns = []
    for i, header in enumerate(headers):
        dtype = _infer_column_type(col_values[i])
        uniques = _unique_values(col_values[i])
        columns.append({
            "index": i,
            "name": header,
            "inferred_type": dtype,
            "unique_count": len(uniques),
            "unique_values": uniques if len(uniques) <= MAX_UNIQUE_VALUES else uniques[:MAX_UNIQUE_VALUES],
            "sample_values": [str(v) if v is not None else "" for v in col_values[i][:5]],
        })

    # Preview rows (first N)
    preview = []
    for row in rows[:MAX_PREVIEW_ROWS]:
        preview.append([str(v) if v is not None else "" for v in row])

    return {
        "columns": columns,
        "preview": preview,
        "headers": headers,
        "total_rows": len(rows),
    }


# ---------------------------------------------------------------------------
# CSV/Excel execution logic
# ---------------------------------------------------------------------------

# In-memory store for parsed CSV file data (keyed by session token)
_parsed_csv_files: dict[str, dict[str, Any]] = {}


def _detect_common_folder(paths: list[str]) -> str:
    """Find the longest common directory prefix of a list of file paths."""
    if not paths:
        return ""
    import os

    dirs = [os.path.dirname(p) for p in paths if p and ("/" in p or "\\" in p)]
    if not dirs:
        return ""
    common = os.path.commonpath(dirs) if len(dirs) > 1 else dirs[0]
    return common


def _execute_csv_new(
    file_bytes: bytes,
    filename: str,
    selected_columns: list[dict[str, Any]],
    project_name: str,
    dataset_name: str,
    table_name: str,
    description: str,
    alias_enabled: bool = True,
    alias_token: str = "",
) -> dict[str, Any]:
    """Create a brand new 3LC table from CSV/Excel columns using TableWriter."""
    import tlc

    _headers, csv_rows = _read_spreadsheet(file_bytes, filename)
    if not csv_rows:
        return {"success": False, "message": "No data rows found in file.", "table_url": None, "details": {}}

    # Build column schemas
    col_schemas: dict[str, Any] = {}
    for col_cfg in selected_columns:
        col_name = col_cfg["name"]
        col_type = col_cfg.get("type", "string")
        is_categorical = col_cfg.get("categorical", False)
        idx = col_cfg["index"]

        raw_values = [row[idx] if idx < len(row) else None for row in csv_rows]

        if col_type == "image_url":
            col_schemas[col_name] = tlc.schemas.ImageSchema(sample_type="url")
        elif is_categorical and col_type == "string":
            unique_strings = sorted({str(v).strip() for v in raw_values if v is not None and str(v).strip()})
            sanitized_classes = [_sanitize_label(s) for s in unique_strings]
            col_schemas[col_name] = tlc.schemas.CategoricalLabelSchema(classes=sanitized_classes)
        elif is_categorical and col_type == "int":
            labels = col_cfg.get("labels", {})
            all_ints: set[int] = set()
            for v in raw_values:
                try:
                    iv = int(float(str(v).strip())) if v is not None and str(v).strip() else 0
                except (ValueError, TypeError):
                    iv = 0
                all_ints.add(iv)
            classes = []
            for i in sorted(all_ints):
                label = _sanitize_label(labels.get(str(i), f"class_{i}"))
                classes.append(label)
            col_schemas[col_name] = tlc.schemas.CategoricalLabelSchema(classes=classes)
        elif col_type == "float":
            col_schemas[col_name] = tlc.schemas.Float32Schema()
        elif col_type == "int":
            col_schemas[col_name] = tlc.schemas.Int32Schema()
        else:
            col_schemas[col_name] = tlc.schemas.StringSchema()

    # Pre-compute string-to-int mappings for categorical string columns
    _str_maps: dict[str, dict[str, int]] = {}
    for col_cfg in selected_columns:
        if col_cfg.get("type") == "string" and col_cfg.get("categorical"):
            idx = col_cfg["index"]
            unique_strings = sorted({
                str(v).strip()
                for v in (r[idx] if idx < len(r) else None for r in csv_rows)
                if v is not None and str(v).strip()
            })
            _str_maps[col_cfg["name"]] = {s: i for i, s in enumerate(unique_strings)}

    writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=dataset_name,
        project_name=project_name,
        description=description or "",
        schema=col_schemas,
    )

    # Add rows
    for csv_row in csv_rows:
        row_data: dict[str, Any] = {}
        for col_cfg in selected_columns:
            idx = col_cfg["index"]
            col_name = col_cfg["name"]
            col_type = col_cfg.get("type", "string")
            is_categorical = col_cfg.get("categorical", False)
            raw_val = csv_row[idx] if idx < len(csv_row) else None

            if col_type == "image_url":
                row_data[col_name] = str(raw_val).strip() if raw_val is not None else ""
            elif is_categorical and col_type == "string":
                s = str(raw_val).strip() if raw_val is not None else ""
                row_data[col_name] = _str_maps[col_name].get(s, 0)
            elif is_categorical and col_type == "int":
                try:
                    row_data[col_name] = (
                        int(float(str(raw_val).strip())) if raw_val is not None and str(raw_val).strip() else 0
                    )
                except (ValueError, TypeError):
                    row_data[col_name] = 0
            elif col_type == "float":
                try:
                    row_data[col_name] = (
                        float(str(raw_val).strip()) if raw_val is not None and str(raw_val).strip() else 0.0
                    )
                except (ValueError, TypeError):
                    row_data[col_name] = 0.0
            elif col_type == "int":
                try:
                    row_data[col_name] = (
                        int(float(str(raw_val).strip())) if raw_val is not None and str(raw_val).strip() else 0
                    )
                except (ValueError, TypeError):
                    row_data[col_name] = 0
            else:
                row_data[col_name] = str(raw_val).strip() if raw_val is not None else ""

        writer.add_row(row_data)

    table = writer.finalize()

    # Register alias if image columns exist and alias is enabled
    if alias_enabled:
        image_cols = [c for c in selected_columns if c.get("type") == "image_url"]
        if image_cols:
            # Detect common image folder from first image column values
            first_img_col_idx = image_cols[0]["index"]
            img_paths = [
                str(row[first_img_col_idx]).strip()
                for row in csv_rows
                if first_img_col_idx < len(row) and row[first_img_col_idx]
            ]
            image_folder = _detect_common_folder(img_paths)
            if image_folder:
                from tlc_plugin_sdk.shared.aliases import default_alias_token, register_alias

                token = alias_token or default_alias_token(project_name)
                register_alias(project_name=project_name, image_folder=image_folder, alias_token=token)

    return {
        "success": True,
        "message": f"Created table '{table_name}' with {len(csv_rows)} rows and {len(selected_columns)} column(s).",
        "table_url": str(table.url),
        "project_name": project_name,
        "dataset_name": dataset_name,
        "details": {
            "row_count": len(csv_rows),
            "columns": [c["name"] for c in selected_columns],
            "project": project_name,
            "dataset": dataset_name,
            "table": table_name,
        },
    }


def _execute_csv_extend(
    table_url: str,
    file_bytes: bytes,
    filename: str,
    selected_columns: list[dict[str, Any]],
    output_table_name: str,
    description: str,
) -> dict[str, Any]:
    """Extend an existing 3LC table with columns from a CSV/Excel file."""
    import tlc

    # Load source table
    source_table = tlc.Table.from_url(table_url)
    source_row_count = source_table.row_count

    # Extract project/dataset names from URL
    url_str = str(source_table.url)
    url_parts = url_str.replace("\\", "/").split("/")
    _src_project = _src_dataset = _src_table_name = source_table.name
    for i, part in enumerate(url_parts):
        if part == "projects" and i + 1 < len(url_parts):
            _src_project = url_parts[i + 1]
        elif part == "datasets" and i + 1 < len(url_parts):
            _src_dataset = url_parts[i + 1]
            if _src_project == source_table.name and i >= 1:
                _src_project = url_parts[i - 1]
        elif part == "tables" and i + 1 < len(url_parts):
            _src_table_name = url_parts[i + 1]

    # Parse the spreadsheet
    headers, csv_rows = _read_spreadsheet(file_bytes, filename)
    csv_row_count = len(csv_rows)

    # Adjust CSV to match source table row count
    if csv_row_count > source_row_count:
        csv_rows = csv_rows[:source_row_count]
    elif csv_row_count < source_row_count:
        num_cols = len(headers) if headers else 0
        empty_row = [None] * num_cols
        csv_rows.extend([list(empty_row) for _ in range(source_row_count - csv_row_count)])

    # Build column schemas and row data
    new_schemas: dict[str, Any] = {}
    new_col_data: dict[str, list[Any]] = {}

    for col_cfg in selected_columns:
        idx = col_cfg["index"]
        col_name = col_cfg["name"]
        col_type = col_cfg.get("type", "string")
        is_categorical = col_cfg.get("categorical", False)

        raw_values = [row[idx] if idx < len(row) else None for row in csv_rows]

        if col_type == "image_url":
            str_values = [str(v).strip() if v is not None else "" for v in raw_values]
            new_schemas[col_name] = tlc.schemas.ImageSchema(sample_type="url")
            new_col_data[col_name] = str_values

        elif is_categorical and col_type == "string":
            unique_strings = sorted({str(v).strip() for v in raw_values if v is not None and str(v).strip()})
            string_to_int = {s: i for i, s in enumerate(unique_strings)}
            sanitized_classes = [_sanitize_label(s) for s in unique_strings]
            new_schemas[col_name] = tlc.schemas.CategoricalLabelSchema(classes=sanitized_classes)
            int_values = []
            for v in raw_values:
                s = str(v).strip() if v is not None else ""
                int_values.append(string_to_int.get(s, 0))
            new_col_data[col_name] = int_values

        elif is_categorical and col_type == "int":
            labels = col_cfg.get("labels", {})
            int_values_list: list[int] = []
            all_ints: set[int] = set()
            for v in raw_values:
                try:
                    iv = int(float(str(v).strip())) if v is not None and str(v).strip() else 0
                except (ValueError, TypeError):
                    iv = 0
                int_values_list.append(iv)
                all_ints.add(iv)
            classes = []
            for i in sorted(all_ints):
                label = _sanitize_label(labels.get(str(i), f"class_{i}"))
                classes.append(label)
            new_schemas[col_name] = tlc.schemas.CategoricalLabelSchema(classes=classes)
            new_col_data[col_name] = int_values_list

        elif col_type == "float":
            float_values = []
            for v in raw_values:
                try:
                    float_values.append(float(str(v).strip()) if v is not None and str(v).strip() else 0.0)
                except (ValueError, TypeError):
                    float_values.append(0.0)
            new_schemas[col_name] = tlc.schemas.Float32Schema()
            new_col_data[col_name] = float_values

        elif col_type == "int":
            int_values_plain: list[int] = []
            for v in raw_values:
                try:
                    int_values_plain.append(int(float(str(v).strip())) if v is not None and str(v).strip() else 0)
                except (ValueError, TypeError):
                    int_values_plain.append(0)
            new_schemas[col_name] = tlc.schemas.Int32Schema()
            new_col_data[col_name] = int_values_plain

        else:
            str_values = [str(v).strip() if v is not None else "" for v in raw_values]
            new_schemas[col_name] = tlc.schemas.StringSchema()
            new_col_data[col_name] = str_values

    # Build a single EditedTable with all new columns
    from tlc._core.objects.tables.from_table.edited_table import EditedTable

    edits: dict[str, Any] = {}
    for col_name_key, values in new_col_data.items():
        runs_and_values: list[Any] = []
        for row_idx, val in enumerate(values):
            runs_and_values.append([row_idx])
            runs_and_values.append(val)
        edits[col_name_key] = {"runs_and_values": runs_and_values}

    base_url = str(source_table.url).rsplit("/", 1)[0]
    new_table = EditedTable(
        input_table_url=source_table,
        override_table_rows_schema={"values": new_schemas},
        edits=edits,
        description=description or f"Extended from {_src_table_name}",
        url=tlc.Url(f"{base_url}/{output_table_name}"),
    )
    new_table.ensure_fully_defined()

    return {
        "success": True,
        "message": (
            f"Created table '{output_table_name}' with {source_row_count} rows "
            f"and {len(selected_columns)} new column(s)."
        ),
        "table_url": str(new_table.url),
        "project_name": _src_project,
        "dataset_name": _src_dataset,
        "details": {
            "row_count": source_row_count,
            "new_columns": [c["name"] for c in selected_columns],
            "project": _src_project,
            "dataset": _src_dataset,
            "table": output_table_name,
        },
    }


# ---------------------------------------------------------------------------
# Import step definitions (declarative form metadata + execute logic)
# ---------------------------------------------------------------------------

IMPORT_STEPS: dict[str, dict[str, Any]] = {
    "yolo": {
        "name": "yolo",
        "display_name": "YOLO Dataset",
        "description": "Import a YOLO-format dataset from a dataset.yaml file (detection, segmentation, pose, OBB).",
        "icon": "⊡",
        "form_fields": [
            {
                "id": "dataset_yaml",
                "label": "Dataset YAML Path",
                "type": "text",
                "placeholder": "/path/to/dataset.yaml",
                "required": True,
                "help": "Path to the YOLO dataset YAML file (e.g. coco128.yaml).",
            },
            {
                "id": "split",
                "label": "Split",
                "type": "select",
                "required": True,
                "options": [
                    {"value": "train", "label": "Train"},
                    {"value": "val", "label": "Validation"},
                    {"value": "test", "label": "Test"},
                ],
                "help": "Which split to import.",
            },
            {
                "id": "task",
                "label": "Task Type",
                "type": "select",
                "required": True,
                "options": [
                    {"value": "detection", "label": "Object Detection"},
                    {"value": "segmentation", "label": "Segmentation"},
                    {"value": "pose", "label": "Pose / Keypoints"},
                    {"value": "obb", "label": "Oriented Bounding Boxes (OBB)"},
                ],
                "help": "YOLO task type.",
            },
            {
                "id": "project_name",
                "label": "Project Name",
                "type": "text",
                "placeholder": "my-project",
                "required": True,
                "help": "3LC project to create or add to.",
            },
            {
                "id": "dataset_name",
                "label": "Dataset Name",
                "type": "text",
                "placeholder": "coco128",
                "required": True,
                "help": "Name for the dataset. Import each split (train, val, test) as a separate dataset.",
            },
            {
                "id": "table_name",
                "label": "Table Name",
                "type": "text",
                "placeholder": "initial",
                "required": False,
                "help": "Table revision name. Leave blank for 'initial'.",
            },
            {
                "id": "description",
                "label": "Description",
                "type": "textarea",
                "placeholder": "Optional description of this dataset...",
                "required": False,
                "help": "Human-readable description.",
            },
        ],
    },
    "coco": {
        "name": "coco",
        "display_name": "COCO Dataset",
        "description": "Import a COCO-format dataset from an annotations JSON file or folder.",
        "icon": "⊡",
        "form_fields": [
            {
                "id": "annotations_file",
                "label": "Annotations Path",
                "type": "text",
                "placeholder": "/path/to/annotations/ or .../instances_train2017.json",
                "required": True,
                "help": "Path to a JSON file, or an annotations folder to auto-detect splits.",
            },
            {
                "id": "image_folder",
                "label": "Images Folder Path",
                "type": "text",
                "placeholder": "/path/to/images/train2017/",
                "required": True,
                "help": "Path to the folder containing the images.",
            },
            {
                "id": "task",
                "label": "Task Type",
                "type": "select",
                "required": True,
                "options": [
                    {"value": "detection", "label": "Object Detection"},
                    {"value": "segmentation", "label": "Instance Segmentation"},
                ],
                "help": "Annotation type to use.",
            },
            {
                "id": "project_name",
                "label": "Project Name",
                "type": "text",
                "placeholder": "my-project",
                "required": True,
                "help": "3LC project to create or add to.",
            },
            {
                "id": "dataset_name",
                "label": "Dataset Name",
                "type": "text",
                "placeholder": "coco2017",
                "required": True,
                "help": "Name for the dataset within the project.",
            },
            {
                "id": "table_name",
                "label": "Table Name",
                "type": "text",
                "placeholder": "initial",
                "required": False,
                "help": "Table revision name. Leave blank for 'initial'.",
            },
            {
                "id": "description",
                "label": "Description",
                "type": "textarea",
                "placeholder": "Optional description...",
                "required": False,
                "help": "Human-readable description.",
            },
        ],
    },
    "folder": {
        "name": "folder",
        "display_name": "Image Folder",
        "description": "Import an image folder dataset (ImageNet/ImageFolder style with class subdirectories, or flat folder).",  # noqa: E501
        "icon": "⊡",
        "form_fields": [
            {
                "id": "folder_path",
                "label": "Folder Path",
                "type": "text",
                "placeholder": "/path/to/images/",
                "required": True,
                "help": "Path to folder of images. Use subdirectory-per-class for classification datasets (e.g. images/cat/, images/dog/).",  # noqa: E501
            },
            {
                "id": "project_name",
                "label": "Project Name",
                "type": "text",
                "placeholder": "my-project",
                "required": True,
                "help": "3LC project name.",
            },
            {
                "id": "dataset_name",
                "label": "Dataset Name",
                "type": "text",
                "placeholder": "my-images",
                "required": True,
                "help": "Dataset name within the project.",
            },
            {
                "id": "table_name",
                "label": "Table Name",
                "type": "text",
                "placeholder": "initial",
                "required": False,
                "help": "Table revision name. Leave blank for 'initial'.",
            },
            {
                "id": "description",
                "label": "Description",
                "type": "textarea",
                "placeholder": "Optional description...",
                "required": False,
                "help": "Human-readable description.",
            },
        ],
    },
    "unlabeled": {
        "name": "unlabeled",
        "display_name": "Unlabeled Images",
        "description": "Create a table from an image folder with empty annotations and weight=0 (for labeling later).",
        "icon": "⊡",
        "form_fields": [
            {
                "id": "folder_path",
                "label": "Folder Path",
                "type": "text",
                "placeholder": "/path/to/images/",
                "required": True,
                "help": "Path to a folder of images (scans recursively).",
            },
            {
                "id": "modality",
                "label": "Modality",
                "type": "select",
                "required": True,
                "help": "Determines the annotation columns created in the table.",
                "options": [
                    {"value": "classification", "label": "Classification"},
                    {"value": "detection", "label": "Object Detection (Bounding Boxes)"},
                    {"value": "instance_segmentation", "label": "Instance Segmentation"},
                ],
            },
            {
                "id": "project_name",
                "label": "Project Name",
                "type": "text",
                "placeholder": "my-project",
                "required": True,
                "help": "3LC project name.",
            },
            {
                "id": "dataset_name",
                "label": "Dataset Name",
                "type": "text",
                "placeholder": "unlabeled-images",
                "required": True,
                "help": "Dataset name within the project.",
            },
            {
                "id": "table_name",
                "label": "Table Name",
                "type": "text",
                "placeholder": "initial",
                "required": False,
                "help": "Table revision name. Leave blank for 'initial'.",
            },
            {
                "id": "description",
                "label": "Description",
                "type": "textarea",
                "placeholder": "Optional description...",
                "required": False,
                "help": "Human-readable description.",
            },
        ],
    },
    "csv_detection": {
        "name": "csv_detection",
        "display_name": "CSV Detection",
        "description": "Import bounding box annotations from a CSV with ImageId, class, and bbox columns + an image folder.",  # noqa: E501
        "icon": "⊡",
        "form_fields": [
            {
                "id": "csv_path",
                "label": "CSV File",
                "type": "file_upload",
                "accept": ".csv",
                "placeholder": "/path/to/annotations.csv",
                "required": True,
                "help": "CSV with columns: ImageId, class, bbox (space-separated x0 y0 x1 y1, normalized 0-1).",
            },
            {
                "id": "image_folder",
                "label": "Image Folder Path",
                "type": "text",
                "placeholder": "/path/to/images/",
                "required": True,
                "help": "Folder containing images as {ImageId}.png (or .jpg).",
            },
            {
                "id": "project_name",
                "label": "Project Name",
                "type": "text",
                "placeholder": "my-project",
                "required": True,
                "help": "3LC project to create or add to.",
            },
            {
                "id": "dataset_name",
                "label": "Dataset Name",
                "type": "text",
                "placeholder": "my-detection-data",
                "required": True,
                "help": "Name for the dataset within the project.",
            },
            {
                "id": "table_name",
                "label": "Table Name",
                "type": "text",
                "placeholder": "initial",
                "required": False,
                "help": "Table revision name. Leave blank for 'initial'.",
            },
            {
                "id": "description",
                "label": "Description",
                "type": "textarea",
                "placeholder": "Optional description...",
                "required": False,
                "help": "Human-readable description.",
            },
        ],
    },
    "csv": {
        "name": "csv",
        "display_name": "CSV / Excel",
        "description": "Import from a CSV or Excel file — create a new table or extend an existing one with new columns.",  # noqa: E501
        "icon": "⊡",
        "custom_ui": True,
        "form_fields": [],
    },
}


# ---------------------------------------------------------------------------
# Execution logic (ported from old 3LCDataTools importers)
# ---------------------------------------------------------------------------


def _validate(step_def: dict[str, Any], form_data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate form data against step field definitions."""
    errors: list[str] = []
    for field in step_def["form_fields"]:
        if field.get("required") and not str(form_data.get(field["id"], "")).strip():
            errors.append(f"{field['label']} is required.")
    return len(errors) == 0, errors


def _maybe_register_alias(form_data: dict[str, Any], image_folder: str) -> dict[str, Any] | None:
    """Register a URL alias if the user opted in.

    Reads ``alias_enabled``, ``alias_token``, and ``alias_folder`` from
    *form_data*.  Returns the alias result dict (with ``token`` and ``path``
    keys), or *None* if aliases are disabled.
    """
    if form_data.get("alias_enabled", "true") not in ("true", True, "1"):
        return None

    from tlc_plugin_sdk.shared.aliases import default_alias_token, register_alias

    project_name = form_data["project_name"].strip()
    token = form_data.get("alias_token", "").strip() or default_alias_token(project_name)
    folder = form_data.get("alias_folder", "").strip() or image_folder
    if not folder:
        return None

    return register_alias(project_name=project_name, image_folder=folder, alias_token=token)


def _enhance_error_message(raw: str) -> str:
    """Add actionable guidance to common import error messages."""
    lowered = raw.lower()
    if "already exists" in lowered or "fileexistserror" in lowered or "table already" in lowered:
        return (
            f"Import failed: {raw}\n\n"
            "The table may already exist from a previous (possibly failed) import. "
            "Try changing the dataset name or table name, or delete the existing table first."
        )
    return f"Import failed: {raw}"


def _infer_coco_images_folder(ann_file: Path) -> str:
    """Infer the COCO images folder from an annotation file path.

    Looks for ``images/<split_suffix>`` or ``<split_suffix>/`` next to the
    annotations directory.  E.g. for ``/data/coco/annotations/instances_train2017.json``
    checks ``/data/coco/images/train2017``, then ``/data/coco/train2017``.
    """
    import re

    stem = ann_file.stem.lower()
    parent = ann_file.parent.parent  # go up from annotations/ to dataset root

    match = re.search(r"((?:train|val|validation|test)\d*)", stem)
    if match:
        suffix = match.group(1)
        if suffix.startswith("validation"):
            suffix = "val" + suffix[len("validation") :]
        candidate = parent / "images" / suffix
        if candidate.is_dir():
            return str(candidate)
        candidate2 = parent / suffix
        if candidate2.is_dir():
            return str(candidate2)

    # Fallback: look for any images/ directory
    candidate3 = parent / "images"
    if candidate3.is_dir():
        return str(candidate3)

    return ""


def _parse_coco_folder(annotations_dir: str) -> dict[str, Any]:
    """Scan a COCO annotations path and return available annotation types and splits.

    Accepts either a directory (scans all JSON files) or a single JSON file path
    (returns just that file with inferred images folder).

    Groups JSON files by annotation type (e.g. ``instances``, ``captions``) and
    extracts split names from filenames like ``instances_train2017.json``.

    Returns:
        Dict with ``name`` (base dataset name from parent folder),
        ``types`` (list of annotation type groups, each with ``type``, ``splits``),
        and ``splits`` (flat list for backward compat — splits from the first/default type).

    """
    import re

    ann_path = Path(annotations_dir.strip())

    # Single JSON file — return it with inferred images folder
    if ann_path.is_file() and ann_path.suffix.lower() == ".json":
        images_hint = _infer_coco_images_folder(ann_path)
        stem = ann_path.stem.lower()
        split_kw = ""
        for kw in ("train", "val", "test", "validation"):
            if kw in stem:
                split_kw = "val" if kw == "validation" else kw
                break
        base_name = ann_path.parent.parent.name or ann_path.parent.name
        for _sfx in ("_train", "_val", "_test", "-train", "-val", "-test"):
            if base_name.endswith(_sfx):
                base_name = base_name[: -len(_sfx)]
                break
        return {
            "name": base_name,
            "root": str(ann_path.parent),
            "types": [],
            "default_type": "",
            "splits": [{"split": split_kw, "file": str(ann_path), "images_hint": images_hint}],
            "images_hint": images_hint,
        }

    if not ann_path.is_dir():
        return {"error": f"Not a directory or JSON file: {ann_path}"}

    json_files = sorted(ann_path.glob("*.json"))

    # Roboflow-style layout: subfolders (train/, val/, test/) each with _annotations.coco.json + images.
    # If no JSON at top level, scan one level down for this pattern.
    if not json_files:
        subfolder_splits: list[dict[str, str]] = []
        for kw in ("train", "val", "test"):
            sub = ann_path / kw
            if sub.is_dir():
                ann_file = sub / "_annotations.coco.json"
                if ann_file.is_file():
                    subfolder_splits.append({"split": kw, "file": str(ann_file), "images_hint": str(sub)})
        if subfolder_splits:
            base_name = ann_path.name
            return {
                "name": base_name,
                "root": str(ann_path),
                "types": [],
                "default_type": "",
                "splits": subfolder_splits,
            }
        return {"error": f"No JSON files found in {ann_path}"}

    split_keywords = ("train", "val", "test", "validation")
    split_normalize = {"validation": "val"}
    parent = ann_path.parent  # e.g. /data/coco/ when annotations is /data/coco/annotations/

    # Parse each file into (annotation_type, split, file, images_hint)
    entries: list[tuple[str, str, str, str]] = []
    for jf in json_files:
        stem = jf.stem.lower()
        matched_split = ""
        for kw in split_keywords:
            if kw in stem:
                matched_split = split_normalize.get(kw, kw)
                break

        # Derive annotation type: strip split+year suffix
        ann_type = re.sub(r"[_\-]?(?:train|val|validation|test)\d*", "", jf.stem, flags=re.IGNORECASE).strip("_- ")
        if not ann_type:
            ann_type = jf.stem

        # Infer matching images folder
        images_hint = _infer_coco_images_folder(jf)

        entries.append((ann_type, matched_split, str(jf), images_hint))

    # Group by annotation type
    type_groups: dict[str, list[dict[str, str]]] = {}
    for ann_type, split, file, images_hint in entries:
        type_groups.setdefault(ann_type, []).append({"split": split, "file": file, "images_hint": images_hint})

    # Build types list, prefer "instances" as default
    types_list: list[dict[str, Any]] = []
    for t in sorted(type_groups.keys()):
        types_list.append({"type": t, "splits": type_groups[t]})

    # Base name from parent folder (e.g. "coco" from /datasets/coco/annotations/)
    base_name = parent.name or ann_path.name
    for _sfx in ("_train", "_val", "_test", "-train", "-val", "-test"):
        if base_name.endswith(_sfx):
            base_name = base_name[: -len(_sfx)]
            break

    # Default type: prefer "instances", else first
    default_type = "instances" if "instances" in type_groups else (types_list[0]["type"] if types_list else "")
    default_splits = type_groups.get(default_type, [])

    return {
        "name": base_name,
        "root": str(parent),
        "types": types_list,
        "default_type": default_type,
        "splits": default_splits,
    }


def _parse_yolo_splits(dataset_yaml: str) -> dict[str, Any]:
    """Parse a YOLO dataset.yaml and return available splits + metadata.

    Returns:
        Dict with ``name`` (dataset base name from YAML), ``splits`` (list of
        available split names), and ``classes`` (number of classes).

    """
    import yaml

    yaml_path = Path(dataset_yaml.strip())
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Dataset base name from YAML filename or the 'path' basename
    ds_root = cfg.get("path", "")
    if ds_root:
        base_name = Path(ds_root).name
    else:
        base_name = yaml_path.stem  # e.g. "coco128" from "coco128.yaml"

    # Strip split suffixes — this is a *base* name; splits get appended later
    for suffix in ("_train", "_val", "_test", "-train", "-val", "-test"):
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break

    # Detect which splits are defined
    splits: list[str] = []
    for s in ("train", "val", "test"):
        if cfg.get(s):
            splits.append(s)

    nc = cfg.get("nc", 0)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = list(names.values())

    root = _parse_yolo_dataset_root(dataset_yaml)

    return {
        "name": base_name,
        "root": root,
        "splits": splits,
        "num_classes": nc,
        "class_names": names[:20] if names else [],  # cap for display
    }


def _parse_yolo_dataset_root(dataset_yaml: str) -> str:
    """Extract the dataset root folder from a YOLO dataset.yaml.

    Returns the ``path`` key from the YAML (the common root for all splits),
    resolved relative to the YAML file location.  This is the correct folder
    for alias registration because it covers train, val, and test splits.
    """
    import yaml

    yaml_path = Path(dataset_yaml.strip())
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    ds_root = cfg.get("path", "")
    if not ds_root:
        return str(yaml_path.parent.resolve())

    root = Path(ds_root)
    if not root.is_absolute():
        root = yaml_path.parent / root

    return str(root.resolve())


def _parse_yolo_image_root(dataset_yaml: str, split: str) -> str:
    """Extract the image root folder for a specific split from a YOLO dataset.yaml.

    Reads ``path`` (dataset root) and the split key (e.g. ``train``, ``val``)
    from the YAML, then returns the resolved folder.
    """
    import yaml

    yaml_path = Path(dataset_yaml.strip())
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    ds_root = cfg.get("path", "")
    split_dir = cfg.get(split, split)

    root = Path(ds_root) / split_dir if ds_root else yaml_path.parent / split_dir
    if not root.is_absolute():
        root = yaml_path.parent / root

    # Go up to parent of "images" if the resolved path contains it
    resolved = root.resolve()
    return str(resolved.parent if resolved.name == "images" else resolved)


_YOLO_TASK_MAP = {
    "detection": "detect",
    "segmentation": "segment",
    "pose": "pose",
    "obb": "obb",
}


def _parse_yolo_yaml_for_split(dataset_yaml: str, split: str) -> tuple[str, dict[int, str] | None]:
    """Resolve the images folder + class names for one split of a YOLO dataset YAML.

    Returns ``(images_url, categories)``. ``categories`` may be None if the YAML
    declares ``nc`` instead of ``names``.
    """
    import yaml

    yaml_path = Path(dataset_yaml.strip())
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    split_value = cfg.get(split)
    if split_value is None:
        msg = f"Split {split!r} not found in {yaml_path} (available keys: {sorted(cfg)})"
        raise ValueError(msg)
    ds_root = cfg.get("path", "")
    split_path = Path(split_value)
    if not split_path.is_absolute():
        root = Path(ds_root) if ds_root else yaml_path.parent
        if not root.is_absolute():
            root = yaml_path.parent / root
        split_path = root / split_path

    # YOLO YAMLs declare class names either as a dict ({0: "cat", 1: "dog"}) or a list
    # (["cat", "dog"], in flow or block style). The list form is index-ordered, so enumerate
    # it; the dict form carries explicit indices. Either way, normalize to {int: str}.
    names = cfg.get("names")
    if isinstance(names, dict):
        categories = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        categories = {i: str(v) for i, v in enumerate(names)}
    else:
        categories = None
    return str(split_path.resolve()), categories


def _execute_yolo(form_data: dict[str, Any]) -> dict[str, Any]:
    """Execute YOLO import for a single split."""
    import tlc

    # The SDK defaults to "detect"; translate the form's task so segmentation/pose/obb
    # datasets are imported with the correct annotation schema.
    split = form_data["split"]
    task = _YOLO_TASK_MAP.get(form_data["task"], form_data["task"])
    images_url, categories = _parse_yolo_yaml_for_split(form_data["dataset_yaml"], split)

    table = tlc.Table.from_yolo_url(
        images_url,
        categories=categories,
        task=task,
        project_name=form_data["project_name"].strip(),
        dataset_name=form_data["dataset_name"].strip(),
        table_name=form_data.get("table_name", "").strip() or "initial",
        description=form_data.get("description", "").strip() or None,
    )

    return {
        "success": True,
        "message": f"Successfully created table '{form_data['table_name']}' from YOLO dataset.",
        "table_url": str(table.url),
        "project_name": form_data["project_name"],
        "dataset_name": form_data["dataset_name"],
        "format": "yolo",
        "split": split,
        "details": {
            "row_count": table.row_count,
            "project": form_data["project_name"],
            "dataset": form_data["dataset_name"],
            "table": form_data.get("table_name", "").strip() or "initial",
            "task": form_data["task"],
        },
    }


def _execute_coco(form_data: dict[str, Any]) -> dict[str, Any]:
    """Execute COCO import."""
    import tlc

    image_folder = form_data["image_folder"].strip()

    # The SDK defaults to "detect"; translate the form's task so segmentation
    # datasets are imported with the correct annotation schema.
    task = _YOLO_TASK_MAP.get(form_data["task"], form_data["task"])

    table = tlc.Table.from_coco(
        annotations_file=form_data["annotations_file"].strip(),
        image_folder=image_folder,
        task=task,
        project_name=form_data["project_name"].strip(),
        dataset_name=form_data["dataset_name"].strip(),
        table_name=form_data.get("table_name", "").strip() or "initial",
        description=form_data.get("description", "").strip() or None,
    )

    return {
        "success": True,
        "message": f"Successfully created table '{form_data['table_name']}' from COCO dataset.",
        "table_url": str(table.url),
        "project_name": form_data["project_name"],
        "dataset_name": form_data["dataset_name"],
        "details": {
            "row_count": table.row_count,
            "project": form_data["project_name"],
            "dataset": form_data["dataset_name"],
            "table": form_data.get("table_name", "").strip() or "initial",
            "task": form_data["task"],
        },
    }


def _execute_folder(form_data: dict[str, Any]) -> dict[str, Any]:
    """Execute Image Folder import."""
    import tlc

    folder_path = form_data["folder_path"].strip()

    table = tlc.Table.from_image_folder(
        root=folder_path,
        project_name=form_data["project_name"].strip(),
        dataset_name=form_data["dataset_name"].strip(),
        table_name=form_data.get("table_name", "").strip() or "initial",
        description=form_data.get("description", "").strip() or None,
    )

    return {
        "success": True,
        "message": f"Successfully created table '{form_data['table_name']}' from image folder.",
        "table_url": str(table.url),
        "project_name": form_data["project_name"],
        "dataset_name": form_data["dataset_name"],
        "details": {
            "row_count": table.row_count,
            "project": form_data["project_name"],
            "dataset": form_data["dataset_name"],
            "table": form_data.get("table_name", "").strip() or "initial",
        },
    }


def _get_image_dimensions(path: Path) -> tuple[int, int]:
    """Read image dimensions without loading full pixel data."""
    from PIL import Image

    with Image.open(path) as img:
        size: tuple[int, int] = img.size  # (width, height)
        return size


def _schemas_and_builder(modality: str, tlc: Any) -> tuple[dict[str, Any], Any]:
    """Return (column_schemas, row_builder_fn) for the given modality."""
    if modality == "classification":
        schemas = {
            "image": tlc.schemas.ImageSchema(sample_type="url"),
            "label": tlc.schemas.CategoricalLabelSchema(classes=["unlabeled"]),
            "weight": tlc.schemas.SampleWeightSchema(),
        }

        def build_row(img_path: Path) -> dict[str, Any]:
            return {"image": str(img_path), "label": 0, "weight": 0.0}

    elif modality == "detection":
        bb_schema = tlc.data_types.BoundingBoxes2D.schema(classes=["unlabeled"])
        schemas = {
            "image": tlc.schemas.ImageSchema(sample_type="url"),
            "bounding_boxes": bb_schema,
            "weight": tlc.schemas.SampleWeightSchema(),
        }

        def build_row(img_path: Path) -> dict[str, Any]:
            w, h = _get_image_dimensions(img_path)
            return {
                "image": str(img_path),
                "bounding_boxes": tlc.data_types.BoundingBoxes2D.create_empty(image_width=w, image_height=h),
                "weight": 0.0,
            }

    elif modality == "instance_segmentation":
        schemas = {
            "image": tlc.schemas.ImageSchema(sample_type="url"),
            "segmentations": tlc.data_types.SegmentationPolygons.schema(classes=["unlabeled"]),
            "weight": tlc.schemas.SampleWeightSchema(),
        }

        def build_row(img_path: Path) -> dict[str, Any]:
            w, h = _get_image_dimensions(img_path)
            return {
                "image": str(img_path),
                "segmentations": tlc.data_types.SegmentationPolygons.create_empty(image_width=w, image_height=h),
                "weight": 0.0,
            }

    else:
        msg = f"Unknown modality: {modality}"
        raise ValueError(msg)

    return schemas, build_row


def _execute_unlabeled(form_data: dict[str, Any]) -> dict[str, Any]:
    """Execute Unlabeled Images import."""
    import tlc

    folder = Path(form_data["folder_path"].strip())
    if not folder.is_dir():
        return {"success": False, "message": f"Folder not found: {folder}", "table_url": None, "details": {}}

    modality = form_data["modality"].strip()
    project_name = form_data["project_name"].strip()
    dataset_name = form_data["dataset_name"].strip()
    table_name = form_data.get("table_name", "").strip() or "initial"
    description = form_data.get("description", "").strip() or None

    image_paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        return {"success": False, "message": f"No images found in {folder}", "table_url": None, "details": {}}

    schemas, row_builder = _schemas_and_builder(modality, tlc)

    writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=dataset_name,
        project_name=project_name,
        description=description or "",
        schema=schemas,
    )

    for img_path in image_paths:
        row = row_builder(img_path)
        writer.add_row(row)

    table = writer.finalize()

    return {
        "success": True,
        "message": (
            f"Created table with {len(image_paths)} images. "
            "All samples have weight=0 — they will be excluded from "
            "training until annotated and re-weighted."
        ),
        "table_url": str(table.url),
        "project_name": project_name,
        "dataset_name": dataset_name,
        "details": {
            "row_count": len(image_paths),
            "modality": modality,
            "project": project_name,
            "dataset": dataset_name,
            "table": table_name,
        },
    }


def _execute_csv_detection(form_data: dict[str, Any]) -> dict[str, Any]:
    """Execute CSV Detection import — bounding boxes from a CSV with ImageId, class, bbox columns."""
    import csv as csv_mod
    from collections import defaultdict

    import tlc

    csv_path = Path(form_data["csv_path"].strip())
    image_folder = Path(form_data["image_folder"].strip())
    project_name = form_data["project_name"].strip()
    dataset_name = form_data["dataset_name"].strip()
    table_name = form_data.get("table_name", "").strip() or "initial"
    description = form_data.get("description", "").strip() or None

    if not csv_path.is_file():
        return {"success": False, "message": f"CSV file not found: {csv_path}", "table_url": None, "details": {}}
    if not image_folder.is_dir():
        return {
            "success": False,
            "message": f"Image folder not found: {image_folder}",
            "table_url": None,
            "details": {},
        }

    # Read CSV
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"success": False, "message": "CSV file is empty.", "table_url": None, "details": {}}

    # Extract sorted unique classes and group rows by ImageId
    classes = sorted({r["class"].strip() for r in rows if r.get("class", "").strip()})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["ImageId"].strip()].append(r)

    # Build schema — BoundingBoxes2D stores absolute XYXY pixel coords
    bb_schema = tlc.data_types.BoundingBoxes2D.schema(classes=classes)
    writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=dataset_name,
        project_name=project_name,
        description=description or "",
        schema={
            "image": tlc.schemas.ImageSchema(sample_type="url"),
            "bounding_boxes": bb_schema,
            "weight": tlc.schemas.SampleWeightSchema(),
        },
    )

    skipped = 0
    for image_id in sorted(grouped):
        # Try .png first, then .jpg
        img_path = image_folder / f"{image_id}.png"
        if not img_path.is_file():
            img_path = image_folder / f"{image_id}.jpg"
        if not img_path.is_file():
            skipped += 1
            continue

        w, h = _get_image_dimensions(img_path)
        bboxes: list[list[float]] = []
        labels_list: list[int] = []
        for r in grouped[image_id]:
            parts = r["bbox"].strip().split()
            if len(parts) < 4:
                continue
            x0_n, y0_n, x1_n, y1_n = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            bboxes.append([x0_n * w, y0_n * h, x1_n * w, y1_n * h])
            labels_list.append(class_to_idx[r["class"].strip()])

        import numpy as np

        bb = tlc.data_types.BoundingBoxes2D(
            bounding_boxes=np.asarray(bboxes, dtype=np.float32).reshape(-1, 4),
            labels=np.asarray(labels_list, dtype=np.int32),
            x_max=float(w),
            y_max=float(h),
        )
        writer.add_row({
            "image": str(img_path),
            "bounding_boxes": bb,
            "weight": 1.0,
        })

    table = writer.finalize()

    image_count = len(grouped) - skipped
    bbox_count = sum(len(v) for v in grouped.values())
    return {
        "success": True,
        "message": f"Created table '{table_name}' with {image_count} images and {bbox_count} bounding boxes.",
        "table_url": str(table.url),
        "project_name": project_name,
        "dataset_name": dataset_name,
        "details": {
            "image_count": image_count,
            "bbox_count": bbox_count,
            "classes": len(classes),
            "skipped_images": skipped,
            "project": project_name,
            "dataset": dataset_name,
            "table": table_name,
        },
    }


_EXECUTORS: dict[str, Any] = {
    "yolo": _execute_yolo,
    "coco": _execute_coco,
    "folder": _execute_folder,
    "unlabeled": _execute_unlabeled,
    "csv_detection": _execute_csv_detection,
}


def _get_image_folder(format_name: str, form_data: dict[str, Any]) -> str:
    """Extract the image folder path from form_data based on import format."""
    if format_name == "yolo":
        try:
            return _parse_yolo_dataset_root(form_data["dataset_yaml"])
        except Exception:
            return str(Path(form_data["dataset_yaml"].strip()).parent)
    elif format_name == "coco":
        coco_folder: str = form_data["image_folder"].strip()
        return coco_folder
    elif format_name in ("folder", "unlabeled"):
        folder_path: str = form_data["folder_path"].strip()
        return folder_path
    elif format_name == "csv_detection":
        csv_folder: str = form_data["image_folder"].strip()
        return csv_folder
    return ""


# ---------------------------------------------------------------------------
# run_job execution — imports run on the host JobManager (CpuQueue-serialized)
# ---------------------------------------------------------------------------


def _unregister_primary_alias(alias_result: dict[str, Any] | None) -> None:
    """Remove the PRIMARY session alias, but only if this job created it.

    Mirrors the legacy runners' cleanup: a PRIMARY alias that already existed
    before this job is left untouched (it was registered for another reason).
    """
    if not (alias_result and alias_result.get("primary_created")):
        return
    try:
        import tlc

        tlc.url.unregister_url_alias(alias_result["token"])
        logger.info("Cleaned up PRIMARY session alias <%s>", alias_result["token"])
    except Exception:
        pass


def _report_result(ctx: JobContext, result: dict[str, Any]) -> None:
    """Drive the generic job surface from an executor's result dict.

    Maps the executor output onto the frontend's generic schema: ``table_url`` →
    the job's result link, ``row_count`` / project / dataset → metric cards, and
    ``message`` → the final progress label + a log line. Nothing plugin-specific
    leaks — the frontend renders it without importer knowledge.
    """
    message = result.get("message", "")
    details = result.get("details") or {}
    row_count = details.get("row_count")
    if row_count is not None:
        ctx.metric("rows", row_count)
    project = result.get("project_name") or details.get("project")
    dataset = result.get("dataset_name") or details.get("dataset")
    if project:
        ctx.metric("project", project)
    if dataset:
        ctx.metric("dataset", dataset)
    table_url = result.get("table_url")
    if table_url:
        ctx.result(run_url=str(table_url))
    ctx.progress(percent=100, label=message)
    if message:
        ctx.log(message)


def _run_format_import(ctx: JobContext, format_name: str) -> None:
    """Run a file-format import (yolo/coco/folder/unlabeled/csv_detection).

    Raises:
        ValueError: Unknown format.
        RuntimeError: The executor failed or reported ``success=False`` — the
            host marks the job failed and surfaces the (enhanced) message.

    """
    executor = _EXECUTORS.get(format_name)
    if executor is None:
        msg = f"No executor for import format: {format_name!r}"
        raise ValueError(msg)

    form_data = ctx.params
    label = f"Importing {format_name}…"
    ctx.log(label)
    # Imports are a single blocking SDK call with no step granularity, so report
    # indeterminate progress (percent=-1 → the panel shows a pulsing bar).
    ctx.progress(percent=-1, label=label, timing={"step_label": "import"})

    # Register the project's URL alias BEFORE the executor runs so the SDK can use
    # the token when encoding image paths; remove the PRIMARY session alias after.
    alias_result = _maybe_register_alias(form_data, _get_image_folder(format_name, form_data))
    try:
        result = executor(form_data)
    except Exception as exc:
        raise RuntimeError(_enhance_error_message(str(exc))) from exc
    finally:
        _unregister_primary_alias(alias_result)

    if not result.get("success"):
        raise RuntimeError(result.get("message") or "Import failed")
    _report_result(ctx, result)


def _run_csv_import(ctx: JobContext) -> None:
    """Run a CSV/Excel import.

    The uploaded file bytes are looked up from the in-process ``_parsed_csv_files``
    store by the ``session_id`` carried in ``ctx.params`` — bytes never travel
    through params. Host-only (in-process), so the store written by ``/csv/parse``
    is visible here.

    Raises:
        RuntimeError: Missing/expired session, no columns, or the executor failed.

    """
    params = ctx.params
    session_id = params.get("session_id", "")
    file_data = _parsed_csv_files.get(session_id)
    if not file_data:
        msg = "File session expired. Please re-upload the file."
        raise RuntimeError(msg)

    selected_columns = params.get("selected_columns", [])
    if not selected_columns:
        msg = "No columns selected."
        raise RuntimeError(msg)

    label = "Importing CSV…"
    ctx.log(label)
    ctx.progress(percent=-1, label=label, timing={"step_label": "import"})

    mode = params.get("mode", "create_new")
    if mode == "extend_existing":
        table_url = params.get("table_url", "").strip()
        if not table_url:
            msg = "No source table URL provided."
            raise RuntimeError(msg)
        result = _execute_csv_extend(
            table_url=table_url,
            file_bytes=file_data["bytes"],
            filename=file_data["filename"],
            selected_columns=selected_columns,
            output_table_name=params.get("output_table_name", "extended").strip(),
            description=params.get("description", "").strip(),
        )
    else:
        project_name = params.get("project_name", "").strip()
        dataset_name = params.get("dataset_name", "").strip()
        if not project_name or not dataset_name:
            msg = "Project name and dataset name are required."
            raise RuntimeError(msg)
        alias_enabled = params.get("alias_enabled", True)
        result = _execute_csv_new(
            file_bytes=file_data["bytes"],
            filename=file_data["filename"],
            selected_columns=selected_columns,
            project_name=project_name,
            dataset_name=dataset_name,
            table_name=params.get("table_name", "").strip() or "initial",
            description=params.get("description", "").strip(),
            alias_enabled=alias_enabled in (True, "true", "1"),
            alias_token=params.get("alias_token", ""),
        )

    if not result.get("success"):
        raise RuntimeError(result.get("message") or "CSV import failed")
    _parsed_csv_files.pop(session_id, None)
    _report_result(ctx, result)


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class ImportPlugin(ComputePlugin):
    """Sidebar plugin for importing datasets into 3LC.

    Behavior only — all metadata lives in ``plugin.toml`` (the manifest). The host
    instantiates this via the manifest's ``runtime.entrypoint`` and stamps the
    display identity onto the instance; the class does not declare it.
    """

    # Display identity stamped onto the instance by the host from the manifest.
    id: str
    name: str
    icon: str

    _ui_cache: str | None = None

    def get_ui_fragment(self) -> str:
        """Return the self-contained import wizard HTML+JS+CSS fragment."""
        if self._ui_cache is None:
            from tlc_plugin_sdk.shared.alias_ui import alias_ui_script
            from tlc_plugin_sdk.shared.job_tracker import job_tracker_script

            ui_path = Path(__file__).resolve().parent / "ui.html"
            raw = ui_path.read_text(encoding="utf-8")
            # Inject the shared alias UI + generic job-tracker JS right after the
            # opening <script> tag (the wizard drives jobs via window.PluginJobs).
            self._ui_cache = raw.replace(
                "<script>\n(function() {\n  'use strict';",
                "<script>\n" + alias_ui_script() + "\n" + job_tracker_script() + "\n(function() {\n  'use strict';",
            )
        return self._ui_cache

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return import format definitions (used by discovery, not execution)."""
        return {"formats": list(IMPORT_STEPS.values())}

    def run_job(self, ctx: JobContext) -> None:
        """Run one import as a host job (CpuQueue-serialized via the JobManager).

        Dispatches on ``ctx.params["format"]``: ``csv`` reads the uploaded bytes
        from the in-process session store by ``session_id``; every other format
        runs its file-format executor. Progress/metrics/result flow only through
        the generic ``ctx`` surface — no plugin-specific events — and a failure is
        raised so the host marks the job failed. The host serializes imports one
        at a time (same as the legacy CPU queue), so multi-split imports stay
        ordered when the UI fires one ``/run`` per split.

        Args:
            ctx: Host-provided job context; ``ctx.params`` carries ``format`` plus
                the format-specific fields the executor needs.

        Raises:
            ValueError: Unknown/missing format.
            RuntimeError: The import failed (message surfaced to the panel).

        """
        format_name = ctx.params.get("format", "")
        if not format_name:
            msg = "Import job is missing a 'format'"
            raise ValueError(msg)
        if format_name == "csv":
            _run_csv_import(ctx)
        else:
            _run_format_import(ctx, format_name)

    def get_route_handlers(self) -> list[Any]:
        """Serve the importer's custom routes as relative Litestar handlers (host + venv)."""
        return _routes.get_route_handlers()
