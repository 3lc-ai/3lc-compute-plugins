# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real native tables: first dataset, incremental releases and leakage checks."""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tlc_plugin_merger.intake import execute


def table(root, name, indices, *, classes=None, group=False):
    import tlc
    from PIL import Image

    root.mkdir(exist_ok=True)
    schema = {
        "image": tlc.schemas.ImageSchema(sample_type="url"),
        "label": tlc.schemas.CategoricalLabelSchema(classes=classes or ["cat", "dog"]),
    }
    if group:
        schema["capture"] = tlc.schemas.StringSchema()
    writer = tlc.TableWriter(
        root_url=str(root), project_name="Intake", dataset_name="incoming", table_name=name, schema=schema
    )
    for i in indices:
        path = root / f"{i}.png"
        Image.new("RGB", (16, 16), (i * 10, 90, 120)).save(path)
        row = {"image": str(path), "label": i % 2}
        if group:
            row["capture"] = str(i // 2)
        writer.add_row(row)
    return writer.finalize()


def run(root, source, **kwargs):
    return execute(
        {
            "operation": "intake_publish",
            "table_url": str(source.url),
            "dataset_name": "training",
            "modality": "classification",
            **kwargs,
        },
        root=str(root),
        state=root / "intake.sqlite3",
        tenant="org",
    )


def test_first_dataset_and_append_are_immutable_and_idempotent(tmp_path):
    import tlc

    root = tmp_path / "projects"
    base = table(root, "first", range(5))
    first = run(root, base)
    assert len(tlc.Table.from_url(first["table_url"])) == 5
    assert first == {**run(root, base), "reused": False}
    batch = table(root, "second", [3, 4, 5, 6, 7])
    second = run(root, batch)
    assert second["rows"] == 3 and second["duplicates"] == 2
    assert len(tlc.Table.from_url(second["table_url"])) == 8
    assert len(base) == 5 and len(batch) == 5


def test_heldout_images_and_groups_are_protected(tmp_path):
    root = tmp_path / "projects"
    heldout = table(root, "val", [0, 1], group=True)
    source = table(root, "overlap", [1, 2], group=True)
    with pytest.raises(ValueError, match="protected"):
        run(root, source, heldout_tables=[str(heldout.url)])
    source = table(root, "same-capture", [0], group=True)
    heldout = table(root, "other-frame", [1], group=True)
    with pytest.raises(ValueError, match="protected"):
        run(root, source, heldout_tables=[str(heldout.url)], group_column="capture")


def test_unknown_classes_fail_before_release(tmp_path):
    root = tmp_path / "projects"
    base = table(root, "base", range(3))
    batch = table(root, "different", range(3, 5), classes=["cat", "bird"])
    with pytest.raises(ValueError, match="Class names"):
        run(root, batch, destination_table=str(base.url))


def test_two_users_do_not_lose_a_batch(tmp_path):
    import tlc

    root = tmp_path / "projects"
    base = table(root, "base", [0, 1])
    batches = [table(root, str(i), [2 * i, 2 * i + 1]) for i in [1, 2]]
    barrier = threading.Barrier(2)

    def publish(batch):
        barrier.wait()
        return run(root, batch, destination_table=str(base.url))

    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(publish, batches))
    assert sorted(len(tlc.Table.from_url(result["table_url"])) for result in results) == [4, 6]


def geometry_table(root, name, kind, indices, ids):
    import numpy as np
    import tlc
    from PIL import Image

    root.mkdir(exist_ok=True)
    classes = {ids[0]: "cat", ids[1]: "dog"}
    factory = {
        "bounding_boxes": tlc.data_types.BoundingBoxes2D,
        "instance_masks": tlc.data_types.SegmentationMasks,
        "semantic_map": tlc.data_types.SemanticSegmentation,
    }[kind]
    schema = (
        tlc.schemas.SemanticSegmentationRleSchema(classes={0: "background", **classes}, background=0)
        if kind == "semantic_map"
        else factory.schema(classes=classes)
    )
    writer = tlc.TableWriter(
        root_url=str(root),
        project_name="Intake",
        dataset_name="incoming",
        table_name=name,
        schema={"image": tlc.schemas.ImageSchema(sample_type="url"), "annotation": schema},
    )
    for i in indices:
        path = root / f"{i}.png"
        Image.new("RGB", (16, 16), (i * 10, 90, 120)).save(path)
        if kind == "bounding_boxes":
            value = factory(
                bounding_boxes=np.asarray([[1, 2, 10, 12]], dtype=np.float32),
                labels=np.asarray([ids[i % 2]], dtype=np.int32),
                image_width=16,
                image_height=16,
            )
        elif kind == "instance_masks":
            value = factory(masks=np.ones((16, 16, 1), dtype=np.uint8), labels=np.asarray([ids[i % 2]], dtype=np.int32))
        else:
            value = factory(np.full((16, 16), ids[i % 2], dtype=np.int32), background_id=0)
        writer.add_row({"image": str(path), "annotation": value})
    return writer.finalize()


@pytest.mark.parametrize("kind", ["bounding_boxes", "instance_masks", "semantic_map"])
def test_geometry_reference_class_ids_are_preserved(tmp_path, kind):
    import tlc

    from tlc_plugin_merger.intake import annotation, class_map

    root = tmp_path / "projects"
    base = geometry_table(root, "base", kind, [0, 1, 2], [3, 7])
    batch = geometry_table(root, "new", kind, [3, 4], [1, 2])
    result = run(root, batch, modality=kind, destination_table=str(base.url), expected_classes=["cat", "dog"])
    merged = tlc.Table.from_url(result["table_url"])
    assert len(merged) == 5
    _, _, schema, path = annotation(merged)
    assert class_map(schema, path)[3] == "cat" and class_map(schema, path)[7] == "dog"


def test_repeated_batch_stops_before_model_calls(tmp_path):
    import tlc

    root = tmp_path / "projects"
    source = table(root, "batch", range(5))
    prepared = run(root, source, operation="intake_prepare", labeled=True)
    published = run(root, tlc.Table.from_url(prepared["table_url"]), batch_url=str(source.url))
    assert published["rows"] == 5
    again = run(root, source, operation="intake_prepare", labeled=True)
    assert again["rows"] == len(tlc.Table.from_url(again["table_url"])) == 0


def test_preparation_rejects_conflicting_existing_labels(tmp_path):
    root = tmp_path / "projects"
    base = table(root, "base", [0, 1])
    batch = table(root, "conflict", [0, 2], classes=["dog", "cat"])
    with pytest.raises(ValueError, match="different annotations"):
        run(root, batch, operation="intake_prepare", labeled=True, destination_table=str(base.url))


def test_separate_batches_do_not_accumulate(tmp_path):
    import tlc

    root = tmp_path / "projects"
    first = table(root, "first", [0, 1])
    second = table(root, "second", [2, 3])
    assert len(tlc.Table.from_url(run(root, first, keep_separate=True)["table_url"])) == 2
    assert len(tlc.Table.from_url(run(root, second, keep_separate=True)["table_url"])) == 2


def test_grouped_split_keeps_capture_together_and_retries_reuse_outputs(tmp_path):
    from types import SimpleNamespace

    import tlc

    from tlc_plugin_splitter.grouped import run_grouped_split

    root = tmp_path / "projects"
    source = table(root, "batch", range(10), group=True)
    outputs = {}
    ctx = SimpleNamespace(
        params={"table_url": str(source.url), "group_column": "capture", "train_pct": 60, "val_pct": 40},
        cancelled=False,
        result=lambda value: None,
        progress=lambda **kwargs: None,
        output=lambda name, value: outputs.update({name: value}),
    )
    run_grouped_split(ctx)
    first = dict(outputs["splits"])
    groups = [{row["capture"] for row in tlc.Table.from_url(part["table_url"]).table_rows} for part in first.values()]
    assert not groups[0] & groups[1]
    assert sum(part["rows"] for part in first.values()) == 10
    run_grouped_split(ctx)
    assert outputs["splits"] == first


@pytest.mark.parametrize("append", [False, True])
def test_reviewed_classification_becomes_native_training_labels(tmp_path, append):
    import tlc

    root = tmp_path / "projects"
    base = table(root, "base", [0, 1])
    source = table(root, "unlabeled", [2, 3])
    coverage = tlc.schemas.BoolSchema()
    coverage.metadata = {"refine_output_coverage": {"column": "classification"}}
    writer = tlc.TableWriter(
        root_url=str(root),
        project_name="Intake",
        dataset_name="incoming",
        table_name="reviewed",
        schema={
            "image": tlc.schemas.ImageSchema(sample_type="url"),
            "classification": tlc.Schema(
                values={
                    "label": tlc.schemas.CategoricalLabelSchema(classes=["cat", "dog"]),
                    "confidence": tlc.schemas.Float64Schema(),
                }
            ),
            "covered": coverage,
        },
    )
    for row in source.table_rows:
        writer.add_row({
            "image": row["image"],
            "classification": {"label": row["label"], "confidence": 0.9},
            "covered": True,
        })
    reviewed = writer.finalize()
    result = run(root, reviewed, **({"destination_table": str(base.url)} if append else {}))
    merged = tlc.Table.from_url(result["table_url"])
    label = "label"
    assert len(merged) == (4 if append else 2)
    assert [row[label] for row in merged.table_rows] == ([0, 1, 0, 1] if append else [0, 1])
    assert set(merged.rows_schema.values) == {"image", label}


@pytest.mark.parametrize("extra", [{"max_rows": 0}, {"labeled": "yes"}, {"heldout_tables": "/val"}])
def test_native_jobs_validate_the_same_inputs_as_mcp(tmp_path, extra):
    root = tmp_path / "projects"
    source = table(root, "batch", [0, 1])
    with pytest.raises(ValueError):
        run(root, source, **extra)


def test_unpublished_rows_cannot_become_negative_training_samples(tmp_path):
    import tlc

    root = tmp_path / "projects"
    source = geometry_table(root, "partial", "bounding_boxes", [0, 1], [3, 7])
    coverage = tlc.schemas.BoolSchema()
    coverage.metadata = {"refine_output_coverage": {"column": "annotation"}}
    from tlc_plugin_merger.intake import revision

    partial = revision(
        source,
        source.url.create_sibling("partial-reviewed"),
        [0, 1],
        overrides={"covered": coverage.to_minimal_dict(include_all=False)},
        edits={"covered": {"runs_and_values": [[0], True, [1], False]}},
        receipt="partial fixture",
    )
    with pytest.raises(ValueError, match="unreviewed or unpublished"):
        run(root, partial, modality="bounding_boxes")


def test_interrupted_publication_reuses_native_output(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import tlc

    import tlc_plugin_merger.intake as intake

    root = tmp_path / "projects"
    source = table(root, "batch", [0, 1, 2])
    original = intake.journal

    @contextmanager
    def interrupted(path):
        with original(path) as db:
            yield db
            message = "interrupted before receipt commit"
            raise RuntimeError(message)

    monkeypatch.setattr(intake, "journal", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(root, source)
    monkeypatch.setattr(intake, "journal", original)
    result = run(root, source)
    assert len(tlc.Table.from_url(result["table_url"])) == 3
    assert run(root, source)["reused"] is True


def test_adding_data_preserves_validation_and_source_tables(tmp_path):
    import tlc

    root = tmp_path / "projects"
    base = geometry_table(root, "train", "bounding_boxes", [0, 1], [3, 7])
    heldout = geometry_table(root, "val", "bounding_boxes", [8, 9], [3, 7])
    batch = geometry_table(root, "new", "bounding_boxes", [2, 3], [1, 2])
    before = {
        str(path.relative_to(str(heldout.url))): path.read_bytes()
        for path in Path(str(heldout.url)).rglob("*")
        if path.is_file()
    }
    result = run(
        root, batch, modality="bounding_boxes", destination_table=str(base.url), heldout_tables=[str(heldout.url)]
    )
    assert len(tlc.Table.from_url(result["table_url"])) == 4
    assert len(base) == 2 and len(batch) == 2
    after = {
        str(path.relative_to(str(heldout.url))): path.read_bytes()
        for path in Path(str(heldout.url)).rglob("*")
        if path.is_file()
    }
    assert before == after
