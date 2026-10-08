# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exporter: a bucket URL from its own picker is a destination, written through ``tlc.Url``.

Real ``tlc`` is not imported: a stand-in ``tlc`` module records what is written where.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tlc_plugin_exporter import routes

WRITTEN: dict[str, bytes] = {}


class _Url:
    def __init__(self, url: str) -> None:
        self.url = url

    def write_bytes(self, data: bytes) -> None:
        WRITTEN[self.url] = data


@pytest.fixture(autouse=True)
def fake_tlc(monkeypatch: pytest.MonkeyPatch) -> None:
    WRITTEN.clear()
    monkeypatch.setitem(sys.modules, "tlc", SimpleNamespace(Url=_Url))


def _csv_writer(calls: list[str]) -> Any:
    def execute(table_url: str, output_path: str, options: dict[str, Any]) -> dict[str, Any]:
        calls.append(output_path)
        p = Path(output_path)
        if p.suffix != ".csv":
            p.mkdir(parents=True, exist_ok=True)
            p = p / "export.csv"
        p.write_text("a,b\n1,2\n")
        return {"success": True, "message": f"Exported 1 rows to CSV at {p}", "details": {"output_path": str(p)}}

    return execute


def _run(monkeypatch: pytest.MonkeyPatch, output_path: str, **extra: Any) -> tuple[dict[str, Any], list[str]]:
    calls: list[str] = []
    monkeypatch.setitem(routes._EXECUTORS, "csv", _csv_writer(calls))
    body = {"format": "csv", "table_url": "s3://b/projects/p/datasets/d/tables/t", "output_path": output_path}
    return routes.run_export({**body, **extra}), calls


def test_a_bucket_prefix_receives_what_the_writer_wrote(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks: list[tuple[int, int]] = []
    calls: list[str] = []
    monkeypatch.setitem(routes._EXECUTORS, "csv", _csv_writer(calls))
    body = {"format": "csv", "table_url": "s3://b/t", "output_path": "s3://b/exports/"}
    result = routes.run_export(body, progress=lambda done, total: ticks.append((done, total)))
    assert result["success"], result
    assert list(WRITTEN) == ["s3://b/exports/export.csv"]
    assert result["message"] == "Exported 1 rows to CSV at s3://b/exports/export.csv"  # no scratch path leaks
    assert result["details"]["output_path"] == "s3://b/exports/export.csv"
    assert ticks == [(1, 1)]
    assert not Path(calls[0]).exists()  # the scratch folder is gone


def test_a_bucket_file_url_names_the_file_in_its_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = _run(monkeypatch, "s3://b/exports/fire.csv")
    assert result["success"], result
    assert list(WRITTEN) == ["s3://b/exports/fire.csv"]


def test_a_bucket_root_with_a_dot_is_a_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = _run(monkeypatch, "s3://my.bucket")
    assert result["success"], result
    assert list(WRITTEN) == ["s3://my.bucket/export.csv"]


def test_a_local_path_is_written_in_place(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, calls = _run(monkeypatch, str(tmp_path / "out"))
    assert result["success"], result
    assert calls == [str(tmp_path / "out")] and (tmp_path / "out" / "export.csv").is_file()
    assert not WRITTEN


def test_a_relative_local_path_is_still_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    result, calls = _run(monkeypatch, "relative/out")
    assert not result["success"] and "must be absolute" in result["message"]
    assert not calls


def test_the_undeclared_alias_overrides_key_is_gone() -> None:
    """Nothing sent ``alias_overrides``; the SDK worker applies the host's ``_alias_overrides`` instead."""
    src = Path(routes.__file__).read_text(encoding="utf-8")
    assert "alias_overrides" not in src
