# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exporter: a run body may only carry the select values the form offers (Bug 19037).

The form offers ``image_strategy`` ignore/copy/symlink, but ``tlc`` also accepts ``move``, which moves the
table's source images out of their folder. A raw job request must not reach the writer with it.
"""

from __future__ import annotations

from typing import Any

import pytest

from tlc_plugin_exporter import EXPORT_FORMATS, routes


def _run(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def execute(table_url: str, output_path: str, options: dict[str, Any]) -> dict[str, Any]:
        calls.append(options)
        return {"success": True, "message": "ok", "details": {}}

    monkeypatch.setitem(routes._EXECUTORS, body["format"], execute)
    return routes.run_export({"table_url": "/t", "output_path": "/tmp/x", **body}), calls


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"format": "yolo", "image_strategy": "move"}, "Image handling must be one of: ignore, copy, symlink."),
        ({"format": "yolo", "split": "../x"}, "Split name must be one of: train, val, test."),
        ({"format": "csv", "separator": "|"}, "Separator must be one of: ,, ;, tab."),
    ],
)
def test_a_select_value_the_form_does_not_offer_is_refused(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, Any], message: str
) -> None:
    result, calls = _run(monkeypatch, body)
    assert result == {"success": False, "message": message}
    assert calls == []


def test_every_offered_choice_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, fmt in EXPORT_FORMATS.items():
        for option in fmt["options"]:
            if option["type"] != "select":
                continue
            for choice in option["choices"]:
                result, calls = _run(monkeypatch, {"format": name, option["name"]: choice})
                assert result["success"], (name, option["name"], choice)
                assert calls[0][option["name"]] == choice


def test_an_absent_select_option_takes_the_writer_default(monkeypatch: pytest.MonkeyPatch) -> None:
    result, calls = _run(monkeypatch, {"format": "yolo"})
    assert result["success"]
    assert len(calls) == 1


def test_the_job_channel_refuses_it_too(monkeypatch: pytest.MonkeyPatch) -> None:
    from tlc_plugin_exporter import ExportPlugin

    calls: list[str] = []
    monkeypatch.setitem(routes._EXECUTORS, "yolo", lambda *a, **k: calls.append("ran"))

    class _Ctx:
        def __init__(self) -> None:
            self.job_id = "job-1"
            self.params = {"format": "yolo", "table_url": "/t", "output_path": "/tmp/x", "image_strategy": "move"}

        def progress(self, **_: Any) -> None:
            pass

        def emit(self, *_: Any) -> None:
            pass

    with pytest.raises(ValueError, match="Image handling must be one of"):
        ExportPlugin().run_job(_Ctx())  # type: ignore[arg-type]
    assert calls == []
