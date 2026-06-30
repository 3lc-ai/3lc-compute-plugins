# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""timm plugin — sidebar plugin for image classification training with timm models.

Job execution uses the unified ``run_job(ctx)`` contract: the host JobManager owns
the queue / cancel / generic progress, while this plugin re-emits its own ``/timm``
SocketIO events (``job_status`` / ``epoch_progress`` / ``job_completed`` /
``job_failed``) via ``ctx.emit`` for its embedded UI. ``ctx.params`` carries the
``config_id``; the config store resolves it to the frozen training params, exactly
as the old runner did.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tlc_plugin_sdk import ComputePlugin

from tlc_plugin_timm import routes as _routes

if TYPE_CHECKING:
    from tlc_plugin_sdk.job_context import JobContext

logger = logging.getLogger(__name__)


class TimmPlugin(ComputePlugin):
    """Sidebar plugin for training image classifiers with timm.

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
        """Return the self-contained timm UI HTML+JS+CSS fragment."""
        if self._ui_cache is None:
            from tlc_plugin_sdk.shared.alias_override_ui import alias_override_ui_script
            from tlc_plugin_sdk.shared.ui_inject import inject_scripts

            ui_path = Path(__file__).resolve().parent / "ui.html"
            raw = ui_path.read_text(encoding="utf-8")
            self._ui_cache = inject_scripts(raw, alias_override_ui_script())
        return self._ui_cache

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Not used — timm uses dedicated REST endpoints + SocketIO."""
        return {"status": "Use /api/plugins/timm/* endpoints"}

    def run_job(self, ctx: JobContext) -> None:
        """Run a timm training or collection job against a host-provided context.

        Reproduces the old runner's ``_execute_job`` setup: ``ctx.params`` carries a
        ``config_id`` which is resolved against the config store to the frozen
        training params, ``.latest()`` table URLs are resolved when requested, and
        the 3LC project / run name are auto-derived. Train-vs-collect is selected by
        the config's ``mode`` (``train`` | ``collect``), exactly as before.

        Driven entirely by ``ctx``: ``ctx.progress`` / ``ctx.metric`` / ``ctx.log``
        feed the generic Queue & Progress panel (percent + label only — no
        training-specific fields), while ``ctx.emit`` re-broadcasts the plugin's own
        ``/timm`` events (``job_status`` / ``epoch_progress`` / ``job_completed`` /
        ``job_failed``) for the embedded UI. Cancellation is cooperative via
        ``ctx.cancelled``.

        Args:
            ctx: Host-provided job context. ``ctx.params`` carries ``config_id`` (and
                optionally already-resolved config params, which override the stored
                config's params).

        """
        import time

        from tlc_plugin_sdk.shared.generic_job import epoch_progress

        from tlc_plugin_timm.config_store import config_store

        params_in = ctx.params
        config_id = str(params_in.get("config_id", "") or "").strip()

        # Resolve config_id → frozen config via the store (as the runner did).
        config = None
        if config_id:
            config = config_store().get_config(config_id)
        if config is None:
            msg = "Config not found" if config_id else "config_id is required"
            ctx.emit("job_failed", {"job_id": ctx.job_id, "error": msg})
            raise ValueError(msg)

        mode = config.mode or "train"
        mode_label = "Collection" if mode == "collect" else "Training"
        ctx.emit("job_status", {"job_id": ctx.job_id, "status": "running", "message": f"{mode_label} started"})

        # Resolve .latest() if requested.
        train_url = config.train_table_url
        val_url = config.val_table_url
        if config.use_latest:
            try:
                import tlc

                train_url = str(tlc.Table.from_url(train_url).latest().url)
                if val_url:
                    val_url = str(tlc.Table.from_url(val_url).latest().url)
            except Exception as e:
                ctx.log(f"Warning: could not resolve .latest(): {e}")

        tables: dict[str, Any] = {"train": train_url, "val": val_url or None}

        # Resolve 3LC project name from the table if not explicitly set.
        tlc_project_name = config.project_name or ""
        if not tlc_project_name:
            try:
                import tlc

                t = tlc.Table.from_url(train_url)
                tlc_project_name = getattr(t, "project_name", "") or ""
            except Exception:
                pass

        # Auto-generate run name if not provided.
        tlc_run_name = config.run_name or ""
        if not tlc_run_name:
            from tlc_plugin_sdk.shared.naming import generate_name

            tlc_run_name = generate_name()

        # ── Timing bookkeeping (lifted from the old runner) ──
        _last_metrics: dict[str, Any] = {}
        _timing: dict[str, Any] = {
            "job_start": time.monotonic(),
            "epoch_start": time.monotonic(),
            "epoch_times": [],
        }

        def _build_timing(epoch: int, total_epochs: int) -> dict[str, Any]:
            now = time.monotonic()
            elapsed = now - _timing["job_start"]
            result: dict[str, Any] = {"elapsed_s": round(elapsed, 1)}
            epoch_times = _timing["epoch_times"]
            if epoch_times:
                avg_epoch = sum(epoch_times) / len(epoch_times)
                result["avg_epoch_s"] = round(avg_epoch, 1)
                remaining = total_epochs - epoch
                result["eta_s"] = round(avg_epoch * remaining, 1) if remaining > 0 else 0
            return result

        # ── Trainer callbacks wired to ctx ──
        def on_epoch(epoch: int, total_epochs: int, metrics: dict[str, Any]) -> None:
            _last_metrics.update(metrics)
            merged = dict(_last_metrics)

            is_batch = "phase" in metrics
            if not is_batch:
                now = time.monotonic()
                _timing["epoch_times"].append(now - _timing["epoch_start"])
                _timing["epoch_start"] = now

            timing = _build_timing(epoch, total_epochs)

            # Plugin-specific event — carries the rich (training) payload the /timm
            # UI listens for. The frontend treats this as opaque.
            ctx.emit(
                "epoch_progress",
                {
                    "job_id": ctx.job_id,
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "metrics": merged,
                    "timing": timing,
                },
            )

            # Generic panel — percent + label only (no epoch/loss/phase leakage).
            progress_raw: dict[str, Any] = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "metrics": merged,
                "timing": timing,
            }
            generic = epoch_progress(progress_raw, step_label="epoch")
            if generic:
                ctx.progress(
                    percent=float(generic.get("percent", 0)),
                    label=str(generic.get("label", "")),
                    timing=generic.get("timing"),
                )

        def on_status(message: str) -> None:
            ctx.log(message)
            is_collecting = "Collecting" in message or "Reducing" in message
            if is_collecting:
                # Generic progress for the collect phase (no training fields).
                ctx.progress(percent=100.0, label="Collecting metrics")
            ctx.emit(
                "job_status",
                {
                    "job_id": ctx.job_id,
                    "status": "collecting" if is_collecting else "running",
                    "message": message,
                },
            )

        def is_cancelled() -> bool:
            return ctx.cancelled

        callbacks = {
            "on_epoch": on_epoch,
            "on_status": on_status,
            "is_cancelled": is_cancelled,
        }

        # Build params with internal fields (frozen config params + run identity).
        params = dict(config.params)
        params["_project_name"] = config.project_name or tlc_project_name
        params["_run_name"] = tlc_run_name
        params["_task_type"] = config.task_type
        params["_image_column"] = config.image_column
        params["_label_column"] = config.label_column
        params["_model_name"] = config.model_name

        # Apply alias overrides if requested (restored in finally).
        alias_originals: list[dict[str, str]] = []
        alias_ov = params.pop("_alias_overrides", None)
        if isinstance(alias_ov, dict) and alias_ov.get("enabled") and alias_ov.get("overrides"):
            from tlc_plugin_sdk.shared.aliases import apply_alias_overrides

            alias_originals = apply_alias_overrides(alias_ov["overrides"])
            if alias_originals:
                ctx.log(f"Applied {len(alias_originals)} alias override(s)")

        try:
            on_status(f"Run name: {tlc_run_name}")

            from tlc_plugin_timm.trainer import collect as timm_collect
            from tlc_plugin_timm.trainer import train as timm_train

            trainer_fn = timm_collect if mode == "collect" else timm_train
            result = trainer_fn(tables, params, callbacks)

            run_url = result.get("run_url")

            if ctx.cancelled:
                ctx.emit("job_status", {"job_id": ctx.job_id, "status": "cancelled", "message": "Job cancelled"})
            else:
                # Generic surface stays percent + label only — final training
                # metrics (best_val_acc, best_epoch, …) ride the plugin-specific
                # job_completed event, never ctx.metric.
                ctx.progress(percent=100.0, label="Done")
                ctx.emit(
                    "job_completed",
                    {"job_id": ctx.job_id, "run_url": run_url, "tlc_project_name": tlc_project_name},
                )

            # Update config last_run timestamp.
            config_store().update_last_run(config.id)

        except Exception as e:
            logger.exception("timm run_job failed")
            ctx.emit("job_failed", {"job_id": ctx.job_id, "error": str(e)})
            raise
        finally:
            if alias_originals:
                from tlc_plugin_sdk.shared.aliases import restore_aliases

                restore_aliases(alias_originals)

    def get_route_handlers(self) -> list[Any]:
        """Serve timm's custom routes as relative Litestar handlers (host + venv)."""
        return _routes.get_route_handlers()
