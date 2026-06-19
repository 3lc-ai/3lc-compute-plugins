# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""Persist training projects as JSON files."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TrainingProject:
    """A saved training configuration."""

    id: str = ""
    name: str = ""
    run_name: str = ""  # 3LC Run name (empty = random)
    project_name: str = ""  # 3LC project for Run association
    model_name: str = ""  # "yolov8", etc.
    task_type: str = ""  # "classification", "detection", ...
    train_table_url: str = ""
    val_table_url: str = ""
    use_latest: bool = False  # Resolve .latest() at train time
    mode: str = "train"  # "train" or "collect"
    params: dict[str, Any] = field(default_factory=dict)
    created: str = ""
    last_run: str | None = None


class ProjectStore:
    """Persist training projects as JSON files on disk."""

    def __init__(self, store_dir: str | None = None) -> None:
        if store_dir:
            self._dir = Path(store_dir)
        else:
            self._dir = Path.home() / ".3lc-training" / "projects"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, project_id: str) -> Path:
        return self._dir / f"{project_id}.json"

    def list_projects(self) -> list[TrainingProject]:
        """Return all saved projects, newest first."""
        projects: list[TrainingProject] = []
        for f in sorted(self._dir.glob("*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                projects.append(
                    TrainingProject(**{k: v for k, v in data.items() if k in TrainingProject.__dataclass_fields__})
                )
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        projects.sort(key=lambda p: p.created or "", reverse=True)
        return projects

    def get_project(self, project_id: str) -> TrainingProject | None:
        """Load a project by id."""
        path = self._path(project_id)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return TrainingProject(**{k: v for k, v in data.items() if k in TrainingProject.__dataclass_fields__})
        except (json.JSONDecodeError, OSError, TypeError):
            return None

    def save_project(self, project: TrainingProject) -> TrainingProject:
        """Save a project (creates new id if empty)."""
        if not project.id:
            project.id = str(uuid.uuid4())
        if not project.created:
            project.created = datetime.now(timezone.utc).isoformat()
        with open(self._path(project.id), "w") as f:
            json.dump(asdict(project), f, indent=2)
        return project

    def delete_project(self, project_id: str) -> bool:
        """Delete a project. Returns True if it existed."""
        path = self._path(project_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def update_last_run(self, project_id: str) -> None:
        """Update last_run timestamp on a project."""
        project = self.get_project(project_id)
        if project:
            project.last_run = datetime.now(timezone.utc).isoformat()
            self.save_project(project)
