"""Project errors: every refusal names its kind and the value.

A project refusal is a fact about the project on disk, never a
guess — the message carries the name, path or value that failed
so the caller can fix exactly that.
"""

from __future__ import annotations

__all__ = ["ProjectError"]


class ProjectError(Exception):
    """A project could not be created, opened, changed or run.

    ``kind`` is one of: bad_name, bad_manifest, bad_config,
    missing_file, unknown_module, unknown_plugin, duplicate_entry,
    unknown_entry, bad_entry, bad_target, bad_mode, not_a_project,
    already_exists, bad_seal, bad_take, bad_rig, bad_scene.
    """

    def __init__(self, message: str, *, kind: str = "bad_project"):
        super().__init__(message)
        self.kind = kind
