"""Show errors — one exception, named kinds, never silent."""

from __future__ import annotations

__all__ = ["ShowError"]


class ShowError(Exception):
    """A live-show failure that names what went wrong.

    ``kind`` is the machine-readable cause (``bad_transition``,
    ``no_rig``, ``missing_resource``, ``unknown_show`` …); the
    message carries the human-readable detail. Callers refuse by
    name; nothing in this organ fails silently.
    """

    def __init__(self, message: str, *, kind: str = "show_error") -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
