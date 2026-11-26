"""The ``file`` brain tool — the file class for the brain.

One tool, one ``action``: info | list | shelve | unshelve | pack |
unpack | packinfo | send | ingest | exhibit. Every action runs the
same ``file_action`` control chat (``/files``) and the console
(``nm file``) run, over the same FileStore — a file shelved by the
brain shows in chat, and vice versa. Packing resolves human refs
(paths, ``museum:<id>``, ``library:<slug>``,
``archive:<path>::<member>``); sending rides the live ``file_send``
tool; ingesting rides the Library.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from .chat import file_action

__all__ = ["register"]


def register(registry: Any) -> None:
    @registry.register(
        "file",
        description=(
            "The file class: one addressable file across the system — "
            "info | list | shelve | unshelve | pack | unpack | packinfo "
            "| send | ingest | exhibit. Refs are plain paths, "
            "museum:<id>, library:<slug>, or archive:<path>::<member>. "
            "Pack builds a manifest-carrying artifact archive (SHA-256 "
            "per member, file cards inside); unpack verifies the "
            "manifest first and reports bad/missing members honestly. "
            "Shelve moves a file onto its kind's shelf under "
            "<home>/data/files; ingest puts a book/document into the "
            "Library; exhibit shows the shelved files in the Museum "
            "census. Refs are plain paths, downloads:<name>, "
            "files:<shelf>/<name>, museum:<id>, library:<slug>, or "
            "archive:<path>::<member>."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "action": ("str — info | list | shelve | unshelve | pack | "
                       "unpack | packinfo | send | ingest | exhibit"),
            "ref": ("str — one file ref: plain path, downloads:<name>, "
                    "files:<shelf>/<name>, museum:<id>, library:<slug>, "
                    "or archive:<path>::<member>"),
            "refs": "list[str] — file refs to pack",
            "dest": "str — pack destination, unpack destination, or unshelve directory",
            "shelf": "str — shelf name (list|shelve|unpack|exhibit; 'downloads' lists the intake)",
            "title": "str — pack title",
            "container": "str — pack container: zip (default) | tar.gz",
            "platform": "str — send platform (telegram | discord | whatsapp | console)",
            "chat_id": "str — send target chat id",
            "caption": "str — send caption",
        },
    )
    def file(context: Any, action: str = "list", ref: str = "",
             refs: Any = None, dest: str = "", shelf: str = "",
             title: str = "", container: str = "zip", platform: str = "",
             chat_id: str = "", caption: str = "") -> dict[str, Any]:
        return file_action(
            context, action, ref=ref,
            refs=tuple(refs or ()), dest=dest, shelf=shelf,
            title=title, container=container, platform=platform,
            chat_id=chat_id, caption=caption)
