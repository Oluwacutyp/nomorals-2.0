"""``nm file`` — the file class from the console.

The console twin of chat ``/files`` (``files/chat.py:
control_files``) over the same ``file_action`` control: bare
``nm file`` lists the shelves, ``nm file info <ref>`` names one file,
``nm file shelve <ref> [shelf]`` moves it onto a shelf, ``nm file pack
<ref...> --dest <pack>`` builds a manifest-carrying artifact archive,
``nm file unpack <pack> [--shelf S]`` verifies then lands it,
``nm file send <ref> <platform> <chat>`` rides the live file-send
path, ``nm file ingest <ref>`` puts a book/document into the Library,
``nm file exhibit [shelf]`` shows the shelved files in the Museum
census. ``nm file send`` is refused in words with chat's exact
wording when no live tool caller is attached: sending needs a live
context.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ...files.chat import file_action

__all__ = ["_cmd_file"]


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _cmd_file(args: argparse.Namespace, context: Any) -> int:
    """``nm file [info|list|shelve|unshelve|pack|unpack|packinfo|send|
    ingest|exhibit ...]``."""
    task = list(getattr(args, "task", None) or [])
    as_json = bool(getattr(args, "json", False))
    # --flag value pairs, same shape as chat's.
    rest: list[str] = []
    flags: dict[str, str] = {}
    i = 0
    while i < len(task):
        tok = task[i]
        if tok.startswith("--") and i + 1 < len(task):
            flags[tok[2:].lower()] = task[i + 1]
            i += 2
        else:
            rest.append(tok)
            i += 1
    verb = rest[0].lower() if rest else "list"
    argv = rest[1:]

    def _emit(out: dict[str, Any]) -> int:
        if as_json:
            print(json.dumps(out, indent=2, default=str))
            return 0 if out.get("ok") else 1
        if out.get("ok"):
            print(out.get("text") or json.dumps(out, indent=2,
                                                default=str))
            return 0
        return _usage_error(f"file failed: {out.get('error')}")

    if verb == "list":
        return _emit(file_action(context, "list",
                                 shelf=argv[0] if argv else ""))
    if verb == "info" and argv:
        return _emit(file_action(context, "info", ref=argv[0]))
    if verb == "shelve" and argv:
        return _emit(file_action(
            context, "shelve", ref=argv[0],
            shelf=argv[1] if len(argv) > 1 else ""))
    if verb == "unshelve" and argv:
        return _emit(file_action(
            context, "unshelve", ref=argv[0],
            dest=argv[1] if len(argv) > 1 else ""))
    if verb == "pack" and argv:
        return _emit(file_action(
            context, "pack", refs=tuple(argv),
            dest=flags.get("dest", ""),
            title=flags.get("title", ""),
            container=flags.get("fmt", "zip")))
    if verb == "unpack" and argv:
        return _emit(file_action(
            context, "unpack", ref=argv[0],
            dest=flags.get("dest", ""),
            shelf=flags.get("shelf", "")))
    if verb == "packinfo" and argv:
        return _emit(file_action(context, "packinfo", ref=argv[0]))
    if verb == "send" and len(argv) >= 3:
        return _emit(file_action(
            context, "send", ref=argv[0], platform=argv[1],
            chat_id=argv[2], caption=" ".join(argv[3:])))
    if verb == "ingest" and argv:
        return _emit(file_action(context, "ingest", ref=argv[0]))
    if verb == "exhibit":
        return _emit(file_action(
            context, "exhibit", shelf=argv[0] if argv else ""))
    return _usage_error(
        "usage: nm file [info <ref> | list [shelf] | shelve <ref> "
        "[shelf] | unshelve <ref> [dest] | pack <ref...> --dest <pack> "
        "[--title T] [--fmt zip|tar.gz] | unpack <pack> [--shelf S] "
        "[--dest DIR] | packinfo <pack> | send <ref> <platform> "
        "<chat> [caption] | ingest <ref> | exhibit [shelf]]")
