"""The file class, driven — one dispatcher for tool, chat, and CLI.

``file_action`` is the real control: every verb routes through it
and returns a dict, so a file packed in chat packs the same way from
the brain tool and the console. ``control_files`` is only the chat
text parser over it (the ``/match`` shape: ``control_match`` owns the
verbs, the tool and the CLI delegate to the same control).
"""

from __future__ import annotations

import logging
from typing import Any

from .exhibit import museum_with_files
from .file import File, FileError, resolve
from .packs import pack_files, pack_info, unpack_pack
from .store import FileStore

__all__ = ["control_files", "file_action", "format_file"]

_log = logging.getLogger(__name__)


def format_file(f: File) -> str:
    return f.describe()


def _store(context: Any, store: FileStore | None) -> FileStore:
    return store or FileStore(context)


def _card_dict(f: File) -> dict[str, Any]:
    card = f.to_card()
    card["sendable"] = f.sendable
    return card


def file_action(context: Any, action: str, *, ref: str = "",
                refs: tuple[str, ...] = (), dest: str = "",
                shelf: str = "", title: str = "",
                container: str = "zip", platform: str = "",
                chat_id: str = "", caption: str = "",
                store: FileStore | None = None) -> dict[str, Any]:
    """One verb, one result dict (``ok`` + the payload, or ``ok:
    False`` + ``error`` naming the reason). Never raises for a bad
    ref — a bad ref is an answer, not a crash."""
    act = (action or "").strip().lower()
    own = _store(context, store)
    try:
        if act == "info":
            f = resolve(context, ref)
            return {"ok": True, "file": _card_dict(f),
                    "text": format_file(f)}
        if act == "list":
            wanted = (shelf or "").strip().lower()
            if wanted == "downloads":
                files = own.downloads()
                return {"ok": True, "shelf": "downloads",
                        "files": [_card_dict(f) for f in files],
                        "count": len(files)}
            if wanted:
                files = own.shelf(wanted)
                return {"ok": True, "shelf": wanted,
                        "files": [_card_dict(f) for f in files],
                        "count": len(files)}
            return {"ok": True, "shelves": own.shelves()}
        if act == "shelve":
            f = own.shelve(ref, shelf)
            return {"ok": True, "file": _card_dict(f),
                    "text": f"shelved {f.name} → {f.shelf}"}
        if act == "unshelve":
            f = own.unshelve(ref, dest)
            return {"ok": True, "file": _card_dict(f),
                    "text": f"unshelved {f.name} → {f.path}"}
        if act == "pack":
            manifest = pack_files(
                context, list(refs) or ([ref] if ref else []),
                dest, title=title, container=container, store=own)
            packed = len(manifest.get("files")
                         or manifest.get("members") or [])
            return {"ok": True, "pack": manifest,
                    "text": (f"packed {packed} "
                             f"file(s) → {manifest.get('path')}")}
        if act == "unpack":
            out = unpack_pack(context, ref, dest, shelf=shelf,
                              store=own)
            return {"ok": True, "unpack": out,
                    "text": (f"unpacked {out.get('extracted', 0)} "
                             f"file(s) → {out.get('dest')}"
                             + (f" — problems: "
                                f"{'; '.join(out.get('problems') or [])}"
                                if out.get("problems") else ""))}
        if act == "packinfo":
            info = pack_info(resolve(context, ref).path)
            return {"ok": True, "pack": info}
        if act == "send":
            f = resolve(context, ref)
            if not f.sendable:
                return {"ok": False,
                        "error": f"{f.title} is not sendable"}
            tools = getattr(context, "tools", None)
            if tools is None:
                return {"ok": False,
                        "error": "no tool caller attached — "
                                 "sending needs a live context"}
            outcome = tools.call(
                "file_send", platform=(platform or "").strip(),
                chat_id=(chat_id or "").strip(), path=str(f.path),
                caption=caption or "")
            if not outcome.ok:
                return {"ok": False,
                        "error": f"file send failed: "
                                 f"{getattr(outcome.error, 'message', outcome.error)}"}
            value = outcome.value or {}
            return {"ok": True, "sent": value,
                    "text": (f"sent {f.name} → "
                             f"{platform}:{chat_id}")}
        if act == "ingest":
            f = resolve(context, ref)
            if f.kind not in ("book", "document"):
                return {"ok": False,
                        "error": (f"{f.name} is a {f.kind} — the "
                                 "library ingests books and "
                                 "documents, not this kind")}
            from ..books.library import Library, LibraryError

            try:
                result = Library(context).ingest(f.path)
            except LibraryError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True,
                    "book": result.to_dict() if hasattr(
                        result, "to_dict") else dict(result),
                    "text": (f"ingested {f.name} → "
                             f"{getattr(result, 'slug', '')}")}
        if act == "exhibit":
            museum = museum_with_files(context, own)
            res = museum.census()
            wanted = (shelf or "").strip().lower()
            files = [c for c in res.get("artifacts") or []
                     if c.get("shelf", "").startswith("files-")]
            if wanted:
                files = [c for c in files
                         if c.get("shelf") == wanted]
            return {"ok": True, "shelves": res.get("by_shelf") or {},
                    "files": files, "count": len(files),
                    "errors": res.get("errors") or {},
                    "text": museum.render(
                        {"artifacts": files,
                         "by_shelf": res.get("by_shelf") or {},
                         "count": len(files),
                         "errors": res.get("errors") or {}})}
        return {"ok": False,
                "error": (f"unknown file action {action!r} — info | "
                          "list | shelve | unshelve | pack | unpack | "
                          "packinfo | send | ingest | exhibit")}
    except FileError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — a named failure, not a crash
        _log.warning("files.file_action %s failed", act, exc_info=True)
        return {"ok": False, "error": f"{act} failed: {exc}"}


_USAGE = ("usage: /files | info <ref> | shelve <ref> [shelf] | "
          "unshelve <ref> [dest] | pack <ref...> --dest <pack> "
          "[--title T] [--fmt zip|tar.gz] | unpack <pack> "
          "[--shelf S] [--dest DIR] | packinfo <pack> | "
          "send <ref> <platform> <chat> [caption] | "
          "ingest <ref> | exhibit [shelf]")


def _split_flags(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    """Pull ``--flag value`` pairs out of a token list."""
    rest: list[str] = []
    flags: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--") and i + 1 < len(tokens):
            flags[tok[2:].lower()] = tokens[i + 1]
            i += 2
        else:
            rest.append(tok)
            i += 1
    return rest, flags


def control_files(tail: str, context: Any = None, chat: Any = None,
                  sender_id: str = "", sender: str = "") -> str:
    """``/files`` — the chat twin of the ``file`` brain tool and
    ``nm file``. Same control, same stores, same answers."""
    _ = (chat, sender_id, sender)
    if context is None:
        return "files needs a live context"
    tail = (tail or "").strip()
    if not tail:
        out = file_action(context, "list")
        if not out.get("ok"):
            return f"files failed: {out.get('error')}"
        shelves = out.get("shelves") or []
        if not shelves:
            return ("the file store is empty — shelve a file with "
                    "/files shelve <ref>, or pack one with "
                    "/files pack <ref...> --dest <pack>")
        lines = ["🗄️ **Files** — shelves:"]
        for s in shelves:
            lines.append(f"  {s['shelf']}: {s['files']} file(s), "
                         f"{s['bytes']:,} B")
        lines.append("  downloads: /files list downloads")
        return "\n".join(lines)
    tokens, flags = _split_flags(tail.split())
    verb = tokens[0].lower() if tokens else ""
    args = tokens[1:]
    if verb == "info" and args:
        out = file_action(context, "info", ref=args[0])
        return out.get("text") or f"files failed: {out.get('error')}"
    if verb == "list":
        shelf = args[0] if args else ""
        out = file_action(context, "list", shelf=shelf)
        if not out.get("ok"):
            return f"files failed: {out.get('error')}"
        files = out.get("files") or []
        if not files:
            return f"shelf {shelf or 'store'} is empty"
        lines = [f"🗄️ {out.get('shelf')} — {len(files)} file(s):"]
        for card in files[:50]:
            lines.append(f"  [{card['id']}] {card['title']} — "
                         f"{card['kind']} · {card['size']:,} B")
        return "\n".join(lines)
    if verb == "shelve" and args:
        out = file_action(context, "shelve", ref=args[0],
                          shelf=args[1] if len(args) > 1 else "")
        return out.get("text") or f"files failed: {out.get('error')}"
    if verb == "unshelve" and args:
        out = file_action(context, "unshelve", ref=args[0],
                          dest=args[1] if len(args) > 1 else "")
        return out.get("text") or f"files failed: {out.get('error')}"
    if verb == "pack" and args:
        out = file_action(context, "pack", refs=tuple(args),
                          dest=flags.get("dest", ""),
                          title=flags.get("title", ""),
                          container=flags.get("fmt", "zip"))
        return out.get("text") or f"files failed: {out.get('error')}"
    if verb == "unpack" and args:
        out = file_action(context, "unpack", ref=args[0],
                          dest=flags.get("dest", ""),
                          shelf=flags.get("shelf", ""))
        return out.get("text") or f"files failed: {out.get('error')}"
    if verb == "packinfo" and args:
        out = file_action(context, "packinfo", ref=args[0])
        if not out.get("ok"):
            return f"files failed: {out.get('error')}"
        pack = out.get("pack") or {}
        manifest = pack.get("manifest") or {}
        members = pack.get("members") or []
        title = (manifest.get("pack") or {}).get("title", "")
        return (f"📦 {pack.get('path')} — "
                f"{'pack' if not pack.get('legacy') else 'legacy bundle'}"
                f"{f' · {title!r}' if title else ''} · "
                f"{len(members)} member(s)")
    if verb == "send" and len(args) >= 3:
        out = file_action(context, "send", ref=args[0],
                          platform=args[1], chat_id=args[2],
                          caption=" ".join(args[3:]))
        return out.get("text") or f"files failed: {out.get('error')}"
    if verb == "ingest" and args:
        out = file_action(context, "ingest", ref=args[0])
        return out.get("text") or f"files failed: {out.get('error')}"
    if verb == "exhibit":
        out = file_action(context, "exhibit",
                          shelf=args[0] if args else "")
        return out.get("text") or f"files failed: {out.get('error')}"
    return _USAGE
