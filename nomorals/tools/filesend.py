"""Research file creator + social sender.

Turns any research, analysis, or result into a well-formatted file
(MD, TXT, JSON, HTML, or a real dependency-free PDF) and sends it to any
active chat platform through the live gateway — the same path TTS uses
for voice notes, with its auto-compression for large files.

Every output is a real deliverable: the HTML is a standalone styled
document, the JSON is pretty-printed and validated, the PDF is a
standards-compliant multi-page file rendered by
:mod:`nomorals.core.pdf`.
"""
from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from .filesystem import safe_path

_log = get_logger(__name__)

__all__ = ["create_file", "send_file", "publish_report", "register",
           "md_to_html", "md_to_text"]

_FORMATS = {
    "md": "markdown",
    "markdown": "markdown",
    "txt": "text",
    "text": "text",
    "json": "json",
    "html": "html",
    "pdf": "pdf",
}

_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(name: str, fallback: str) -> str:
    name = (name or "").strip()
    name = _SLUG.sub("-", name).strip("-.")
    return name[:80] or f"{fallback}-{int(time.time())}"


# ── markdown helpers (small, honest, no external parser) ────────────────────


def md_to_text(md: str) -> str:
    """Strip markdown chrome to clean plain text (keep the substance)."""
    lines: list[str] = []
    in_fence = False
    for line in (md or "").split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            lines.append("")
            continue
        if in_fence:
            lines.append("    " + line)
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            lines.append(title.upper() if level <= 2 else title)
            if level <= 2:
                lines.append("-" * min(60, max(4, len(title))))
            continue
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[image: \1]", line)
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", line)
        lines.append(line.rstrip())
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def md_to_html(md: str, *, title: str = "Report") -> str:
    """Minimal, honest markdown→HTML: headings, bold/italic, code, fences,
    lists, links, blockquotes, paragraphs.  Output is a standalone doc
    with embedded CSS — opens in any browser or phone viewer."""
    lines = (md or "").split("\n")
    body: list[str] = []
    in_fence = False
    fence_buf: list[str] = []
    list_stack: list[str] = []  # "ul" | "ol"

    def close_lists() -> None:
        while list_stack:
            body.append(f"</{list_stack.pop()}>")

    def inline(text: str) -> str:
        text = html.escape(text, quote=False)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
        text = re.sub(
            r"\[([^\]]+)\]\(([^)\s]+)\)",
            r'<a href="\2">\1</a>', text)
        return text

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                body.append("<pre><code>" + html.escape("\n".join(fence_buf))
                            + "</code></pre>")
                fence_buf = []
                in_fence = False
            else:
                close_lists()
                in_fence = True
            continue
        if in_fence:
            fence_buf.append(line)
            continue
        if not stripped:
            close_lists()
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            close_lists()
            level = len(m.group(1))
            body.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            continue
        m = re.match(r"^[-*+]\s+(.*)$", stripped)
        if m:
            if not list_stack or list_stack[-1] != "ul":
                close_lists()
                list_stack.append("ul")
                body.append("<ul>")
            body.append(f"<li>{inline(m.group(1))}</li>")
            continue
        m = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if m:
            if not list_stack or list_stack[-1] != "ol":
                close_lists()
                list_stack.append("ol")
                body.append("<ol>")
            body.append(f"<li>{inline(m.group(1))}</li>")
            continue
        if stripped.startswith(">"):
            close_lists()
            body.append(f"<blockquote>{inline(stripped.lstrip('> '))}</blockquote>")
            continue
        body.append(f"<p>{inline(line.rstrip())}</p>")
    if in_fence and fence_buf:
        body.append("<pre><code>" + html.escape("\n".join(fence_buf)) + "</code></pre>")
    close_lists()

    css = (
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "max-width:760px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#1c1e21}"
        "h1,h2,h3{line-height:1.25}pre{background:#f6f8fa;padding:.8rem;border-radius:6px;"
        "overflow-x:auto;font-size:.85em}code{background:#f6f8fa;padding:.1em .3em;"
        "border-radius:4px;font-size:.9em}pre code{background:none;padding:0}"
        "blockquote{border-left:3px solid #d0d7de;margin:0;padding:.2rem 1rem;color:#57606a}"
        "a{color:#0969da}hr{border:none;border-top:1px solid #d0d7de;margin:1.4rem 0}"
    )
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n"
        f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{css}</style></head>\n"
        f"<body>\n" + "\n".join(body) + "\n</body></html>\n"
    )


# ── file creation ───────────────────────────────────────────────────────────


def create_file(
    context: Any,
    name: str,
    content: str,
    *,
    format: str = "md",
    title: str = "",
) -> dict[str, Any]:
    """Write ``content`` to ``workspace/files/<name>.<ext>`` in ``format``.

    Formats: md | txt | json | html | pdf.  JSON is validated and
    pretty-printed; HTML gets a standalone styled document; PDF renders
    through the pure-Python writer.  Returns path, size, and a preview.
    """
    fmt = (format or "md").lower()
    if fmt not in _FORMATS:
        raise ToolError(
            f"unknown format {format!r}; use one of: "
            + ", ".join(sorted({v for v in _FORMATS.values()} | set(_FORMATS))))
    content = content or ""
    if not content.strip():
        raise ToolError("refusing to create an empty file")
    base_title = title or name or "file"

    slug = _slug(name, "file")
    ext = fmt if fmt in {"md", "txt", "json", "html", "pdf"} else "txt"
    target: Path = safe_path(context, f"files/{slug}.{ext}")

    if ext == "json":
        try:
            parsed = json.loads(content)
            body = json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"
        except (ValueError, TypeError):
            body = json.dumps(
                {"note": "content was not valid JSON; stored as-is in 'text'",
                 "text": content}, indent=2, ensure_ascii=False) + "\n"
        data: bytes = body.encode("utf-8")
    elif ext == "html":
        data = md_to_html(content, title=base_title).encode("utf-8")
    elif ext == "pdf":
        from ..core.pdf import render_pdf

        # content with multiple top-level sections (a book) gets the real
        # book layout: bold headings, chapter page breaks, a TOC with true
        # page numbers.  A plain report (≤1 top heading) keeps the classic
        # flat rendering.
        n_h1 = len(re.findall(r"(?m)^#\s+\S", content))
        if n_h1 >= 2:
            data = render_pdf(content, title=base_title, headings=True,
                              chapter_break=True, toc=True)
        else:
            data = render_pdf(content, title=base_title)
    elif ext == "txt":
        data = md_to_text(content).encode("utf-8")
    else:  # md
        data = content.encode("utf-8")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    try:
        target.chmod(0o644)
    except OSError:
        pass
    _log.info("file created: %s (%d bytes, %s)", target, len(data), ext)
    return {
        "path": str(target),
        "format": ext,
        "bytes": len(data),
        "title": base_title,
        "preview": (content if ext == "md" else md_to_text(content))[:400],
    }


# ── social sending ──────────────────────────────────────────────────────────


def _gateway(context: Any) -> Any:
    gateway = (getattr(context, "extras", None) or {}).get("gateway")
    if gateway is None:
        raise ToolError(
            "chat gateway not attached to this context — file sending "
            "requires the running runtime")
    return gateway


def send_file(
    context: Any,
    platform: str,
    chat_id: str,
    path: str,
    *,
    caption: str = "",
) -> dict[str, Any]:
    """Send a file to a chat on any live platform via the gateway.

    ``path`` may be a workspace-relative path (resolved safely) or an
    absolute path outside the workspace (the owner's own files are fair
    game).  Large files ride the gateway's auto-compression.
    """
    gateway = _gateway(context)
    target = (path or "").strip()
    if not target:
        raise ToolError("send_file needs a path")
    resolved: str = target
    if not target.startswith("/") and not target.startswith("~"):
        try:
            resolved = str(safe_path(context, target))
        except ToolError:
            pass  # fall through to the literal path; the send will fail loudly
    adapters = getattr(gateway, "adapters", None) or {}
    if platform not in adapters:
        live = ", ".join(sorted(adapters)) or "none"
        raise ToolError(f"no live adapter for platform {platform!r}; live: {live}")
    # the gateway keys chats as 'platform:chat_id'; a bare id gets prefixed
    if ":" not in chat_id:
        chat_id = f"{platform}:{chat_id}"
    try:
        result = gateway.send_file(platform, chat_id, resolved,
                                   caption=caption or "")
    except Exception as exc:  # noqa: BLE001 - report, don't crash the caller
        raise ToolError(f"send failed: {exc}") from exc
    if not getattr(result, "ok", False):
        raise ToolError(f"send failed: {getattr(result, 'error', 'unknown error')}")
    return {
        "sent": True,
        "platform": platform,
        "chat": chat_id,
        "path": resolved,
        "message_id": getattr(result, "message_id", ""),
        "bytes": _size(resolved),
    }


def _size(path: str) -> int:
    import os

    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def publish_report(
    context: Any,
    content: str,
    *,
    title: str,
    platform: str = "",
    chat_id: str = "",
    format: str = "pdf",
    name: str = "",
) -> dict[str, Any]:
    """Create a research report file and (when a destination is given)
    send it straight to the chat.  One call = the whole research→social
    loop."""
    base_name = name or _slug(title, "report")
    created = create_file(context, base_name, content, format=format, title=title)
    out: dict[str, Any] = dict(created)
    if platform and chat_id:
        sent = send_file(context, platform, chat_id, created["path"],
                         caption=f"📄 {title}")
        out.update(sent)
    return out


# ── tool registration ───────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "file_create",
        description=(
            "Turn research/analysis/results into a formatted file: md, txt, "
            "json (validated+pretty), standalone html, or pdf (dependency-free "
            "writer). Writes to workspace/files/."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "name": "str — file base name",
            "content": "str — the markdown-ish report content",
            "format": "str (optional, md) — md | txt | json | html | pdf",
            "title": "str (optional) — document title (used by html/pdf)",
        },
    )
    def file_create(name: str, content: str, *, format: str = "md",
                    title: str = "") -> dict[str, Any]:
        return create_file(context, name, content, format=format, title=title)

    @registry.register(
        "file_send",
        description=(
            "Send a file to any active chat platform (telegram, discord, "
            "whatsapp, console) through the live gateway. Big files are "
            "auto-compressed."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "platform": "str — telegram | discord | whatsapp | console | …",
            "chat_id": "str — the target chat id",
            "path": "str — workspace path or absolute path",
            "caption": "str (optional)",
        },
    )
    def file_send(platform: str, chat_id: str, path: str,
                  *, caption: str = "") -> dict[str, Any]:
        return send_file(context, platform, chat_id, path, caption=caption)

    @registry.register(
        "report_publish",
        description=(
            "Research→file→social in one step: create a formatted report "
            "(pdf by default) and send it to a chat."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "content": "str — markdown report content",
            "title": "str — report title",
            "platform": "str (optional) — where to send it",
            "chat_id": "str (optional)",
            "format": "str (optional, pdf) — md | txt | json | html | pdf",
            "name": "str (optional) — file base name",
        },
    )
    def report_publish(content: str, *, title: str, platform: str = "",
                       chat_id: str = "", format: str = "pdf",
                       name: str = "") -> dict[str, Any]:
        return publish_report(context, content, title=title, platform=platform,
                              chat_id=chat_id, format=format, name=name)
