"""Create-and-deliver: report → styled HTML (+PDF) → zip → chat.

The richer create-and-deliver flow for "send me a report on X":

    generate_report(context, topic, sections)   # HTML + PDF under workspace/reports/<slug>/
    + zip (archive.zip_create)                 # workspace/archives/report-<slug>.zip
    + send (filesend.send_file)                # the same live-gateway path the rest of the bot uses

PDF is not faked: it renders through :func:`nomorals.core.pdf.render_pdf`,
the dependency-free pure-Python writer, so a real ``.pdf`` ships next to
the styled HTML.  No reportlab / weasyprint needed (and none is installed).

Sections are ``(title, markdown body)`` triples — accepted as
:class:`ReportSection`, ``(title, body)`` tuples, or ``{"title": …,
"body": …}`` dicts.  Empty topics and empty section lists fail fast;
there is no synthetic filler content — the caller (research, an agent,
the CLI) supplies the substance.

The send edge is mockable: pass ``sender=`` to :func:`deliver_report`
and the fake sees the exact zip bytes that would have gone to the chat
(the same injection pattern as ``media_pipeline.run_pipeline``).
"""

from __future__ import annotations

import html as _html
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..core.text import slugify
from .filesystem import safe_path

_log = get_logger(__name__)

__all__ = [
    "ReportSection",
    "ReportBundle",
    "normalize_sections",
    "compose_markdown",
    "render_report_html",
    "generate_report",
    "deliver_report",
    "register",
]

#: directory (under the workspace) where generated reports land
REPORTS_DIR = "reports"
#: directory (under the workspace) where the zipped deliverable lands
ARCHIVES_DIR = "archives"


# ── section / bundle models ───────────────────────────────────────────────


@dataclass
class ReportSection:
    """One report section: a title and a markdown body."""

    title: str
    body: str = ""


@dataclass
class ReportBundle:
    """Everything :func:`generate_report` produced."""

    topic: str
    title: str
    slug: str
    sections: list[str] = field(default_factory=list)
    html_path: str = ""
    pdf_path: str = ""
    zip_path: str = ""
    zip_files: list[str] = field(default_factory=list)
    html_bytes: int = 0
    pdf_bytes: int = 0
    zip_bytes: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "title": self.title,
            "slug": self.slug,
            "sections": list(self.sections),
            "html_path": self.html_path,
            "pdf_path": self.pdf_path,
            "zip_path": self.zip_path,
            "zip_files": list(self.zip_files),
            "html_bytes": self.html_bytes,
            "pdf_bytes": self.pdf_bytes,
            "zip_bytes": self.zip_bytes,
            "seconds": self.seconds,
        }


def normalize_sections(
    sections: Sequence[ReportSection | tuple[str, str] | dict[str, Any]],
) -> list[ReportSection]:
    """Coerce the flexible section input into :class:`ReportSection` list.

    Accepts dataclasses, ``(title, body)`` tuples, and ``{"title": …,
    "body": …}`` dicts.  Raises :class:`ToolError` on anything else, and
    on an empty list — a report with no sections is a bug in the caller,
    not something to paper over.
    """
    if sections is None:
        raise ToolError("report needs at least one section")
    out: list[ReportSection] = []
    for i, raw in enumerate(sections):
        if isinstance(raw, ReportSection):
            title, body = raw.title, raw.body
        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
            title, body = raw[0], raw[1]
        elif isinstance(raw, dict) and "title" in raw:
            title, body = raw.get("title", ""), raw.get("body", "")
        else:
            raise ToolError(
                f"section {i}: expected ReportSection, (title, body) tuple, "
                f"or {{'title': …, 'body': …}} dict; got {type(raw).__name__}")
        title = str(title or "").strip()
        body = str(body or "").strip()
        if not title:
            raise ToolError(f"section {i}: title is required")
        out.append(ReportSection(title=title, body=body))
    if not out:
        raise ToolError("report needs at least one section")
    if not any(s.body for s in out):
        raise ToolError("report sections are all empty — nothing to deliver")
    return out


def _report_slug(topic: str, *, name: str = "") -> str:
    slug = slugify(name or topic, limit=60, keep_case=True,
                   extra="._-", strip="-.") or f"report-{int(time.time())}"
    return slug


# ── markdown composition ──────────────────────────────────────────────────


def compose_markdown(
    topic: str,
    sections: Sequence[ReportSection | tuple[str, str] | dict[str, Any]],
    *,
    title: str = "",
    generated: str = "",
) -> str:
    """Compose the report's markdown source: ``# title`` + ``##`` sections.

    The same source feeds both renderers — the styled HTML and the
    pure-Python PDF — so the two artifacts can never disagree on content.
    """
    topic = (topic or "").strip()
    if not topic:
        raise ToolError("report needs a topic")
    norm = normalize_sections(sections)
    doc_title = (title or "").strip() or topic
    stamp = generated or time.strftime("%Y-%m-%d %H:%M")
    parts = [f"# {doc_title}", "", f"*Topic: {topic} · generated {stamp}*", ""]
    for section in norm:
        parts.append(f"## {section.title}")
        parts.append("")
        parts.append(section.body or "_(no content provided)_")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


# ── styled HTML rendering ─────────────────────────────────────────────────


_REPORT_CSS = """
:root{color-scheme:light}
body{font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
max-width:820px;margin:0 auto;padding:0 1.25rem 4rem;line-height:1.6;color:#1c1e21;
background:#fff}
.report-cover{background:linear-gradient(135deg,#0b1e3a,#123a6d);color:#fff;
margin:0 -1.25rem 2rem;padding:3rem 1.25rem 2.25rem;border-radius:0 0 18px 18px}
.report-kicker{font-size:.78rem;letter-spacing:.18em;text-transform:uppercase;
opacity:.75;margin-bottom:.6rem}
.report-cover h1{color:#fff;font-size:2.1rem;line-height:1.2;margin:.2rem 0 .8rem}
.report-meta{font-size:.9rem;opacity:.8}
.report-toc{background:#f6f8fa;border:1px solid #d0d7de;border-radius:10px;
padding:1rem 1.25rem;margin:0 0 2rem}
.report-toc h2{margin:.2rem 0 .6rem;font-size:1.05rem}
.report-toc ol{margin:0;padding-left:1.4rem}
.report-toc li{margin:.25rem 0}
.report-toc a{color:#0969da;text-decoration:none}
.report-toc a:hover{text-decoration:underline}
h1{font-size:1.9rem;line-height:1.25;margin:2.2rem 0 1rem}
h2{font-size:1.4rem;line-height:1.3;margin:2rem 0 .8rem;
padding-bottom:.35rem;border-bottom:2px solid #eaeef2}
h3{font-size:1.15rem;margin:1.5rem 0 .6rem}
p{margin:.7rem 0}
pre{background:#f6f8fa;padding:.9rem;border-radius:8px;overflow-x:auto;
font-size:.85em;border:1px solid #eaeef2}
code{background:#f6f8fa;padding:.1em .35em;border-radius:4px;font-size:.9em}
pre code{background:none;padding:0}
blockquote{border-left:3px solid #0969da;margin:1rem 0;padding:.3rem 1rem;
color:#4b5563;background:#f6f8fa;border-radius:0 6px 6px 0}
a{color:#0969da}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92em}
th,td{border:1px solid #d0d7de;padding:.45rem .7rem;text-align:left}
th{background:#f6f8fa}
hr{border:none;border-top:1px solid #d0d7de;margin:2rem 0}
.report-footer{margin-top:3rem;padding-top:1.2rem;border-top:1px solid #d0d7de;
color:#57606a;font-size:.82rem;text-align:center}
"""

_STYLE_RE = re.compile(r"<style>.*?</style>", re.S)
_HEADING_RE = re.compile(r"<h([123])>(.*?)</h\1>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _heading_id(inner_html: str) -> str:
    text = _html.unescape(_TAG_RE.sub("", inner_html)).strip()
    return slugify(text, limit=60, fallback="section") or "section"


def render_report_html(
    topic: str,
    sections: Sequence[ReportSection | tuple[str, str] | dict[str, Any]],
    *,
    title: str = "",
    generated: str = "",
) -> str:
    """Render the styled standalone HTML report.

    Reuses the honest dependency-free markdown renderer from
    :mod:`nomorals.tools.filesend` for the body, then layers the report
    chrome on top: the report theme CSS, a cover header with metadata, a
    table of contents with working anchors, and a footer.  The result is
    a single self-contained ``.html`` — opens on any phone or browser.
    """
    from .filesend import md_to_html

    topic = (topic or "").strip()
    if not topic:
        raise ToolError("report needs a topic")
    norm = normalize_sections(sections)
    doc_title = (title or "").strip() or topic
    stamp = generated or time.strftime("%Y-%m-%d %H:%M")

    doc = md_to_html(compose_markdown(topic, norm, title=doc_title,
                                      generated=stamp),
                     title=doc_title)

    # 1. report theme instead of the plain renderer CSS
    doc = _STYLE_RE.sub(f"<style>{_REPORT_CSS}</style>", doc, count=1)
    # 2. anchor ids on every heading (the TOC links to the section ones)
    doc = _HEADING_RE.sub(
        lambda m: f'<h{m.group(1)} id="{_heading_id(m.group(2))}">'
                  f"{m.group(2)}</h{m.group(1)}>",
        doc,
    )
    # 3. cover header + table of contents
    toc_items = "\n".join(
        f'      <li><a href="#{_heading_id(s.title)}">'
        f"{_html.escape(s.title)}</a></li>"
        for s in norm
    )
    cover = (
        '<header class="report-cover">\n'
        '  <div class="report-kicker">📄 Devon report</div>\n'
        f'  <div class="report-meta">topic: {_html.escape(topic)} · '
        f"generated {stamp} · {len(norm)} section"
        f"{'s' if len(norm) != 1 else ''}</div>\n"
        "</header>\n"
        '<nav class="report-toc">\n  <h2>Contents</h2>\n  <ol>\n'
        f"{toc_items}\n  </ol>\n</nav>\n"
    )
    doc = doc.replace("<body>\n", "<body>\n" + cover, 1)
    # 4. footer
    footer = (
        '<footer class="report-footer">\n'
        f"  generated by Devon · {stamp} · {_html.escape(doc_title)}\n"
        "</footer>\n"
    )
    head, sep, tail = doc.rpartition("</body>")
    if sep:
        doc = head + footer + sep + tail
    return doc


# ── generate → zip ────────────────────────────────────────────────────────


def _unique_report_dir(context: Any, slug: str) -> Path:
    base = safe_path(context, f"{REPORTS_DIR}/{slug}")
    if not base.exists():
        return base
    for i in range(2, 1000):
        candidate = safe_path(context, f"{REPORTS_DIR}/{slug}-{i}")
        if not candidate.exists():
            return candidate
    raise ToolError(f"could not find a free report dir for {slug!r}")


def generate_report(
    context: Any,
    topic: str,
    sections: Sequence[ReportSection | tuple[str, str] | dict[str, Any]],
    *,
    title: str = "",
    name: str = "",
    include_pdf: bool = True,
    generated: str = "",
) -> ReportBundle:
    """Generate the report files and zip them.  No sending — see :func:`deliver_report`.

    Writes ``report.html`` (styled) and ``report.pdf`` (pure-Python writer;
    skipped with an honest note when rendering fails) into
    ``workspace/reports/<slug>/``, then zips the directory into
    ``workspace/archives/report-<slug>.zip`` via :func:`archive.zip_create`.
    """
    from .archive import zip_create

    started = time.perf_counter()
    topic = (topic or "").strip()
    if not topic:
        raise ToolError("report needs a topic")
    norm = normalize_sections(sections)
    doc_title = (title or "").strip() or topic
    stamp = generated or time.strftime("%Y-%m-%d %H:%M")
    slug = _report_slug(topic, name=name)

    report_dir = _unique_report_dir(context, slug)
    report_dir.mkdir(parents=True, exist_ok=True)

    html_text = render_report_html(topic, norm, title=doc_title, generated=stamp)
    html_file = report_dir / "report.html"
    html_file.write_text(html_text, encoding="utf-8")

    pdf_file = report_dir / "report.pdf"
    pdf_bytes = 0
    pdf_note = ""
    if include_pdf:
        try:
            from ..core.pdf import render_pdf

            md = compose_markdown(topic, norm, title=doc_title, generated=stamp)
            data = render_pdf(md, title=doc_title, headings=True,
                              chapter_break=True, toc=True)
            pdf_file.write_bytes(data)
            pdf_bytes = len(data)
        except Exception as exc:  # noqa: BLE001 - PDF is a bonus; say so, keep the HTML
            pdf_note = (f"PDF render skipped: {type(exc).__name__}: {exc}; "
                        "the HTML report is the deliverable")
            _log.warning("generate_report %s", pdf_note)
            pdf_file.unlink(missing_ok=True)

    rel = f"{REPORTS_DIR}/{report_dir.name}"
    created = zip_create(context, [rel], name=f"report-{report_dir.name}")
    zip_path = Path(created["path"])
    with zipfile.ZipFile(zip_path) as zf:
        zip_files = zf.namelist()

    bundle = ReportBundle(
        topic=topic,
        title=doc_title,
        slug=report_dir.name,
        sections=[s.title for s in norm],
        html_path=str(html_file),
        pdf_path=str(pdf_file) if pdf_bytes else "",
        zip_path=str(zip_path),
        zip_files=zip_files,
        html_bytes=html_file.stat().st_size,
        pdf_bytes=pdf_bytes,
        zip_bytes=zip_path.stat().st_size,
        seconds=round(time.perf_counter() - started, 2),
    )
    _log.info("report generated: %s (%d sections, html=%d B, pdf=%d B, zip=%d B)%s",
              bundle.slug, len(norm), bundle.html_bytes, pdf_bytes,
              bundle.zip_bytes, f" — {pdf_note}" if pdf_note else "")
    return bundle


# ── deliver: generate → zip → send ────────────────────────────────────────


def deliver_report(
    context: Any,
    topic: str,
    sections: Sequence[ReportSection | tuple[str, str] | dict[str, Any]],
    platform: str,
    chat_id: str,
    *,
    title: str = "",
    caption: str = "",
    include_pdf: bool = True,
    sender: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate the report, zip it, and send the archive to a chat — one call.

    The send edge is :func:`filesend.send_file` — the same live-gateway
    path the rest of the bot uses.  ``sender`` is injectable for tests
    (a fake records the zip bytes it would have sent).  Fails fast: an
    empty topic, empty sections, or a failed send raises :class:`ToolError`
    instead of reporting a phantom delivery.
    """
    from .filesend import send_file as _send_file

    started = time.perf_counter()
    if not (platform or "").strip():
        raise ToolError("deliver_report needs a platform (telegram | whatsapp | …)")
    if not (chat_id or "").strip():
        raise ToolError("deliver_report needs a chat_id")

    bundle = generate_report(context, topic, sections, title=title,
                             include_pdf=include_pdf)

    send = sender or _send_file
    try:
        sent = send(context, platform, chat_id, bundle.zip_path,
                    caption=caption or f"📄 {bundle.title}")
    except TypeError:
        # injected fakes may not accept the full keyword surface
        sent = send(context, platform, chat_id, bundle.zip_path)
    except Exception as exc:  # noqa: BLE001 - fail fast, but name the stage
        raise ToolError(f"report generated but send failed: {exc}") from exc
    if isinstance(sent, dict) and sent.get("sent") is False:
        raise ToolError(f"report generated but send reported failure: {sent}")

    out = bundle.to_dict()
    out.update({
        "ok": True,
        "platform": platform,
        "chat_id": chat_id,
        "caption": caption or f"📄 {bundle.title}",
        "message_id": (sent or {}).get("message_id", "")
        if isinstance(sent, dict) else "",
        "total_seconds": round(time.perf_counter() - started, 2),
    })
    return out


def register(registry: Any) -> None:
    """Attach the create-and-deliver report tool to a registry."""
    context = registry.context

    @registry.register(
        "deliver_report",
        description=(
            "Create-and-deliver in one call: generate a styled report from a "
            "topic plus (title, markdown body) sections — HTML plus a real "
            "PDF (pure-Python writer) — zip them, and send the archive to an "
            "active chat. Use when the owner asks for a report on something."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "topic": "str — what the report is about",
            "sections": "list — one per section: {'title': str, 'body': str (markdown)}",
            "platform": "str — telegram | whatsapp | discord | console | …",
            "chat_id": "str — the target chat id",
            "title": "str (optional) — report title (defaults to the topic)",
            "caption": "str (optional) — caption for the sent archive",
            "include_pdf": "bool (optional, True) — also render the PDF",
        },
    )
    def deliver(topic: str, sections: list, platform: str, chat_id: str, *,
                title: str = "", caption: str = "",
                include_pdf: bool = True) -> dict[str, Any]:
        return deliver_report(
            context, topic, sections, platform, chat_id, title=title,
            caption=caption, include_pdf=include_pdf)
