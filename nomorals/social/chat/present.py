"""The shared presentation layer (Formatting Pass F2).

One formatter, per-transport renderers. Producers describe *what* —
a :class:`Doc` — and never how it renders on Telegram::

    from nomorals.social.chat.present import Doc, Section, KV, render

    doc = Doc(
        title="Finance alerts", icon="⏰", tone="ok",
        summary="3 price alerts checked — none fired.",
        sections=[Section("Results",
                          rows=[KV("Checked", "3"), KV("Fired", "0")])],
        footer="next run", footer_unix=1792905600.0,
        actions=[("Jobs", "schedule list")],
    )
    text, parse_mode = render(doc, "telegram-bot")   # HTML + "HTML"
    text, _          = render(doc, "whatsapp")       # *bold* dialect

This module extends — never replaces — the four existing formatting
pieces (Phase-0 survey §E.1):

* :mod:`.platforms` converters are the renderers' engine — every prose
  field (summary, lines, quote lines, footer) is transcoded per
  transport by ``to_telegram`` / ``to_whatsapp`` / ``to_markdown`` /
  ``to_plain``, so producer text is escaped correctly everywhere and
  the house dialect (``**bold**``, `` `code` ``, ``[text](url)``)
  keeps working inside a Doc exactly as it does in a plain string.
* :mod:`.style`'s vocabulary (``▍`` quote prefix, the divider) shapes
  the plain/terminal rendering.
* :mod:`nomorals.social.format_conformance` is the guardrail:
  :func:`chunks` splits rendered output with its markup-balanced
  splitter, and the F2 conformance tests run its ``check_message``
  over every chunk of every renderer.
* :mod:`nomorals.core.output_style`'s header + divider discipline is
  folded into the plain renderer (title, divider, ``Label: value``
  rows). The module itself stays — chat tool results still use it;
  F7 owns that convergence.

Structural text (titles, KV labels, section heads) is *data*, not
prose: it is escaped literally per dialect, never dialect-converted,
so a label can neither inject markup nor break a marker pair.

Renderers never raise (the platforms.py contract): an unexpected
failure degrades to the plain rendering, then to the bare title.
Construction, by contrast, fails fast — an unknown ``tone`` is a
producer bug and raises immediately.
"""

from __future__ import annotations

import html as _html
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from .platforms import to_markdown, to_plain, to_telegram, to_whatsapp

_log = logging.getLogger(__name__)

__all__ = [
    "Doc", "Section", "KV", "Field", "Rendered",
    "TONE_ICONS", "TONES",
    "render", "render_text", "chunks", "body_text", "actions_rows",
    "from_dict", "humanize_key",
]

#: tone → banner icon. The same status vocabulary as
#: ``agents/render.py``'s ICONS (✅/⚠️/❌/ℹ️) — one visual language;
#: duplicated here because agents (L5) sits above social (L4).
TONE_ICONS: dict[str, str] = {
    "ok": "✅",
    "warn": "⚠️",
    "error": "❌",
    "info": "ℹ️",
}

#: accepted tones (``fail`` is accepted as render.ICONS' spelling of
#: the error tone and normalized to ``error``).
TONES = frozenset(TONE_ICONS)

_TONE_ALIASES = {"fail": "error"}

#: The output_style divider — the plain renderer's header rule.
_DIVIDER = "━━━━━━━━━━━━━━━"


# ── the document model ───────────────────────────────────────────────

@dataclass(frozen=True)
class KV:
    """One label/value row. ``value`` is data: stringified at render
    (bools as yes/no), escaped literally, never dialect-converted.
    ``mono`` renders the value as code on rich platforms; ``url``
    turns it into a link."""

    label: str
    value: Any = ""
    mono: bool = False
    url: str = ""


def _coerce_kv(item: Any) -> KV:
    if isinstance(item, KV):
        return item
    if isinstance(item, (tuple, list)):
        if len(item) == 2:
            return KV(str(item[0]), item[1])
        if len(item) == 3:
            return KV(str(item[0]), item[1], mono=bool(item[2]))
    raise TypeError(
        f"section rows must be KV or (label, value) pairs, got {item!r}")


@dataclass(frozen=True)
class Section:
    """A named block of a Doc: KV rows, prose lines, and/or quote
    (detail) lines. ``expandable`` collapses the quote block on
    Telegram (``<blockquote expandable>``); other transports render
    the same lines visibly — they have no collapse affordance."""

    title: str = ""
    rows: tuple[KV, ...] = ()
    lines: tuple[str, ...] = ()
    quote: tuple[str, ...] = ()
    expandable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rows", tuple(_coerce_kv(r) for r in self.rows or ()))
        object.__setattr__(
            self, "lines", tuple(str(x) for x in self.lines or ()))
        object.__setattr__(
            self, "quote", tuple(str(x) for x in self.quote or ()))


@dataclass(frozen=True)
class Doc:
    """A platform-neutral owner-facing document.

    ``summary`` is always human and always first. ``footer_unix``
    (epoch seconds) renders as a ``<tg-time>`` element on Telegram —
    localized by the client — and as a labeled UTC string elsewhere.
    ``actions`` are gateway buttons: ``(label, callback_data)`` pairs
    (one row) or a list of rows. ``overflow_text`` is the
    full-fidelity tail (the document-route payload): Telegram folds
    it into a trailing expandable quote; other transports append it
    as a final plain block.
    """

    title: str
    icon: str = ""
    tone: str = "info"
    summary: str = ""
    sections: tuple[Section, ...] = ()
    footer: str = ""
    footer_unix: float | None = None
    actions: tuple = ()
    overflow_text: str = ""

    def __post_init__(self) -> None:
        tone = _TONE_ALIASES.get(str(self.tone or "info").lower(),
                                 str(self.tone or "info").lower())
        if tone not in TONE_ICONS:
            raise ValueError(
                f"Doc tone must be one of {sorted(TONE_ICONS)}, "
                f"got {self.tone!r}")
        object.__setattr__(self, "tone", tone)
        object.__setattr__(
            self, "sections", tuple(self.sections or ()))
        object.__setattr__(self, "actions", _normalize_actions(self.actions))

    @property
    def resolved_icon(self) -> str:
        """The explicit icon wins; otherwise the tone's icon."""
        return self.icon or TONE_ICONS[self.tone]


def _is_pair(item: Any) -> bool:
    return (isinstance(item, (list, tuple)) and len(item) == 2
            and all(isinstance(x, str) for x in item))


def _normalize_actions(actions: Any) -> tuple:
    """Canonical internal form: a tuple of rows, each row a tuple of
    ``(label, callback_data)`` pairs. Accepts a flat pair list (one
    row) or a list of rows. Anything else is a producer bug."""
    items = list(actions or ())
    if not items:
        return ()
    if _is_pair(items[0]):
        return (tuple((str(a), str(b)) for a, b in items),)
    rows = []
    for row in items:
        if not isinstance(row, (list, tuple)):
            raise TypeError(
                f"Doc actions must be (label, callback) pairs or rows "
                f"of pairs, got {row!r}")
        rows.append(tuple((str(a), str(b)) for a, b in row))
    return tuple(rows)


def actions_rows(doc: Doc) -> list[list[tuple[str, str]]]:
    """The Doc's actions as gateway button rows (``[]`` = none)."""
    return [[(str(a), str(b)) for a, b in row]
            for row in doc.actions or () if row]


# ── schema-driven dict rendering ─────────────────────────────────────

@dataclass(frozen=True)
class Field:
    """One field of a dict→Doc schema.

    ``label`` defaults to the humanized key (``skipped_snoozed`` →
    "Skipped snoozed"). ``hide_empty`` drops None/""/[]/{} values;
    ``hide_zero`` additionally drops numeric 0. ``fmt`` (value → str)
    overrides the default stringification; ``unit`` is appended after
    the formatted value; ``mono`` code-styles the value; ``fields``
    is the nested schema for a dict value (default: humanized keys,
    empties hidden).
    """

    key: str
    label: str = ""
    hide_empty: bool = True
    hide_zero: bool = False
    fmt: Callable[[Any], Any] | None = None
    unit: str = ""
    mono: bool = False
    fields: tuple["Field", ...] | None = None


def humanize_key(key: Any) -> str:
    """``skipped_snoozed`` → ``Skipped snoozed``. Never raises."""
    text = str(key or "").replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    return text[:1].upper() + text[1:] if text else ""


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return len(value) == 0
    return False


def _is_zero(value: Any) -> bool:
    return (isinstance(value, (int, float))
            and not isinstance(value, bool) and value == 0)


def _scalar_text(value: Any) -> str:
    """The one stringification used by every renderer (bools yes/no,
    floats trimmed — the house ``ok: yes`` convention)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)  # nan/inf: no trimmed form, never raise
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _format_value(fld: Field, value: Any) -> str:
    if fld.fmt is not None:
        try:
            text = _scalar_text(fld.fmt(value))
        except Exception:  # noqa: BLE001 - a bad fmt must not eat the doc
            _log.debug("present: fmt for field %r failed; using default",
                       fld.key, exc_info=True)
            text = _scalar_text(value)
    else:
        text = _scalar_text(value)
    if fld.unit and text:
        text = f"{text} {fld.unit}"
    return text


def _coerce_fields(fields: Any) -> list[Field]:
    if fields is None:
        return []
    if isinstance(fields, Mapping):
        out: list[Field] = []
        for key, spec in fields.items():
            if spec is None:
                out.append(Field(str(key)))
            elif isinstance(spec, Field):
                out.append(spec)
            elif isinstance(spec, Mapping):
                out.append(Field(key=str(key), **dict(spec)))
            else:
                raise TypeError(
                    f"field spec for {key!r} must be Field/dict/None, "
                    f"got {spec!r}")
        return out
    return [f if isinstance(f, Field) else Field(str(f)) for f in fields]


def _dict_rows(data: Mapping[str, Any],
               fields: list[Field]) -> tuple[list[KV], list[Section]]:
    """(rows, extra sections) for one dict level under ``fields``."""
    rows: list[KV] = []
    sections: list[Section] = []
    for fld in fields:
        if fld.key not in data:
            continue
        value = data[fld.key]
        label = fld.label or humanize_key(fld.key)
        if _is_empty(value):
            if not fld.hide_empty:
                rows.append(KV(label, "", mono=fld.mono))
            continue
        if _is_zero(value) and fld.hide_zero:
            continue
        if isinstance(value, Mapping):
            sub_fields = (list(fld.fields) if fld.fields is not None
                          else [Field(str(k)) for k in value])
            sub_rows, sub_sections = _dict_rows(value, sub_fields)
            if sub_rows or sub_sections:
                sections.append(Section(label, rows=tuple(sub_rows)))
                sections.extend(sub_sections)
            elif not fld.hide_empty:
                rows.append(KV(label, ""))
            continue
        if isinstance(value, (list, tuple)):
            scalars = [x for x in value if not isinstance(x, Mapping)]
            dicts = [x for x in value if isinstance(x, Mapping)]
            if scalars and not dicts:
                text = ", ".join(_scalar_text(x) for x in scalars
                                 if not _is_empty(x))
                if text:
                    rows.append(KV(label, text, mono=fld.mono))
                elif not fld.hide_empty:
                    rows.append(KV(label, "", mono=fld.mono))
            if dicts:
                lines = []
                for item in dicts:
                    parts = [f"{humanize_key(k)}: {_scalar_text(v)}"
                             for k, v in item.items()
                             if not _is_empty(v)
                             and not isinstance(v, (Mapping, list, tuple))]
                    lines.append(" · ".join(parts) if parts
                                 else _scalar_text(item))
                if scalars:
                    rows.append(KV(
                        label, ", ".join(_scalar_text(x) for x in scalars)))
                if lines:
                    sections.append(Section(label, lines=tuple(lines)))
            continue
        rows.append(KV(label, _format_value(fld, value), mono=fld.mono))
    return rows, sections


def from_dict(data: Mapping[str, Any], fields: Any = None, *,
              title: str = "", icon: str = "", tone: str = "info",
              summary: str = "", section_title: str = "",
              include_unlisted: bool = True) -> Doc:
    """Build a Doc from a result dict + a field schema.

    This is the path that retires ``core/tool_render.render_tool_value``
    at owner-facing boundaries: internal keys become human labels
    (``skipped_snoozed`` → "Skipped snoozed"), empty collections and
    (per-field) zero counters are suppressed instead of dumped, nested
    dicts become sections. Keys missing from an explicit schema are
    still rendered — humanized, empties hidden — unless
    ``include_unlisted=False``; nothing is ever silently dropped that
    the schema didn't classify. ``render_tool_value`` itself stays
    as the internal/debug renderer.
    """
    if not isinstance(data, Mapping):
        raise TypeError(
            f"from_dict needs a mapping, got {type(data).__name__}")
    schema = _coerce_fields(fields)
    if not schema:
        schema = [Field(str(k)) for k in data]
    elif include_unlisted:
        covered = {f.key for f in schema}
        schema = schema + [Field(str(k)) for k in data
                           if str(k) not in covered]
    rows, sections = _dict_rows(data, schema)
    all_sections: list[Section] = []
    if rows or section_title:
        all_sections.append(Section(section_title, rows=tuple(rows)))
    all_sections.extend(sections)
    return Doc(title=title, icon=icon, tone=tone, summary=summary,
               sections=tuple(all_sections))


# ── renderers ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rendered:
    """One rendered Doc: the text plus the parse mode the transport
    must be told to use (``"HTML"`` for the Telegram family, ``""``
    everywhere else)."""

    text: str
    parse_mode: str = ""
    platform: str = ""


def _family(platform: Any) -> tuple[str, str]:
    """platform name → (dialect, parse_mode). Unknown → plain, the
    rendering that is safe on every surface."""
    p = str(platform or "").strip().lower()
    if p.startswith("telegram") or p in ("tg", "tgbot", "tgpersonal"):
        return "html", "HTML"
    if p in ("whatsapp", "wa"):
        return "whatsapp", ""
    if p.startswith("discord") or p in ("web", "webapp"):
        return "markdown", ""
    return "plain", ""


def _prose(text: str, dialect: str) -> str:
    """Transcode one prose field through the platforms engine."""
    if dialect == "html":
        return to_telegram(text)
    if dialect == "whatsapp":
        return to_whatsapp(text)
    if dialect == "markdown":
        return to_markdown(text)
    if dialect == "canonical":
        return text
    return to_plain(text)


def _literal(text: Any, dialect: str) -> str:
    """Escape one structural string (title/label/head) per dialect.
    Structural text is data: on marker dialects the characters that
    would break the wrapper are dropped; on HTML it is escaped."""
    text = _scalar_text(text)
    if dialect == "html":
        return _html.escape(text)
    if dialect in ("whatsapp", "markdown", "canonical"):
        return text.replace("*", "").replace("`", "")
    return text


def _time_text(unix: float) -> str:
    """The labeled fallback string for a footer timestamp."""
    try:
        dt = datetime.fromtimestamp(float(unix), tz=timezone.utc)
        return dt.strftime("%a %d %b %H:%M UTC")
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def _kv_value(kv: KV, dialect: str) -> str:
    value = _scalar_text(kv.value)
    if dialect == "html":
        esc = _html.escape(value)
        if kv.url:
            return (f'<a href="{_html.escape(kv.url, quote=True)}">'
                    f"{esc}</a>")
        return f"<code>{esc}</code>" if kv.mono else esc
    if dialect == "whatsapp":
        if kv.url:
            return f"{value}: {kv.url}"
        return f"```{value}```" if kv.mono else value
    if dialect == "markdown":
        if kv.url:
            return f"[{value}]({kv.url})"
        return f"`{value}`" if kv.mono else value
    if dialect == "canonical":
        if kv.url:
            return f"[{value}]({kv.url})"
        return f"`{value}`" if kv.mono else value
    # plain
    if kv.url and value != kv.url:
        return f"{value}: {kv.url}"
    return value


def _kv_line(kv: KV, dialect: str) -> str:
    label = _literal(kv.label, dialect)
    value = _kv_value(kv, dialect)
    if dialect == "html":
        return f"<b>{label}</b>: {value}"
    if dialect == "whatsapp":
        return f"*{label}*: {value}"
    if dialect in ("markdown", "canonical"):
        return f"**{label}**: {value}"
    return f"{label}: {value}"


def _section_lines(section: Section, dialect: str) -> list[str]:
    lines: list[str] = []
    if section.title:
        head = _literal(section.title, dialect)
        if dialect == "html":
            lines.append(f"<b>{head}</b>")
        elif dialect == "whatsapp":
            lines.append(f"*{head}*")
        elif dialect in ("markdown", "canonical"):
            lines.append(f"**{head}**")
        else:
            lines.append(head)
    for kv in section.rows:
        line = _kv_line(kv, dialect)
        lines.append(f"  {line}" if dialect == "plain" else line)
    lines.extend(_prose(x, dialect) for x in section.lines)
    if section.quote:
        if dialect == "html":
            tag = ("<blockquote expandable>" if section.expandable
                   else "<blockquote>")
            inner = "\n".join(_prose(x, dialect) for x in section.quote)
            lines.append(f"{tag}{inner}</blockquote>")
        elif dialect == "markdown":
            lines.extend(f"> {_prose(x, dialect)}"
                         for x in section.quote)
        elif dialect == "whatsapp" or dialect == "plain":
            lines.extend(f"▍ {_prose(x, dialect)}"
                         for x in section.quote)
        else:  # canonical
            lines.extend(f"> {_prose(x, dialect)}"
                         for x in section.quote)
    return lines


def _render(doc: Doc, dialect: str, *, include_title: bool = True) -> str:
    groups: list[str] = []
    if include_title and (doc.title or doc.resolved_icon):
        icon = doc.resolved_icon
        title = _literal(doc.title, dialect)
        head = f"{icon} {title}".strip()
        if dialect == "html":
            head = (f"{icon} <b>{title}</b>" if title
                    else icon)
        elif dialect == "whatsapp":
            head = f"{icon} *{title}*" if title else icon
        elif dialect in ("markdown", "canonical"):
            head = f"{icon} **{title}**" if title else icon
        if dialect == "plain":
            head = f"{head}\n{_DIVIDER}"  # output_style: header+rule
        groups.append(head)
    if doc.summary:
        groups.append(_prose(doc.summary, dialect))
    for section in doc.sections:
        block = _section_lines(section, dialect)
        if block:
            groups.append("\n".join(block))
    if doc.footer or doc.footer_unix is not None:
        groups.append(_footer(doc, dialect))
    if doc.overflow_text:
        groups.append(_overflow(doc, dialect))
    return "\n\n".join(g for g in groups if g)


def _footer(doc: Doc, dialect: str) -> str:
    text = _prose(doc.footer, dialect) if doc.footer else ""
    stamp = _time_text(doc.footer_unix) if doc.footer_unix is not None \
        else ""
    if dialect == "html":
        parts: list[str] = []
        if text:
            parts.append(f"<i>{text}</i>")
        if doc.footer_unix is not None:
            parts.append(
                f'<tg-time unix="{int(doc.footer_unix)}" '
                f'format="wDT">{_html.escape(stamp)}</tg-time>')
        return " ".join(parts)
    joined = " · ".join(x for x in (text, stamp) if x)
    if dialect == "whatsapp" and joined:
        return f"_{joined}_"
    if dialect == "markdown" and joined:
        return f"_{joined}_"
    return joined


def _overflow(doc: Doc, dialect: str) -> str:
    if dialect == "html":
        return ("<blockquote expandable>"
                f"{to_telegram(doc.overflow_text)}</blockquote>")
    if dialect == "whatsapp":
        return to_whatsapp(doc.overflow_text)
    if dialect == "markdown":
        return to_markdown(doc.overflow_text)
    if dialect == "canonical":
        return doc.overflow_text
    return doc.overflow_text


def render(doc: Doc, platform: Any = "telegram-bot") -> Rendered:
    """Render ``doc`` for ``platform``. Never raises: an unexpected
    failure degrades to the plain rendering, then to the bare title —
    a formatting bug must never eat the message itself."""
    dialect, parse_mode = _family(platform)
    try:
        return Rendered(_render(doc, dialect), parse_mode,
                        str(platform or ""))
    except Exception:  # noqa: BLE001 - degrade, never drop
        _log.debug("present: %s render failed; degrading to plain",
                   dialect, exc_info=True)
    try:
        return Rendered(_render(doc, "plain"), "", str(platform or ""))
    except Exception:  # noqa: BLE001
        return Rendered(str(getattr(doc, "title", "") or ""), "",
                        str(platform or ""))


def render_text(doc: Doc, platform: Any = "telegram-bot") -> str:
    """Convenience: just the rendered text for ``platform``."""
    return render(doc, platform).text


def body_text(doc: Doc) -> str:
    """The stored form of a Doc alert: the canonical house-dialect
    body *without* the title (the Notifier stores title and body
    separately and its banner re-adds the title on redelivery, where
    the gateway's F1 presentation converts the dialect exactly like
    any string alert)."""
    try:
        return _render(doc, "canonical", include_title=False)
    except Exception:  # noqa: BLE001
        return _render(doc, "plain", include_title=False)


def chunks(doc: Doc, platform: Any = "telegram-bot") -> list[Rendered]:
    """The rendered Doc split for ``platform`` with the balanced
    splitter (:func:`nomorals.social.format_conformance.chunk`):
    every chunk is independently valid markup for its transport and
    within the platform cap (10 chars reserved for the gateway's
    ``(i/n)`` labels, mirroring the gateway's formatted path)."""
    rendered = render(doc, platform)
    try:
        from ..format_conformance import chunk as _chunk
        from ..format_conformance import max_text_for

        budget = max(64, int(max_text_for(str(platform or ""))) - 10)
        parts = _chunk(rendered.text, str(platform or ""),
                       max_len=budget)
        if parts:
            return [Rendered(p, rendered.parse_mode, rendered.platform)
                    for p in parts]
    except Exception:  # noqa: BLE001 - fall back to the whole render
        _log.debug("present: balanced chunking failed", exc_info=True)
    return [rendered]
