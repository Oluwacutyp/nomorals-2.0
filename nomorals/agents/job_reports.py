"""Job-report rendering (Formatting Pass F3).

The scheduler's alert for a finished job used to be the raw
``render_tool_value`` dump of the tool result — internal field names,
empty collections, zero counters, no summary.  The owner's finance-
alert screenshot, verbatim: ``finance_alert: / ok: yes / delivered:
yes / messages: [] / checked: 0 / fired: 0 / details: [] /
skipped_snoozed: 0``.  This module is the scheduler's render
boundary: it turns a tool result value into a presentation
:class:`~nomorals.social.chat.present.Doc` — summary first, human
labels, noise suppressed, detail expandable — which the scheduler
publishes through the Notifier's Doc path (Formatting F2).

Stored vs published — deliberately different forms:

* ``schedule_jobs.last_result`` keeps the legacy string
  (``"{tool}:\\n" + render_tool_value(...)``) byte-identical.  That
  string is a contract, not a rendering: the Pass #71/#75 suites pin
  its content (``last_result.startswith("finance_alert:")``,
  ``"fired: 1" in last_result``, failure text preserved), the
  scheduler's own failure-transition logic keys on the
  ``"job failed"`` prefix of its failure counterpart, the
  dependency gate and ``health()`` read the same prefix, and the
  jobs/status views slice it for display (their presentation is
  F7's sweep, not this pass).
* The Doc built here is the *published* form only.  The Notifier
  stores the Doc's canonical body on the notification row, so
  ``/notify`` and redelivery carry the report too.

Builders: hand-built Docs for the high-volume finance jobs
(``finance_alert`` — the screenshot case — and ``finance_digest``,
whose payload is a prose ``text`` field), and a generic fallback for
every other tool: schema-derived from the result itself (humanized
labels, empty collections hidden, numeric zero counters hidden,
lists capped via the ``Field.list_limit`` option F3 added to the
presentation layer), with the summary taken from the tool's own
summary-ish field when it wrote one, else composed from the first
visible facts, else an honest "finished — nothing to report" when
every field rendered empty.  No summary is ever invented beyond
what the result carries.

Nothing here raises: a formatting bug degrades to the generic Doc,
then to a minimal Doc — it must never fail the job itself (the
scheduler additionally guards the call).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from ..social.chat.present import (Doc, Field, KV, Section, from_dict,
                                   humanize_key)

_log = logging.getLogger(__name__)

__all__ = ["job_report_doc", "failure_doc"]

#: Lists longer than this render their head + an explicit tail count.
_LIST_LIMIT = 10
#: Long prose (a digest text, a fired-alert tail) caps its lines too.
_PROSE_LINE_LIMIT = 40

#: Dict keys a tool uses to state its own headline, in preference
#: order.  A short single-line value becomes the report's summary
#: verbatim (the tool's own words — never paraphrased into facts it
#: didn't state) and is not repeated as a row.
_SUMMARY_KEYS = ("summary", "message", "note", "headline")

#: Dict keys that carry long-form prose rather than a fact.
_PROSE_KEYS = ("text", "body", "content", "report")


def _count(value: Any) -> int:
    """An int counter out of a result field (bools are not counts)."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == int(value):
        return int(value)
    return 0


def _n(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _plain(value: Any) -> str:
    """One-line scalar text (bools yes/no — the house convention)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return " ".join(str(value).split())


# ── finance_alert — the owner's screenshot case ──────────────────────

def _finance_alert_doc(tool: str, value: Mapping[str, Any], *,
                       title: str) -> Doc:
    """``finance_alert(action="check")`` →
    ``{ok, delivered, messages, checked, fired, details,
    skipped_snoozed}`` (``finance/alerts.py::evaluate_alerts``).

    ``ok``/``delivered`` are classified presentation noise: success
    is the Doc's tone, and delivery is the tool's own per-alert
    publish (``price_alert`` kind), which the owner already sees.
    The two counters the job exists to report — checked and fired —
    always show, even at zero: at zero they ARE the report, and the
    summary line carries them too.
    """
    checked = _count(value.get("checked"))
    fired = _count(value.get("fired"))
    snoozed = _count(value.get("skipped_snoozed"))
    if fired:
        summary = (f"{_n(fired, 'alert')} fired "
                   f"of {_n(checked, 'alert')} checked.")
    elif checked:
        summary = f"{_n(checked, 'alert')} checked — none fired."
    elif snoozed:
        summary = (f"Nothing checked — {_n(snoozed, 'alert')} snoozed, "
                   f"none fired.")
    else:
        summary = "No alerts checked — none fired."
    rows = [KV("Checked", checked), KV("Fired", fired)]
    if snoozed:
        rows.append(KV("Skipped (snoozed)", snoozed))
    sections: list[Section] = [Section("Results", rows=tuple(rows))]
    # The fired alerts' own texts (title + body, as the tool published
    # them) are the detail — collapsed on Telegram, visible elsewhere.
    lines: list[str] = []
    for msg in value.get("messages") or []:
        lines.extend(str(msg).splitlines())
    lines = [line for line in lines if line.strip()]
    if not lines:
        for detail in value.get("details") or []:
            if isinstance(detail, Mapping):
                target = _plain(detail.get("target"))
                message = _plain(detail.get("message"))
                line = " — ".join(x for x in (target, message) if x)
                if line:
                    lines.append(line)
    if lines:
        shown = lines[:_PROSE_LINE_LIMIT]
        if len(lines) > _PROSE_LINE_LIMIT:
            shown = shown + [
                f"… (+{len(lines) - _PROSE_LINE_LIMIT} more lines)"]
        sections.append(Section("Fired alerts", quote=tuple(shown),
                                expandable=True))
    return Doc(title=title, icon="✅", tone="ok", summary=summary,
               sections=tuple(sections))


# ── finance_digest — a prose ``text`` payload ────────────────────────

def _finance_digest_doc(tool: str, value: Mapping[str, Any], *,
                        title: str) -> Doc:
    """``finance_digest`` → ``{ok, delivered, text}`` where ``text``
    is the digest prose itself (``finance/budgets.py::weekly_digest``
    plus the overspend / safe-to-spend / bills riders the tool
    appends).  The text is the report: its first figure-bearing line
    becomes the summary, the rest renders as body lines.  The tool's
    own header line (``💰 weekly money digest``) carries no figures
    and duplicates the Doc's title, so it folds away.
    """
    lines = str(value.get("text") or "").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and not any(ch.isdigit() for ch in lines[0]):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    summary = ""
    if lines:
        summary = lines[0].strip()
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines.pop()
    sections: list[Section] = []
    if any(line.strip() for line in lines):
        sections.append(Section(lines=tuple(lines)))
    delivered = value.get("delivered")
    if isinstance(delivered, bool):
        sections.append(Section(rows=(KV("Delivered", delivered),)))
    if not summary:
        summary = "Finance digest finished — nothing to report."
    return Doc(title=title, icon="✅", tone="ok", summary=summary,
               sections=tuple(sections))


# ── the generic fallback ─────────────────────────────────────────────

def _derive_summary(tool: str, doc: Doc, prose: list[str]) -> str:
    """A summary the data supports, for a result that stated none.

    Preference: the first visible facts (up to three KV rows — the
    ``Ok`` row is skipped: success is the tone, not a fact); then the
    first prose line; then "finished"; and when literally every
    field rendered empty, "finished — nothing to report" — which is
    exactly what an all-empty/zero result means.
    """
    facts: list[str] = []
    has_content = bool(prose)
    for section in doc.sections:
        for row in section.rows:
            text = _plain(row.value)
            if text and row.label != "Ok":
                facts.append(f"{row.label}: {text}")
        # a lone "Ok: yes" row is the tone restated, not content
        if any(r.label != "Ok" for r in section.rows) \
                or section.lines or section.quote:
            has_content = True
    if facts:
        return " · ".join(facts[:3])
    for line in prose:
        if line.strip():
            return line.strip()[:200]
    name = humanize_key(tool) or "Job"
    if has_content:
        return f"{name} finished."
    return f"{name} finished — nothing to report."


def _generic_doc(tool: str, value: Mapping[str, Any], *,
                 title: str) -> Doc:
    data = dict(value)
    summary = ""
    # 1) the tool's own headline, when it wrote a short one
    for key in _SUMMARY_KEYS:
        v = data.get(key)
        if isinstance(v, str) and v.strip() \
                and len(v.strip()) <= 240 and "\n" not in v.strip():
            summary = v.strip()
            del data[key]
            break
    # 2) a long prose field becomes an expandable detail section
    prose: list[str] = []
    for key in _PROSE_KEYS + _SUMMARY_KEYS:
        v = data.get(key)
        if isinstance(v, str) and v.strip() \
                and (len(v.strip()) > 240 or "\n" in v.strip()):
            prose = v.strip().splitlines()
            del data[key]
            break
    fields = [
        Field(str(k), hide_zero=isinstance(v, (int, float))
              and not isinstance(v, bool), list_limit=_LIST_LIMIT)
        for k, v in data.items()
    ]
    doc = from_dict(data, fields, title=title, icon="✅", tone="ok",
                    summary=summary, include_unlisted=False)
    sections = list(doc.sections)
    if prose:
        shown = prose[:_PROSE_LINE_LIMIT]
        if len(prose) > _PROSE_LINE_LIMIT:
            shown = shown + [
                f"… (+{len(prose) - _PROSE_LINE_LIMIT} more lines)"]
        sections.append(Section("Detail", quote=tuple(shown),
                                expandable=True))
    if not summary:
        summary = _derive_summary(tool, doc, prose)
    return Doc(title=title, icon="✅", tone="ok", summary=summary,
               sections=tuple(sections))


def _string_doc(tool: str, text: str, *, title: str) -> Doc:
    """A tool that returned prose: its first line leads (the lead is
    the summary), the rest stays as body lines — the string IS the
    human output already, so nothing is re-worded."""
    lines = text.splitlines()
    summary = ""
    rest: tuple[str, ...] = ()
    if len(lines) > 1 and lines[0].strip() and len(lines[0]) <= 200:
        summary = lines[0].strip()
        rest = tuple(lines[1:])
    elif lines and lines[0].strip() and len(lines[0]) <= 400:
        summary = lines[0].strip()
    elif lines:
        rest = tuple(lines)
    sections: list[Section] = []
    if any(line.strip() for line in rest):
        sections.append(Section(lines=rest))
    if not summary and not sections:
        summary = "No output."
    return Doc(title=title, icon="✅", tone="ok", summary=summary,
               sections=tuple(sections))


def _item_line(item: Any) -> str:
    if isinstance(item, Mapping):
        parts = [f"{humanize_key(k)}: {_plain(v)}"
                 for k, v in item.items()
                 if not isinstance(v, (Mapping, list, tuple))
                 and _plain(v)]
        return " · ".join(parts) if parts else _plain(dict(item))
    if isinstance(item, (list, tuple)):
        return ", ".join(_plain(x) for x in item)
    return _plain(item)


def _value_doc(tool: str, value: Any, *, title: str) -> Doc:
    """Non-dict, non-string results: None, a top-level list, a scalar."""
    if value is None:
        return Doc(title=title, icon="✅", tone="ok",
                   summary="No output.")
    if isinstance(value, (list, tuple)):
        lines = [_item_line(x) for x in value[:_LIST_LIMIT]]
        lines = [line for line in lines if line]
        if len(value) > _LIST_LIMIT:
            lines.append(f"… (+{len(value) - _LIST_LIMIT} more)")
        sections = (Section(lines=tuple(lines)),) if lines else ()
        return Doc(title=title, icon="✅", tone="ok",
                   summary=f"{_n(len(value), 'item')} returned.",
                   sections=sections)
    return Doc(title=title, icon="✅", tone="ok",
               summary=_plain(value) or "No output.")


_BUILDERS = {
    "finance_alert": _finance_alert_doc,
    "finance_digest": _finance_digest_doc,
}


def job_report_doc(tool: str, value: Any, *, job_name: str = "") -> Doc:
    """The published report for one successful tool run.

    Dispatches to the tool's hand builder when one exists, else the
    generic schema fallback (dicts), the prose path (strings), or
    the plain value path.  Never raises.
    """
    title = f"scheduled: {job_name}" if job_name else f"scheduled: {tool}"
    try:
        builder = _BUILDERS.get(tool)
        if builder is not None and isinstance(value, Mapping):
            return builder(tool, value, title=title)
        if isinstance(value, str):
            return _string_doc(tool, value, title=title)
        if isinstance(value, Mapping):
            return _generic_doc(tool, value, title=title)
        return _value_doc(tool, value, title=title)
    except Exception:  # noqa: BLE001 - degrade, never fail the job
        _log.debug("job_reports: builder failed for %s; minimal Doc",
                   tool, exc_info=True)
    try:
        return Doc(title=title, icon="✅", tone="ok",
                   summary=f"{humanize_key(tool) or 'Job'} finished.",
                   overflow_text=str(value)[:2000])
    except Exception:  # noqa: BLE001
        return Doc(title=title, icon="✅", tone="ok")


def failure_doc(job_name: str, summary: str) -> Doc:
    """The published report for a failed run: tone error, and the
    summary is the run's own failure text (``"job failed: …"``) —
    the failure named plainly, exactly as the stored record states
    it.  The ``last_result`` string itself is untouched by this Doc;
    it exists only for the publish path."""
    return Doc(title=f"scheduled: {job_name}", icon="❌", tone="error",
               summary=str(summary or ""))
