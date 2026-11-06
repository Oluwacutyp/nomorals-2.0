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
whose payload is a prose ``text`` field), for the agency shift
(``__agency_shift__`` — Formatting F4's autonomy digest: the nested
cycle/expansion structure rendered once per fact, the triplicated
``knowledge_items`` collapsed to the freshest count), and a generic
fallback for every other tool: schema-derived from the result itself
(humanized labels, empty collections hidden, numeric zero counters
hidden, lists capped via the ``Field.list_limit`` option F3 added to
the presentation layer), with the summary taken from the tool's own
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

__all__ = ["job_report_doc", "failure_doc",
           "command_report_doc", "command_failure_doc",
           "tool_result_doc", "summarize_last_result", "noop_reason"]

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


# ── scheduled silence (owner law, 2026-11-05) ────────────────────────
#
# A scheduled job with nothing to report sends NOTHING: the run is
# journaled (its stored last_result string is untouched — the jobs
# views still show every run), but no report posts to the owner's
# chats.  The finance-alerts "Checked: 0 / Fired: 0" post is the
# named class.  Findings and owner-actionable failures still post;
# the classification below is conservative by construction — only
# shapes that provably carry no finding are silent, and a shape it
# doesn't understand is never silenced.

#: Dict keys that state the run's own bookkeeping, never a finding.
_NOISE_KEYS = frozenset({"ok", "delivered"})


def _value_is_empty(value: Any) -> bool:
    """A result field carrying nothing: None, blank, empty
    collection, numeric zero, False."""
    if value is None:
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def noop_reason(tool: str, value: Any) -> str | None:
    """Why a successful run has nothing to report — None when it
    does (and therefore posts).  Never raises.
    """
    try:
        if value is None:
            return "no output"
        if isinstance(value, bool):
            return "no findings" if value else None
        if isinstance(value, str):
            return "no output" if not value.strip() else None
        if isinstance(value, (list, tuple)):
            return "empty result" if not value else None
        if isinstance(value, (int, float)):
            return "zero result" if value == 0 else None
        if not isinstance(value, Mapping):
            return None
        if tool == "finance_alert":
            # The named class: nothing checked, nothing fired (a
            # snoozed-only run is the same nothing — the snooze was
            # the owner's own act).  A run that CHECKED and found
            # nothing still posts its negative; a fired alert
            # always posts (and also publishes itself individually).
            if _count(value.get("fired")) == 0 \
                    and _count(value.get("checked")) == 0:
                return "no alerts checked, none fired"
            return None
        if tool == "finance_digest":
            if not str(value.get("text") or "").strip():
                return "empty digest"
            return None
        if tool == "__agency_shift__":
            shift = value.get("shift")
            cycle = (shift.get("cycle")
                     if isinstance(shift, Mapping) else None)
            if isinstance(shift, Mapping) \
                    and isinstance(cycle, Mapping):
                expansion = value.get("expansion")
                expansion_ran = isinstance(expansion, Mapping)
                if not expansion_ran:
                    # The missing expansion is itself a finding
                    # (the digest's warn tone) — never silenced.
                    return None
                triggers = _name_list(shift.get("triggers"))
                fired = _count(shift.get("triggers_fired")) \
                    or len(triggers)
                quiet = (
                    _count(cycle.get("systems_run")) == 0
                    and _count(cycle.get("useful_turns")) == 0
                    and fired == 0
                    and _count(expansion.get("systems_run")) == 0)
                return "quiet shift — nothing ran, nothing fired" \
                    if quiet else None
            # unparseable shift → the generic rule below
        content = [k for k, v in value.items()
                   if k not in _NOISE_KEYS and not _value_is_empty(v)]
        if not content:
            return "nothing to report"
        return None
    except Exception:  # noqa: BLE001 - unclassifiable posts, never hides
        _log.debug("job_reports: noop classification failed for %s",
                   tool, exc_info=True)
        return None


def _stamped(doc: Doc) -> Doc:
    """The always-present stamp (owner law, 2026-11-05) rides the
    Doc's footer slot — the presentation layer renders a footer
    quiet-italic per platform (``<i>`` on Telegram, ``_…_`` on
    WhatsApp).  A producer-set footer is never overwritten."""
    try:
        if doc.footer:
            return doc
        from dataclasses import replace

        from ..social.chat.typography import DEFAULT_STAMP_TEXT

        return replace(doc, footer=DEFAULT_STAMP_TEXT)
    except Exception:  # noqa: BLE001 - the stamp never fails a report
        return doc


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


# ── __agency_shift__ — the autonomy digest (Formatting F4) ──────────

def _system_universes() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The system-name universes the two cycles rotate over.

    Imported lazily from the agency layer (agents L5 → agency L0 is
    downward; the agency layer itself must never import this L5
    presentation code).  Guarded: without the names the digest
    degrades to counts only — it never invents a total.
    """
    core: tuple[str, ...] = ()
    expansion: tuple[str, ...] = ()
    try:
        from ..agency.systems import SYSTEM_NAMES

        core = tuple(str(n) for n in SYSTEM_NAMES)
    except Exception:  # noqa: BLE001 - names are context, not content
        pass
    try:
        from ..agency.expansion import EXPANSION_SYSTEMS

        expansion = tuple(str(n) for n in EXPANSION_SYSTEMS)
    except Exception:  # noqa: BLE001
        pass
    return core, expansion


def _name_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    return []


def _stood_down_names(universe: tuple[str, ...], ran: list[str],
                      stood_down: int) -> list[str]:
    """The stood-down systems' names — ONLY when they reconcile.

    ``stood_down`` counts systems with no work at the cycle's final
    runnable computation.  A system WITH work can still miss its
    turn (rotation cooldown), so universe − ran is not automatically
    the stood-down set: the names are listed only when their count
    equals the reported ``stood_down`` exactly.  Otherwise the count
    stands alone — the digest never guesses a name.
    """
    if not universe or stood_down <= 0:
        return []
    missing = [n for n in universe if n not in set(ran)]
    return missing if len(missing) == stood_down else []


def _cycle_section(heading: str, cycle: Mapping[str, Any],
                   universe: tuple[str, ...]) -> Section:
    """One rotation cycle as a digest section: counts as rows, the
    per-system lines (ran / stood down) underneath."""
    ran = _name_list(cycle.get("ran"))
    run = _count(cycle.get("systems_run"))
    total = len(universe)
    rows = [KV("Systems run", f"{run} of {total}" if total else run),
            KV("Useful turns", _count(cycle.get("useful_turns"))),
            KV("Stood down", _count(cycle.get("stood_down")))]
    preemptions = _count(cycle.get("preemptions"))
    if preemptions:
        rows.append(KV("Preemptions", preemptions))
    lines: list[str] = []
    if ran:
        lines.append("Ran: " + " · ".join(ran))
    down = _stood_down_names(universe, ran,
                             _count(cycle.get("stood_down")))
    if down:
        lines.append("Stood down: " + " · ".join(down))
    return Section(heading, rows=tuple(rows), lines=tuple(lines))


def _agency_shift_doc(tool: str, value: Mapping[str, Any], *,
                      title: str) -> Doc:
    """``__agency_shift__`` → ``{ok, shift, expansion}`` where
    ``shift`` is ``ActiveSystems.work_shift()`` and ``expansion`` is
    ``ExpansionSystems.run_cycle()`` (or None — the tool swallows an
    expansion failure into None).

    The raw dump of this value printed ``knowledge_items`` three
    times (the shift, its nested cycle, and the expansion each carry
    a copy) and both cycles' counters as bare keys.  The digest
    renders each fact once: the two cycles as parallel sections with
    human labels and per-system lines, the fired triggers by name,
    and the knowledge-base size ONCE — the expansion's count when
    the expansion ran (it is read last), else the shift's.

    The cycle dicts carry no per-system failure field (tick failures
    are absorbed inside the agency layer; a machinery failure fails
    the whole run and publishes through :func:`failure_doc`), so the
    only in-value degradation is the missing expansion: it is named
    plainly and carries the warn tone.  A cycle where everything
    stood down is a quiet shift, not an error.
    """
    shift = value.get("shift")
    cycle = shift.get("cycle") if isinstance(shift, Mapping) else None
    if not isinstance(shift, Mapping) or not isinstance(cycle, Mapping):
        return _generic_doc(tool, value, title=title)
    core_names, expansion_names = _system_universes()
    expansion = value.get("expansion")
    expansion_ran = isinstance(expansion, Mapping)

    triggers = _name_list(shift.get("triggers"))
    fired = _count(shift.get("triggers_fired")) or len(triggers)
    run = _count(cycle.get("systems_run"))
    useful = _count(cycle.get("useful_turns"))
    stood = _count(cycle.get("stood_down"))

    # the knowledge-base size, once: the freshest copy wins
    knowledge: int | None = None
    if expansion_ran and "knowledge_items" in expansion:
        knowledge = _count(expansion.get("knowledge_items"))
    elif "knowledge_items" in shift:
        knowledge = _count(shift.get("knowledge_items"))

    sections: list[Section] = [
        _cycle_section("Cycle", cycle, core_names)]
    if triggers:
        sections.append(Section("Triggers", lines=tuple(triggers)))
    elif fired:
        sections.append(Section(
            "Triggers", rows=(KV("Fired", fired),)))
    if expansion_ran:
        sections.append(
            _cycle_section("Expansion", expansion, expansion_names))
    else:
        sections.append(Section("Expansion", lines=(
            "Did not run — the expansion cycle was unavailable "
            "this shift.",)))
    if knowledge is not None:
        sections.append(Section(
            rows=(KV("Knowledge base", f"{knowledge} items"),)))

    quiet = (run == 0 and useful == 0 and fired == 0
             and (not expansion_ran
                  or _count(expansion.get("systems_run")) == 0))
    if quiet:
        summary = ("Quiet shift — no triggers fired and every "
                   "system stood down")
        if not expansion_ran:
            summary += "; the expansion cycle did not run"
        summary += "."
    else:
        total = len(core_names)
        core_part = (f"{run} of {total} systems ran" if total
                     else f"{_n(run, 'system')} ran")
        if run and not useful:
            core_part += " (no useful turns)"
        elif useful:
            core_part += f" ({_n(useful, 'useful turn')})"
        core_part += f", {stood} stood down"
        parts: list[str] = []
        if fired:
            parts.append(f"{_n(fired, 'trigger')} fired")
        parts.append(core_part)
        if expansion_ran:
            e_run = _count(expansion.get("systems_run"))
            e_total = len(expansion_names)
            parts.append(f"expansion ran {e_run} of {e_total}"
                         if e_total else f"expansion ran {e_run}")
        else:
            parts.append("expansion did not run")
        summary = "; ".join(parts) + "."
    tone = "ok" if expansion_ran else "warn"
    return Doc(title=title, icon="✅" if expansion_ran else "⚠️",
               tone=tone, summary=summary, sections=tuple(sections))


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
    "__agency_shift__": _agency_shift_doc,
}


def job_report_doc(tool: str, value: Any, *, job_name: str = "") -> Doc:
    """The published report for one successful tool run.

    Dispatches to the tool's hand builder when one exists, else the
    generic schema fallback (dicts), the prose path (strings), or
    the plain value path.  Never raises.
    """
    title = f"scheduled: {job_name}" if job_name else f"scheduled: {tool}"
    doc: Doc | None = None
    try:
        builder = _BUILDERS.get(tool)
        if builder is not None and isinstance(value, Mapping):
            doc = builder(tool, value, title=title)
        elif isinstance(value, str):
            doc = _string_doc(tool, value, title=title)
        elif isinstance(value, Mapping):
            doc = _generic_doc(tool, value, title=title)
        else:
            doc = _value_doc(tool, value, title=title)
    except Exception:  # noqa: BLE001 - degrade, never fail the job
        _log.debug("job_reports: builder failed for %s; minimal Doc",
                   tool, exc_info=True)
    if doc is None:
        try:
            doc = Doc(title=title, icon="✅", tone="ok",
                      summary=f"{humanize_key(tool) or 'Job'} finished.",
                      overflow_text=str(value)[:2000])
        except Exception:  # noqa: BLE001
            doc = Doc(title=title, icon="✅", tone="ok")
    return _stamped(doc)


def failure_doc(job_name: str, summary: str) -> Doc:
    """The published report for a failed run: tone error, and the
    summary is the run's own failure text (``"job failed: …"``) —
    the failure named plainly, exactly as the stored record states
    it.  The ``last_result`` string itself is untouched by this Doc;
    it exists only for the publish path."""
    return _stamped(Doc(title=f"scheduled: {job_name}", icon="❌",
                        tone="error", summary=str(summary or "")))


# ── command payloads (Formatting F7 — the F3 residue) ───────────────

def _quote_section(heading: str, text: str) -> Section | None:
    """Prose as an expandable quote section with the house counted
    tail past the prose line limit; None when the text is empty."""
    lines = [line for line in str(text or "").splitlines()
             if line.strip()]
    if not lines:
        return None
    shown = lines[:_PROSE_LINE_LIMIT]
    if len(lines) > _PROSE_LINE_LIMIT:
        shown = shown + [
            f"… (+{len(lines) - _PROSE_LINE_LIMIT} more lines)"]
    return Section(heading, quote=tuple(shown), expandable=True)


def command_report_doc(job_name: str, command: str,
                       value: Mapping[str, Any]) -> Doc:
    """The published report for one successful command run (F7).

    The stored ``last_result`` string (``exit=N`` + the head of
    stdout) is unchanged — this Doc is the published form: the
    command named as a mono row, its exit code as a fact, stdout as
    prose/detail in an expandable section (stderr likewise when the
    command wrote any), and a summary taken from stdout's own first
    line — never invented.
    """
    value = value or {}
    exit_code = value.get("exit_code")
    stdout = str(value.get("stdout") or "")
    stderr = str(value.get("stderr") or "")
    first = next((line.strip() for line in stdout.splitlines()
                  if line.strip()), "")
    if first:
        summary = first[:200]
    elif exit_code is not None:
        summary = f"Command finished (exit {exit_code})."
    else:
        summary = "Command finished."
    rows = []
    if exit_code is not None:
        rows.append(KV("Exit code", exit_code))
    rows.append(KV("Command", str(command or ""), mono=True))
    sections: list[Section] = [Section("Run", rows=tuple(rows))]
    out_section = _quote_section("Output", stdout)
    if out_section is not None:
        sections.append(out_section)
    err_section = _quote_section("Errors", stderr)
    if err_section is not None:
        sections.append(err_section)
    return _stamped(Doc(title=f"scheduled: {job_name}", icon="✅",
                        tone="ok", summary=summary,
                        sections=tuple(sections)))


def command_failure_doc(job_name: str, command: str,
                        summary: str) -> Doc:
    """:func:`failure_doc` for a command payload: the failure text
    is the summary (exactly as stored), and the command that failed
    is named as a mono row — a failed run names what failed."""
    return _stamped(Doc(
        title=f"scheduled: {job_name}", icon="❌", tone="error",
        summary=str(summary or ""),
        sections=(Section("Run", rows=(
            KV("Command", str(command or ""), mono=True),)),)))


# ── chat tool results (Formatting F7 — the C2 convergence) ──────────

def tool_result_doc(tool: str, value: Any) -> Doc:
    """The F3-style generic Doc for an interactive tool result.

    The same builders the scheduler publishes (hand builders where
    they exist, the schema fallback otherwise), titled by the tool
    instead of the job: chat surfaces render this Doc instead of the
    raw ``render_tool_value`` dump, so an interactive result gets
    the same summary-first, noise-suppressed treatment as a
    scheduled report.  Never raises (via :func:`job_report_doc`'s
    own degradation)."""
    doc: Doc | None = None
    try:
        builder = _BUILDERS.get(tool)
        if builder is not None and isinstance(value, Mapping):
            doc = builder(tool, value, title=tool)
        elif isinstance(value, str):
            doc = _string_doc(tool, value, title=tool)
        elif isinstance(value, Mapping):
            doc = _generic_doc(tool, value, title=tool)
        else:
            doc = _value_doc(tool, value, title=tool)
    except Exception:  # noqa: BLE001 - degrade, never fail the reply
        _log.debug("job_reports: chat builder failed for %s", tool,
                   exc_info=True)
    if doc is None:
        doc = Doc(title=tool, icon="✅", tone="ok",
                  summary=f"{humanize_key(tool) or 'Tool'} finished.")
    return _stamped(doc)


def summarize_last_result(job: Mapping[str, Any], *,
                          with_mark: bool = True) -> str:
    """One display line for a job row's last outcome (Formatting F7).

    The jobs/status views used to slice the stored
    ``schedule_jobs.last_result`` contract string at 80 chars —
    mid-word, mid-fact.  This re-presents the SAME row data: the
    outcome mark from the row's own failure contract (the documented
    ``"job failed"`` prefix the health gates read), then the stored
    string's own first content line whole — for a tool job that is
    the result's first fact line, with the redundant ``"{tool}:"``
    header line (the F3 storage shape) folded away.  The stored
    string itself is never touched.  ``with_mark=False`` for views
    that already carry their own outcome mark.  Never raises.
    """
    try:
        raw = str(job.get("last_result") or "")
        if not raw.strip():
            return ""
        failed = raw.startswith("job failed")
        lines = [line.strip() for line in raw.splitlines()
                 if line.strip()]
        if not lines:
            return ""
        first = lines[0]
        # the "{tool}:\n" storage header is a label, not a result —
        # the next line is the first fact (when there is one)
        if first.endswith(":") and len(lines) > 1:
            first = lines[1]
            # a leading "ok: yes" restates the mark — the next line
            # is the first fact the run actually reported
            if first.lower().startswith("ok:") and len(lines) > 2:
                first = lines[2]
        if not with_mark:
            return first
        mark = "❌" if failed else "✅"
        return f"{mark} {first}"
    except Exception:  # noqa: BLE001 - a view must never break on data
        return ""
