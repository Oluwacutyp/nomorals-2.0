"""Alert templates (Formatting Pass F5).

The Doc builders for the producer families that page the owner with
hand-composed strings: immune notices, watcher hits and digests, ops
alerts, backup reports, research results, trial notices, directive
(task) notices, and upgrade proposals.  Each builder turns the data
its producer *already computes* into a presentation
:class:`~nomorals.social.chat.present.Doc` — a truthful summary line
first, the structured facts as labeled rows (ids mono-styled), long
detail in expandable sections — and the producer publishes it
through the Notifier's Doc path (Formatting F2) alongside the exact
historical ``kind`` and title string, so dedupe, quiet hours, the
Pass 7 alert policy, and redelivery all keep their semantics.  The
stored notification body becomes the Doc's canonical body — the
shape F2 proved for scheduler reports.

Placement: this module lives in ``agents`` (L5) beside
:mod:`.job_reports` because every F5 consumer is L5 — the agents
producers themselves, ``tools/autonomy`` (the immune sampler tool,
pinned L5), and ``backup`` (L5).  The one producer below the layer,
``immune/wiring.py`` (Layer 0), stays presentation-free exactly as
its Pass 7 design requires: :func:`summarize_tick_event` returns
plain data (title / body / critical / subject / hints — plus, since
F5, ``tone`` and structured ``facts``), and the immune Doc is shaped
here from that data.  Facts are consumed as values, never parsed
back out of composed prose.

Nothing here raises: a builder failure degrades to a minimal Doc
carrying the producer's own title and prose, so a formatting bug can
never eat an alert (the producers' publish calls are best-effort
already; this keeps the template layer to the same standard).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

from ..social.chat.present import Doc, KV, Section, from_dict

_log = logging.getLogger(__name__)

__all__ = [
    "immune_event_doc",
    "watcher_hit_doc", "watcher_notice_doc", "watcher_digest_doc",
    "ops_finding_doc",
    "backup_result_doc", "backup_failure_doc", "backup_drill_doc",
    "backup_queued_doc",
    "research_loop_doc", "research_job_doc",
    "trial_notice_doc",
    "task_notice_doc",
    "upgrade_proposed_doc", "upgrade_merged_doc", "upgrade_decided_doc",
    "upgrade_implemented_doc",
]

#: Long prose tails (a directive result, a research brief) ride in an
#: expandable section, capped with an explicit counted tail — the
#: house pattern from job_reports, never a silent mid-sentence cut.
_PROSE_LINE_LIMIT = 40


def _split_icon(title: str) -> tuple[str, str]:
    """Split a leading emoji token off a historical alert title.

    Producers title their alerts ``"🛡️ immune gate ENFORCED"`` /
    ``"👁 watcher: …"`` — the emoji is the Doc's icon, the rest is
    its title, so the rendered head carries the glyph once.  A first
    token containing any alphanumeric character is not an icon and
    the title is returned whole.
    """
    text = str(title or "")
    head, _, rest = text.partition(" ")
    if head and rest and not any(ch.isalnum() for ch in head):
        return head, rest.strip()
    return "", text


def _prose_sections(text: str, *, section_title: str = "Detail",
                    limit: int = _PROSE_LINE_LIMIT) -> list[Section]:
    """Prose lines beyond the summary as one expandable section.

    Lines are carried complete; past ``limit`` the tail is an
    explicit ``… (+N more lines)`` marker, never a mid-sentence cut.
    """
    lines = [line for line in str(text or "").splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return []
    shown = lines[:limit]
    if len(lines) > limit:
        shown = shown + [f"… (+{len(lines) - limit} more lines)"]
    return [Section(section_title, quote=tuple(shown), expandable=True)]


def _prose_doc(title: str, body: str, *, tone: str = "info",
               icon: str = "") -> Doc:
    """The shared prose template: the body's first line is the
    summary, the remainder rides as expandable detail.  Used by the
    families whose producers compose prose and carry no further
    structured facts at the publish site (trial, directives)."""
    text = str(body or "")
    lines = text.splitlines()
    summary = ""
    rest = ""
    for i, line in enumerate(lines):
        if line.strip():
            summary = line.strip()
            rest = "\n".join(lines[i + 1:])
            break
    sections = _prose_sections(rest) if rest.strip() else []
    return Doc(title=title, icon=icon, tone=tone, summary=summary,
               sections=tuple(sections))


def _safe(builder: str, fallback: Doc, fn) -> Doc:
    """Run a builder; on any failure return ``fallback`` (never raise)."""
    try:
        return fn()
    except Exception:  # noqa: BLE001 - a template bug must not eat the alert
        _log.debug("alert_templates: %s builder failed; minimal Doc",
                   builder, exc_info=True)
        return fallback


# ── immune notices ───────────────────────────────────────────────────

def immune_event_doc(event: Mapping[str, Any]) -> Doc:
    """The Doc for one immune sampler event.

    ``event`` is :func:`nomorals.immune.wiring.summarize_tick_event`'s
    plain-data dict: the historical title/body (the stored title is
    the title string itself, passed to ``publish`` unchanged — this
    Doc only shapes delivery), the Pass 7 ``tone`` the wiring chose,
    and ``facts``: the sample's structured values (spawned / growth /
    quarantined / baseline samples) as labeled rows.  The lead prose
    line (the gate's reason, the error text, an announcement's tail)
    is the summary; further announcement lines stay as lines.
    """
    event = event or {}
    title = str(event.get("title") or "")
    icon, doc_title = _split_icon(title)
    body = str(event.get("body") or "")
    tone = str(event.get("tone") or "info")

    def _build() -> Doc:
        rows = tuple(
            KV(str(label), value)
            for label, value in (event.get("facts") or [])
            if value is not None and str(value) != "")
        sections: list[Section] = []
        lines = [ln for ln in body.splitlines() if ln.strip()]
        # The context lines the wiring composed from the same facts
        # ("sample: …", "quarantined: … · baseline samples: …") are
        # the prose form of the rows — when rows exist they are not
        # repeated as lines.
        prose = lines
        if rows:
            prose = [ln for ln in lines
                     if not ln.startswith("sample:")
                     and not ln.startswith("quarantined:")]
        summary = prose[0].strip() if prose else ""
        rest = prose[1:]
        if rows:
            sections.append(Section("Sample", rows=rows))
        if rest:
            sections.append(Section(lines=tuple(rest)))
        return Doc(title=doc_title, icon=icon, tone=tone,
                   summary=summary, sections=tuple(sections))

    return _safe("immune", Doc(title=doc_title, icon=icon, tone=tone,
                               summary=body), _build)


# ── watchers ─────────────────────────────────────────────────────────

def watcher_hit_doc(*, title: str, watcher_name: str, watcher_id: str,
                    severity: str, kind: str, summary: str) -> Doc:
    """One watcher hit: the check's own summary prose leads (first
    line as the summary, any remainder as expandable detail — the
    full text always rides; the retired ``[:3500]`` cut is not
    reproduced), with the watcher's facts as labeled rows."""
    icon, doc_title = _split_icon(title)
    tone = "warn" if severity == "urgent" else "info"

    def _build() -> Doc:
        rows = [KV("Watcher", watcher_name),
                KV("Severity", severity),
                KV("Kind", kind),
                KV("Watcher id", watcher_id, mono=True)]
        text = str(summary or "")
        first, _, remainder = text.partition("\n")
        sections: list[Section] = [Section(rows=tuple(rows))]
        if remainder.strip():
            sections.extend(_prose_sections(remainder))
        return Doc(title=doc_title, icon=icon or "👁", tone=tone,
                   summary=first.strip(), sections=tuple(sections))

    return _safe("watcher_hit",
                 Doc(title=doc_title, icon=icon or "👁", tone=tone,
                     summary=str(summary or "")), _build)


def watcher_notice_doc(title: str, body: str, *, tone: str = "warn") -> Doc:
    """A watcher lifecycle notice (erroring / paused): the producer's
    prose, templated — title kept whole, body prose-led."""
    icon, doc_title = _split_icon(title)
    return _safe("watcher_notice",
                 Doc(title=doc_title, icon=icon, tone=tone,
                     summary=str(body or "")),
                 lambda: _prose_doc(doc_title, body, tone=tone,
                                    icon=icon))


def watcher_digest_doc(count: int,
                       groups: Sequence[tuple[str, Sequence[str]]]
                       ) -> Doc:
    """The held-alerts digest: one section per watcher, every held
    line carried as an expandable quote line.  The old digest cut
    the joined body at 3500 chars mid-sentence; the Doc path chunks
    losslessly instead, so nothing is dropped."""
    def _build() -> Doc:
        sections = tuple(
            Section(str(name), quote=tuple(str(ln) for ln in lines),
                    expandable=True)
            for name, lines in groups if lines)
        watchers = len(sections)
        summary = (f"{count} held alert{'s' if count != 1 else ''} "
                   f"across {watchers} watcher"
                   f"{'s' if watchers != 1 else ''}.")
        return Doc(title=f"watcher digest ({count})", icon="👁",
                   tone="info", summary=summary, sections=sections)

    return _safe("watcher_digest",
                 Doc(title=f"watcher digest ({count})", icon="👁",
                     tone="info"), _build)


# ── ops alerts ───────────────────────────────────────────────────────

def ops_finding_doc(finding: Mapping[str, Any]) -> Doc:
    """One ops finding (stuck mission / chronic routing role /
    unrecovered pivot) from its ``{kind, title, body, data}`` dict:
    the structured facts the scan already computed become rows; the
    composed prose stays out — the summary is derived from the same
    data, never parsed back out of the body string."""
    finding = finding or {}
    title = str(finding.get("title") or "")
    kind = str(finding.get("kind") or "")
    data = finding.get("data") or {}

    def _build() -> Doc:
        if kind == "stuck_mission":
            mid = str(data.get("id") or "")[:12]
            steps = len(data.get("completed") or [])
            rows = [KV("Mission", mid, mono=True)]
            goal = str(data.get("goal") or "").strip()
            if goal:
                rows.append(KV("Goal", goal))
            rows.append(KV("Iterations", data.get("iterations", 0)))
            rows.append(KV("Steps done", steps))
            last_error = str(data.get("last_error") or "").strip()
            if last_error:
                rows.append(KV("Last error", last_error[:200]))
            summary = (f"Mission {mid} is stuck — "
                       f"{data.get('iterations', 0)} iteration(s), "
                       f"{steps} step(s) done.")
            sections = [Section(rows=tuple(rows)),
                        Section(lines=(
                            f"Resumable: `nm missions --resume {mid}`",))]
            return Doc(title=title, tone="warn", summary=summary,
                       sections=tuple(sections))
        if kind == "chronic_role":
            name = str(data.get("name") or "")
            role = name.split(":", 1)[1] if ":" in name else name
            total = int(data.get("success_count") or 0) + \
                int(data.get("failure_count") or 0)
            rate = float(data.get("success_rate") or 0.0)
            rows = [KV("Role", role),
                    KV("Success rate", f"{rate:.0%}"),
                    KV("Runs", total),
                    KV("Successes", int(data.get("success_count") or 0)),
                    KV("Failures", int(data.get("failure_count") or 0))]
            summary = (f"Routing role '{role}' is chronically failing "
                       f"— {rate:.0%} success over {total} runs. The "
                       "router is discounting this role (-0.15 prior); "
                       "check the agent behind it or retrain the goal "
                       "wording.")
            return Doc(title=title, tone="warn", summary=summary,
                       sections=(Section(rows=tuple(rows)),))
        if kind == "unrecovered_pivot":
            task = str(data.get("task") or "?")
            rows = [KV("Task", task, mono=True)]
            sections: list[Section] = [Section(rows=tuple(rows))]
            change = str(data.get("pivot") or "").strip()
            if change:
                sections.append(Section(
                    "Approach change", lines=(change[:300],)))
            summary = (f"Pivoted step '{task}' did not recover — the "
                       "approach was changed and the step still "
                       "failed. Needs new information, not another "
                       "retry.")
            return Doc(title=title, tone="warn", summary=summary,
                       sections=tuple(sections))
        doc = _prose_doc(title, str(finding.get("body") or ""),
                         tone="warn")
        return doc

    return _safe("ops", Doc(title=title, tone="warn",
                            summary=str(finding.get("body") or "")),
                 _build)


# ── backup reports ───────────────────────────────────────────────────

def backup_result_doc(result: Mapping[str, Any]) -> Doc:
    """A shipped backup: id / assets / encrypted size / release /
    manifest state as rows (the prose body's facts, as values)."""
    result = result or {}
    backup_id = str(result.get("backup_id", "unknown"))
    assets = result.get("assets") or []

    def _build() -> Doc:
        rows = [KV("Backup", backup_id, mono=True),
                KV("Assets", len(assets)),
                KV("Encrypted bytes",
                   int(result.get("total_encrypted_bytes", 0) or 0)),
                KV("Release", str(result.get("tag", "")), mono=True),
                KV("Manifest committed",
                   bool(result.get("manifest_pushed", False)))]
        return Doc(title=f"backup shipped: {backup_id}", tone="ok",
                   summary=(f"Backup {backup_id} shipped to GitHub "
                            "Releases."),
                   sections=(Section(rows=tuple(
                       r for r in rows if str(r.value) != "")),))

    return _safe("backup_result",
                 Doc(title=f"backup shipped: {backup_id}", tone="ok"),
                 _build)


def backup_failure_doc(backup_id: str, error: BaseException) -> Doc:
    """A failed backup run — error tone; the previous generation is
    intact, and the body says so."""
    def _build() -> Doc:
        rows = [KV("Backup", str(backup_id), mono=True),
                KV("Error", type(error).__name__, mono=True)]
        return Doc(
            title=f"BACKUP FAILED: {backup_id}", tone="error",
            summary=f"Backup {backup_id} failed: "
                    f"{type(error).__name__}: {error}",
            sections=(Section(rows=tuple(rows)),
                      Section(lines=(
                          "The previous generation is intact. Check "
                          "the run log and re-run; a queued snapshot "
                          "(if any) will retry with backoff.",))))

    return _safe("backup_failure",
                 Doc(title=f"BACKUP FAILED: {backup_id}", tone="error"),
                 _build)


def backup_drill_doc(drill_report: Any) -> Doc:
    """A restore-drill outcome: the drill's own summary prose leads;
    on failure the failed step names are labeled lines."""
    ok = bool(getattr(drill_report, "ok", False))
    backup_id = str(getattr(drill_report, "backup_id", "unknown"))
    title = (f"restore drill PASS: {backup_id}" if ok
             else f"RESTORE DRILL FAIL: {backup_id}")

    def _build() -> Doc:
        summary_text = (drill_report.summary()
                        if hasattr(drill_report, "summary")
                        else str(drill_report))
        steps = drill_report.to_dict().get("steps", []) \
            if hasattr(drill_report, "to_dict") else []
        failed = [str(s["name"]) for s in steps if not s.get("ok")]
        doc = _prose_doc(title, str(summary_text),
                         tone="ok" if ok else "error")
        if not ok:
            lines = []
            if failed:
                lines.append(f"Failed steps: {', '.join(failed)}")
            lines.append(
                "A backup never restored is a hypothesis — "
                "investigate before the next generation ships.")
            sections = list(doc.sections) + \
                [Section(lines=tuple(lines))]
            return Doc(title=doc.title, icon=doc.icon, tone="error",
                       summary=doc.summary, sections=tuple(sections))
        return doc

    return _safe("backup_drill", Doc(title=title,
                                     tone="ok" if ok else "error"),
                 _build)


def backup_queued_doc(result: Mapping[str, Any]) -> Doc:
    """A snapshot queued in the outbox (network down): warn tone,
    the reason as a row — explicitly not a silent skip."""
    result = result or {}
    backup_id = str(result.get("backup_id", "unknown"))

    def _build() -> Doc:
        rows = [KV("Backup", backup_id, mono=True),
                KV("Reason", str(result.get("reason", "network down")))]
        return Doc(
            title=f"backup queued (offline): {backup_id}", tone="warn",
            summary=(f"Backup {backup_id} snapshotted + encrypted "
                     "locally but NOT shipped. It waits in the local "
                     "outbox and retries with backoff — this is not "
                     "a silent skip."),
            sections=(Section(rows=tuple(rows)),))

    return _safe("backup_queued",
                 Doc(title=f"backup queued (offline): {backup_id}",
                     tone="warn"), _build)


# ── research results ─────────────────────────────────────────────────

def research_loop_doc(*, title: str, briefs: Sequence[str],
                      findings: int, proposals: int,
                      signals: str) -> Doc:
    """The research loop's per-cycle digest: counts as rows, the
    briefs complete in an expandable section (the old body cut them
    at 3000 chars), the usefulness-signals line as the summary, and
    the review pointer as the footer."""
    def _build() -> Doc:
        quote: list[str] = []
        for brief in briefs:
            quote.extend(str(brief).splitlines())
        sections: list[Section] = [Section(rows=(
            KV("Findings", int(findings)), KV("Proposals", int(proposals))))]
        if quote:
            sections.append(Section("Briefs", quote=tuple(quote),
                                    expandable=True))
        return Doc(title=title, tone="info",
                   summary=f"signals: {signals}",
                   sections=tuple(sections),
                   footer=("Review: /proposals — upgrade tickets also "
                           "via /upgrade list"))

    return _safe("research_loop", Doc(title=title, tone="info"),
                 _build)


def research_job_doc(job: Mapping[str, Any]) -> Doc:
    """One background research job's completion (agents/search/jobs):
    done → the report's summary leads, its sources as link rows (all
    of them — the old string capped at 8); failed → error tone with
    the job's error as the summary."""
    job = job or {}
    query = str(job.get("query", "") or "")
    if job.get("state") == "done":
        title = f"Research ready: {query[:80]}"
        report = job.get("report") or {}

        def _build() -> Doc:
            summary = str(report.get("summary") or "").strip()
            first, _, remainder = summary.partition("\n")
            sections: list[Section] = []
            sources = report.get("sources") or []
            if sources:
                rows = tuple(
                    KV(f"[{s.get('n')}] {s.get('title') or s.get('url')}",
                       str(s.get("url") or ""), url=str(s.get("url")
                                                        or ""))
                    for s in sources)
                sections.append(Section("Sources", rows=rows))
            if remainder.strip():
                sections.extend(_prose_sections(remainder))
            return Doc(title=title, tone="ok",
                       summary=first.strip() or "Research complete.",
                       sections=tuple(sections))

        return _safe("research_job", Doc(title=title, tone="ok"),
                     _build)
    title = f"Research failed: {query[:80]}"
    return Doc(title=title, tone="error",
               summary=("The background research run failed: "
                        f"{job.get('error') or 'unknown error'}"))


# ── trial flow notices ───────────────────────────────────────────────

def trial_notice_doc(title: str, body: str) -> Doc:
    """A trial-flow owner notice: the producer's prose, templated.
    Tone follows the family's own signals — an interrupted/expired
    watch or a ⚠️-led body is a warning; a 📩 code delivery is ok."""
    text = str(body or "")
    title = str(title or "")
    if ("interrupted" in title or "expired" in title
            or text.lstrip().startswith(("⚠", "⏸"))):
        tone = "warn"
    elif text.lstrip().startswith("📩"):
        tone = "ok"
    else:
        tone = "info"
    return _safe("trial", Doc(title=title, tone=tone, summary=text),
                 lambda: _prose_doc(title, text, tone=tone))


# ── directives / task notices ────────────────────────────────────────

def task_notice_doc(title: str, body: str) -> Doc:
    """A directive outcome: ``task done:`` → ok, ``task failed:`` →
    error.  The result prose rides complete — first line as the
    summary, the tail expandable (the old publish cut it at 1500
    chars)."""
    tone = "error" if str(title).startswith("task failed") else "ok"
    return _safe("task", Doc(title=title, tone=tone,
                             summary=str(body or "")),
                 lambda: _prose_doc(title, body, tone=tone))


# ── upgrade proposals ────────────────────────────────────────────────

def upgrade_proposed_doc(*, pid: str, title: str, rationale: str,
                         source: str = "",
                         files: Sequence[str] = (),
                         tests: Sequence[str] = (),
                         claim_ids: Sequence[str] = (),
                         conflicts: Sequence[Mapping[str, Any]] = ()
                         ) -> Doc:
    """A newly filed proposal: the rationale leads (complete — the
    old body cut it at 400 chars; the tail rides expandable), the
    filing's facts as rows, conflicts named as lines."""
    doc_title = f"new upgrade proposal: {title}"

    def _build() -> Doc:
        rows = [KV("Proposal", pid, mono=True)]
        if source:
            rows.append(KV("Source", source))
        if claim_ids:
            rows.append(KV("Claims", len(claim_ids)))
        sections: list[Section] = [Section(rows=tuple(rows))]
        if files:
            sections.append(Section("Files", lines=tuple(
                str(f) for f in files)))
        if tests:
            sections.append(Section("Tests", lines=tuple(
                str(t) for t in tests)))
        if conflicts:
            sections.append(Section("Conflicts with", lines=tuple(
                f"{c.get('id')}: {str(c.get('title') or '')[:60]}"
                for c in conflicts)))
        text = str(rationale or "")
        first, _, remainder = text.partition("\n")
        if remainder.strip():
            sections.extend(_prose_sections(remainder))
        return Doc(title=doc_title, tone="info",
                   summary=first.strip(), sections=tuple(sections))

    return _safe("upgrade_proposed", Doc(title=doc_title, tone="info"),
                 _build)


def upgrade_merged_doc(*, pid: str, title: str, target_id: str) -> Doc:
    """A near-duplicate filing merged into an open proposal."""
    doc_title = f"upgrade proposal merged: {title}"

    def _build() -> Doc:
        rows = [KV("Filed as", pid, mono=True),
                KV("Merged into", target_id, mono=True)]
        return Doc(title=doc_title, tone="info",
                   summary=(f"Near-duplicate of open proposal "
                            f"{target_id} — merged in with provenance "
                            "instead of a new queue entry."),
                   sections=(Section(rows=tuple(rows)),))

    return _safe("upgrade_merged", Doc(title=doc_title, tone="info"),
                 _build)


def upgrade_decided_doc(*, pid: str, title: str, by: str,
                        approved: bool, reason: str = "") -> Doc:
    """An approve / deny decision: the decider and (for a denial,
    required) the reason as rows."""
    doc_title = (f"upgrade approved: {title}" if approved
                 else f"upgrade denied: {title}")

    def _build() -> Doc:
        rows = [KV("Proposal", pid, mono=True),
                KV("Decided by", by)]
        if not approved and reason:
            rows.append(KV("Reason", reason))
        summary = ("Dispatching to the evolution agent for "
                   "implementation." if approved
                   else f"Denied by {by}.")
        return Doc(title=doc_title,
                   tone="ok" if approved else "warn",
                   summary=summary, sections=(Section(rows=tuple(rows)),))

    return _safe("upgrade_decided", Doc(title=doc_title, tone="info"),
                 _build)


def upgrade_implemented_doc(*, pid: str, status: str, evo_id: str,
                            result: Mapping[str, Any]) -> Doc:
    """An implementation outcome: status tone (implemented → ok,
    failed → error), the evolution proposal id as a row, and the
    result dict's own fields as a schema section (the old body was
    a raw ``json.dumps`` dump cut at 400 chars)."""
    doc_title = f"upgrade {status}: {pid}"
    ok = status == "implemented"

    def _build() -> Doc:
        rows = [KV("Proposal", pid, mono=True)]
        if evo_id:
            rows.append(KV("Evolution proposal", evo_id, mono=True))
        sections: list[Section] = [Section(rows=tuple(rows))]
        if result:
            sub = from_dict(dict(result), title="", include_unlisted=True)
            sections.extend(sub.sections)
        return Doc(title=doc_title, tone="ok" if ok else "error",
                   summary=(f"Implementation {'finished' if ok else 'failed'} "
                            f"for proposal {pid}."),
                   sections=tuple(sections))

    return _safe("upgrade_implemented",
                 Doc(title=doc_title, tone="ok" if ok else "error"),
                 _build)
