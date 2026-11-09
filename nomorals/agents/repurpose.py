"""Repurposing through the content cell (WS3-P7).

One asset, other forms: a piece that already earned its place in
one shape is raw material for the others — a POSTED draft becomes
a short script and a thread, a DAY RECORD becomes a short, a
SHORT becomes a text post.  Every adaptation runs through the
shipped ContentCell (P6) with the same discipline as the weekly
run: the source is entered as evidence, the adapted draft is a
claim the skeptic number-traces against the SOURCE (a repurposed
piece may re-shape, never invent), and each landed piece leaves a
receipt whose chain reads contract → draft.

Each repurposed piece is a NEW draft in the review queue at
``pending_review`` carrying provenance metadata (source kind +
source handle + the cell contract) — never a mutation of the
source, never a post.  #43 is untouched: the owner still approves
every outward step on the /drafts board.

The adaptation itself has the house ladder: the brain rung when a
brain is reachable (``dayrecord.brain_llm_fn``), a deterministic
template rung otherwise — and the rung that drafted is recorded
in the draft's metadata, never disguised.  Thread part-numbering
is formatting applied AFTER the skeptic gate, so the structural
numbers (1/, 2/, …) are never mistaken for factual claims.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .content_cell import ContentCell, ReceiptStore, _first_line

_log = logging.getLogger(__name__)

#: The forms a repurpose can take.
FORM_SHORT_SCRIPT = "short_script"
FORM_THREAD = "thread"
FORM_POST = "post"

#: Which forms each source kind can spawn.
FORMS_FOR_KIND: dict[str, tuple[str, ...]] = {
    "posted_draft": (FORM_SHORT_SCRIPT, FORM_THREAD),
    "day": (FORM_SHORT_SCRIPT, FORM_THREAD, FORM_POST),
    "short": (FORM_POST,),
}

_FORM_LABEL = {
    FORM_SHORT_SCRIPT: "short script",
    FORM_THREAD: "thread",
    FORM_POST: "text post",
}


@dataclass
class SourceMaterial:
    """One repurposable asset, read — never mutated."""

    kind: str           # posted_draft | day | short
    ref: str            # structured handle (draft id / day / post id)
    label: str          # human name for boards and receipts
    text: str           # the asset's own words (adaptation input)
    basis: str          # the evidence text numbers must trace to
    platforms: list[str] = field(default_factory=list)


# ── the cell ─────────────────────────────────────────────────────────

class RepurposeCell(ContentCell):
    """The P6 skeleton specialised for one-source repurposing."""

    def __init__(self, contracts: Any, receipts: ReceiptStore) -> None:
        super().__init__(contracts, kind="repurpose")
        self.receipts = receipts

    def begin(self, source: SourceMaterial) -> Any:
        contract = self.open(
            goal=(f"repurpose {source.kind} {source.label!r} into"
                  " adapted drafts of other forms — the source's"
                  " facts only, re-shaped, never invented"),
            deliverable=("new drafts in the review queue at"
                         " pending_review, each carrying its"
                         " provenance and a readable receipt chain"),
            required_evidence=("source", "draft"))
        self.evidence(
            "source",
            f"source: {source.kind} {source.label} "
            f"({len(source.text)} chars)",
            producer=self.researcher,
            evidence_ref=f"{source.kind}:{source.ref}",
            verify=True)
        return contract

    def review(self, text: str, *, basis: str, form: str) -> str | None:
        """The skeptic gate, identical in shape to the weekly
        cell's: numbers in the adaptation must trace to the
        source.  Returns the objection string when blocked."""
        from .content_cell import _fmt_numbers, _sha, untraced_numbers

        claim = self.evidence(
            "draft",
            f"repurposed {_FORM_LABEL.get(form, form)} draft "
            f"({len(text)} chars)",
            producer=self.drafter,
            evidence_ref=f"draft:{_sha(text)}")
        bad = untraced_numbers(text, basis)
        if bad:
            objection = (
                f"adaptation states number(s) {_fmt_numbers(bad)}"
                " with no source in the original asset")
            self.block(claim.id, objection,
                       note="a repurposed piece never invents facts")
            return objection
        self.contracts.verify_claim(
            claim.id, validator=self.verifier, verdict="supported")
        return None

    def landed(self, pieces: list[dict[str, Any]],
               source: SourceMaterial) -> dict[str, Any]:
        receipts = [
            self.receipts.record(
                contract_id=self.contract_id, kind="repurpose",
                label=_first_line(str(piece.get("text") or "")),
                platform=str(piece.get("platform") or ""),
                draft_id=str(piece.get("draft_id") or ""),
                content=str(piece.get("text") or ""),
                chain={"source_kind": source.kind,
                       "source_ref": source.ref,
                       "form": str(piece.get("form") or "")})
            for piece in pieces]
        if not pieces:
            return {
                "contract_id": self.contract_id, "receipts": [],
                "completed": False,
                "note": "the adaptation was blocked by the skeptic"
                        " — nothing was queued"}
        self.deliver(
            "draft",
            f"delivery: {len(pieces)} repurposed draft(s) landed"
            f" in the review queue from {source.kind}"
            f" {source.label}",
            evidence_ref=f"repurpose:{source.kind}:{source.ref}")
        decision = self.finish()
        return {
            "contract_id": self.contract_id, "receipts": receipts,
            "completed": bool(decision.ok),
            "note": "" if decision.ok else (
                "drafts landed, but the repurpose contract could"
                " not complete: " + "; ".join(decision.reasons or []))}


def make_repurpose_cell(db: Any) -> RepurposeCell | None:
    """Composition factory, the make_weekly_cell shape.  None when
    there is no database to hold contracts."""
    if db is None:
        return None
    from .delegation import DelegationStore

    return RepurposeCell(DelegationStore(db), ReceiptStore(db))


# ── adaptation: brain rung, then the deterministic template ─────────

def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines()
            if ln.strip()]


def _template_short_script(source: SourceMaterial) -> str:
    # Number-free by construction: the skeptic traces every digit
    # in the draft against the source, so the template carries no
    # structural numbering (beats are bullets, the hook is a
    # label) — structure is formatting, never a factual claim.
    lines = _lines(source.text) or [source.label]
    hook = lines[0]
    beats = lines[1:6] if len(lines) > 1 else lines[:1]
    out = [f"🎬 short script — {hook}", "", f"hook: {hook}"]
    for beat in beats:
        out.append(f"• {beat}")
    out.append(f"close: {lines[-1]}")
    return "\n".join(out)


def _template_thread_body(source: SourceMaterial) -> str:
    """The thread's gated body: the source's own paragraphs,
    re-flowed — part numbers are added after the gate."""
    paras = [p.strip() for p in (source.text or "").split("\n\n")
             if p.strip()]
    if not paras:
        paras = _lines(source.text) or [source.label]
    return "\n\n".join(paras)


def _number_thread(body: str) -> str:
    paras = [p.strip() for p in (body or "").split("\n\n") if p.strip()]
    if len(paras) <= 1:
        return body
    return "\n\n".join(
        f"{i}/ {para}" for i, para in enumerate(paras, 1))


def _template_post(source: SourceMaterial) -> str:
    return (source.text or source.label).strip()


_TEMPLATE = {
    FORM_SHORT_SCRIPT: _template_short_script,
    FORM_THREAD: _template_thread_body,
    FORM_POST: _template_post,
}

_BRAIN_BRIEF = {
    FORM_SHORT_SCRIPT: (
        "Rewrite the source as a vertical short-video SCRIPT: a"
        " spoken hook line first, then short spoken beats, then a"
        " closing line. Plain text, one beat per line, no camera"
        " directions. Keep every fact and number exactly as the"
        " source states it; add nothing."),
    FORM_THREAD: (
        "Rewrite the source as a thread: a strong opening post,"
        " then one idea per post, separated by a blank line. Keep"
        " every fact and number exactly as the source states it;"
        " add nothing. Do not number the posts."),
    FORM_POST: (
        "Rewrite the source as one text post in the owner's"
        " voice: hook first line, then the substance, hashtags"
        " last if the source has them. Keep every fact and number"
        " exactly as the source states it; add nothing."),
}


def adapt(source: SourceMaterial, form: str, *,
          llm_fn: Callable[[str], str] | None = None
          ) -> tuple[str, str]:
    """Adapt ``source`` into ``form``.  Returns (text, rung) —
    rung is "brain" or "template", recorded on the draft either
    way.  Thread text comes back UNNUMBERED (the gate runs on it);
    the caller numbers after review."""
    template = _TEMPLATE[form]
    if llm_fn is not None:
        try:
            out = llm_fn(
                _BRAIN_BRIEF[form] + "\n\nSOURCE:\n" + source.text)
            out = (out or "").strip()
            if out:
                return out, "brain"
        except Exception as exc:  # noqa: BLE001 - fall to template
            _log.debug("repurpose brain rung failed: %s", exc)
    return template(source), "template"


# ── the run ──────────────────────────────────────────────────────────

def repurpose_source(db: Any, queue: Any, source: SourceMaterial,
                     form: str, *,
                     llm_fn: Callable[[str], str] | None = None
                     ) -> dict[str, Any]:
    """One source → one adapted draft, through the cell.

    Returns a result dict: ``ok`` with ``draft_id`` when the piece
    landed at pending_review, or ``ok`` False with the skeptic's
    ``objection`` (or a plain ``error``) when nothing landed.  The
    source is only ever read.
    """
    if form not in FORMS_FOR_KIND.get(source.kind, ()):
        return {"ok": False, "draft_id": "",
                "error": f"a {source.kind} does not repurpose into"
                         f" a {_FORM_LABEL.get(form, form)}"}
    cell = make_repurpose_cell(db)
    if cell is None:
        return {"ok": False, "draft_id": "",
                "error": "no database to hold the repurpose"
                         " contract"}
    cell.begin(source)
    text, rung = adapt(source, form, llm_fn=llm_fn)
    objection = cell.review(text, basis=source.basis, form=form)
    if objection:
        outcome = cell.landed([], source)
        return {"ok": False, "draft_id": "",
                "objection": objection,
                "contract_id": cell.contract_id,
                "completed": outcome["completed"]}
    final = _number_thread(text) if form == FORM_THREAD else text
    platforms = (list(source.platforms)
                 if form in (FORM_THREAD, FORM_POST) else [])
    draft = queue.create_draft(
        final, platforms=platforms or None,
        metadata={
            "origin": "repurpose",
            "source_kind": source.kind,
            "source_ref": source.ref,
            "source_label": source.label,
            "form": form,
            "contract_id": cell.contract_id,
            "drafter": rung,
        })
    queue.propose(draft.id)
    outcome = cell.landed(
        [{"text": final, "draft_id": draft.id,
          "platform": platforms[0] if platforms else "",
          "form": form}], source)
    return {"ok": True, "draft_id": draft.id,
            "contract_id": cell.contract_id,
            "completed": outcome["completed"],
            "note": outcome["note"], "drafter": rung,
            "form": form}


# ── source loaders (read-only) ───────────────────────────────────────

def posted_draft_sources(queue: Any, *, limit: int = 8
                         ) -> list[SourceMaterial]:
    """Posted drafts as repurpose sources, newest first."""
    from ..social.drafts import POSTED

    out: list[SourceMaterial] = []
    for draft in queue.list_by_status(POSTED, limit=limit):
        out.append(SourceMaterial(
            kind="posted_draft", ref=draft.id,
            label=_first_line(draft.content) or "posted draft",
            text=draft.content, basis=draft.content,
            platforms=list(draft.platforms or [])))
    return out


def day_source(db: Any, day: str) -> SourceMaterial | None:
    """One day record as a repurpose source — the STORED record
    (fragments + derived sections), never a recompute.  None when
    the day holds nothing."""
    from .dayrecord import DayStore

    store = DayStore(db)
    fragments = store.fragments(day)
    sections = store.sections(day)
    frag_text = "\n".join(str(f.get("text") or "")
                          for f in fragments)
    section_text = "\n".join(
        "\n".join(sec.lines) for sec in sections if sec.present)
    basis = "\n".join(part for part in (frag_text, section_text)
                      if part)
    if not basis.strip():
        return None
    return SourceMaterial(
        kind="day", ref=day, label=f"the {day} day record",
        text=frag_text.strip() or section_text.strip(),
        basis=basis, platforms=[])


def short_sources(*, limit: int = 8) -> list[SourceMaterial]:
    """Planned shorts as repurpose sources, from the shorts
    calendar — the same lazy, contained read the /calendar
    surface makes.  Post copy joins the basis when the short's
    run produced it."""
    out: list[SourceMaterial] = []
    try:
        from ..media.contentops.pipeline import ShortPipeline

        pipe = ShortPipeline()
        posts = list(pipe.calendar.list())
    except Exception as exc:  # noqa: BLE001 - contained, like /calendar
        _log.debug("repurpose short sources unavailable: %s", exc)
        return out
    for post in posts[-int(limit):][::-1]:
        parts = [str(post.topic or "")]
        text = str(post.topic or "")
        try:
            job = pipe.jobs.get(post.job_id) if post.job_id else None
            if job is not None:
                from .shorts_publisher import _load_post_copy

                copy = _load_post_copy(pipe, job)
                copy_text = "\n".join(
                    str(copy.get(k) or "") for k in
                    ("title", "description", "hashtags")).strip()
                if copy_text:
                    parts.append(copy_text)
                    text = copy_text
        except Exception as exc:  # noqa: BLE001 - copy is a bonus
            _log.debug("repurpose post copy read failed: %s", exc)
        basis = "\n".join(p for p in parts if p.strip())
        out.append(SourceMaterial(
            kind="short", ref=post.id,
            label=_first_line(str(post.topic or "")) or "short",
            text=text, basis=basis,
            platforms=list(post.platforms or [])))
    return out


__all__ = [
    "FORM_POST", "FORM_SHORT_SCRIPT", "FORM_THREAD", "FORMS_FOR_KIND",
    "RepurposeCell", "SourceMaterial", "adapt", "day_source",
    "make_repurpose_cell", "posted_draft_sources", "repurpose_source",
    "short_sources",
]
