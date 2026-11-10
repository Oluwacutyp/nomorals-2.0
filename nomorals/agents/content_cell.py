"""The content delivery cell (WS3 P6) — the §6 preset, formalized.

Devon's content work stops being a set of pipeline functions and
becomes ONE reusable cooperative machine: a delivery cell on the
shipped delegation substrate (``agents/delegation.py``). Every
content job — the Monday batch, a day's derivation, an idea's
development, a due short — runs as a work contract with the same
role shape:

    researcher → drafter → skeptic → verifier → delivery steward
    (arbiter: the owner; planner and intake are functions/surfaces
    around the contract, not roles inside it — the day cell's
    planner precedent)

* **Researcher** submits findings as evidence claims with
  provenance; a claim with no evidence ref is a statement, never a
  fact the cell will build on.
* **Drafter** (the worker) submits each piece as a draft claim.
* **Skeptic** challenges every factual claim against its evidence.
  The gate is mechanical where it can be: a number stated in a
  draft that does not trace to the run's evidence base is
  challenged and resolved *contradicted*, and the piece never
  lands. A contradicted claim blocks contract completion until a
  later delivery claim supersedes it (the store's latest-per-scope
  rule — the day cell's settled design).
* **Verifier** (never the worker — the store raises on it) checks
  pieces by read-back or rule, verifies claims *supported*, and
  signs off before completion.
* **Delivery steward** lands the pieces (review queue, publish
  ledger) and submits the delivery claim that closes the chain.
* **Learning steward** is the measurement spine (WS3 P1): it is not
  a contract role — post metrics flow back through
  ``social/metrics.py`` and are READ into the receipt chain here.

**What is shared, what is not.** This module is the shared skeleton:
contract opening with the cell role set, the evidence flow, the
skeptic block, the delivery supersede, sign-off/completion, and the
receipt store + chain reader. The shipped day cell (``dayrecord``)
and idea cell (``idea_development``) are NOT rewritten onto it:
each carries cell-specific gates the skeleton parameterizes but
does not absorb — the day cell's verifier checks section receipts
by read-back from its own store and its allowed-number set is built
from day sections (spelled-out numbers, day-of-month, percentages
of recorded values); the idea cell's verdict exits and artifact
fencing are its own. Rewriting either would risk shipped behavior
for zero functional gain. The pipelines that had NO contract at
all — the weekly content pipeline and the shorts factory — adopt
the skeleton here: the weekly pipeline through an injected cell
(the composition root installs :func:`make_weekly_cell` as the
factory; ``social/content_pipeline.py`` stays L4 and imports
nothing from this layer), the shorts sweep in
``agents/shorts_publisher.py`` driving this module directly.

**Receipts.** Every piece a cell delivers gets one row in
``content_receipts`` (runtime-created, Batch B pattern —
``storage/migrations.py`` stays frozen): contract → draft → post →
metrics, resolvable end-to-end by :func:`receipt_chain` and
renderable as one readable record per piece by
:func:`render_receipt`. Ids ride the structured record only; the
rendered text names pieces by label and platform.

Nothing here posts on its own: the cell lands drafts for review
and publishes only where the owner's own scheduling act is the
authorization (the #43 rule — it is his directive, not a gate this
module adds or removes).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from typing import Any, Callable

_log = logging.getLogger(__name__)

__all__ = [
    "numbers_in",
    "untraced_numbers",
    "ContentCell",
    "WeeklyCell",
    "make_weekly_cell",
    "ReceiptStore",
    "receipt_chain",
    "render_receipt",
]


# ── the skeptic's mechanical rule: numbers must trace ─────────────────

_NUM_TOKEN = re.compile(r"[-+]?[$₦€£]?\d[\d,]*(?:\.\d+)?%?")


def numbers_in(text: str) -> list[float]:
    """Every numeric value a text states, normalised (currency,
    commas, percent and sign stripped to the bare value). Same
    semantics as the day cell's tracer."""
    out: list[float] = []
    for token in _NUM_TOKEN.findall(text or ""):
        cleaned = token.replace(",", "").replace("%", "")
        cleaned = cleaned.strip("$₦€£")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def untraced_numbers(text: str, basis: str) -> list[float]:
    """Numbers stated in ``text`` that appear nowhere in ``basis``
    (the run's evidence base: findings, the record, the topic —
    whatever the cell can actually stand behind)."""
    allowed = {round(n, 4) for n in numbers_in(basis)}
    return [n for n in numbers_in(text) if round(n, 4) not in allowed]


def _fmt_numbers(nums: list[float]) -> str:
    def one(n: float) -> str:
        return str(int(n)) if float(n).is_integer() else str(n)

    return ", ".join(one(n) for n in nums)


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _first_line(text: str, limit: int = 70) -> str:
    line = (text or "").strip().split("\n", 1)[0].strip()
    return line[: limit - 1] + "…" if len(line) > limit else line


# ── the cell skeleton ─────────────────────────────────────────────────

class ContentCell:
    """One delivery cell: a contract plus the role discipline.

    ``kind`` names the job family (``weekly``, ``shorts``, …); the
    role holders derive from it (``weekly-drafter`` …), with the
    arbiter always the owner. Construct through :meth:`open`.
    """

    def __init__(self, contracts: Any, *, kind: str,
                 mission_id: str = "") -> None:
        self.contracts = contracts
        self.kind = kind
        self.mission_id = mission_id
        self.researcher = f"{kind}-researcher"
        self.drafter = f"{kind}-drafter"
        self.skeptic = f"{kind}-skeptic"
        self.verifier = f"{kind}-verifier"
        self.steward = f"{kind}-steward"
        self.contract: Any = None

    # -- lifecycle ----------------------------------------------------

    def open(self, *, goal: str, deliverable: str,
             required_evidence: tuple[str, ...],
             consequential: bool = False) -> Any:
        self.contract = self.contracts.create_contract(
            goal=goal, deliverable=deliverable,
            mission_id=self.mission_id,
            worker=self.drafter, verifier=self.verifier,
            skeptic=self.skeptic, arbiter="owner",
            required_evidence=tuple(required_evidence),
            consequential=consequential)
        return self.contract

    @property
    def contract_id(self) -> str:
        return self.contract.id if self.contract is not None else ""

    # -- evidence flow --------------------------------------------------

    def evidence(self, kind: str, claim: str, *, producer: str,
                 evidence_ref: str = "",
                 verify: bool | Callable[[], Any] = False) -> Any:
        """Submit one evidence claim. ``verify=True`` marks it
        supported by the verifier immediately; a callable is CALLED
        and the claim is verified only when it returns truthy — the
        verifier judges the artifact, never the producer's word."""
        entry = self.contracts.submit_evidence(
            self.contract_id, kind=kind, claim=claim,
            producer=producer, evidence_ref=evidence_ref)
        ok = verify() if callable(verify) else verify
        if ok:
            self.contracts.verify_claim(
                entry.id, validator=self.verifier, verdict="supported")
            entry.verdict = "supported"
        return entry

    def block(self, claim_id: str, objection: str, *,
              note: str = "the challenged claim does not land") -> Any:
        """The skeptic gate: challenge a claim and resolve it
        contradicted. The piece it belongs to never lands, and the
        contradicted claim stands in the ledger until a delivery
        claim supersedes its scope."""
        challenge = self.contracts.challenge(
            self.contract_id, challenger=self.skeptic, claim=objection,
            target_claim_id=claim_id)
        return self.contracts.resolve_challenge(
            challenge.id, resolver=self.verifier,
            verdict="contradicted", note=note)

    def deliver(self, kind: str, claim: str, *,
                evidence_ref: str = "") -> Any:
        """The steward's delivery claim: the latest claim of its
        scope, so a completed delivery supersedes any contradicted
        piece claim in the completion gate."""
        return self.evidence(kind, claim, producer=self.steward,
                             evidence_ref=evidence_ref, verify=True)

    def finish(self) -> Any:
        """Sign-off + the store's completion gate."""
        self.contracts.sign_off(
            self.contract_id, verifier=self.verifier,
            note=f"{self.kind} cell: evidence verified, delivery"
                 " recorded")
        return self.contracts.complete(
            self.contract_id, by=self.verifier)

    def fail(self, detail: str, *, by: str = "") -> Any:
        """Record a worker failure; the contract stays open for a
        re-plan — it is never quietly completed over a failure."""
        return self.contracts.report_failure(
            self.contract_id, by=by or self.drafter, detail=detail)


# ── receipts: contract → draft → post → metrics ───────────────────────

_RECEIPT_DDL = """
CREATE TABLE IF NOT EXISTS content_receipts (
  id TEXT PRIMARY KEY,
  contract_id TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL DEFAULT '',
  label TEXT NOT NULL DEFAULT '',
  platform TEXT NOT NULL DEFAULT '',
  draft_id TEXT NOT NULL DEFAULT '',
  post_ref TEXT NOT NULL DEFAULT '',
  content_sha TEXT NOT NULL DEFAULT '',
  chain_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_receipts_contract
  ON content_receipts(contract_id);
"""


class ReceiptStore:
    """One receipt row per delivered piece, on the shared Database
    (runtime-created — the Batch B pattern)."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_RECEIPT_DDL)

    def record(self, *, contract_id: str, kind: str, label: str,
               platform: str, draft_id: str = "", post_ref: str = "",
               content: str = "",
               chain: dict[str, Any] | None = None) -> dict[str, Any]:
        now = time.time()
        receipt = {
            "id": f"cr-{uuid.uuid4().hex[:12]}",
            "contract_id": contract_id, "kind": kind, "label": label,
            "platform": platform, "draft_id": draft_id,
            "post_ref": post_ref,
            "content_sha": _sha(content) if content else "",
            "chain": dict(chain or {}),
            "created_at": now, "updated_at": now,
        }
        self.db.execute(
            "INSERT INTO content_receipts (id, contract_id, kind,"
            " label, platform, draft_id, post_ref, content_sha,"
            " chain_json, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (receipt["id"], receipt["contract_id"], receipt["kind"],
             receipt["label"], receipt["platform"], receipt["draft_id"],
             receipt["post_ref"], receipt["content_sha"],
             json.dumps(receipt["chain"]), receipt["created_at"],
             receipt["updated_at"]))
        return receipt

    @staticmethod
    def _row(row: Any) -> dict[str, Any]:
        out = dict(row)
        try:
            out["chain"] = json.loads(out.pop("chain_json") or "{}")
        except Exception:  # noqa: BLE001 - a torn chain reads as empty
            out["chain"] = {}
        return out

    def get(self, receipt_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM content_receipts WHERE id = ?",
            (receipt_id,))
        return self._row(row) if row else None

    def for_contract(self, contract_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM content_receipts WHERE contract_id = ?"
            " ORDER BY created_at", (contract_id,))
        return [self._row(r) for r in rows]

    def recent(self, limit: int = 50, *,
               kind: str = "") -> list[dict[str, Any]]:
        if kind:
            rows = self.db.query(
                "SELECT * FROM content_receipts WHERE kind = ?"
                " ORDER BY created_at DESC LIMIT ?",
                (kind, int(limit)))
        else:
            rows = self.db.query(
                "SELECT * FROM content_receipts"
                " ORDER BY created_at DESC LIMIT ?", (int(limit),))
        return [self._row(r) for r in rows]

    def attach_post(self, receipt_id: str, *, post_ref: str = "",
                    chain: dict[str, Any] | None = None
                    ) -> dict[str, Any] | None:
        """Move a receipt's chain forward when its piece posts:
        the post reference + any chain facts the poster learned."""
        receipt = self.get(receipt_id)
        if receipt is None:
            return None
        merged = dict(receipt["chain"])
        merged.update(chain or {})
        self.db.execute(
            "UPDATE content_receipts SET post_ref = ?, chain_json = ?,"
            " updated_at = ? WHERE id = ?",
            (post_ref or receipt["post_ref"], json.dumps(merged),
             time.time(), receipt_id))
        receipt["post_ref"] = post_ref or receipt["post_ref"]
        receipt["chain"] = merged
        return receipt


def receipt_chain(db: Any, queue: Any,
                  receipt: dict[str, Any]) -> dict[str, Any]:
    """Resolve one receipt end-to-end, live: the contract's state
    from the delegation store, the draft's state from the queue,
    the post from the social ledger (the spine's honest
    content+platform join — publishing records no draft id on post
    rows), and the post's measured metrics as the spine last wrote
    them. Stages that have not happened yet read as None and are
    NAMED as such by the render — never dressed up."""
    chain: dict[str, Any] = {
        "receipt": receipt, "contract": None, "draft": None,
        "post": None, "metrics": None}
    contract_id = receipt.get("contract_id") or ""
    if contract_id and db is not None:
        try:
            from .delegation import DelegationStore

            contract = DelegationStore(db).get_contract(contract_id)
            if contract is not None:
                chain["contract"] = {
                    "status": contract.status,
                    "title": contract.title}
        except Exception as exc:  # noqa: BLE001 - a read never raises
            _log.debug("receipt contract read failed: %s", exc)
    draft = None
    if receipt.get("draft_id") and queue is not None:
        try:
            draft = queue.get(receipt["draft_id"])
        except Exception as exc:  # noqa: BLE001
            _log.debug("receipt draft read failed: %s", exc)
    if draft is not None:
        chain["draft"] = {
            "status": draft.status,
            "scheduled_time": draft.scheduled_time,
            "posted_at": draft.posted_at}
        if db is not None and draft.status == "posted":
            try:
                rows = db.query(
                    "SELECT external_id, posted_at, metrics, metadata"
                    " FROM social_posts WHERE platform = ?"
                    " AND content = ? AND status = 'posted'"
                    " ORDER BY posted_at DESC LIMIT 1",
                    (receipt.get("platform", ""), draft.content))
                if rows:
                    row = rows[0]
                    chain["post"] = {
                        "external_id": row["external_id"] or "",
                        "posted_at": row["posted_at"]}
                    metrics = row["metrics"]
                    if isinstance(metrics, str):
                        metrics = json.loads(metrics or "{}")
                    if isinstance(metrics, dict) and metrics:
                        meta = row["metadata"]
                        if isinstance(meta, str):
                            meta = json.loads(meta or "{}")
                        chain["metrics"] = {
                            "values": metrics,
                            "fetched_at": (meta or {}).get(
                                "metrics_fetched_at")}
            except Exception as exc:  # noqa: BLE001
                _log.debug("receipt post join failed: %s", exc)
    if chain["post"] is None and receipt.get("post_ref"):
        chain["post"] = {
            "external_id": str(
                (receipt.get("chain") or {}).get("external_id") or ""),
            "ref": receipt["post_ref"]}
    return chain


def render_receipt(chain: dict[str, Any]) -> str:
    """One piece's chain as readable text. Internal ids never
    render — pieces are named by label + platform; a post is named
    by its platform-side id, which is public by construction."""
    receipt = chain["receipt"]
    label = receipt.get("label") or "(untitled piece)"
    platform = receipt.get("platform") or ""
    head = f"🧾 {label}" + (f" — {platform}" if platform else "")
    stages: list[str] = []
    contract = chain.get("contract")
    stages.append(
        f"contract {contract['status']}" if contract
        else "contract not recorded")
    draft = chain.get("draft")
    if receipt.get("draft_id"):
        if draft is None:
            stages.append("draft gone from the queue")
        elif draft["status"] == "scheduled" and draft.get(
                "scheduled_time"):
            when = time.strftime(
                "%a %H:%M", time.localtime(draft["scheduled_time"]))
            stages.append(f"draft scheduled for {when}")
        else:
            stages.append(f"draft {draft['status']}")
    post = chain.get("post")
    if post:
        ext = post.get("external_id") or ""
        stages.append(f"post live{(' · ' + ext) if ext else ''}")
    elif receipt.get("draft_id"):
        stages.append("not posted yet")
    metrics = chain.get("metrics")
    if metrics:
        values = metrics.get("values") or {}
        total = sum(v for v in values.values()
                    if isinstance(v, (int, float)))
        fetched = metrics.get("fetched_at")
        when = (time.strftime("%b %d %H:%M", time.localtime(fetched))
                if fetched else "time unrecorded")
        stages.append(
            f"metrics: {total:g} engagements (refreshed {when})")
    elif post:
        stages.append("metrics not measured yet")
    return head + "\n   " + " · ".join(stages)


# ── the weekly pipeline's cell (the injected protocol) ────────────────

class WeeklyCell(ContentCell):
    """The cell as ``social/content_pipeline.weekly_content``
    drives it — the pipeline (L4) calls this duck-typed protocol;
    it imports nothing from this layer:

    * ``research(findings)`` — findings enter the ledger with
      their provenance; the run's summary claim is the one the
      completion gate reads.
    * ``review(text, basis=…, fit=…)`` — the skeptic gate over one
      drafted text; returns an objection string (the text must NOT
      be queued) or None.
    * ``landed(pieces)`` — receipts per queued draft + the delivery
      claim + completion; the run's closing record.
    """

    def __init__(self, contracts: Any, receipts: ReceiptStore, *,
                 niche: str,
                 niche_context: dict[str, Any] | None = None) -> None:
        super().__init__(contracts, kind="weekly")
        self.receipts = receipts
        self.niche = niche
        #: The run's niche read through the shared registry (WS4
        #: P5, contract 2): the effective record (floor plugin or
        #: owner-adopted overlay) when the run's niche names one,
        #: else None. A proposed niche never resolves — drafting
        #: context is production state, and production state is
        #: the floor plus what the owner adopted.
        self.niche_context = niche_context

    def begin(self) -> Any:
        goal = (f"this week's content for {self.niche}: bounded"
                " research → drafts in the owner's voice → the"
                " review queue")
        if self.niche_context:
            thesis = str(self.niche_context.get("thesis") or "")
            if thesis:
                goal += f" — niche thesis: {thesis}"
        return self.open(
            goal=goal,
            deliverable=("per-platform drafts landed in the review"
                         " queue, each with a readable receipt"
                         " chain"),
            required_evidence=("research", "draft"))

    def research(self, findings: list[dict[str, Any]]) -> None:
        if self.niche_context:
            slug = str(self.niche_context.get("slug") or self.niche)
            origin = str(self.niche_context.get("origin") or "")
            self.evidence(
                "research",
                f"niche context: {slug}"
                + (f" ({origin})" if origin else "")
                + f" — {self.niche_context.get('thesis') or ''}",
                producer=self.researcher,
                evidence_ref=f"niche:{slug}",
                verify=True)
        for finding in findings:
            title = str(finding.get("title") or "").strip()
            if not title:
                continue
            self.evidence(
                "research", f"finding: {title}",
                producer=self.researcher,
                evidence_ref=str(finding.get("url") or ""))
        if findings:
            refs = [str(f.get("url") or "") for f in findings]
            self.evidence(
                "research",
                f"research: {len(findings)} finding(s) from the"
                f" bounded run on {self.niche}",
                producer=self.researcher,
                evidence_ref=next((u for u in refs if u), ""),
                verify=True)
        else:
            self.evidence(
                "research",
                f"research: the bounded run on {self.niche} returned"
                " no findings — this batch is template-drafted and"
                " says so",
                producer=self.researcher,
                evidence_ref=f"weekly:{self.niche}:no-findings",
                verify=True)

    def review(self, text: str, *, basis: str, fit: str = "") -> str | None:
        claim = self.evidence(
            "draft",
            f"weekly draft ({len(text)} chars"
            + (f"; {fit}" if fit else "") + ")",
            producer=self.drafter,
            evidence_ref=f"draft:{_sha(text)}")
        bad = untraced_numbers(text, basis)
        if bad:
            objection = (
                f"draft states number(s) {_fmt_numbers(bad)} with"
                " no source in this week's research")
            self.block(claim.id, objection,
                       note="untraced numbers never queue")
            return objection
        self.contracts.verify_claim(
            claim.id, validator=self.verifier, verdict="supported")
        return None

    def landed(self, pieces: list[dict[str, Any]]) -> dict[str, Any]:
        receipts = [
            self.receipts.record(
                contract_id=self.contract_id, kind="weekly",
                label=_first_line(str(piece.get("text") or "")),
                platform=str(piece.get("platform") or ""),
                draft_id=str(piece.get("draft_id") or ""),
                content=str(piece.get("text") or ""))
            for piece in pieces]
        if not pieces:
            return {
                "contract_id": self.contract_id, "receipts": [],
                "completed": False,
                "note": "every draft was blocked by the skeptic —"
                        " nothing was queued"}
        platforms = ", ".join(
            sorted({str(p.get("platform") or "") for p in pieces}))
        self.deliver(
            "draft",
            f"delivery: {len(pieces)} draft(s) landed in the"
            f" review queue ({platforms})",
            evidence_ref=f"weekly:{self.niche}:delivery")
        decision = self.finish()
        return {
            "contract_id": self.contract_id, "receipts": receipts,
            "completed": bool(decision.ok),
            "note": "" if decision.ok else (
                "drafts landed, but the weekly contract could not"
                " complete: " + "; ".join(decision.reasons or []))}


def make_weekly_cell(db: Any, niche: str) -> WeeklyCell | None:
    """The composition factory ``social.content_pipeline`` asks
    for a cell through (installed by ``agents/context.py`` — the
    factory-inversion pattern: the L4 pipeline may not import this
    L5 module). None when there is no database to hold contracts.

    The cell's niche context resolves through the WS4 P5 adapter
    (``media/explore/feed.py``) over the explore store bound to
    this database's KV — the shared-registry read (contract 2):
    an adopted overlay niche's thesis frames the run; a proposed
    one is invisible. Resolution failure degrades to no context;
    the cell runs exactly as before."""
    if db is None:
        return None
    from .delegation import DelegationStore

    niche_context = None
    try:
        from ..media.explore.feed import niche_drafting_context
        from ..media.explore.store import ExploreStore
        from ..storage.kv import KVStore

        niche_context = niche_drafting_context(
            ExploreStore(KVStore(db)), niche or "")
    except Exception as exc:  # noqa: BLE001 - context is an enhancement
        _log.debug("weekly cell niche context unresolved: %s", exc)
    return WeeklyCell(
        DelegationStore(db), ReceiptStore(db), niche=niche or "general",
        niche_context=niche_context)
