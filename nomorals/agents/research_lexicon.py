"""Dynamic vocabulary expansion: a scored, versioned, DB-backed lexicon.

Research (the swarm, partner phrasing experiments, domain modules) constantly
produces vocabulary that consuming modules should use *without a code deploy*.
The lexicon lives in the database (``lexicon_terms`` / ``lexicon_versions``
from migration 62), versioned per consuming module, so terms can be acquired,
scored, retired, and reloaded at runtime.

Data flow:

    research findings -> mine_candidates() -> LexiconStore.acquire()
        -> lexicon_terms (scored, versioned)
        -> consumers read via dynamic_terms(db, module, category)

Scoring formula (``score_term``), all components in [0, 1]:

    score = 0.5 * relevance + 0.3 * novelty + 0.2 * quality

* **relevance** — fraction of the term's content words (letters/digits,
  >= 4 chars, not stopwords) that also appear in the category keywords
  (matched at word level, so multi-word keywords work). When no keywords
  are given, relevance is a neutral 0.5.
* **novelty** — ``1.0 - max_fuzzy_similarity`` against the terms the module
  already has. An exact match scores 0.0. Fuzzy similarity is the
  ``difflib.SequenceMatcher`` ratio on normalized text.
* **quality** — length sweet spot 3..60 chars (shorter/longer decay),
  alpha ratio >= 0.6 (digit/symbol soup penalized), no URLs, and no
  single-character dominance > 0.6 (``aaaaaa``-style junk).

Degenerate candidates — empty, < 3 chars after normalization, containing
a URL, or dominated by one repeated character — score 0.0 outright.

Layering: this module is L5 (agents/) but depends only downward
(core logging/ids/errors/policy). It never imports ``partner`` — the
consumer hook in ``partner/style.py`` calls *into* this module (or uses
``dynamic_terms``), never the reverse, so no import cycle is possible.
"""

from __future__ import annotations

import difflib
import re
import time
from types import SimpleNamespace
from typing import Any

from ..core.errors import ToolError
from ..core.ids import new_id
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "LexiconStore",
    "acquire_from_findings",
    "dynamic_terms",
    "mine_candidates",
    "normalize_term",
    "register",
    "score_term",
]

#: Normalization: collapse internal whitespace, lowercase.
_WS = re.compile(r"\s+")

#: Terms longer than this are junk, not vocabulary.
MAX_TERM_LEN = 80

#: Normalized edit similarity at or above this counts as a duplicate.
FUZZY_DUPLICATE = 0.85

#: Words that carry no meaning for relevance / mining.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has",
    "had", "was", "were", "are", "will", "would", "about", "into",
    "over", "under", "than", "then", "them", "they", "their", "what",
    "when", "where", "which", "while", "who", "how", "why", "can",
    "could", "should", "does", "doing", "done", "best", "new", "using",
    "use", "used", "based", "such", "like", "also", "more", "most",
    "much", "many", "some", "any", "all", "its", "it's", "you", "your",
    "yours", "our", "ours", "their", "theirs", "been", "being", "these",
    "those", "between", "through", "after", "before", "other", "another",
    "first", "second", "however", "because", "just", "even", "very",
    "well", "really", "make", "made", "get", "getting", "got", "take",
    "takes", "taken", "people", "things", "thing", "much", "many",
    "often", "always", "never", "still", "already", "yet", "both",
    "each", "every", "either", "neither", "whether", "within", "without",
    "among", "along", "across", "behind", "beyond", "during", "until",
    "again", "once", "twice", "here", "there", "then", "thus", "hence",
    "maybe", "perhaps", "quite", "rather", "seems", "seemed", "seem",
    "shown", "show", "shows", "found", "find", "finds", "said", "says",
    "say", "told", "tell", "according", "report", "reports", "reported",
    "source", "sources", "claim", "claims", "study", "studies", "data",
}

#: Filler words that make an n-gram generic even when 4+ chars long.
_GENERIC = _STOPWORDS | {
    "with", "that", "this", "from", "have", "more", "most", "many",
    "some", "over", "under", "when", "where", "which", "about", "into",
    "will", "would", "should", "could", "using", "used", "based",
    "also", "such", "like", "make", "made", "even", "just", "than",
    "these", "those", "been", "being", "what", "while", "other",
    "first", "however", "because", "people", "things", "thing",
    "much", "often", "still", "every", "within", "without", "during",
}

_QUOTED_DOUBLE = re.compile(r'"([^"\n]{3,80})"')
_QUOTED_SINGLE = re.compile(r"(?<!\w)'([A-Za-z][^'\n]{2,79})'")
_TITLE_CASE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})\b")
_WORD = re.compile(r"[a-z][a-z0-9']*")
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)


def normalize_term(raw: Any) -> str:
    """Normalize a candidate: strip, lowercase, collapse whitespace."""
    return _WS.sub(" ", str(raw or "").strip().lower())


def _similarity(a: str, b: str) -> float:
    """Normalized edit similarity in [0, 1] (difflib ratio)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _content_words(text: str) -> set[str]:
    """Words that carry meaning: >= 4 chars, not stopwords."""
    return {w for w in _WORD.findall(text) if len(w) >= 4 and w not in _STOPWORDS}


def _is_degenerate(text: str) -> bool:
    """True for junk that must score 0: too short, a URL, or one-char mush."""
    if not text or len(text) < 3:
        return True
    if _URL.search(text):
        return True
    chars = [c for c in text if not c.isspace()]
    if chars and max(chars.count(c) for c in set(chars)) / len(chars) > 0.6:
        return True
    return False


def _quality(text: str) -> float:
    """Quality in [0, 1]: length sweet spot, alpha ratio, repetition."""
    q = 1.0
    n = len(text)
    if n < 3:
        q *= 0.2
    elif n > 60:
        q *= max(0.2, 1.0 - (n - 60) / 40.0)
    chars = [c for c in text if not c.isspace()]
    if chars:
        alpha = sum(1 for c in chars if c.isalpha())
        ratio = alpha / len(chars)
        if ratio < 0.6:
            q *= ratio / 0.6
        top = max(chars.count(c) for c in set(chars)) / len(chars)
        if top > 0.4:
            q *= max(0.2, 1.0 - top)
    return max(0.0, min(1.0, q))


def score_term(
    term: str,
    category_keywords: list[str] | None = None,
    existing_terms: Any = (),
) -> float:
    """Score a candidate term in [0, 1].

    ``0.5 * relevance + 0.3 * novelty + 0.2 * quality`` (see module
    docstring). ``category_keywords`` may be None (neutral 0.5 relevance)
    or a list of words/phrases; matching is at the word level so
    multi-word keywords work. ``existing_terms`` is any iterable of
    already-acquired terms used for the novelty component. Degenerate
    input (URL, < 3 chars, single-char mush) scores 0.0.
    """
    text = normalize_term(term)
    if _is_degenerate(text):
        return 0.0
    keywords = {
        w
        for kw in (category_keywords or [])
        for w in _WORD.findall(normalize_term(kw))
    }
    content = _content_words(text)
    if not keywords:
        relevance = 0.5
    elif not content:
        relevance = 0.0
    else:
        relevance = len(content & keywords) / len(content)
    sims = [
        _similarity(text, normalize_term(e))
        for e in (existing_terms or ())
        if normalize_term(e)
    ]
    novelty = 1.0 - max(sims, default=0.0)
    quality = _quality(text)
    return round(0.5 * relevance + 0.3 * novelty + 0.2 * quality, 3)


def _finding_claims(findings: Any) -> list[str]:
    """Extract claim strings from SwarmFinding objects or plain dicts."""
    claims: list[str] = []
    for f in findings or []:
        claim = getattr(f, "claim", None)
        if claim is None and isinstance(f, dict):
            claim = f.get("claim")
        if claim:
            claims.append(str(claim))
    return claims


def _quoted_spans(claims: list[str]) -> list[str]:
    out: list[str] = []
    for claim in claims:
        for m in _QUOTED_DOUBLE.finditer(claim):
            out.append(m.group(1))
        for m in _QUOTED_SINGLE.finditer(claim):
            out.append(m.group(1))
    return [s for s in (_WS.sub(" ", s.strip()) for s in out) if s]


def _title_case_phrases(claims: list[str]) -> list[str]:
    out: list[str] = []
    for claim in claims:
        out.extend(_TITLE_CASE.findall(claim))
    return out


def _repeated_ngrams(claims: list[str]) -> list[str]:
    """Bigrams/trigrams (4+ char content words) seen in >= 2 findings,
    skipping stopword-heavy/generic runs."""
    per_claim: list[set[str]] = []
    for claim in claims:
        tokens = [w for w in _WORD.findall(claim.lower()) if len(w) >= 4]
        grams: set[str] = set()
        for size in (2, 3):
            for i in range(len(tokens) - size + 1):
                run = tokens[i : i + size]
                if sum(1 for w in run if w in _GENERIC) * 2 >= len(run):
                    continue
                grams.add(" ".join(run))
        per_claim.append(grams)
    counts: dict[str, int] = {}
    for grams in per_claim:
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
    return sorted(
        (g for g, c in counts.items() if c >= 2),
        key=lambda g: (-counts[g], g),
    )


def mine_candidates(findings: Any, *, max_terms: int = 60) -> list[str]:
    """Pull candidate phrases from SwarmFinding claims.

    Sources, in priority order:

    a. quoted spans — ``"..."`` or ``'...'`` in the claim text;
    b. Title Case multi-word phrases (``Phi Three``);
    c. distinctive bigrams/trigrams — consecutive 4+ char content-word
       runs appearing in >= 2 findings, skipping stopword-heavy runs.

    Results are normalized, deduped (case-insensitive), and capped at
    ``max_terms``. Pure function — no DB, no context.
    """
    claims = _finding_claims(findings)
    seen: set[str] = set()
    out: list[str] = []

    def _emit(raw: str) -> None:
        norm = normalize_term(raw)
        if (
            not norm
            or len(norm) > MAX_TERM_LEN
            or _is_degenerate(norm)
            or norm in seen
        ):
            return
        seen.add(norm)
        out.append(norm)

    for span in _quoted_spans(claims):
        _emit(span)
    for phrase in _title_case_phrases(claims):
        _emit(phrase)
    for gram in _repeated_ngrams(claims):
        _emit(gram)
    return out[: max(1, int(max_terms or 60))]


class LexiconStore:
    """DB-backed lexicon for one deployment. Constructed with an agent
    ``context`` and using ``context.db``."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = context.db

    # ── internals ────────────────────────────────────────────────────
    def _active_terms(self, module: str, category: str) -> list[str]:
        rows = self.db.query(
            "SELECT term FROM lexicon_terms "
            "WHERE module = ? AND category = ? AND status = 'active' "
            "ORDER BY score DESC",
            (module, category),
        )
        return [str(r["term"]) for r in rows]

    # ── public API ───────────────────────────────────────────────────
    def acquire(
        self,
        candidates: list[str],
        *,
        module: str,
        category: str,
        source: str,
        threshold: float = 0.55,
        category_keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        """Score candidates and insert those >= ``threshold``.

        Normalizes each candidate (strip, lowercase, collapse
        whitespace; skip empties and terms > 80 chars as ``invalid``),
        dedupes against existing active terms for (module, category) —
        exact AND fuzzy (normalized edit similarity >= 0.85) — as
        ``duplicate``, and scores the rest via :func:`score_term`
        (below threshold -> ``low_score``). Inserts carry the new
        version number; ``lexicon_versions`` is bumped (version+1,
        active term count, timestamp) only when at least one term was
        added.
        """
        module = (module or "").strip()
        if not module:
            raise ToolError("lexicon acquire needs a module name")
        category = (category or "general").strip() or "general"
        source = (source or "").strip()
        threshold = float(threshold)

        current_version = self.version(module)
        new_version = current_version + 1
        existing = self._active_terms(module, category)
        added: list[str] = []
        skipped: list[dict[str, str]] = []
        now = time.time()

        for raw in candidates or []:
            norm = normalize_term(raw)
            if not norm or len(norm) > MAX_TERM_LEN:
                skipped.append(
                    {"term": str(raw or "")[:MAX_TERM_LEN], "reason": "invalid"}
                )
                continue
            if norm in existing or any(
                _similarity(norm, e) >= FUZZY_DUPLICATE for e in existing
            ):
                skipped.append({"term": norm, "reason": "duplicate"})
                continue
            score = score_term(norm, category_keywords, existing)
            if score < threshold:
                skipped.append({"term": norm, "reason": "low_score"})
                continue
            self.db.execute(
                "INSERT INTO lexicon_terms "
                "(id, term, category, module, score, source, version, "
                " status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)",
                (
                    new_id("lex"),
                    norm,
                    category,
                    module,
                    score,
                    source,
                    new_version,
                    now,
                ),
            )
            existing.append(norm)
            added.append(norm)

        if added:
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM lexicon_terms "
                "WHERE module = ? AND status = 'active'",
                (module,),
            )
            term_count = int(row["n"]) if row else 0
            self.db.execute(
                "INSERT INTO lexicon_versions "
                "(module, version, term_count, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(module) DO UPDATE SET "
                "version = excluded.version, "
                "term_count = excluded.term_count, "
                "updated_at = excluded.updated_at",
                (module, new_version, term_count, now),
            )
        return {
            "added": added,
            "skipped": skipped,
            "version": new_version if added else current_version,
        }

    def terms_for(
        self, module: str, category: str, limit: int = 50, *, min_score: float = 0.0
    ) -> list[dict[str, Any]]:
        """Active terms for (module, category), score desc."""
        rows = self.db.query(
            "SELECT term, score, version FROM lexicon_terms "
            "WHERE module = ? AND category = ? AND status = 'active' "
            "AND score >= ? ORDER BY score DESC LIMIT ?",
            (module, category, float(min_score), int(limit)),
        )
        return [
            {"term": str(r["term"]), "score": r["score"], "version": r["version"]}
            for r in rows
        ]

    def retire(self, term: str, module: str, *, reason: str = "") -> bool:
        """Mark a term retired. True when a row changed. The reason is
        recorded in the log (the schema carries no reason column)."""
        norm = normalize_term(term)
        cur = self.db.execute(
            "UPDATE lexicon_terms SET status = 'retired' "
            "WHERE term = ? AND module = ? AND status = 'active'",
            (norm, module),
        )
        changed = cur.rowcount > 0
        if changed:
            _log.info("lexicon retired %r from module %s: %s", norm, module, reason)
        return changed

    def version(self, module: str) -> int:
        """Current lexicon version for a module (0 when none)."""
        row = self.db.query_one(
            "SELECT version FROM lexicon_versions WHERE module = ?", (module,)
        )
        return int(row["version"]) if row else 0

    def stats(self, module: str = "") -> dict[str, Any]:
        """Counts by status and category, optionally for one module."""
        where: str
        params: tuple[Any, ...]
        if module:
            where, params = "WHERE module = ?", (module,)
        else:
            where, params = "", ()
        rows = self.db.query(
            "SELECT status, category, COUNT(*) AS n FROM lexicon_terms "
            f"{where} GROUP BY status, category",
            params,
        )
        by_status: dict[str, int] = {}
        by_category: dict[str, int] = {}
        total = 0
        for r in rows:
            n = int(r["n"])
            total += n
            by_status[str(r["status"])] = by_status.get(str(r["status"]), 0) + n
            by_category[str(r["category"])] = by_category.get(
                str(r["category"]), 0
            ) + n
        return {"total": total, "by_status": by_status, "by_category": by_category}


def acquire_from_findings(
    findings: Any,
    store: LexiconStore,
    *,
    module: str,
    category: str,
    source: str,
    category_keywords: list[str] | None = None,
    threshold: float = 0.55,
) -> dict[str, Any]:
    """One-call research -> lexicon path: mine candidates from findings,
    then acquire them into the store."""
    candidates = mine_candidates(findings)
    return store.acquire(
        candidates,
        module=module,
        category=category,
        source=source,
        threshold=threshold,
        category_keywords=category_keywords,
    )


def dynamic_terms(db: Any, module: str, category: str, limit: int = 50) -> tuple[str, ...]:
    """Cheap read helper for consumers (no context needed).

    Returns active terms, score desc. Never raises — consumer hot paths
    must not break when the DB is unavailable, so any error yields ().
    """
    try:
        if db is None:
            return ()
        rows = db.query(
            "SELECT term FROM lexicon_terms "
            "WHERE module = ? AND category = ? AND status = 'active' "
            "ORDER BY score DESC LIMIT ?",
            (module, category, int(limit)),
        )
        return tuple(str(r["term"]) for r in rows)
    except Exception as exc:  # noqa: BLE001 - consumer hot paths must never break
        _log.debug("dynamic_terms read failed: %s", exc)
        return ()


# ── tool registration ────────────────────────────────────────────────


def register(registry: Any) -> None:
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "research_lexicon",
        description=(
            "dynamic vocabulary expansion: score research-mined terms "
            "and store them in a versioned per-module lexicon (no "
            "restart needed for consumers). Actions: acquire new "
            "candidates, terms (list active), retire a term, version, "
            "stats, mine (extract candidates from claim text)."
        ),
        capability=Capability.DB_WRITE,
        parameters={
            "action": "str — acquire|terms|retire|version|stats|mine",
            "module": "str — consuming module (e.g. partner)",
            "category": "str (optional) — term category, default general",
            "terms": "list (optional) — candidate strings for acquire",
            "term": "str (optional) — term for retire",
            "source": "str (optional) — provenance label for acquire",
            "reason": "str (optional) — retire reason",
            "limit": "int (optional) — terms listing limit, default 50",
            "claims": "list (optional) — claim strings for mine",
            "keywords": "list (optional) — category keywords for scoring",
            "threshold": "float (optional) — acquire threshold, default 0.55",
        },
    )
    def research_lexicon(
        action: str = "terms",
        module: str = "",
        category: str = "general",
        terms: Any = None,
        term: str = "",
        source: str = "research",
        reason: str = "",
        limit: int = 50,
        claims: Any = None,
        keywords: Any = None,
        threshold: float = 0.55,
        **_: Any,
    ) -> dict[str, Any]:
        store = LexiconStore(context)
        action = (action or "terms").strip().lower()
        if action == "acquire":
            if not module:
                raise ToolError("research_lexicon acquire needs module")
            return store.acquire(
                list(terms or []),
                module=module,
                category=category,
                source=source,
                threshold=float(threshold or 0.55),
                category_keywords=list(keywords or []) or None,
            )
        if action == "terms":
            return {
                "terms": store.terms_for(
                    module, category, limit=int(limit or 50)
                )
            }
        if action == "retire":
            if not (term and module):
                raise ToolError("research_lexicon retire needs term and module")
            return {"retired": store.retire(term, module, reason=reason)}
        if action == "version":
            return {"version": store.version(module)}
        if action == "stats":
            return store.stats(module)
        if action == "mine":
            pseudo = [
                SimpleNamespace(angle="adhoc", claim=str(c))
                for c in (claims or [])
                if str(c).strip()
            ]
            return {"candidates": mine_candidates(pseudo)}
        raise ToolError(f"unknown research_lexicon action {action!r}")
