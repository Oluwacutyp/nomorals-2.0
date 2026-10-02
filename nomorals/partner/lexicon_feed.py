"""The partner's dynamic voice feed: lexicon store -> phrasing choices.

This is the consumer side of the Wave C dynamic lexicon
(``nomorals/agents/research_lexicon.py``). The store holds scored,
versioned terms per (module, category); this module reads the
``partner`` module's categories and blends them into the reply path:

* **prompt voice note** — catchphrases, pet names, openers,
  transitions, acknowledgments, and mood expressions, mixed with the
  persona's own configured banks and handed to the model in the system
  prompt (see :meth:`LexiconFeed.voice_note`);
* **fallback lines** — the infrastructure-failure path in
  ``responder._fallback_parts`` draws from the ``fallback`` category
  when it has terms;
* **guard phrases** — the ``robotic_phrase`` category feeds
  ``style.strip_robotic``'s ``extra_phrases`` hook (support-voice
  detection only; the identity/character gate stays hardcoded).

Fallback visibility: when the store is empty for a category — or the DB
is unavailable — the static banks are used and the category is reported
in the ``fallback_categories`` return value plus a debug log. The reply
bundle carries ``lexicon_dynamic`` / ``lexicon_terms_used`` so "dynamic"
is measurable, never a claim.

Owner control: the persona's own banks (``SpeechProfile.catchphrases``,
``pet_names``) are the base the dynamic terms *add* to — owner-set
values are never replaced. ``settings.partner.lexicon_voice`` disables
all dynamic influence; ``settings.partner.lexicon_seed`` disables the
one-time starter seed. Retired terms are never re-acquired by the seed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

from ..agents.research_lexicon import LexiconStore, dynamic_terms
from ..core.logging_setup import get_logger

__all__ = [
    "LEXICON_MODULE",
    "CATEGORIES",
    "CATEGORY_KEYWORDS",
    "LexiconFeed",
    "seed_partner_lexicon",
]

_log = get_logger(__name__)

#: The module key this consumer uses in ``lexicon_terms`` / ``lexicon_versions``.
LEXICON_MODULE = "partner"

#: Categories the partner voice path reads. ``mood_expression:<label>``
#: (e.g. ``mood_expression:happy``) is also read for the current mood.
CATEGORIES: tuple[str, ...] = (
    "catchphrase",
    "pet_name",
    "opener",
    "transition",
    "acknowledgment",
    "mood_expression",
    "fallback",
    "robotic_phrase",
)

#: Scoring keywords per category for the starter seed (see ``score_term``:
#: relevance is the fraction of the term's content words that hit these).
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "catchphrase": ["catchphrase", "phrase", "saying", "habit", "quip"],
    "pet_name": ["pet", "name", "nickname", "endearment", "call", "them"],
    "opener": ["opener", "open", "start", "begin", "lead"],
    "transition": ["transition", "bridge", "segue", "shift", "subject"],
    "acknowledgment": ["acknowledge", "agree", "confirm", "mhm", "yeah", "hear"],
    "mood_expression": ["mood", "feeling", "expression", "honest", "voice"],
    "fallback": ["fallback", "spare", "glitch", "stall", "cover"],
    "robotic_phrase": ["robotic", "support", "assistant", "help", "article"],
}

#: Starter vocab — small on purpose. The primary mechanism is the
#: acquire -> score -> integrate -> reload pipeline; this just keeps the
#: two embarrassingly thin banks (4 catchphrases, 3 pet names) and the
#: model-carried slots (openers, transitions, acknowledgments) from
#: starting at zero. Every term goes through ``LexiconStore.acquire``
#: (scored, deduped, versioned) — never straight into a prompt.
#:
#: The sets are lexically diverse on purpose (low pairwise similarity):
#: the scorer's novelty component penalizes terms that resemble already-
#: acquired ones, so near-identical seeds would cannibalize each other.
_SEED_TERMS: dict[str, tuple[str, ...]] = {
    "catchphrase": (
        "okay wait",
        "tell me everything",
        "i'm listening",
        "say more",
        "hold on tho",
    ),
    "pet_name": (
        "love",
        "my person",
        "sweetheart",
        "cutie",
        # NOTE: single-word pet names like "b" trip the scorer's
        # degenerate-term rule (< 3 chars), so they stay out of the seed —
        # the pipeline's junk filter wins.
    ),
    "opener": (
        "okay so",
        "real talk",
        "so listen",
        "anyway",
        "look",
    ),
    "transition": (
        "also tho",
        "oh and",
        "on another note",
        "speaking of",
        "btw",
    ),
    "acknowledgment": (
        "yeah no i get it",
        "that's fair",
        "okay yeah",
        "fair enough",
        # NOTE: "mhm" trips the scorer's degenerate-term rule (one-char
        # dominance), so it stays out of the seed — the pipeline's junk
        # filter wins.
    ),
    "mood_expression": (
        "not gonna lie",
        "honestly tho",
        "lowkey though",
    ),
    "fallback": (
        "brain's buffering, one sec",
        "say that again? i spaced",
        "i'm here, just glitching a little",
    ),
    "robotic_phrase": (
        "i'm here to help",
        "how can i assist you",
    ),
}

# Seed threshold: 0.35, not the research default 0.55. Colloquial voice
# terms almost never share content words with category keywords, so their
# relevance lands at the neutral-to-zero end; a term with decent novelty
# and full quality scores ~0.35-0.5. These are starter terms — hand-picked
# for the voice, scored for novelty and quality only — which the explicit
# threshold documents. The junk filter (score 0.0) still rejects
# degenerate terms at any threshold, and fuzzy dedupe still kills dupes.
_SEED_THRESHOLD = 0.35


class LexiconFeed:
    """Read side of the partner lexicon. Never raises on DB trouble."""

    def __init__(self, db: Any = None) -> None:
        self.db = db

    @property
    def available(self) -> bool:
        """False when there is no DB to read — the static banks take over."""
        return self.db is not None

    def terms(self, category: str, limit: int = 20) -> tuple[str, ...]:
        """Active terms for a category, score desc. Never raises."""
        category = (category or "").strip().lower()
        if not category:
            return ()
        return dynamic_terms(self.db, LEXICON_MODULE, category, limit)

    def has(self, category: str) -> bool:
        """True when at least one active term exists for the category."""
        return bool(self.terms(category, limit=1))

    def status(self) -> dict[str, Any]:
        """Owner-inspectable snapshot: availability, version, per-category counts."""
        info: dict[str, Any] = {
            "available": self.available,
            "enabled": True,
            "module": LEXICON_MODULE,
            "version": 0,
            "categories": {},
        }
        if not self.available:
            return info
        try:
            row = self.db.query_one(
                "SELECT version FROM lexicon_versions WHERE module = ?",
                (LEXICON_MODULE,),
            )
            info["version"] = int(row["version"]) if row else 0
            rows = self.db.query(
                "SELECT category, COUNT(*) AS n FROM lexicon_terms "
                "WHERE module = ? AND status = 'active' GROUP BY category",
                (LEXICON_MODULE,),
            )
            info["categories"] = {str(r["category"]): int(r["n"]) for r in rows}
        except Exception as exc:  # noqa: BLE001 - status must never break
            _log.debug("lexicon status read failed: %s", exc)
            info["available"] = False
        return info

    def reload(self) -> dict[str, Any]:
        """Explicit reload step of the acquire loop.

        Term reads are uncached, so there is no stale copy to flush —
        reload re-reads the module version and per-category counts from
        the store and returns them, letting the loop (and ``nm`` tooling)
        confirm newly acquired terms are live.
        """
        return self.status()

    def voice_note(
        self,
        persona: Any,
        label: str,
        *,
        exclude: Sequence[str] = (),
    ) -> tuple[str, int, list[str]]:
        """Build the dynamic voice note for the system prompt.

        Blends the lexicon's terms with the persona's own configured
        banks (owner-set values are the base; dynamic terms only add).
        Returns ``(note, dynamic_terms_used, fallback_categories)`` —
        ``note`` is "" when nothing dynamic exists for any category.

        ``exclude`` skips categories (e.g. ``("catchphrase", "pet_name")``
        when the caller already blended those into the persona's own
        speech block) so no term is listed twice in the prompt.
        """
        excluded = {str(c).strip().lower() for c in (exclude or ())}
        lines: list[str] = []
        used = 0
        fallback: list[str] = []

        def _dyn(category: str, limit: int) -> list[str]:
            nonlocal used
            if category in excluded:
                return []
            terms = list(self.terms(category, limit=limit))
            if terms:
                used += len(terms)
            else:
                fallback.append(category)
            return terms

        speech = getattr(persona, "speech", None)
        # Static banks are the base the dynamic terms ADD to — the persona's
        # own prompt block already carries the static banks, so the note
        # only mentions a bank when dynamic terms exist for it. Empty store
        # -> no note at all (the prompt is byte-identical to the static
        # path), with the fallback visible via ``fallback_categories``.
        dyn_catch = _dyn("catchphrase", 6)
        if dyn_catch:
            static_catch = list(getattr(speech, "catchphrases", ()) or ())
            blended = (dyn_catch + static_catch)[:8]
            lines.append(
                "catchphrases you actually use (rarely — never stack them): "
                + ", ".join(f"{c!r}" for c in blended)
            )

        dyn_pets = _dyn("pet_name", 4)
        if dyn_pets:
            static_pets = list(getattr(speech, "pet_names", ()) or ())
            blended = (dyn_pets + static_pets)[:6]
            lines.append("what you call them: " + ", ".join(f"{p!r}" for p in blended))

        openers = _dyn("opener", 5)
        if openers:
            lines.append("ways you open a thought: " + ", ".join(f"{o!r}" for o in openers))

        transitions = _dyn("transition", 4)
        if transitions:
            lines.append(
                "ways you change the subject: " + ", ".join(f"{t!r}" for t in transitions)
            )

        acks = _dyn("acknowledgment", 4)
        if acks:
            lines.append("ways you acknowledge them: " + ", ".join(f"{a!r}" for a in acks))

        mood_bits = _dyn("mood_expression", 4)
        mood_bits += _dyn(f"mood_expression:{label}", 4)
        if mood_bits:
            lines.append(
                f"how a {label} you actually phrases it: "
                + ", ".join(f"{m!r}" for m in mood_bits[:6])
            )

        if not lines:
            return "", 0, fallback
        note = (
            "Your voice right now — these are YOUR actual words and habits. "
            "Use them when they fit; never force one in, never list them:\n"
            + "\n".join(f"  - {ln}" for ln in lines)
        )
        return note, used, fallback


def seed_partner_lexicon(db: Any, *, source: str = "partner_seed") -> dict[str, Any]:
    """One-time starter vocab through the normal pipeline.

    Every term goes through ``LexiconStore.acquire`` (scored, deduped,
    versioned) — the same path research findings take. Idempotent: skips
    when the partner module already has a version, and never re-adds
    owner-retired terms. Never raises.
    """
    if db is None:
        return {"seeded": False, "reason": "no db"}
    try:
        store = LexiconStore(SimpleNamespace(db=db))
        if store.version(LEXICON_MODULE) > 0:
            return {"seeded": False, "reason": "already versioned"}
        retired = {
            str(r["term"])
            for r in db.query(
                "SELECT term FROM lexicon_terms WHERE module = ? AND status = 'retired'",
                (LEXICON_MODULE,),
            )
        }
    except Exception as exc:  # noqa: BLE001 - seeding must never break boot
        _log.warning("lexicon seed precheck failed: %s", exc)
        return {"seeded": False, "reason": f"precheck failed: {exc}"}

    added: list[str] = []
    try:
        for category, seeds in _SEED_TERMS.items():
            candidates = [s for s in seeds if s not in retired]
            if not candidates:
                continue
            result = store.acquire(
                candidates,
                module=LEXICON_MODULE,
                category=category,
                source=source,
                threshold=_SEED_THRESHOLD,
                category_keywords=CATEGORY_KEYWORDS.get(category),
            )
            added.extend(result["added"])
            skipped = [s for s in result["skipped"] if s["reason"] != "duplicate"]
            if skipped:
                _log.debug(
                    "lexicon seed: %d %r terms below threshold: %s",
                    len(skipped),
                    category,
                    ", ".join(s["term"] for s in skipped),
                )
    except Exception as exc:  # noqa: BLE001 - seeding must never break boot
        _log.warning("lexicon seed acquire failed: %s", exc)
        return {"seeded": False, "reason": f"acquire failed: {exc}", "added": added}
    try:
        version = store.version(LEXICON_MODULE)
    except Exception:  # noqa: BLE001
        version = 0
    _log.info("lexicon seed: %d starter terms acquired (version %d)", len(added), version)
    return {"seeded": True, "added": added, "version": version}
