"""Brief-first content pipeline (build-map #99).

Frase/Surfer rule, enforced: no brief, no draft, no schedule.

Pipeline order is mandatory:
    brief → draft → score → schedule

1. build_brief(topic) — search backends (injectable) → keyword
   extraction → questions people ask → competitor angles → outline.
2. score_draft(draft, brief) — term coverage (brief keywords in the
   draft), readability, brand-voice match (reuses #44 virality_score).
   Surfer pattern: write → score updates → term suggestions.
3. schedule_draft() REFUSES any draft with no brief attached.
4. predict_performance() — pre-publish engagement prediction
   (Anyword pattern) before scheduling.
5. record_outcome() — published performance feeds back into the
   scorer (#40 resurfacing pattern).

Every public function never raises.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

_log = logging.getLogger("nomorals.marketing.briefs")


# ── data ────────────────────────────────────────────────────────────────────


@dataclass
class ContentBrief:
    """SERP-derived brief for one topic."""

    brief_id: str = ""
    topic: str = ""
    keywords: list[str] = field(default_factory=list)   # ranked, deduped
    questions: list[str] = field(default_factory=list)  # people ask these
    angles: list[str] = field(default_factory=list)     # competitor angles
    outline: list[str] = field(default_factory=list)    # section headings
    created_at: float = 0.0

    def summary(self) -> str:
        try:
            lines = [f"📋 Brief: {self.topic or '(untitled)'}"]
            if self.keywords:
                lines.append("keywords: " + ", ".join(self.keywords[:8]))
            if self.questions:
                lines.append("people ask: " + " | ".join(self.questions[:3]))
            if self.angles:
                lines.append("angles: " + " | ".join(self.angles[:3]))
            if self.outline:
                lines.append("outline: " + " → ".join(self.outline[:5]))
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return "📋 Brief"


@dataclass
class ContentScore:
    """score_draft() result."""

    score: float = 0.0            # 0-100
    term_coverage: float = 0.0    # 0-1, brief keywords present in draft
    readability: float = 0.0      # 0-1, Flesch-ish
    brand_voice: float = 0.0      # 0-100, from #44 virality_score
    missing_terms: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    source: str = "brief_pipeline"  # learned | heuristic

    def format(self) -> str:
        try:
            lines = [
                f"📊 draft score: {self.score:.0f}/100",
                f"  term coverage {self.term_coverage:.0%} · "
                f"readability {self.readability:.0%} · "
                f"brand voice {self.brand_voice:.0f}/100",
            ]
            if self.missing_terms:
                lines.append("  missing terms: " + ", ".join(self.missing_terms[:6]))
            for s in self.suggestions[:4]:
                lines.append(f"  • {s}")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return "📊 draft score unavailable"


@dataclass
class BriefDraft:
    """A draft produced through the brief-first pipeline."""

    draft_id: str = ""
    topic: str = ""
    content: str = ""
    brief_id: str = ""
    platform: str = "x"
    score: float = 0.0
    predicted_engagement: float = 0.0  # pre-publish prediction 0-100
    scheduled_for: float = 0.0        # 0 = unscheduled
    published_at: float = 0.0
    outcome_metrics: dict = field(default_factory=dict)  # feedback loop
    created_at: float = 0.0


# ── 1. brief building ───────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z][a-z0-9'’-]{2,}")
_QUESTION_HINTS = re.compile(
    r"\b(how|what|why|when|where|which|who|can|should|is|are|do|does)\b",
    re.IGNORECASE,
)

_STOPWORDS = frozenset(
    "the and for are with you your from that this have has was were will would "
    "there their they them our out about into over after also not but all any one "
    "two get got can just like more most new use used using make made many much "
    "what when where which who why how does did its it's his her she him are is "
    "was be been being been of to in on at by as an a".split()
)


def _extract_keywords(texts: list[str], topic: str, top_n: int = 12) -> list[str]:
    """Keyword extraction: frequency over stopword-filtered unigrams+bigrams."""
    try:
        freq: dict[str, int] = {}
        bigram_freq: dict[str, int] = {}
        for text in texts:
            words = [
                w for w in _WORD_RE.findall((text or "").lower())
                if w not in _STOPWORDS
            ]
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            for a, b in zip(words, words[1:]):
                bigram_freq[f"{a} {b}"] = bigram_freq.get(f"{a} {b}", 0) + 1
        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        keywords = [w for w, _ in ranked[: top_n // 2]]
        for bg, _ in sorted(bigram_freq.items(), key=lambda kv: -kv[1])[: top_n // 2]:
            if bg not in keywords:
                keywords.append(bg)
        # Always seed the topic's own words first (they must appear).
        topic_words = [
            w for w in _WORD_RE.findall((topic or "").lower())
            if w not in _STOPWORDS
        ]
        seed = [w for w in topic_words if w not in keywords]
        return (seed + keywords)[:top_n]
    except Exception:  # noqa: BLE001
        return []


def _extract_questions(texts: list[str], top_n: int = 6) -> list[str]:
    """Questions people ask: interrogative sentences + question hints."""
    try:
        seen: list[str] = []
        for text in texts:
            for sent in re.split(r"(?<=[.?!])\s+", text or ""):
                s = sent.strip()
                if len(s) > 140:
                    continue
                if s.endswith("?") or _QUESTION_HINTS.match(s):
                    norm = s if s.endswith("?") else s.rstrip(".") + "?"
                    if norm not in seen and len(norm.split()) > 3:
                        seen.append(norm)
                if len(seen) >= top_n:
                    break
            if len(seen) >= top_n:
                break
        return seen[:top_n]
    except Exception:  # noqa: BLE001
        return []


def _extract_angles(texts: list[str], topic: str, top_n: int = 5) -> list[str]:
    """Competitor angles: distinctive claims/positions from source snippets."""
    try:
        angles: list[str] = []
        for text in texts:
            first = (text or "").strip().split(".")[0].strip()
            if 20 < len(first) < 160 and first not in angles:
                angles.append(first)
            if len(angles) >= top_n:
                break
        if not angles:
            return [f"{topic}: the honest take", f"{topic}: what nobody tells you"]
        return angles[:top_n]
    except Exception:  # noqa: BLE001
        return [f"{topic}: the honest take"]


def _build_outline(topic: str, keywords: list[str], questions: list[str],
                   angles: list[str]) -> list[str]:
    try:
        outline = ["The hook: why this matters now"]
        for q in questions[:2]:
            outline.append(f"Answered: {q[:60]}")
        if keywords:
            outline.append("The essentials: " + ", ".join(keywords[:4]))
        for a in angles[:2]:
            outline.append(f"Angle: {a[:60]}")
        outline.append("The takeaway")
        return outline[:7]
    except Exception:  # noqa: BLE001
        return ["The takeaway"]


def _default_search(topic: str) -> list[dict]:
    """No-search honesty: brief from the topic alone, flagged heuristic."""
    return []


class BriefStore:
    """SQLite store for briefs and brief-pipeline drafts. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            import os

            path = db_path or os.path.expanduser(
                "~/.nomorals/marketing/briefs.db")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS briefs(
                  brief_id TEXT PRIMARY KEY, topic TEXT, keywords TEXT,
                  questions TEXT, angles TEXT, outline TEXT, created_at REAL);
                CREATE TABLE IF NOT EXISTS drafts(
                  draft_id TEXT PRIMARY KEY, topic TEXT, content TEXT,
                  brief_id TEXT, platform TEXT, score REAL,
                  predicted_engagement REAL, scheduled_for REAL,
                  published_at REAL, outcome_metrics TEXT, created_at REAL);
                CREATE TABLE IF NOT EXISTS score_feedback(
                  draft_id TEXT, score REAL, outcome REAL, ts REAL);
                """
            )
            self._db.commit()
        except Exception:  # noqa: BLE001
            self._db = None

    # ── briefs ──

    def build_brief(self, topic: str,
                    search_fn=None) -> ContentBrief | None:
        """Search → keywords → questions → angles → outline. Never raises."""
        try:
            topic = (topic or "").strip()
            if not topic:
                return None
            texts: list[str] = []
            try:
                results = (search_fn(topic) if search_fn
                           else _default_search(topic)) or []
                for r in results:
                    if isinstance(r, dict):
                        blob = " ".join(str(r.get(k, "")) for k in
                                        ("title", "snippet", "text", "url"))
                    else:
                        blob = str(r)
                    if blob.strip():
                        texts.append(blob)
            except Exception:  # noqa: BLE001
                texts = []
            keywords = _extract_keywords(texts, topic)
            questions = _extract_questions(texts)
            if not questions:
                # Heuristic fallbacks still derived from the topic.
                questions = [
                    f"What is {topic} and why does it matter?",
                    f"How does {topic} actually work in practice?",
                ]
            angles = _extract_angles(texts, topic)
            outline = _build_outline(topic, keywords, questions, angles)
            brief = ContentBrief(
                brief_id="brief_" + uuid.uuid4().hex[:8],
                topic=topic, keywords=keywords, questions=questions,
                angles=angles, outline=outline, created_at=time.time(),
            )
            if self._db is not None:
                import json

                self._db.execute(
                    "INSERT INTO briefs VALUES (?,?,?,?,?,?,?)",
                    (brief.brief_id, brief.topic, json.dumps(brief.keywords),
                     json.dumps(brief.questions), json.dumps(brief.angles),
                     json.dumps(brief.outline), brief.created_at),
                )
                self._db.commit()
            return brief
        except Exception:  # noqa: BLE001
            return None

    def get_brief(self, brief_id: str) -> ContentBrief | None:
        try:
            if self._db is None or not brief_id:
                return None
            import json

            row = self._db.execute(
                "SELECT * FROM briefs WHERE brief_id = ?", (brief_id,)).fetchone()
            if not row:
                return None
            return ContentBrief(
                brief_id=row["brief_id"], topic=row["topic"],
                keywords=json.loads(row["keywords"] or "[]"),
                questions=json.loads(row["questions"] or "[]"),
                angles=json.loads(row["angles"] or "[]"),
                outline=json.loads(row["outline"] or "[]"),
                created_at=row["created_at"] or 0.0,
            )
        except Exception:  # noqa: BLE001
            return None

    def latest_brief(self, topic: str = "") -> ContentBrief | None:
        try:
            if self._db is None:
                return None
            import json

            if topic:
                row = self._db.execute(
                    "SELECT * FROM briefs WHERE topic LIKE ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (f"%{topic}%",)).fetchone()
            else:
                row = self._db.execute(
                    "SELECT * FROM briefs ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
            if not row:
                return None
            return ContentBrief(
                brief_id=row["brief_id"], topic=row["topic"],
                keywords=json.loads(row["keywords"] or "[]"),
                questions=json.loads(row["questions"] or "[]"),
                angles=json.loads(row["angles"] or "[]"),
                outline=json.loads(row["outline"] or "[]"),
                created_at=row["created_at"] or 0.0,
            )
        except Exception:  # noqa: BLE001
            return None

    # ── 2. scoring ──

    def score_draft(self, draft: str, brief: ContentBrief,
                    platform: str = "x") -> ContentScore:
        """Term coverage + readability + brand-voice (from #44). Never raises."""
        try:
            text = (draft or "").strip()
            if not text:
                return ContentScore(
                    suggestions=["empty draft — nothing to score"])
            keywords = [k.lower() for k in (brief.keywords if brief else [])]
            low = text.lower()

            # Term coverage: Surfer-style keyword presence.
            hits = [k for k in keywords if k and k in low]
            coverage = len(hits) / max(1, len(keywords))
            missing = [k for k in (brief.keywords if brief else [])
                       if k and k.lower() not in low][:12]

            # Readability: Flesch-ish, normalized to 0-1.
            words = _WORD_RE.findall(low)
            sentences = max(1, len(re.split(r"[.!?]+", text)))
            syll = sum(max(1, len(re.findall(r"[aeiouy]+", w)))
                       for w in words[:300])
            asl = len(words) / sentences
            asw = syll / max(1, len(words))
            flesch = 206.835 - 1.015 * asl - 84.6 * asw
            readability = max(0.0, min(1.0, flesch / 100.0))

            # Brand voice: #44's virality_score (heuristic signals).
            brand_voice = 50.0
            try:
                from ...social.voice import virality_score
                vs = virality_score(text, platform=platform)
                brand_voice = float(vs.score)
            except Exception:  # noqa: BLE001
                pass

            # Learned calibration from feedback (#40 resurfacing pattern).
            source = "heuristic"
            calib = self._calibration()
            raw = (0.45 * coverage * 100
                   + 0.25 * readability * 100
                   + 0.30 * brand_voice)
            if calib is not None:
                raw = raw * calib
                source = "learned"

            suggestions: list[str] = []
            if coverage < 0.5:
                suggestions.append(
                    f"add the missing terms ({len(missing)}): "
                    + ", ".join(missing[:4]))
            if readability < 0.4:
                suggestions.append("shorten sentences — readability is low")
            if brand_voice < 50:
                suggestions.append("strengthen the hook and add a call to action")

            return ContentScore(
                score=round(max(0.0, min(100.0, raw)), 1),
                term_coverage=round(coverage, 2),
                readability=round(readability, 2),
                brand_voice=round(brand_voice, 1),
                missing_terms=missing,
                suggestions=suggestions,
                source=source,
            )
        except Exception:  # noqa: BLE001
            return ContentScore(suggestions=["scoring unavailable"])

    def _calibration(self) -> float | None:
        """Feedback-loop multiplier: predicted vs actual outcomes."""
        try:
            if self._db is None:
                return None
            rows = self._db.execute(
                "SELECT score, outcome FROM score_feedback").fetchall()
            if len(rows) < 5:
                return None
            # ratio of actual engagement to score-implied engagement
            ratios = [r["outcome"] / max(1.0, r["score"]) for r in rows
                      if r["score"] and r["score"] > 0]
            if not ratios:
                return None
            return max(0.5, min(1.5, sum(ratios) / len(ratios)))
        except Exception:  # noqa: BLE001
            return None

    # ── 3. drafting (brief-gated) ──

    def draft_from_brief(self, brief: ContentBrief,
                         llm_fn=None,
                         platform: str = "x") -> BriefDraft | None:
        """Draft MUST have a brief. Never raises."""
        try:
            if brief is None or not brief.brief_id:
                return None  # no brief, no draft
            outline_txt = "\n".join(f"- {h}" for h in brief.outline)
            if llm_fn is not None:
                try:
                    content = llm_fn(
                        f"Write a short post about '{brief.topic}' "
                        f"following this outline:\n{outline_txt}\n"
                        f"Must include these terms: "
                        f"{', '.join(brief.keywords[:6])}.")
                except Exception:  # noqa: BLE001
                    content = ""
            else:
                content = ""
            if not content:
                # Honest template draft — clearly from the brief, not fake.
                content = (
                    f"{brief.topic}: here's the honest take.\n"
                    + "\n".join(f"• {q}" for q in brief.questions[:2])
                    + f"\nKey terms: {', '.join(brief.keywords[:5])}"
                )
            sc = self.score_draft(content, brief, platform=platform)
            pred = self.predict_performance(content, brief, sc)
            draft = BriefDraft(
                draft_id="bd_" + uuid.uuid4().hex[:8],
                topic=brief.topic, content=content, brief_id=brief.brief_id,
                platform=platform, score=sc.score,
                predicted_engagement=pred,
                created_at=time.time(),
            )
            self._save_draft(draft)
            return draft
        except Exception:  # noqa: BLE001
            return None

    def _save_draft(self, draft: BriefDraft) -> None:
        try:
            if self._db is None:
                return
            import json

            self._db.execute(
                """INSERT OR REPLACE INTO drafts VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (draft.draft_id, draft.topic, draft.content, draft.brief_id,
                 draft.platform, draft.score, draft.predicted_engagement,
                 draft.scheduled_for, draft.published_at,
                 json.dumps(draft.outcome_metrics or {}), draft.created_at),
            )
            self._db.commit()
        except Exception:  # noqa: BLE001
            pass

    def get_draft(self, draft_id: str) -> BriefDraft | None:
        try:
            if self._db is None or not draft_id:
                return None
            import json

            row = self._db.execute(
                "SELECT * FROM drafts WHERE draft_id = ?", (draft_id,)).fetchone()
            if not row:
                return None
            return BriefDraft(
                draft_id=row["draft_id"], topic=row["topic"] or "",
                content=row["content"] or "", brief_id=row["brief_id"] or "",
                platform=row["platform"] or "x", score=row["score"] or 0.0,
                predicted_engagement=row["predicted_engagement"] or 0.0,
                scheduled_for=row["scheduled_for"] or 0.0,
                published_at=row["published_at"] or 0.0,
                outcome_metrics=json.loads(row["outcome_metrics"] or "{}"),
                created_at=row["created_at"] or 0.0,
            )
        except Exception:  # noqa: BLE001
            return None

    # ── 4. pre-publish prediction (Anyword pattern) ──

    def predict_performance(self, draft: str, brief: ContentBrief,
                            scored: ContentScore | None = None) -> float:
        """Predicted engagement 0-100 before scheduling. Never raises."""
        try:
            sc = scored or self.score_draft(draft, brief)
            # Anyword-style: score + brief-quality + hook signals → band.
            hook_bonus = 0.0
            text = (draft or "").strip()
            if re.match(r"^(here's|stop|the truth|nobody|why)\b", text,
                        re.IGNORECASE):
                hook_bonus = 6.0
            if "?" in text[:140]:
                hook_bonus += 3.0
            brief_bonus = min(6.0, len(brief.keywords if brief else []) / 2.0)
            pred = 0.7 * sc.score + 0.2 * 50.0 + hook_bonus + brief_bonus
            return round(max(0.0, min(100.0, pred)), 1)
        except Exception:  # noqa: BLE001
            return 0.0

    # ── 5. scheduling (enforced: no brief → refuse) ──

    def schedule_draft(self, draft_id: str, when_ts: float,
                       scheduler=None, queue=None) -> dict:
        """Schedule a draft. REFUSES when no brief is attached. Never raises."""
        try:
            draft = self.get_draft(draft_id)
            if draft is None:
                return {"ok": False, "reason": "draft not found"}
            if not draft.brief_id:
                return {
                    "ok": False,
                    "reason": "refused: no brief attached — "
                              "brief → draft → score → schedule. "
                              "Build a brief first (/brief <topic>).",
                }
            brief = self.get_brief(draft.brief_id)
            if brief is None:
                return {
                    "ok": False,
                    "reason": "refused: brief record missing — "
                              "brief → draft → score → schedule.",
                }
            # Prediction must exist before scheduling.
            if not draft.predicted_engagement:
                draft.predicted_engagement = self.predict_performance(
                    draft.content, brief)
            # Wire into the existing #45/#43 queue when available.
            if queue is not None:
                try:
                    qd = queue.create_draft(
                        draft.content, platforms=[draft.platform],
                        metadata={"brief_id": draft.brief_id,
                                  "score": draft.score,
                                  "predicted_engagement":
                                      draft.predicted_engagement,
                                  "pipeline": "brief_first"})
                    queue.propose(qd.id)  # → pending_review: asks, never posts
                except Exception:  # noqa: BLE001
                    pass
            if scheduler is not None:
                try:
                    from ...social.content_pipeline import schedule_post
                    schedule_post(scheduler, queue, draft_id, when_ts,
                                  platforms=[draft.platform])
                except Exception:  # noqa: BLE001
                    pass
            draft.scheduled_for = when_ts
            self._save_draft(draft)
            return {"ok": True, "scheduled_for": when_ts,
                    "predicted_engagement": draft.predicted_engagement}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"scheduling failed: {exc}"}

    # ── 6. feedback loop (#40 resurfacing pattern) ──

    def record_outcome(self, draft_id: str,
                       metrics: dict | None) -> bool:
        """Published performance feeds back into the scorer. Never raises."""
        try:
            draft = self.get_draft(draft_id)
            if draft is None:
                return False
            metrics = dict(metrics or {})
            draft.outcome_metrics = metrics
            draft.published_at = time.time()
            self._save_draft(draft)
            if self._db is not None:
                outcome = float(metrics.get("engagement_score",
                                            metrics.get("score", 0)) or 0)
                self._db.execute(
                    "INSERT INTO score_feedback VALUES (?,?,?,?)",
                    (draft_id, draft.score, outcome, time.time()))
                self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ── full pipeline entry ──

    def run(self, topic: str, *, search_fn=None, llm_fn=None,
            platform: str = "x", schedule_at: float = 0.0,
            scheduler=None, queue=None) -> dict:
        """brief → draft → score → (schedule). Never raises."""
        try:
            brief = self.build_brief(topic, search_fn=search_fn)
            if brief is None:
                return {"ok": False, "reason": "could not build brief"}
            draft = self.draft_from_brief(brief, llm_fn=llm_fn,
                                          platform=platform)
            if draft is None:
                return {"ok": False, "reason": "could not draft from brief"}
            scored = self.score_draft(draft.content, brief, platform=platform)
            result: dict = {
                "ok": True, "brief": brief, "draft": draft,
                "score": scored,
                "predicted_engagement": draft.predicted_engagement,
            }
            if schedule_at:
                result["schedule"] = self.schedule_draft(
                    draft.draft_id, schedule_at,
                    scheduler=scheduler, queue=queue)
            return result
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"pipeline failed: {exc}"}


# ── chat ────────────────────────────────────────────────────────────────────

_store: BriefStore | None = None


def _get_store() -> BriefStore:
    global _store
    if _store is None:
        _store = BriefStore()
    return _store


def _usage() -> str:
    return (
        "📋 brief-first content pipeline\n"
        "/brief <topic> — build a SERP-derived brief\n"
        "/score <draft text> — score a draft against the latest brief\n"
        "/content run <topic> — full pipeline: brief → draft → score\n"
        "/content schedule <draft_id> <when> — schedule (refuses without a brief)\n"
        "no brief, no draft, no schedule."
    )


def control_brief(tail: str, context=None, chat=None, **kwargs) -> str:
    """/brief — brief-first content pipeline. Owner-only; never raises."""
    store = kwargs.get("store") or _get_store()
    try:
        rest = (tail or "").strip()
        if not rest or rest.lower() in ("help", "?"):
            return _usage()
        low = rest.lower()

        if low.startswith("run "):
            topic = rest[4:].strip()
            if not topic:
                return _usage()
            res = store.run(topic)
            if not res.get("ok"):
                return f"pipeline failed: {res.get('reason', '?')}"
            brief, draft, scored = res["brief"], res["draft"], res["score"]
            return (
                brief.summary() + "\n\n"
                f"✍️ draft {draft.draft_id} (score {scored.score:.0f}/100, "
                f"predicted engagement {draft.predicted_engagement:.0f}/100)\n"
                f"{draft.content[:280]}\n\n"
                f"schedule with: /content schedule {draft.draft_id} <when>"
            )

        # Default: build a brief.
        brief = store.build_brief(rest)
        if brief is None:
            return "couldn't build a brief from that — try a clearer topic."
        return brief.summary() + "\n\nno brief, no draft, no schedule."

    except Exception:  # noqa: BLE001
        return "brief pipeline hiccup — try again."


def control_content(tail: str, context=None, chat=None, **kwargs) -> str:
    """/content — score + schedule. Owner-only; never raises."""
    store = kwargs.get("store") or _get_store()
    try:
        rest = (tail or "").strip()
        if not rest or rest.lower() in ("help", "?"):
            return _usage()
        low = rest.lower()

        if low.startswith("score "):
            draft_text = rest[6:].strip()
            brief = store.latest_brief()
            if brief is None:
                return ("no brief on record — score against what? "
                        "build one first: /brief <topic>")
            scored = store.score_draft(draft_text, brief)
            return scored.format()

        if low.startswith("schedule "):
            parts = rest[9:].strip().split()
            if len(parts) < 2:
                return "usage: /content schedule <draft_id> <when>"
            draft_id, when = parts[0], " ".join(parts[1:])
            try:
                when_ts = float(when)
            except ValueError:
                # "in 2h" / "tomorrow 9am" style — best effort.
                when_ts = _parse_when(when) or (time.time() + 3600)
            res = store.schedule_draft(draft_id, when_ts)
            if not res.get("ok"):
                return f"❌ {res.get('reason')}"
            return (f"✅ scheduled for {time.ctime(res['scheduled_for'])} "
                    f"(predicted engagement {res.get('predicted_engagement', 0):.0f}/100)")

        if low.startswith("run "):
            return control_brief("run " + rest[4:], context=context, chat=chat,
                                 **kwargs)

        return _usage()
    except Exception:  # noqa: BLE001
        return "content pipeline hiccup — try again."


def _parse_when(text: str) -> float | None:
    """Best-effort natural time parse. Never raises."""
    try:
        import datetime

        t = (text or "").strip().lower()
        now = time.time()
        m = re.match(r"in (\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)",
                     t)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            mult = 60 if unit.startswith("m") else 3600 if unit.startswith("h") else 86400
            return now + n * mult
        if t.startswith("tomorrow"):
            return now + 86400
        return None
    except Exception:  # noqa: BLE001
        return None
