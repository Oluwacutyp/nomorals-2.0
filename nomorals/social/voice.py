"""Voice-trained brand voice + virality scoring + evergreen recycling + auto-plug.

The creator moat (Lately / Opus Clip / Hypefury / SocialBee patterns), built
on Devon's own memory at zero cost:

- **Voice training**: :class:`VoiceProfile` learns the owner's writing style
  from past posts (pure heuristics — no LLM needed). :func:`apply_voice`
  rewrites a draft toward the profile (real style-transfer with ``llm_fn``,
  honest rule-based shaping without). :meth:`VoiceProfile.feedback` folds
  corrections like "too formal, make it punchier" back into the profile.
- **Virality scoring**: :func:`virality_score` grades a draft 0-100 before
  publishing (hook strength, CTA presence, length vs platform optimum,
  emoji/hashtag balance, novelty vs the owner's past winners).
  :func:`check_virality` warns below threshold — it never blocks.
- **Evergreen recycling**: :class:`EvergreenQueue` holds category queues of
  proven winners; :func:`install_evergreen_job` wires a cron job that reposts
  whatever is past cooldown.
- **Auto-plug**: :class:`AutoPlug` watches engagement and auto-replies with
  the owner's CTA once a post passes threshold. The CTA is owner-configured —
  it is never invented.

All persistence lives under ``~/.nomorals/social/``. Everything degrades
honestly without an LLM or a configured publisher.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core.logging_setup import get_logger

__all__ = [
    "VoiceProfile",
    "apply_voice",
    "ViralityScore",
    "virality_score",
    "check_virality",
    "VIRALITY_WARN_THRESHOLD",
    "EvergreenQueue",
    "install_evergreen_job",
    "AutoPlug",
    "voice_profile_path",
]

_log = get_logger(__name__)

#: Below this virality score, :func:`check_virality` warns (never blocks).
VIRALITY_WARN_THRESHOLD = 40


def voice_profile_path() -> Path:
    p = Path.home() / ".nomorals" / "social" / "voice_profile.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _home_db(name: str) -> Path:
    p = Path.home() / ".nomorals" / "social" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── text analysis helpers ────────────────────────────────────────────────────

_WORD = re.compile(r"[A-Za-z0-9']+")
_SENT = re.compile(r"[^.!?…]+[.!?…]+|[^.!?…]+$")
_HASHTAG = re.compile(r"#\w+")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]"
)
_URL = re.compile(r"https?://\S+")

_STOPWORDS = frozenset(
    "a an the and or but if then else when at by for with about into of to in "
    "on is are was were be been being i you he she it we they my your his her "
    "its our their this that these those as so do does did not no yes just "
    "very really quite more most some any all can will would should could "
    "have has had me him us them what which who whom how why where".split()
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text or "") if s.strip()]


def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


# ── 1. voice training ────────────────────────────────────────────────────────


@dataclass
class VoiceProfile:
    """The owner's learned writing style. Pure heuristics, zero LLM cost."""

    n_posts: int = 0
    avg_sentence_len: float = 15.0      # words per sentence
    emoji_per_100: float = 0.0         # emojis per 100 chars
    hashtags_per_post: float = 0.0
    question_rate: float = 0.0         # share of posts ending in "?"
    caps_rate: float = 0.0             # share of ALL-CAPS words
    exclaim_rate: float = 0.0          # share of sentences ending in "!"
    opener_patterns: list[str] = field(default_factory=list)   # top openers
    closer_patterns: list[str] = field(default_factory=list)   # top closers
    vocab: list[str] = field(default_factory=list)              # fingerprint
    corrections: list[str] = field(default_factory=list)       # feedback notes

    # -- training -----------------------------------------------------------

    def train(self, posts: Sequence[str]) -> "VoiceProfile":
        """Learn style signals from past posts. Returns self (chainable)."""
        posts = [p for p in (posts or []) if (p or "").strip()]
        if not posts:
            return self
        sents, words_total = [], 0
        emoji_total = hashtag_total = caps_total = words_all = 0
        exclaim_total = q_total = 0
        openers: Counter[str] = Counter()
        closers: Counter[str] = Counter()
        vocab: Counter[str] = Counter()
        chars = 0
        for post in posts:
            post = post.strip()
            chars += len(post)
            ss = _sentences(post)
            sents.extend(ss)
            ws = _words(post)
            words_total += len(ws)
            words_all += len(ws)
            emoji_total += len(_EMOJI.findall(post))
            hashtag_total += len(_HASHTAG.findall(post))
            caps_total += sum(1 for w in _WORD.findall(post)
                              if len(w) > 1 and w.isupper())
            exclaim_total += sum(1 for s in ss if s.endswith("!"))
            if post.rstrip().endswith("?"):
                q_total += 1
            wseq = ws
            if len(wseq) >= 3:
                openers[" ".join(wseq[:3])] += 1
                closers[" ".join(wseq[-3:])] += 1
            for w in ws:
                if w not in _STOPWORDS and len(w) > 3:
                    vocab[w] += 1
        n = len(posts)
        # Blend with prior (running average) so retraining refines, not resets.
        prior_n = self.n_posts
        total = prior_n + n
        def blend(old: float, new: float) -> float:
            return (old * prior_n + new * n) / total if total else new
        new_avg_sent = (sum(len(_words(s)) for s in sents) / len(sents)
                        if sents else self.avg_sentence_len)
        self.avg_sentence_len = blend(self.avg_sentence_len, new_avg_sent)
        self.emoji_per_100 = blend(self.emoji_per_100,
                                   emoji_total / max(1, chars) * 100)
        self.hashtags_per_post = blend(self.hashtags_per_post,
                                       hashtag_total / n)
        self.question_rate = blend(self.question_rate, q_total / n)
        self.caps_rate = blend(self.caps_rate, caps_total / max(1, words_all))
        self.exclaim_rate = blend(self.exclaim_rate,
                                  exclaim_total / max(1, len(sents)))
        self.opener_patterns = [o for o, _ in openers.most_common(5)]
        self.closer_patterns = [c for c, _ in closers.most_common(5)]
        self.vocab = [w for w, _ in vocab.most_common(40)]
        self.n_posts = total
        return self

    # -- description / persistence ------------------------------------------

    def describe(self) -> str:
        """Human/LLM-readable style brief for style-transfer prompts."""
        bits = [
            f"~{self.avg_sentence_len:.0f} words per sentence",
            "asks questions often" if self.question_rate > 0.25
            else "rarely asks questions",
            f"uses emojis {'often' if self.emoji_per_100 > 1.5 else 'sparingly' if self.emoji_per_100 > 0.2 else 'almost never'}",
            f"~{self.hashtags_per_post:.1f} hashtags per post",
            "uses ALL-CAPS emphasis" if self.caps_rate > 0.02
            else "avoids ALL-CAPS",
        ]
        if self.opener_patterns:
            bits.append(f"often opens with: '{self.opener_patterns[0]}'")
        if self.vocab:
            bits.append("signature words: " + ", ".join(self.vocab[:8]))
        for c in self.corrections[-3:]:
            bits.append(f"correction: {c}")
        return "; ".join(bits)

    def save(self, path: Path | str | None = None) -> Path:
        p = Path(path) if path else voice_profile_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
        return p

    @classmethod
    def load(cls, path: Path | str | None = None) -> "VoiceProfile":
        p = Path(path) if path else voice_profile_path()
        if not p.is_file():
            return cls()
        try:
            data = json.loads(p.read_text())
            known = {f for f in cls.__dataclass_fields__}
            return cls(**{k: v for k, v in data.items() if k in known})
        except Exception as exc:  # noqa: BLE001 - corrupt profile → fresh
            _log.warning("voice profile load failed (%s), starting fresh", exc)
            return cls()

    # -- feedback ------------------------------------------------------------

    def feedback(self, note: str, *, facts: Any = None) -> "VoiceProfile":
        """Fold a correction ("too formal, make it punchier") into the profile.

        Nudges the numeric signals toward the requested direction and records
        the note. When a FactStore is passed, the correction is also stored as
        a ``preference`` fact for the memory layer.
        """
        note = (note or "").strip()
        if not note:
            return self
        low = note.lower()
        # Directional nudges — small, honest, documented.
        if any(w in low for w in ("punchier", "shorter", "concise", "tighter")):
            self.avg_sentence_len = max(6.0, self.avg_sentence_len * 0.85)
        if any(w in low for w in ("longer", "elaborate", "detail")):
            self.avg_sentence_len = min(40.0, self.avg_sentence_len * 1.15)
        if any(w in low for w in ("more emojis", "more emoji", "emoji")):
            self.emoji_per_100 = min(8.0, self.emoji_per_100 + 1.0)
        if any(w in low for w in ("fewer emojis", "less emoji", "no emojis")):
            self.emoji_per_100 = max(0.0, self.emoji_per_100 - 1.0)
        if "hashtag" in low and any(w in low for w in ("more", "add")):
            self.hashtags_per_post = min(10.0, self.hashtags_per_post + 1.0)
        if "hashtag" in low and any(w in low
                                    for w in ("fewer", "less", "no", "drop")):
            self.hashtags_per_post = max(0.0, self.hashtags_per_post - 1.0)
        if any(w in low for w in ("more questions", "ask more")):
            self.question_rate = min(1.0, self.question_rate + 0.15)
        self.corrections.append(note)
        if facts is not None:
            try:
                facts.add_fact(f"my writing style preference: {note}",
                               confidence=0.9)
            except Exception:  # noqa: BLE001 - memory is best-effort here
                _log.debug("storing style preference failed", exc_info=True)
        return self


def apply_voice(
    text: str,
    profile: VoiceProfile,
    *,
    llm_fn: Callable[[str], str] | None = None,
) -> str:
    """Rewrite ``text`` toward ``profile``.

    With ``llm_fn``: real style transfer via the caller's model (facts
    preserved, style shifted). Without: conservative rule-based shaping
    toward the profile's hashtag/emoji/caps habits — documented as the
    fallback, never disguised as a rewrite.
    """
    text = (text or "").strip()
    if not text or profile.n_posts == 0:
        return text
    if llm_fn is not None:
        prompt = (
            "Rewrite the post below in the author's voice. "
            f"Style brief: {profile.describe()}. "
            "Keep every fact identical — change style only, never add claims.\n\n"
            f"{text}"
        )
        try:
            rewritten = (llm_fn(prompt) or "").strip()
        except Exception as exc:  # noqa: BLE001 - LLM failure → rules
            _log.warning("voice llm failed, using rules: %s", exc)
            rewritten = ""
        if rewritten:
            return rewritten
        # fall through to rules on empty/exception
    return _rule_voice_shape(text, profile)


def _rule_voice_shape(text: str, profile: VoiceProfile) -> str:
    """Conservative mechanical shaping toward the profile. Honest, small."""
    # Hashtag count toward the profile mean.
    tags = _HASHTAG.findall(text)
    target_tags = round(profile.hashtags_per_post)
    if len(tags) > target_tags:
        keep = set(tags[:target_tags])
        kept, seen = [], set()
        for t in tags:
            if t in keep and t not in seen:
                seen.add(t)
                kept.append(t)
        text = _HASHTAG.sub("", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        if kept:
            text = f"{text}\n\n{' '.join(kept)}".strip()
    # Emoji density cap toward the profile (only trims excess).
    target_emoji = profile.emoji_per_100 / 100 * max(1, len(text))
    emojis = _EMOJI.findall(text)
    if len(emojis) > target_emoji + 2 and target_emoji < 3:
        # Strip trailing emoji runs first — least destructive.
        text = re.sub(r"(\s*[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]){2,}\s*$", "", text).strip()
    # ALL-CAPS normalization when the profile avoids it.
    if profile.caps_rate < 0.01:
        text = re.sub(r"\b([A-Z]{2,})\b",
                      lambda m: m.group(1).capitalize(), text)
    return text


# ── 2. pre-publish virality scoring ──────────────────────────────────────────

#: Words that open strong hooks (number, provocation, direct address).
_HOOK_OPENERS = re.compile(
    r"^(i |you |we |stop |never |always |here'?s|the |why |how |\d)",
    re.IGNORECASE,
)
#: Tired openers that underperform.
_CLICHE_OPENERS = (
    "excited to announce", "thrilled to share", "happy to announce",
    "proud to announce", "just dropped",
)
_CTA_WORDS = frozenset(
    "comment reply share retweet repost follow subscribe sign click link "
    "join try buy download check read watch listen dm message".split()
)


@dataclass
class ViralityScore:
    """0-100 grade for a draft, with the reasons that built it."""

    score: float
    reasons: list[str] = field(default_factory=list)  # "+12 strong hook"

    @property
    def grade(self) -> str:
        if self.score >= 75:
            return "strong"
        if self.score >= 55:
            return "decent"
        if self.score >= VIRALITY_WARN_THRESHOLD:
            return "weak"
        return "likely to flop"


def virality_score(
    draft: str,
    *,
    platform: str = "x",
    past_winners: Sequence[str] | None = None,
) -> ViralityScore:
    """Grade a draft 0-100 before publishing.

    Heuristic signals: hook strength (first 8 words), question/CTA
    presence, length vs platform optimum, emoji/hashtag balance, novelty
    vs the owner's past winners. Heuristics, not prophecy — documented.
    """
    from .tone import _profile  # local import: avoid cycle at module load

    text = (draft or "").strip()
    reasons: list[str] = []
    if not text:
        return ViralityScore(0.0, ["empty draft"])
    score = 50.0  # neutral start; signals move it

    def add(pts: float, why: str) -> None:
        nonlocal score
        score += pts
        reasons.append(f"{pts:+.0f} {why}")

    words = _words(text)
    first8 = " ".join(words[:8])
    low = text.lower()

    # Hook: the first 8 words decide the scroll.
    if _HOOK_OPENERS.match(text):
        add(12, "strong hook opening")
    if any(c in low[:60] for c in _CLICHE_OPENERS):
        add(-10, "cliché opener")
    if re.match(r"^\d+", text):
        add(6, "opens with a number/stat")
    if "?" in text[:120]:
        add(5, "early question pulls readers in")

    # Question / CTA: posts that invite response outperform.
    if text.rstrip().endswith("?"):
        add(6, "ends with a question")
    if _CTA_WORDS & set(words):
        add(6, "has a call to action")

    # Length vs platform optimum (brevity wins on short-form networks).
    profile = _profile(platform)
    ratio = len(text) / max(1, profile.max_chars)
    if profile.max_chars <= 500:
        if ratio < 0.5:
            add(8, "short and punchy for the platform")
        elif ratio > 0.9:
            add(-8, "near the character limit — will get trimmed")
    else:
        if 100 <= len(text) <= 1200:
            add(5, "good long-form length")

    # Emoji / hashtag balance.
    n_emoji = len(_EMOJI.findall(text))
    n_tags = len(_HASHTAG.findall(text))
    if 1 <= n_emoji <= 4:
        add(4, "tasteful emoji use")
    elif n_emoji > 8:
        add(-6, "emoji overload")
    if 1 <= n_tags <= 3:
        add(4, "good hashtag count")
    elif n_tags > 5:
        add(-6, "hashtag stuffing")

    # Novelty vs past winners: near-duplicates of old posts underperform.
    if past_winners:
        toks = set(words) - _STOPWORDS
        best = 0.0
        for w in past_winners:
            wtoks = set(_words(w)) - _STOPWORDS
            if toks and wtoks:
                best = max(best, len(toks & wtoks) / len(toks | wtoks))
        if best > 0.7:
            add(-12, "too similar to a past post")
        elif best < 0.2:
            add(5, "fresh angle vs past winners")

    # Readability basics.
    sents = _sentences(text)
    if sents:
        avg = sum(len(_words(s)) for s in sents) / len(sents)
        if avg > 35:
            add(-6, "sentences run long — hard to skim")
    if _URL.search(text):
        add(3, "link gives it somewhere to go")

    return ViralityScore(round(max(0.0, min(100.0, score)), 1),
                         reasons)


def check_virality(
    draft: str,
    *,
    platform: str = "x",
    threshold: float = VIRALITY_WARN_THRESHOLD,
    past_winners: Sequence[str] | None = None,
) -> str | None:
    """Warn (never block) when a draft scores below threshold.

    Returns the warning string, or None when the draft is fine.
    """
    vs = virality_score(draft, platform=platform, past_winners=past_winners)
    if vs.score < threshold:
        top = "; ".join(vs.reasons[:3])
        return (
            f"this might underperform (virality {vs.score:.0f}/100, "
            f"{vs.grade}) — want a punch-up? [{top}]"
        )
    return None


# ── 3. evergreen recycling ───────────────────────────────────────────────────


@dataclass
class EvergreenPost:
    id: str
    content: str
    category: str
    engagement: float
    last_posted_at: float | None = None
    cooldown_days: float = 30.0


class EvergreenQueue:
    """Category queues of proven winners, reposted past cooldown.

    SQLite-backed. ``add`` a winner with its measured engagement;
    ``next_due`` returns whatever is past cooldown and above its floor.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _home_db("evergreen.db")
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init(self) -> None:
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS evergreen (
                       id TEXT PRIMARY KEY,
                       content TEXT NOT NULL,
                       category TEXT NOT NULL,
                       engagement REAL NOT NULL DEFAULT 0,
                       min_engagement REAL NOT NULL DEFAULT 0,
                       last_posted_at REAL,
                       cooldown_days REAL NOT NULL DEFAULT 30,
                       created_at REAL NOT NULL
                   )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_evergreen_cat "
                      "ON evergreen(category)")

    def add(self, content: str, category: str, *,
            engagement: float = 0.0, min_engagement: float = 50.0,
            cooldown_days: float = 30.0) -> str:
        """Queue a winner. Returns the id. Never raises."""
        try:
            import uuid
            pid = uuid.uuid4().hex[:12]
            with self._conn() as c:
                c.execute(
                    "INSERT INTO evergreen (id, content, category, engagement,"
                    " min_engagement, last_posted_at, cooldown_days,"
                    " created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (pid, (content or "").strip(), (category or "general"),
                     float(engagement), float(min_engagement), None,
                     float(cooldown_days), time.time()))
            return pid
        except Exception:  # noqa: BLE001
            _log.debug("evergreen add failed", exc_info=True)
            return ""

    def next_due(self, category: str = "",
                 *, limit: int = 5) -> list[EvergreenPost]:
        """Winners past cooldown and above their engagement floor."""
        try:
            now = time.time()
            q = ("SELECT * FROM evergreen WHERE engagement >= min_engagement"
                 " AND (last_posted_at IS NULL OR last_posted_at +"
                 " cooldown_days * 86400 <= ?)")
            args: list[Any] = [now]
            if category:
                q += " AND category = ?"
                args.append(category)
            q += " ORDER BY last_posted_at NULLS FIRST LIMIT ?"
            args.append(limit)
            with self._conn() as c:
                rows = c.execute(q, args).fetchall()
            return [EvergreenPost(
                id=r["id"], content=r["content"], category=r["category"],
                engagement=r["engagement"],
                last_posted_at=r["last_posted_at"],
                cooldown_days=r["cooldown_days"]) for r in rows]
        except Exception:  # noqa: BLE001
            _log.debug("evergreen next_due failed", exc_info=True)
            return []

    def mark_posted(self, post_id: str) -> None:
        try:
            with self._conn() as c:
                c.execute("UPDATE evergreen SET last_posted_at = ?"
                          " WHERE id = ?",
                          (time.time(), post_id))
        except Exception:  # noqa: BLE001
            _log.debug("evergreen mark_posted failed", exc_info=True)


def install_evergreen_job(
    scheduler: Any,
    queue: EvergreenQueue,
    publish_fn: Callable[[str, str], Any],
    *,
    cron: str = "0 9 * * *",
    categories: Sequence[str] = ("tips", "wins", "questions"),
    task_id: str = "social.evergreen",
) -> Any:
    """Register the recycling cron: each run reposts due winners.

    ``publish_fn(content, category)`` does the actual posting (Postiz
    adapter, drafts.execute_post, …). Follows the ``social.post_draft``
    action pattern from drafts.py.
    """
    async def _fire(**params: Any) -> None:
        for cat in categories:
            for post in queue.next_due(cat, limit=1):
                try:
                    res = publish_fn(post.content, post.category)
                    if hasattr(res, "__await__"):
                        res = await res
                    queue.mark_posted(post.id)
                    _log.info("evergreen reposted %s (%s)", post.id, cat)
                except Exception as exc:  # noqa: BLE001 - one failure
                    _log.warning("evergreen repost failed: %s", exc)

    scheduler.register_action(task_id, _fire)
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        return loop.create_task(
            scheduler.schedule_cron(task_id, cron, task_id))
    # No running loop (tests / sync callers): register only.
    _log.info("evergreen job registered (cron %s); schedule when a loop runs",
              cron)
    return None


# ── 4. auto-plug ─────────────────────────────────────────────────────────────


class AutoPlug:
    """A post passes its engagement threshold → auto-reply with the CTA.

    The CTA is owner-configured via :meth:`configure_cta`. Without one,
    :meth:`check` never fires — the CTA is never invented.
    """

    def __init__(self, *,
                 cta: str = "",
                 thresholds: dict[str, float] | None = None,
                 default_threshold: float = 100.0) -> None:
        self.cta = (cta or "").strip()
        self.thresholds = dict(thresholds or {})
        self.default_threshold = default_threshold
        self._plugged: set[str] = set()  # post ids already plugged

    def configure_cta(self, cta: str) -> "AutoPlug":
        self.cta = (cta or "").strip()
        return self

    def threshold_for(self, platform: str) -> float:
        return self.thresholds.get((platform or "").lower(),
                                   self.default_threshold)

    def check(self, post_id: str, engagement: float, *,
              platform: str = "x",
              reply_fn: Callable[[str, str], Any] | None = None) -> bool:
        """Maybe auto-reply with the CTA. Returns True when it fired.

        Never raises. Never fires twice for the same post, never fires
        without a configured CTA, never fires below threshold.
        """
        try:
            if not post_id or post_id in self._plugged:
                return False
            if not self.cta:
                _log.debug("auto-plug: no CTA configured, skipping %s",
                           post_id)
                return False
            if float(engagement) < self.threshold_for(platform):
                return False
            if reply_fn is not None:
                res = reply_fn(post_id, self.cta)
                if hasattr(res, "__await__"):
                    # Sync callers can't await; run it to completion.
                    import asyncio
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        asyncio.run(res)
            self._plugged.add(post_id)
            _log.info("auto-plug fired on %s (engagement %s)", post_id,
                      engagement)
            return True
        except Exception:  # noqa: BLE001
            _log.debug("auto-plug check failed", exc_info=True)
            return False
