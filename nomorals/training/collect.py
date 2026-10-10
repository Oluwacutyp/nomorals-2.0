"""Harvest safe training examples from the durable runtime stores.

Collection is intentionally boring and local.  It reads the rows that already
form the system's audit trail; it does not call a network provider and it never
copies tool arguments or results (the tool log stores digests for this reason).
The result is suitable for :func:`nomorals.training.preprocess.prepare`.

The scrubbing pass is a defence in depth measure, not a promise that arbitrary
text is anonymous.  It covers the high-value accidental leaks we can recognise
without a third-party PII package: email addresses, phone numbers, common
secrets, bearer tokens, credit cards, SSNs, and IP addresses.  Unknown personal
information is still subject to the normal review/quality-filter step.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.logging_setup import get_logger
from .dataset import (
    DatasetRegistry,
    Example,
    Turn,
    write_format_bundles,
    write_jsonl,
)
from .preprocess import clean_text, hamming, simhash

__all__ = [
    "CollectionStats",
    "CollectionResult",
    "TrainingCollector",
    "ConversationMiner",
    "Collector",
    "LiveCollector",
    "collect",
    "collect_training_examples",
    "mine_conversations",
    "scrub_pii",
    "scrub_example",
    "scrub_report",
    "presidio_available",
    "trigram_overlap",
]


def presidio_available() -> bool:
    """Is Microsoft Presidio importable here (the NER-grade scrub path)?"""
    try:
        import presidio_analyzer  # noqa: F401
        import presidio_anonymizer  # noqa: F401
        return True
    except ImportError:
        return False

_log = get_logger(__name__)

# Order matters: a bearer/API token may contain punctuation that also resembles a
# phone number.  These expressions deliberately err on the side of redaction.
# Each entry is (entity_name, pattern, replacement).
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("bearer", re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1[REDACTED_TOKEN]"),
    ("api_key", re.compile(r"(?i)\b(?:sk|rk|pk|ghp|gho|github_pat|xox[baprs])[-_][A-Za-z0-9_-]{12,}\b"), "[REDACTED_TOKEN]"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    # A leading + means a phone number, not a card — the phone pattern
    # below owns those (without this guard "+2348012345678" scrubs as a
    # card and leaves the + dangling).
    ("card", re.compile(r"(?<!\+)\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_CARD]"),
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[REDACTED_EMAIL]"),
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
    # Nigerian national identifiers: NIN and BVN are both 11 digits.  The
    # keyword-anchored form is high-precision; the bare 11-digit form errs
    # toward redaction (it also catches un-prefixed 080 mobile numbers).
    ("national_id", re.compile(r"(?i)\b(?:nin|bvn|national[\s_]?id)[\s:=\-]*\d{11}\b"), "[REDACTED_NIN]"),
    ("national_id", re.compile(r"(?<!\d)\d{11}(?!\d)"), "[REDACTED_NIN]"),
    # Nigerian 10-digit bank account numbers.
    ("bank_account", re.compile(r"(?i)\b(?:account[\s_]?(?:no|number)?)[\s:=\-]*\d{10}\b"), "[REDACTED_ACCOUNT]"),
    # Require separators or a leading country code, but do not mistake an
    # ISO/calendar date for a telephone number.
    ("phone", re.compile(r"(?<!\w)(?!(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})(?!\w))\+?\d[\d(). -]{8,}\d(?!\w)"), "[REDACTED_PHONE]"),
)


def _hash_placeholder(entity: str, value: str) -> str:
    """Deterministic pseudonym: the same raw value always maps to the same
    placeholder, so rows stay joinable without ever storing the value."""
    digest = hashlib.sha256(f"{entity}:{value}".encode("utf-8")).hexdigest()[:10]
    return f"[HASH_{entity.upper()}_{digest}]"


def scrub_pii(
    text: str,
    *,
    allow: Sequence[str] = (),
    mode: str = "replace",
    engine: str = "auto",
) -> str:
    """Replace directly identifying values with stable placeholders.

    ``allow`` — literal tokens that survive redaction (e.g. the owner's
    public handle); ``mode="hash"`` emits deterministic pseudonyms instead
    of flat placeholders, so the same value stays joinable across rows;
    ``engine="presidio"`` uses Microsoft Presidio's analyzer+anonymizer
    when importable (NER catches PERSON names the regexes cannot) and
    falls back to the regexes otherwise.
    """
    value = str(text or "")
    if engine in {"presidio", "auto"}:
        presidio_hit = _presidio_scrub(value, allow=allow, mode=mode)
        if presidio_hit is not None:
            return presidio_hit
        if engine == "presidio":
            raise RuntimeError("presidio requested but presidio_analyzer is not installed")
    for entity, pattern, replacement in _PII_PATTERNS:
        if mode == "hash":
            value = pattern.sub(lambda m: _hash_placeholder(entity, m.group(0)), value)
        else:
            value = pattern.sub(replacement, value)
    if allow:
        # restore allow-listed literals (they were redacted above if they
        # happened to match a pattern — e.g. an email-shaped handle)
        for token in allow:
            token = str(token)
            if token:
                value = value.replace(token, token)
    return value


def _presidio_scrub(text: str, *, allow: Sequence[str], mode: str) -> str | None:
    """Presidio pass.  Returns None when Presidio is not importable so the
    caller falls back to the regexes — never a hard dependency."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        return None
    try:
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(
            text=text, language="en",
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                      "IBAN_CODE", "IP_ADDRESS", "US_SSN", "NRP"],
        )
        anonymizer = AnonymizerEngine()
        if allow:
            results = [r for r in results
                       if text[r.start:r.end] not in set(allow)]
        if mode == "hash":
            from presidio_anonymizer.entities import OperatorConfig
            operators = {r.entity_type: OperatorConfig("hash") for r in results}
            return anonymizer.anonymize(
                text=text, analyzer_results=results,
                operators=operators).text
        return anonymizer.anonymize(text=text, analyzer_results=results).text
    except Exception:  # noqa: BLE001 — a broken Presidio install is not a crash
        _log.debug("presidio scrub failed, falling back to regexes")
        return None


def scrub_report(text: str) -> dict[str, int]:
    """Per-entity redaction counts for a text — the audit trail.

    Reports *how many* of each entity were found, never the values (the
    raw value must never be logged or persisted, per the module docstring).
    """
    value = str(text or "")
    return {
        entity: len(pattern.findall(value))
        for entity, pattern, _ in _PII_PATTERNS
    }


def scrub_example(
    example: Example,
    *,
    allow: Sequence[str] = (),
    mode: str = "replace",
    engine: str = "auto",
) -> Example:
    """Return a scrubbed copy, preserving the example's source and weight."""
    return Example(
        turns=[Turn(t.role, scrub_pii(t.content, allow=allow, mode=mode, engine=engine))
               for t in example.turns],
        weight=example.weight,
        source=scrub_pii(example.source, allow=allow, mode=mode, engine=engine),
    )


@dataclass
class CollectionStats:
    """Counters explaining exactly what the collector considered and kept."""

    memories: int = 0
    messages: int = 0
    tool_calls: int = 0
    reflections: int = 0
    social_history: int = 0
    input_examples: int = 0
    pii_scrubbed: int = 0
    duplicates: int = 0
    kept: int = 0
    existing_datasets: int = 0

    def __getitem__(self, key: str) -> int:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memories": self.memories,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "reflections": self.reflections,
            "social_history": self.social_history,
            "input_examples": self.input_examples,
            "pii_scrubbed": self.pii_scrubbed,
            "duplicates": self.duplicates,
            "kept": self.kept,
            "existing_datasets": self.existing_datasets,
        }


@dataclass
class CollectionResult:
    """Collected examples and optional durable raw-dataset registration."""

    examples: list[Example] = field(default_factory=list)
    stats: CollectionStats = field(default_factory=CollectionStats)
    dataset_id: str = ""
    path: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def count(self) -> int:
        return len(self.examples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "stats": self.stats.to_dict(),
            "dataset_id": self.dataset_id,
            "path": self.path,
            "created_at": self.created_at,
        }


class TrainingCollector:
    """Read live-use rows and turn them into scrubbed, simhash-deduplicated examples.

    ``db`` is deliberately typed as ``Any`` so a small fake database can be used
    by offline tests.  A real :class:`~nomorals.storage.db.Database` is all that
    is required at runtime.
    """

    def __init__(self, db: Any, *, simhash_threshold: int = 3) -> None:
        self.db = db
        self.simhash_threshold = max(0, int(simhash_threshold))

    def collect(
        self,
        *,
        since: float = 0.0,
        limit: int = 0,
        output_dir: str | Path | None = None,
        name: str = "",
        register: bool = False,
        include_messages: bool = True,
        include_social_history: bool = True,
        social_platforms: Sequence[str] = ("telegram",),
        social_min_score: float = 0.4,
    ) -> CollectionResult:
        """Harvest rows, scrub them, remove duplicates, and optionally persist them.

        ``since`` is an ingestion watermark, not a privacy boundary.  Existing
        datasets are always inspected for simhashes, so a repeated collection can
        never append the same lesson just because the source row is old.

        ``include_social_history`` also mines the companion's own social chat
        history (``telegram:`` and other platform-prefixed conversations) into
        quality-scored instruction pairs — the "personal model" data source for
        continuous fine-tuning.  It is PII-scrubbed like every other source.
        """
        stats = CollectionStats()
        raw: list[Example] = []
        raw.extend(self._memory_examples(stats, since=since, limit=limit))
        social_prefixes = tuple(
            str(p).strip().lower() + ":"
            for p in (social_platforms if include_social_history else ())
            if str(p).strip())
        if include_messages:
            raw.extend(self._message_examples(
                stats, since=since, limit=limit,
                exclude_prefixes=social_prefixes))
        raw.extend(self._tool_examples(stats, since=since, limit=limit))
        raw.extend(self._reflection_examples(stats, since=since, limit=limit))
        if include_social_history and social_platforms:
            raw.extend(self._social_history_examples(
                stats, since=since, limit=limit,
                platforms=tuple(social_platforms), min_score=social_min_score,
            ))
        stats.input_examples = len(raw)

        scrubbed: list[Example] = []
        for example in raw:
            clean = scrub_example(example)
            # Count examples changed rather than exposing which PII expression
            # matched.  The raw value must never be logged or persisted.
            if clean.to_dict() != example.to_dict():
                stats.pii_scrubbed += 1
            for turn in clean.turns:
                turn.content = clean_text(turn.content)
            clean.source = clean_text(clean.source)
            if any(turn.content for turn in clean.turns):
                scrubbed.append(clean)

        existing = self._existing_fingerprints(stats)
        kept: list[Example] = []
        seen = list(existing)
        for example in scrubbed:
            fingerprint = _example_simhash(example)
            if any(hamming(fingerprint, prior) <= self.simhash_threshold for prior in seen):
                stats.duplicates += 1
                continue
            seen.append(fingerprint)
            kept.append(example)
        stats.kept = len(kept)

        result = CollectionResult(examples=kept, stats=stats)
        if output_dir is not None and kept:
            target_name = name or f"collected-{int(result.created_at)}"
            metadata = {
                "collected": True,
                "source": "live-use",
                "provider": "mock",
                "collected_at": result.created_at,
                "collector_stats": stats.to_dict(),
            }
            if register:
                dataset = DatasetRegistry(self.db).register_examples(
                    target_name, kept, output_dir, kind="chat", metadata=metadata
                )
                result.dataset_id = dataset.id
                result.path = dataset.path
            else:
                target = Path(output_dir).expanduser() / f"{target_name}.jsonl"
                result.path = str(target)
                write_jsonl(target, (example.to_dict() for example in kept))
        _log.info("collected %d/%d training examples", result.count, stats.input_examples)
        return result

    def harvest(self, **kwargs: Any) -> CollectionResult:
        """Alias for integrations that call live-use collection a harvest."""
        return self.collect(**kwargs)

    # ── source readers -----------------------------------------------------
    def _memory_examples(self, stats: CollectionStats, *, since: float, limit: int) -> list[Example]:
        rows = self._rows(
            "SELECT * FROM memories WHERE created_at >= ? ORDER BY created_at, rowid",
            since,
            limit=limit,
        )
        out: list[Example] = []
        for row in rows:
            content = str(row.get("content") or "")
            if not content.strip():
                continue
            stats.memories += 1
            metadata = _json(row.get("metadata"), {})
            role = str(metadata.get("role") or "assistant")
            if role not in {"system", "user", "assistant", "tool"}:
                role = "assistant"
            out.append(
                Example(
                    turns=[
                        Turn("user", f"Recall the {row.get('kind') or 'memory'} from live use."),
                        Turn(role, content),
                    ],
                    source=f"memory:{row.get('id') or 'unknown'}",
                )
            )
        return out

    def _message_examples(self, stats: CollectionStats, *, since: float, limit: int,
                          exclude_prefixes: Sequence[str] = ()) -> list[Example]:
        # Messages are the richer conversation representation when available.
        # The memory rows remain the required fallback/source of truth.
        # ``exclude_prefixes`` (e.g. ("telegram:",)) hands social-platform
        # conversations to the quality-pair miner instead of the whole-convo
        # reader, so the same chat is not trained on twice.
        rows = self._rows(
            "SELECT * FROM messages WHERE created_at >= ? ORDER BY conversation_id, created_at, rowid",
            since,
            limit=limit,
        )
        grouped: dict[str, list[Turn]] = {}
        ids: dict[str, str] = {}
        for row in rows:
            content = str(row.get("content") or "")
            if not content.strip():
                continue
            role = str(row.get("role") or "user")
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            conversation = str(row.get("conversation_id") or row.get("id") or "unknown")
            if any(conversation.startswith(p) for p in exclude_prefixes):
                continue
            grouped.setdefault(conversation, []).append(Turn(role, content))
            ids[conversation] = str(row.get("id") or conversation)
        out: list[Example] = []
        for conversation, turns in grouped.items():
            if len(turns) < 2:
                # A single message is already represented by memories in normal
                # operation; do not create a low-context duplicate here.
                continue
            stats.messages += len(turns)
            out.append(Example(turns=turns, source=f"conversation:{ids[conversation]}"))
        return out

    def _tool_examples(self, stats: CollectionStats, *, since: float, limit: int) -> list[Example]:
        rows = self._rows(
            "SELECT * FROM tool_calls WHERE created_at >= ? ORDER BY created_at, rowid",
            since,
            limit=limit,
        )
        out: list[Example] = []
        for row in rows:
            tool = scrub_pii(str(row.get("tool") or "tool"))
            status = scrub_pii(str(row.get("status") or "pending"))
            decision = scrub_pii(str(row.get("decision") or ""))
            error = scrub_pii(str(row.get("error") or ""))
            if not tool:
                continue
            stats.tool_calls += 1
            outcome = f"Tool {tool} finished with status {status}."
            if decision:
                outcome += f" Policy decision: {decision}."
            if error:
                outcome += f" Error class: {error[:240]}"
            out.append(
                Example(
                    turns=[
                        Turn("user", f"Use the {tool} tool and report its outcome."),
                        Turn("assistant", outcome),
                    ],
                    source=f"tool:{row.get('id') or 'unknown'}",
                )
            )
        return out

    def _reflection_examples(self, stats: CollectionStats, *, since: float, limit: int) -> list[Example]:
        rows = self._rows(
            "SELECT * FROM reflections WHERE created_at >= ? ORDER BY created_at, rowid",
            since,
            limit=limit,
        )
        out: list[Example] = []
        for row in rows:
            summary = scrub_pii(str(row.get("summary") or "")).strip()
            lessons = _json(row.get("lessons"), [])
            if not isinstance(lessons, list):
                lessons = [lessons]
            lesson_text = "; ".join(scrub_pii(str(item)) for item in lessons if str(item).strip())
            if not summary and not lesson_text:
                continue
            stats.reflections += 1
            try:
                score = f"{float(row.get('score') or 0.0):.3f}"
            except (TypeError, ValueError):
                score = "0.000"
            response = f"Mission outcome score: {score}."
            if summary:
                response += f" {summary}"
            if lesson_text:
                response += f" Lessons: {lesson_text}"
            out.append(
                Example(
                    turns=[
                        Turn("user", "Review the mission outcome and extract a reusable lesson."),
                        Turn("assistant", response),
                    ],
                    source=f"reflection:{row.get('id') or 'unknown'}",
                    weight=max(0.1, min(2.0, float(row.get("score") or 0.5) + 0.5)),
                )
            )
        return out

    def _social_history_examples(
        self,
        stats: CollectionStats,
        *,
        since: float,
        limit: int,
        platforms: Sequence[str] = ("telegram",),
        min_score: float = 0.4,
    ) -> list[Example]:
        """Mine the companion's own social chat history into instruction pairs.

        Social conversations (``telegram:123``, ``discord:456``, …) are stored
        in the same ``messages`` table as every other conversation, keyed by a
        ``{platform}:{chat_id}`` conversation id. This reader pulls only the
        platform-prefixed ones, pairs each user turn with the following
        assistant reply, and keeps the pairs that score at or above
        ``min_score`` — the high-signal slice that makes a personal fine-tune
        actually resemble the owner's real interactions. PII is scrubbed by the
        surrounding :meth:`collect` pass like every other source.
        """
        out: list[Example] = []
        for platform in platforms:
            prefix = str(platform).strip().lower()
            if not prefix:
                continue
            sql = (
                "SELECT id, conversation_id, role, content, created_at FROM messages "
                "WHERE conversation_id LIKE ? AND created_at >= ? "
                "ORDER BY conversation_id, created_at, rowid"
            )
            params: tuple[Any, ...] = (f"{prefix}:%", since)
            if limit:
                sql += " LIMIT ?"
                params = (f"{prefix}:%", since, limit * 10)
            try:
                rows = [dict(r) for r in self.db.query(sql, params)]
            except Exception as exc:  # noqa: BLE001 - optional table absent
                if "no such table" in str(exc).lower():
                    continue
                raise
            grouped: dict[str, list[Turn]] = {}
            for row in rows:
                role = str(row.get("role") or "")
                if role not in {"user", "assistant"}:
                    continue
                content = clean_text(scrub_pii(str(row.get("content") or "")))
                if not content.strip():
                    continue
                grouped.setdefault(
                    str(row.get("conversation_id")),
                    []).append(Turn(role, content))
            for conversation, turns in grouped.items():
                chat = conversation.split(":", 1)[1] if ":" in conversation else conversation
                for i, turn in enumerate(turns):
                    if turn.role != "user":
                        continue
                    answer = next(
                        (t.content for t in turns[i + 1:] if t.role == "assistant"), "")
                    user_text = turn.content.strip()
                    if user_text.startswith("/"):
                        continue
                    if len(user_text) < 8 or len(answer) < 24:
                        continue
                    score = _quality_score(user_text, answer)
                    if score < min_score:
                        continue
                    stats.social_history += 1
                    out.append(Example(
                        turns=[Turn("user", user_text), Turn("assistant", answer)],
                        weight=score,
                        source=f"social:{platform}:{chat}",
                    ))
                    if limit and stats.social_history >= limit:
                        break
                if limit and stats.social_history >= limit:
                    break
        return out

    def _rows(self, sql: str, since: float, *, limit: int) -> list[dict[str, Any]]:
        # A limit is applied after the source query so each source contributes a
        # bounded amount.  ``rowid`` exists on the shipped SQLite tables.
        if limit:
            sql += " LIMIT ?"
            params: tuple[Any, ...] = (since, limit)
        else:
            params = (since,)
        try:
            return [dict(row) for row in self.db.query(sql, params)]
        except Exception as exc:  # absent optional tables should not stop harvest
            if "no such table" in str(exc).lower():
                return []
            raise

    def _existing_fingerprints(self, stats: CollectionStats) -> list[int]:
        fingerprints: list[int] = []
        try:
            datasets = DatasetRegistry(self.db).list(limit=5000)
        except Exception as e:
            _log.debug("dataset registry list failed: %s", e)
            return fingerprints
        stats.existing_datasets = len(datasets)
        for dataset in datasets:
            try:
                for example in dataset.examples():
                    fingerprints.append(_example_simhash(scrub_example(example)))
            except Exception as exc:  # a missing old artifact should not block live use
                _log.warning("could not inspect dataset %s for dedup: %s", dataset.name, exc)
        return fingerprints


# The shorter names are convenient for integrations and preserve room for a
# future specialised collector implementation.
Collector = TrainingCollector
LiveCollector = TrainingCollector


def collect_training_examples(
    db: Any,
    *,
    since: float = 0.0,
    limit: int = 0,
    simhash_threshold: int = 3,
    output_dir: str | Path | None = None,
    name: str = "",
    register: bool = False,
    include_messages: bool = True,
) -> CollectionResult:
    """Functional wrapper used by small integrations and tests."""
    return TrainingCollector(db, simhash_threshold=simhash_threshold).collect(
        since=since,
        limit=limit,
        output_dir=output_dir,
        name=name,
        register=register,
        include_messages=include_messages,
    )


# Concise functional spelling for callers that treat collection as an operation.
collect = collect_training_examples


# ── conversation miner: quality-curated Q&A pairs from live chats ───────────


# Replies the bot sends for control commands, not for conversation — none of
# them are teaching material.
_CONTROL_NOISE = (
    re.compile(r"^\s*(usage:|control error)"),
    re.compile(r"^\s*[●■▶📄🐝🔎👁]"),
    re.compile(r"\b(swarm done|macro \w+:|saved macro|recording \w+|not recording)"),
    re.compile(r"^\s*(ok|okay|k|cool|sure|yes|no|yep|nope)[.!]?\s*$", re.I),
    re.compile(r"\b(failed|crashed|error:|timed out)\b.*$", re.I),
)

_CODE_HINT = re.compile(r"(```|\bdef \b|\bimport \b|SELECT |npm |git )")
_QUESTION = re.compile(r"\?")


def trigram_overlap(left: str, right: str) -> float:
    """Jaccard overlap of word trigram sets — the Self-Instruct diversity
    filter in cheap form.  A new instruction is only kept when its overlap
    with every kept instruction is below ~0.7; without this, mined corpora
    mode-collapse into variations of the same handful of questions."""
    def trigrams(text: str) -> set[str]:
        words = re.findall(r"\w+", text.lower())
        return {" ".join(words[i:i + 3]) for i in range(max(0, len(words) - 2))}
    left_t, right_t = trigrams(left), trigrams(right)
    if not left_t or not right_t:
        return 0.0
    return len(left_t & right_t) / len(left_t | right_t)


def _quality_score(user_text: str, answer_text: str) -> float:
    """Heuristic 0..1: how much this pair teaches a fine-tune.

    Rewards real Q→A structure, balanced lengths, substance (code,
    numbers, lists), and penalises chit-chat and bot boilerplate.
    """
    score = 0.3  # being a coherent pair at all is worth something
    u, a = len(user_text), len(answer_text)
    if u < 8 or a < 24:
        score -= 0.25
    if _QUESTION.search(user_text):
        score += 0.15
    ratio = min(u, a) / max(u, a, 1)
    score += 0.1 * ratio  # balanced exchanges read like instruction data
    if _CODE_HINT.search(user_text) or _CODE_HINT.search(answer_text):
        score += 0.15
    if any(ch.isdigit() for ch in answer_text):
        score += 0.05
    if answer_text.count("\n") >= 2:
        score += 0.05  # structured answer
    words = len(user_text.split())
    if 3 <= words <= 60:
        score += 0.05
    for pattern in _CONTROL_NOISE:
        if pattern.search(answer_text):
            score -= 0.3
            break
    return max(0.0, min(1.0, round(score, 3)))


class ConversationMiner:
    """Permanently-linked conversation → training-data agent.

    Reads the partner's own chat history (the ``messages`` rows the runtime
    persists for every conversation), pairs each user turn with the next
    assistant reply, scrubs PII, removes near-duplicates, quality-scores
    every pair, and writes fine-tune-ready bundles in four formats:
    internal JSONL, Alpaca, ShareGPT, and ChatML.
    """

    def __init__(self, db: Any, *, simhash_threshold: int = 3) -> None:
        self.db = db
        self.simhash_threshold = max(0, int(simhash_threshold))

    # ── source ──────────────────────────────────────────────────────────────
    def conversations(self, *, since: float = 0.0) -> list[list[Turn]]:
        """All partner conversations, oldest message first."""
        sql = (
            "SELECT m.* FROM messages m "
            "JOIN conversations c ON c.id = m.conversation_id "
            "WHERE c.agent = 'partner' AND m.created_at >= ? "
            "ORDER BY m.conversation_id, m.created_at, m.rowid"
        )
        try:
            rows = [dict(r) for r in self.db.query(sql, (since,))]
        except Exception as exc:
            if "no such table" in str(exc).lower():
                return []
            raise
        grouped: dict[str, list[Turn]] = {}
        for row in rows:
            role = str(row.get("role") or "")
            if role not in {"user", "assistant"}:
                continue
            content = clean_text(scrub_pii(str(row.get("content") or "")))
            if not content.strip():
                continue
            grouped.setdefault(str(row.get("conversation_id")), []).append(
                Turn(role, content))
        return [turns for turns in grouped.values() if len(turns) >= 2]

    # ── mining ──────────────────────────────────────────────────────────────
    def mine(
        self,
        *,
        since: float = 0.0,
        min_score: float = 0.3,
        limit: int = 400,
        diversity_overlap: float = 0.7,
        judge: Any = None,
        judge_weight: float = 0.4,
    ) -> dict[str, Any]:
        """Pair, scrub, dedupe, score.  Returns examples + stats.

        ``diversity_overlap`` — the Self-Instruct filter: a pair is dropped
        when its trigram overlap with an already-kept user turn exceeds
        this (default 0.7), so the corpus cannot mode-collapse into forty
        variations of one question.  ``judge`` — an optional
        ``(user, answer) -> float`` in [0, 1] (the AlpaGasus/DEITA hook:
        an LLM judge scoring pair quality); it blends with the heuristic
        at ``judge_weight``.
        """
        stats = {"conversations": 0, "pairs_seen": 0, "noise_dropped": 0,
                 "duplicates": 0, "below_score": 0, "kept": 0,
                 "diversity_dropped": 0, "judge_scored": 0}
        raw_pairs: list[tuple[str, str, str]] = []
        for turns in self.conversations(since=since):
            stats["conversations"] += 1
            # walk the conversation; each user turn pairs with the next
            # assistant turn that follows it
            for i, turn in enumerate(turns):
                if turn.role != "user":
                    continue
                answer = next((t.content for t in turns[i + 1:]
                               if t.role == "assistant"), "")
                if not answer:
                    continue
                stats["pairs_seen"] += 1
                user_text = turn.content.strip()
                if user_text.startswith("/"):
                    stats["noise_dropped"] += 1
                    continue
                if len(user_text) < 8 or len(answer) < 24:
                    stats["noise_dropped"] += 1
                    continue
                raw_pairs.append((user_text, answer, turns[0].content[:40]))
        if limit:
            raw_pairs = raw_pairs[-limit:]

        seen: list[int] = []
        kept_users: list[str] = []
        scored: list[tuple[float, Example, str]] = []
        for user_text, answer, source in raw_pairs:
            score = _quality_score(user_text, answer)
            if judge is not None:
                try:
                    judge_score = max(0.0, min(1.0, float(judge(user_text, answer))))
                    stats["judge_scored"] += 1
                    score = round((1.0 - judge_weight) * score
                                  + judge_weight * judge_score, 3)
                except Exception:  # noqa: BLE001 — a judge failure keeps the heuristic
                    pass
            if score < min_score:
                stats["below_score"] += 1
                continue
            if any(trigram_overlap(user_text, kept) > diversity_overlap
                   for kept in kept_users):
                stats["diversity_dropped"] += 1
                continue
            example = Example(
                turns=[Turn("user", user_text), Turn("assistant", answer)],
                weight=score,
                source=f"chat-pair:{source!r}:{user_text[:24]!r}",
            )
            fingerprint = _example_simhash(example)
            if any(hamming(fingerprint, prior) <= self.simhash_threshold
                   for prior in seen):
                stats["duplicates"] += 1
                continue
            seen.append(fingerprint)
            kept_users.append(user_text)
            scored.append((score, example, user_text))
        scored.sort(key=lambda item: item[0], reverse=True)
        stats["kept"] = len(scored)
        return {"examples": [e for _, e, _ in scored],
                "scores": [s for s, _, _ in scored],
                "stats": stats}

    def export(self, *, output_dir: str | Path, name: str = "",
               since: float = 0.0, min_score: float = 0.3,
               limit: int = 400, register: bool = True,
               diversity_overlap: float = 0.7,
               judge: Any = None, judge_weight: float = 0.4) -> dict[str, Any]:
        """Mine and write the four-format bundle + manifest; register the
        internal JSONL in the dataset registry when ``register`` is set."""
        result = self.mine(since=since, min_score=min_score, limit=limit,
                           diversity_overlap=diversity_overlap,
                           judge=judge, judge_weight=judge_weight)
        examples = result["examples"]
        out_dir = Path(output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        if not name:
            name = f"conversations-{int(time.time())}"
        base = out_dir / name
        counts = write_format_bundles(base, examples) if examples else {}
        manifest = {
            "name": name,
            "created_at": time.time(),
            "min_score": min_score,
            "formats": counts,
            "stats": result["stats"],
            "score_histogram": {
                "high": sum(1 for s in result["scores"] if s >= 0.7),
                "mid": sum(1 for s in result["scores"] if 0.5 <= s < 0.7),
                "low": sum(1 for s in result["scores"] if s < 0.5),
            },
            "files": {
                "internal": f"{base}.jsonl",
                "alpaca": f"{base}.alpaca.json",
                "sharegpt": f"{base}.sharegpt.json",
                "chatml": f"{base}.chatml.jsonl",
            },
        }
        (out_dir / f"{name}.manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        out: dict[str, Any] = {
            "name": name, "dir": str(out_dir),
            "examples": len(examples), "stats": result["stats"],
            "score_histogram": manifest["score_histogram"],
            "files": manifest["files"],
            "dataset_id": "",
        }
        if register and examples:
            try:
                dataset = DatasetRegistry(self.db).register_examples(
                    name, examples, out_dir, kind="chat",
                    metadata={"mined": True, "min_score": min_score,
                              "collector_stats": result["stats"]})
                out["dataset_id"] = dataset.id
            except Exception as exc:  # files are written; registration is bonus
                _log.warning("could not register mined dataset: %s", exc)
        _log.info("conversation miner: %d/%d pairs kept (name=%s)",
                  len(examples), result["stats"]["pairs_seen"], name)
        return out


def mine_conversations(
    db: Any,
    *,
    output_dir: str | Path,
    name: str = "",
    since: float = 0.0,
    min_score: float = 0.3,
    limit: int = 400,
    register: bool = True,
    simhash_threshold: int = 3,
    diversity_overlap: float = 0.7,
    judge: Any = None,
    judge_weight: float = 0.4,
) -> dict[str, Any]:
    """Functional entry point for the conversation→training pipeline."""
    return ConversationMiner(db, simhash_threshold=simhash_threshold).export(
        output_dir=output_dir, name=name, since=since, min_score=min_score,
        limit=limit, register=register, diversity_overlap=diversity_overlap,
        judge=judge, judge_weight=judge_weight)


def _example_simhash(example: Example) -> int:
    text = "\n".join(f"{turn.role}:{turn.content}" for turn in example.turns)
    return simhash(clean_text(text))


def _json(raw: Any, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return default
    return value
