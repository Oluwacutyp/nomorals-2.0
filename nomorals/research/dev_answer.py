"""Developer answer engine — the Phind core (52-websites #40).

Phind's idea: ask a technical question (or paste a full stack trace)
and get a direct answer — explanation + working code + citations you
can hover to verify — from live web search biased to developer
sources. Devon's equivalent is built on her own organs and exceeds
the reference in one place Phind does not have: she can RUN the code
she answers with.

The pipeline, end to end:

1. ``parse_dev_query`` understands the ask as a ``DevQuery`` — a
   pasted traceback is recognised by shape (Python, Go panic, JS,
   Rust, Java, .NET), the language is detected from fence tags,
   traceback shape and code tokens, and framework + versions are
   extracted (``django 4.2``, ``go 1.22``).
2. Retrieval runs the federated engine over the dev source pack
   (``DEV_PACK_SOURCES`` / ``DEV_PACK_WEIGHTS`` in
   ``nomorals.search.sources``): Devon's own code index first — for
   "why is MY bridge doing X" her codebase is the most authoritative
   source there is — then the live web backends. Hits are re-ranked
   by a curated domain tier: official docs, Stack Overflow and
   GitHub issues/discussions score highest. The tier map is a
   boost, never a closed list — an unknown domain scores by its
   retrieval rank with boost 1.0, never penalised.
3. Synthesis follows the citation contract of
   ``nomorals.research.grounded``: every claim carries a [n] that
   maps to a real, checkable source. Without a model the answer is
   extractive (top sentences by query-term overlap, code blocks
   carried verbatim from the sources); with a model the model is
   handed the sources labelled [S1]... and its citations are mapped
   back to [n] numbers, invented citations stripped, and a model
   answer with no surviving citation falls back to the extractive
   path.
4. Execution verification: every self-contained code block in the
   final answer is executed through the injected ``run_fn`` (the
   brain tool wires it to the ``run_code`` tool; the CLI wires it to
   ``CodeRunner`` directly). The result is recorded on the block —
   ``verified_by_execution`` true/false with the exit code and
   stderr excerpt. A block that did not run green is labelled
   "not verified" in the reply, NEVER silently presented as
   working. Fragments (``...`` placeholders) are not executed at
   all: they cannot run, so they are labelled as fragments.
5. The answer is persisted in ``DevAnswerStore``
   (``<home>/data/dev_answers.json``) with its ``answer_id``;
   follow-ups pass that id and carry the prior question + answer
   forward instead of restarting from zero.

Layering: this module is L4 (research). It never imports the search
engine (L5) or the code runner (L7) statically — retrieval is
resolved lazily by dotted name through :func:`_default_search_fn`
(the sanctioned dynamic-loading contract; the registry's books
probe is the precedent) and execution is always an injected
callable. ``answer_dev_question`` never raises on a dead network or
an empty corpus: an unanswered question returns ok False with the
reason named, and the miss is still recorded so it is visible.
"""

from __future__ import annotations

import importlib
import json
import re
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

log = get_logger(__name__)

DEV_ANSWERS_FILENAME = "dev_answers.json"

DEFAULT_LIMIT = 5
MAX_CODE_BLOCKS = 3
MAX_SENTENCES = 6

# ── the dev source pack (weights live in search/sources.py) ──────────
# The pack itself is defined there so the federated engine owns the
# registry; this module resolves it lazily by dotted name (L4 must not
# import L5 statically). The fallback below is only used when the
# import fails outright — it keeps the organ honest instead of dead.

_DEV_PACK_FALLBACK_SOURCES = ["code", "web_searxng", "web_ddgs"]
_DEV_PACK_FALLBACK_WEIGHTS = {
    "code": 2.0, "web_searxng": 1.5, "web_ddgs": 1.5,
}


def _dev_pack() -> tuple[list[str], dict[str, float]]:
    """Resolve (sources, weights) for the dev pack, lazily."""
    try:
        sources_mod = importlib.import_module("nomorals.search.sources")
        return (list(sources_mod.DEV_PACK_SOURCES),
                dict(sources_mod.DEV_PACK_WEIGHTS))
    except Exception:  # noqa: BLE001 — the fallback keeps the organ alive
        log.warning("dev_answer: dev pack import failed; using fallback")
        return (list(_DEV_PACK_FALLBACK_SOURCES),
                dict(_DEV_PACK_FALLBACK_WEIGHTS))


def _default_search_fn() -> Callable[..., Any]:
    """The federated search callable, resolved lazily by dotted name.

    Module-level seam: tests monkeypatch this name to inject a fake
    retrieval; production resolves the real engine. The organ never
    imports the L5 search package statically.
    """
    federated = importlib.import_module("nomorals.search.federated")
    return federated.federated_search


# ── domain tiers ──────────────────────────────────────────────────────
# Curated boost map, NOT a closed list. Tier 1 is the canonical dev
# reference set (official docs, Stack Overflow, GitHub/GitLab);
# tier 2 is strong dev writing. Any domain not listed scores by its
# retrieval rank with boost 1.0 — never penalised for being unknown.

DEV_DOMAIN_TIERS: dict[str, float] = {
    # tier 1 — canonical dev references
    "stackoverflow.com": 1.5,
    "github.com": 1.5,
    "gitlab.com": 1.5,
    "docs.python.org": 1.5,
    "developer.mozilla.org": 1.5,
    "go.dev": 1.5,
    "pkg.go.dev": 1.5,
    "docs.rs": 1.5,
    "learn.microsoft.com": 1.5,
    "react.dev": 1.5,
    "nodejs.org": 1.5,
    "docs.djangoproject.com": 1.5,
    # tier 2 — strong dev writing
    "geeksforgeeks.org": 1.25,
    "w3schools.com": 1.25,
    "realpython.com": 1.25,
    "medium.com": 1.25,
    "dev.to": 1.25,
    "readthedocs.io": 1.25,
    "docs.oracle.com": 1.25,
    "kotlinlang.org": 1.25,
    "swift.org": 1.25,
}


def dev_boost(url: str) -> float:
    """The domain-tier boost for a hit URL (1.0 when unlisted)."""
    try:
        host = (urllib.parse.urlparse(url or "").hostname or "").lower()
    except Exception:  # noqa: BLE001 — an unparsable URL is unboosted
        return 1.0
    if host.startswith("www."):
        host = host[4:]
    if host in DEV_DOMAIN_TIERS:
        return DEV_DOMAIN_TIERS[host]
    for domain, boost in DEV_DOMAIN_TIERS.items():
        if host.endswith("." + domain):
            return boost
    return 1.0


# ── the query shape ───────────────────────────────────────────────────

_FRAMEWORK_ORDER = (
    "django", "flask", "fastapi", "react", "next", "vue", "angular",
    "express", "laravel", "rails", "spring", "numpy", "pandas",
    "pytorch", "tensorflow", "node",
)

_VERSION_RE = re.compile(
    r"\b(django|flask|fastapi|react|go|python|node|rust|java|next|vue"
    r"|angular)\s+(\d+(?:\.\d+){0,2})\b", re.IGNORECASE)

_FENCE_RE = re.compile(r"```\s*([A-Za-z0-9_+-]+)")

_FENCE_LANG = {
    "py": "python", "python": "python", "python3": "python",
    "js": "javascript", "javascript": "javascript", "node": "javascript",
    "ts": "typescript", "typescript": "typescript",
    "go": "go", "golang": "go",
    "rs": "rust", "rust": "rust",
    "java": "java", "rb": "ruby", "ruby": "ruby",
    "php": "php", "c": "c", "cpp": "cpp", "c++": "cpp",
    "sh": "bash", "bash": "bash", "shell": "bash",
}

_LANGUAGE_HINTS: dict[str, tuple[str, ...]] = {
    "python": (r"\bdef \w+\(", r"\bimport \w+", r"\bprint\(",
               r"\.py\b", r"\bfrom \w+ import\b"),
    "go": (r"\bpackage \w+", r"\bfunc \w+\(", r":=",
           r"\.go\b", r"\bfmt\.Print"),
    "javascript": (r"\bconst \w+", r"\blet \w+", r"console\.",
                   r"=>", r"\.js\b", r"\bdocument\."),
    "typescript": (r"\binterface \w+", r": string\b", r"\.ts\b"),
    "rust": (r"\bfn \w+\(", r"let mut\b", r"\bcargo\b", r"\.rs\b"),
    "java": (r"public class\b", r"System\.out", r"\.java\b"),
    "ruby": (r"\bdef \w+", r"\.rb\b", r"\brequire ['\"]"),
    "php": (r"<\?php", r"\$\w+->", r"\.php\b"),
}

_JS_ERROR_RE = re.compile(
    r"^(TypeError|ReferenceError|SyntaxError|RangeError|Error):",
    re.MULTILINE)
_JAVA_EXC_RE = re.compile(r"Exception in thread .*?:\s*([\w.$]+)",
                          re.MULTILINE)


@dataclass
class DevQuery:
    """The ask, understood as a developer question."""

    question: str
    language: str = ""
    framework: str = ""
    versions: dict[str, str] = field(default_factory=dict)
    error_text: str = ""
    error_kind: str = ""
    is_traceback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "language": self.language,
            "framework": self.framework,
            "versions": dict(self.versions),
            "error_text": self.error_text,
            "error_kind": self.error_kind,
            "is_traceback": self.is_traceback,
        }


def _detect_language(text: str) -> str:
    """Language from fence tags first, then code tokens, then names."""
    fence = _FENCE_RE.search(text or "")
    if fence:
        lang = _FENCE_LANG.get(fence.group(1).lower())
        if lang:
            return lang
    lowered = text or ""
    for lang, hints in _LANGUAGE_HINTS.items():
        for hint in hints:
            if re.search(hint, lowered):
                return lang
    # A bare language name in the ask ("in Python", "a Rust crate"):
    # "go" is left out — as a bare word it is far too common to trust.
    for lang, pattern in (
            ("javascript", r"\bjavascript\b"),
            ("typescript", r"\btypescript\b"),
            ("python", r"\bpython\b"),
            ("rust", r"\brust\b"),
            ("ruby", r"\bruby\b"),
            ("php", r"\bphp\b"),
            ("java", r"\bjava\b")):
        if re.search(pattern, lowered, re.IGNORECASE):
            return lang
    return ""


def parse_dev_query(raw: str) -> DevQuery:
    """Parse a raw ask (plain question or pasted traceback)."""
    text = (raw or "").strip()
    query = DevQuery(question=text)
    lines = [ln for ln in text.splitlines() if ln.strip()]

    # ── traceback shapes, most specific first ──
    if "Traceback (most recent call last)" in text:
        query.is_traceback = True
        query.language = "python"
        query.error_text = lines[-1] if lines else ""
        head = query.error_text.split(":", 1)[0].strip()
        if re.fullmatch(r"[A-Za-z_][\w.]*", head or ""):
            query.error_kind = head
    elif re.search(r"^panic:", text, re.MULTILINE):
        query.is_traceback = True
        query.language = "go"
        query.error_kind = "panic"
        query.error_text = next(
            (ln.strip() for ln in lines if ln.startswith("panic:")), "")
    elif "panicked at" in text:
        query.is_traceback = True
        query.language = "rust"
        query.error_kind = "panic"
        query.error_text = next(
            (ln.strip() for ln in lines if "panicked at" in ln), "")
    elif _JS_ERROR_RE.search(text) and re.search(
            r"^\s+at ", text, re.MULTILINE):
        query.is_traceback = True
        query.language = "javascript"
        match = _JS_ERROR_RE.search(text)
        query.error_kind = match.group(1) if match else "Error"
        query.error_text = next(
            (ln.strip() for ln in lines
             if _JS_ERROR_RE.match(ln.strip())), lines[0] if lines else "")
    elif "Exception in thread" in text:
        query.is_traceback = True
        query.language = "java"
        match = _JAVA_EXC_RE.search(text)
        query.error_kind = match.group(1) if match else "Exception"
        query.error_text = next(
            (ln.strip() for ln in lines
             if "Exception in thread" in ln), lines[0] if lines else "")
    elif "Unhandled exception." in text:
        query.is_traceback = True
        query.language = "csharp"
        query.error_kind = "Exception"
        query.error_text = next(
            (ln.strip() for ln in lines
             if "Unhandled exception" in ln), "")

    if not query.language:
        query.language = _detect_language(text)

    lowered = text.lower()
    for framework in _FRAMEWORK_ORDER:
        if re.search(r"\b" + re.escape(framework) + r"\b", lowered):
            query.framework = framework
            break
    for match in _VERSION_RE.finditer(text):
        query.versions[match.group(1).lower()] = match.group(2)
    return query


def _retrieval_query(dev_query: DevQuery,
                     prior: dict[str, Any] | None) -> str:
    """What the federated engine is actually asked."""
    parts: list[str] = []
    if prior:
        parts.append(
            f"Prior question: {prior.get('question', '')}. "
            f"Prior answer: {str(prior.get('answer', ''))[:300]}.")
    if dev_query.is_traceback:
        parts.append(dev_query.error_text)
        if dev_query.language:
            parts.append(dev_query.language)
        if dev_query.framework:
            parts.append(dev_query.framework)
        parts.append("how to fix")
    else:
        parts.append(dev_query.question)
    return " ".join(p for p in parts if p).strip()


# ── the store ─────────────────────────────────────────────────────────

def _default_home(home: str | Path | None) -> Path:
    return Path(home) if home else Path.home() / ".nomorals"


class DevAnswerStore:
    """Durable dev answers: <home>/data/dev_answers.json.

    Atomic writes (temp file + replace). ``get`` accepts a full id or
    a unique prefix, so a chat reply can cite the short id.
    """

    def __init__(self, path: str | Path = "") -> None:
        self.path = str(path) if path else str(
            _default_home(None) / "data" / DEV_ANSWERS_FILENAME)
        self._data: dict[str, Any] = {"answers": {}}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(Path(self.path).read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(
                    data.get("answers"), dict):
                self._data["answers"] = data["answers"]
        except (OSError, ValueError):
            self._data = {"answers": {}}

    def _save(self) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def save(self, record: dict[str, Any]) -> None:
        self._data["answers"][str(record["answer_id"])] = record
        self._save()

    def get(self, answer_id: str) -> dict[str, Any] | None:
        row = self._data["answers"].get(answer_id or "")
        if isinstance(row, dict):
            return row
        matches = [r for k, r in self._data["answers"].items()
                   if k.startswith(answer_id or "")]
        return matches[0] if len(matches) == 1 else None

    def list(self) -> list[dict[str, Any]]:
        rows = [r for r in self._data["answers"].values()
                if isinstance(r, dict)]
        rows.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
        return [{
            "answer_id": str(r.get("answer_id") or ""),
            "question": str(r.get("question") or ""),
            "created_at": str(r.get("created_at") or ""),
            "ok": bool(r.get("ok")),
        } for r in rows]


def open_dev_answer_store(home: str | Path | None = None) -> DevAnswerStore:
    """Open (creating if needed) the durable dev-answer store."""
    return DevAnswerStore(_default_home(home) / "data"
                          / DEV_ANSWERS_FILENAME)


# ── code blocks + verification ────────────────────────────────────────

_FENCE_BLOCK_RE = re.compile(
    r"```\s*([A-Za-z0-9_+-]*)\s*\n(.*?)\n```", re.DOTALL)

_ELLIPSIS_RE = re.compile(r"\.\.\.")


def _extract_blocks(text: str) -> list[dict[str, Any]]:
    """Fenced code blocks from a snippet or an answer, in order."""
    blocks = []
    for match in _FENCE_BLOCK_RE.finditer(text or ""):
        code = match.group(2).strip("\n")
        if not code.strip():
            continue
        lang = _FENCE_LANG.get(match.group(1).lower(), "")
        blocks.append({"code": code, "language": lang})
    return blocks


def _is_fragment(code: str) -> bool:
    """True when the block is a placeholder, not runnable code.

    ``...`` means "fill in the rest" in every dev context (a stub
    body, a ``# ... more setup ...`` line, an elided middle) — a
    block carrying it cannot run as written, so it is never executed
    and never presented as verified.
    """
    return bool(_ELLIPSIS_RE.search(code or ""))


def _block_language(code: str, declared: str) -> str:
    if declared:
        return declared
    return _detect_language(code) or "python"


def _verify_block(block: dict[str, Any],
                  run_fn: Callable[..., Any] | None) -> dict[str, Any]:
    """Execute one block (or say honestly why it was not executed).

    Returns the verification record stored on the block. ``None``
    from the runner, or no runner at all, is "unverified" — never
    presented as working.
    """
    code = block["code"]
    if _is_fragment(code):
        return {"verified": False,
                "reason": "fragment — needs surrounding code to run"}
    if run_fn is None:
        return {"verified": False,
                "reason": "no code runner available — block not executed"}
    lang = _block_language(code, block.get("language", ""))
    try:
        result = run_fn(code, lang)
    except Exception as exc:  # noqa: BLE001 — the runner's failure is data
        return {"verified": False,
                "reason": f"runner error: {exc.__class__.__name__}: {exc}"}
    if result is None:
        return {"verified": False,
                "reason": "code runner unavailable — block not executed"}
    if not isinstance(result, dict):
        return {"verified": False,
                "reason": "runner returned an unreadable result"}
    exit_code = result.get("exit_code")
    timed_out = bool(result.get("timed_out"))
    ok = bool(result.get("ok")) and exit_code == 0 and not timed_out
    record: dict[str, Any] = {
        "verified": ok,
        "language": lang,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "seconds": result.get("seconds"),
    }
    if not ok:
        stderr = str(result.get("stderr") or "").strip()
        stdout = str(result.get("stdout") or "").strip()
        record["reason"] = (
            "timed out" if timed_out else
            f"exited {exit_code}: {stderr or stdout or 'no output'}")
        if stderr:
            record["stderr"] = stderr[:500]
    return record


# ── synthesis ─────────────────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9_]+")


def _terms(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower())
            if len(w) > 2}


def _extractive_answer(dev_query: DevQuery, hits: list[dict[str, Any]],
                       citations: list[dict[str, Any]]) -> str:
    """Deterministic synthesis: top sentences by query-term overlap.

    Every sentence carries its source's [n]. Code blocks are NOT
    inlined here — they live on the answer as structured blocks with
    their verification records.
    """
    query_terms = _terms(dev_query.question) | _terms(
        dev_query.error_text) | _terms(dev_query.language) | _terms(
        dev_query.framework)
    scored: list[tuple[int, int, str, int]] = []
    for idx, hit in enumerate(hits):
        prose = _FENCE_BLOCK_RE.sub(" ", hit.get("snippet", ""))
        for pos, sentence in enumerate(
                _SENTENCE_SPLIT_RE.split(prose.strip())):
            sentence = sentence.strip()
            if len(sentence) < 12:
                continue
            overlap = len(_terms(sentence) & query_terms)
            scored.append((overlap, -pos, sentence, idx))
    scored.sort(key=lambda row: (-row[0], row[1]))
    picked: list[str] = []
    for _overlap, _neg_pos, sentence, idx in scored[:MAX_SENTENCES]:
        picked.append(f"{sentence} [{idx + 1}]")
    if not picked:
        picked = [f"{hits[0].get('title', 'See the sources.')} [1]"]
    return " ".join(picked)


def _llm_answer(dev_query: DevQuery, hits: list[dict[str, Any]],
                citations: list[dict[str, Any]],
                llm_fn: Callable[[str], str],
                prior: dict[str, Any] | None) -> str | None:
    """Model synthesis over labelled sources, citations mapped to [n].

    The model sees the sources as [S1]... and must cite those labels.
    The labels are mapped back to citation numbers; any citation the
    model invents ([S99], a bare [7] with no source) is stripped, and
    an answer with no surviving citation falls back to the
    extractive path. Returns None when the model produced nothing
    usable.
    """
    labelled = "\n\n".join(
        f"[S{n + 1}] {hit.get('title', '')}\n{hit.get('snippet', '')}"
        for n, hit in enumerate(hits))
    prior_text = ""
    if prior:
        prior_text = (
            f"Prior question: {prior.get('question', '')}\n"
            f"Prior answer: {str(prior.get('answer', ''))[:600]}\n\n")
    prompt = (
        "Answer this developer question from the sources below. "
        "Give a direct explanation, then the working code in a fenced "
        "code block when the sources carry one. Cite every claim with "
        "the source label it came from, like [S1]. Use ONLY the labels "
        "shown — never invent a source. If the sources do not answer "
        "the question, say so plainly.\n\n"
        f"{prior_text}"
        f"Question: {dev_query.question}\n"
        f"Language: {dev_query.language or 'unknown'}\n"
        f"Error: {dev_query.error_text or 'none'}\n\n"
        f"Sources:\n{labelled}"
    )
    try:
        raw = llm_fn(prompt) or ""
    except Exception as exc:  # noqa: BLE001 — a dead model falls back
        log.warning("dev_answer: llm_fn failed: %s", exc)
        return None
    raw = raw.strip()
    if not raw:
        return None

    def _map_label(match: re.Match[str]) -> str:
        num = int(match.group(1))
        return f"[{num}]" if 1 <= num <= len(citations) else ""

    mapped = re.sub(r"\[S(\d+)\]", _map_label, raw)
    mapped = re.sub(r"\[\d+\]", lambda m: m.group(0)
                    if 1 <= int(m.group(0)[1:-1]) <= len(citations)
                    else "", mapped)
    mapped = re.sub(r"\s{2,}", " ", mapped).strip()
    if not re.search(r"\[\d+\]", mapped):
        return None
    return mapped


# ── the question flow ─────────────────────────────────────────────────

def answer_dev_question(
        home: str | Path | None, raw: str, *,
        store: DevAnswerStore | None = None,
        limit: int = DEFAULT_LIMIT,
        llm_fn: Callable[[str], str] | None = None,
        run_fn: Callable[..., Any] | None = None,
        search_fn: Callable[..., Any] | None = None,
        pack: tuple[list[str], dict[str, float]] | None = None,
        answer_id: str | None = None) -> dict[str, Any]:
    """Ask a dev question (or paste a traceback) and get a cited answer.

    Retrieval → domain-boosted ranking → synthesis → execution
    verification of every code block → durable record. Never raises
    on a dead network or an empty corpus: an unanswered question
    returns ok False with the reason named, and the miss is still
    recorded.
    """
    from ..core.ids import new_id

    store = store or open_dev_answer_store(home)
    dev_query = parse_dev_query(raw)
    prior = store.get(answer_id) if answer_id else None
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record: dict[str, Any] = {
        "answer_id": new_id("dev"),
        "question": dev_query.question,
        "dev_query": dev_query.to_dict(),
        "created_at": created_at,
        "follow_up_of": prior["answer_id"] if prior else None,
        "prior_question": prior["question"] if prior else None,
        "ok": False,
    }
    try:
        sources, weights = pack if pack is not None else _dev_pack()
        search = search_fn or _default_search_fn()
        response = search(
            _retrieval_query(dev_query, prior),
            sources=sources, fusion="rrf", source_weights=weights,
            limit=max(limit, DEFAULT_LIMIT))
        hits = getattr(response, "hits", []) or []
        searched = list(getattr(response, "sources_searched", []) or [])
        skipped = dict(getattr(response, "sources_skipped", {}) or {})
    except Exception as exc:  # noqa: BLE001 — retrieval failure is data
        record["error"] = f"dev search failed: {exc.__class__.__name__}: {exc}"
        record["reply"] = (
            "I could not search for that just now: "
            f"{exc.__class__.__name__}: {exc}")
        store.save(record)
        return record

    if not hits:
        skipped_text = "; ".join(
            f"{name}: {reason}" for name, reason in skipped.items())
        record["error"] = (
            "no dev sources returned anything for this question"
            + (f" (skipped: {skipped_text})" if skipped_text else "")
            + (f" (searched: {', '.join(searched)})" if searched else ""))
        record["reply"] = (
            "No dev source returned anything for that — "
            "nothing to cite, so I would rather say so than guess. "
            + (f"Skipped sources: {skipped_text}. " if skipped_text else "")
            + "Try rewording the question, or paste the full error.")
        record["sources_searched"] = searched
        record["sources_skipped"] = skipped
        store.save(record)
        return record

    # Rank: retrieval score scaled by the domain tier, top-N cited.
    def _hit_url(hit: Any) -> str:
        prov = getattr(hit, "provenance", None) or {}
        return str(prov.get("url") or "")

    def _hit_score(hit: Any) -> float:
        base = float(getattr(hit, "score", 0.0) or 0.0)
        raw = getattr(hit, "raw_score", None)
        if raw is not None:
            base = max(base, float(raw))
        return base * dev_boost(_hit_url(hit))

    ranked = sorted(hits, key=_hit_score, reverse=True)[:limit]
    citations: list[dict[str, Any]] = []
    hit_dicts: list[dict[str, Any]] = []
    for hit in ranked:
        url = _hit_url(hit)
        citations.append({
            "title": str(getattr(hit, "title", "") or ""),
            "url": url,
            "source": str(getattr(hit, "source", "") or ""),
        })
        hit_dicts.append({
            "title": str(getattr(hit, "title", "") or ""),
            "snippet": str(getattr(hit, "snippet", "") or ""),
        })

    answer_text = None
    grounded = False
    if llm_fn is not None:
        answer_text = _llm_answer(dev_query, hit_dicts, citations,
                                  llm_fn, prior)
    if answer_text is None:
        answer_text = _extractive_answer(dev_query, hit_dicts, citations)
    grounded = True

    # Code blocks: from the model's answer when it has them, else
    # carried verbatim from the top sources. Every block is executed.
    blocks = _extract_blocks(answer_text)
    if not blocks:
        for hit in hit_dicts:
            blocks.extend(_extract_blocks(hit["snippet"]))
            if len(blocks) >= MAX_CODE_BLOCKS:
                break
        blocks = blocks[:MAX_CODE_BLOCKS]
    code_blocks: list[dict[str, Any]] = []
    for block in blocks[:MAX_CODE_BLOCKS]:
        verification = _verify_block(block, run_fn)
        code_blocks.append({
            "code": block["code"],
            "language": _block_language(block["code"],
                                        block.get("language", "")),
            "verified_by_execution": bool(verification.get("verified")),
            "verification": verification,
        })

    record.update({
        "ok": True,
        "answer": answer_text,
        "code_blocks": code_blocks,
        "citations": citations,
        "sources_searched": searched,
        "sources_skipped": skipped,
        "grounded": grounded,
    })
    record["reply"] = format_dev_answer_reply(record)
    store.save(record)
    return record


def format_dev_answer_reply(answer: dict[str, Any]) -> str:
    """Render a stored answer as the chat reply.

    The answer prose first, then each code block with its
    verification label (✅ verified / ⚠️ not verified, with the
    reason), then the Sources section with real URLs.
    """
    if not answer.get("ok"):
        return str(answer.get("reply") or answer.get("error")
                   or "No answer.")
    lines = [str(answer.get("answer") or "").strip()]
    for block in answer.get("code_blocks") or []:
        verification = block.get("verification") or {}
        lang = block.get("language") or ""
        lines.append("")
        if block.get("verified_by_execution"):
            seconds = verification.get("seconds")
            when = (f" in {seconds:.2f}s" if isinstance(seconds, float)
                    else "")
            lines.append(f"```{lang}\n{block.get('code', '')}\n```")
            lines.append(f"✅ verified by execution{when} (exit 0)")
        else:
            reason = verification.get("reason") or "not verified"
            lines.append(f"```{lang}\n{block.get('code', '')}\n```")
            lines.append(f"⚠️ not verified — {reason}")
    citations = answer.get("citations") or []
    if citations:
        lines.append("")
        lines.append("Sources:")
        for n, cite in enumerate(citations, 1):
            title = cite.get("title") or cite.get("url") or ""
            lines.append(f"[{n}] {title} — {cite.get('url', '')}")
    return "\n".join(lines).strip()
