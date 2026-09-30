"""Task-type classification (wave 65) — does this text want a BUILD?

This is how the system "knows to build" from pure text.  Every task entry
point (devon's router, goal -> project, project create) classifies the plain
text BEFORE deciding what to do, so the decision never depends on a human
typing the right magic words:

* **deterministic layer** — word-boundary keyword rules (offline-safe, zero
  model calls): a build verb hitting an artifact noun, investigation markers
  (why did / error / crash / broken), research markers (look up / latest /
  what is).  Order-agnostic, so "create a parser for csv files" and
  "the deploy is broken, why" both route correctly.
* **model tiebreak** — ONLY when the keywords are silent and a model is live:
  one small JSON call (kind + confidence + artifact + verify).  Never spent
  when the deterministic layer already has an answer.
* **default** — chat, low confidence, reason "no signal".

A ``build`` classification carries an *artifact* (the concrete thing that
must exist when done — "todo cli", "csv parser") and a *verify* hint.  The
project layer uses both to force real, runnable output (draft -> sandbox
run -> fix, real exit code) instead of a model narrating that it "built"
the thing.

The kinds:

* ``build`` — wants a new artifact created and working (code, script, app).
* ``investigate`` — wants to know what happened / why / where, in an
  existing system (logs, state, error trace).
* ``research`` — wants external information gathered and summarized.
* ``chat`` — anything else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["TaskType", "TASK_KINDS", "classify_task",
          "artifact_filename", "acceptance_command"]

#: The four task kinds, in routing priority order.
TASK_KINDS: tuple[str, ...] = ("build", "investigate", "research", "chat")

# Build verbs: a task wants an artifact when one of these hits...
_BUILD_VERBS = (
    "build", "create", "write", "make", "develop", "implement",
    "scaffold", "generate", "code", "construct", "set up", "set-up",
)

# ...and one of these is the thing to produce.  Multi-word nouns are
# matched as phrases; single words use word boundaries.
_BUILD_ARTIFACTS = (
    "cli", "command-line tool", "command line tool", "script", "app",
    "application", "api", "rest api", "tool", "bot", "agent", "site",
    "website", "webpage", "web page", "page", "program", "module",
    "service", "server", "dashboard", "extension", "plugin", "game",
    "crawler", "scraper", "parser", "generator", "function", "class",
    "library", "package", "file", "calculator", "todo", "tracker",
    "wizard", "snippet", "template", "pipeline", "cli tool",
)

# Strong build phrases that route on their own (verb implied).
_BUILD_STRONG = (
    "build a", "build the", "build me", "make a", "make the", "make me",
    "write a", "write the", "write me", "create a", "create the",
    "create me", "develop a", "develop the", "i want you to build",
    "i want you to write", "i want you to create", "big build",
    "code a", "code the", "scaffold a",
)

# Investigation markers: something happened / is wrong in an existing system.
_INVESTIGATE = (
    "why did", "why is", "why the", "why does", "error", "crash",
    "crashed", "failed", "failing", "broken", "traceback", "exception",
    "not working", "stopped", "what happened", "check if", "investigate",
    "diagnose", "root cause", "went wrong", "debug", "bug in", "no response",
    "silent", "hangs", "stuck",
)

# Research markers: external information to gather.
_RESEARCH = (
    "research", "look up", "what is", "who is", "latest", "news about",
    "find out", "compare", "state of the art", "how do i know",
    "best way to find", "deep dive", "survey",
)


@dataclass
class TaskType:
    """The classification of one piece of task text."""

    kind: str = "chat"
    confidence: float = 0.3
    reason: str = ""
    source: str = "default"  # keywords | model | default
    artifact: str = ""       # build only: the concrete thing that must exist
    verify: str = ""         # build only: how to prove it works

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "confidence": self.confidence,
            "reason": self.reason,
            "source": self.source,
            "artifact": self.artifact,
            "verify": self.verify,
        }


def _word_hit(text: str, words: tuple[str, ...]) -> bool:
    """True when any phrase (substring) or word (word-boundary) matches."""
    for w in words:
        if " " in w:
            if w in text:
                return True
        elif re.search(rf"\b{re.escape(w)}\b", text):
            return True
    return False


def _extract_artifact(t: str) -> str:
    """Best-effort artifact span: up to two preceding words + the noun.

    "build a todo cli" -> "todo cli"; "write a python script" -> "python
    script".  Articles/determiners are stripped.  Returns the bare noun when
    nothing precedes it.
    """
    best: tuple[int, str] | None = None  # (end offset, noun)
    for noun in _BUILD_ARTIFACTS:
        m = re.search(rf"({re.escape(noun)})\b", t)
        if m and (best is None or m.end() > best[0]
                  or (m.end() == best[0] and len(noun) > len(best[1]))):
            best = (m.end(), noun)
    if best is None:
        return ""
    noun = best[1]
    start = t.rfind(noun)
    prefix_words = re.findall(r"[a-z0-9-]+", t[:start])[-2:]
    # drop leading build verbs / determiners that leaked into the prefix
    while prefix_words and prefix_words[0] in {
        "build", "make", "write", "create", "develop", "implement",
        "scaffold", "generate", "code", "construct", "a", "an", "the",
        "my", "new",
    }:
        prefix_words.pop(0)
    span = " ".join(prefix_words + [noun]).strip()
    return span[:40]


def _kw_classify(text: str) -> TaskType | None:
    """Deterministic layer.  Returns None when the keywords are silent."""
    t = (text or "").lower().strip()
    if not t:
        return None

    # 1) build: verb + artifact, or a strong build phrase.
    verb = _word_hit(t, _BUILD_VERBS)
    artifact = _word_hit(t, _BUILD_ARTIFACTS)
    if artifact and verb:
        return TaskType(
            kind="build", confidence=0.9, reason="build verb + artifact noun",
            source="keywords", artifact=_extract_artifact(t),
        )
    if _word_hit(t, _BUILD_STRONG):
        return TaskType(
            kind="build", confidence=0.8, reason="strong build phrase",
            source="keywords", artifact=_extract_artifact(t),
        )

    # 2) investigate: something is wrong / happened.
    if _word_hit(t, _INVESTIGATE):
        return TaskType(
            kind="investigate", confidence=0.8,
            reason="investigation markers (error/crash/why did)",
            source="keywords",
        )

    # 3) research: external information wanted.
    if _word_hit(t, _RESEARCH):
        return TaskType(
            kind="research", confidence=0.7,
            reason="research markers (look up/latest/what is)",
            source="keywords",
        )

    return None


def _model_classify(context: Any, text: str) -> TaskType | None:
    """Model tiebreak — one small JSON call, only when keywords are silent."""
    router = getattr(context, "router", None)
    if router is None:
        return None
    prompt = (
        "Classify this task into exactly one kind.\n"
        "- build: wants a NEW artifact created and working (code, script, app, file)\n"
        "- investigate: wants to know what happened / why / where in an existing system\n"
        "- research: wants external information gathered and summarized\n"
        "- chat: anything else\n\n"
        f"TASK: {text[:400]}\n\n"
        'Respond with JSON ONLY: {"kind": "build|investigate|research|chat", '
        '"confidence": 0.0-1.0, '
        '"artifact": "<for build only: the concrete file/thing that must exist '
        'when done, else empty>", '
        '"verify": "<for build only: ONE runnable shell command that proves it '
        'works (e.g. python3 app.py --check, or python3 -m pytest app.py -q), '
        'else empty>"}'
    )
    try:
        from ..llm.base import Message, SamplingParams

        resp = router.chat(
            [Message.system("You are a task classifier. Reply with JSON only."),
             Message.user(prompt)],
            SamplingParams(temperature=0.0, max_tokens=120),
        )
        if not getattr(resp, "ok", False):
            return None
        from .reasoning import _extract_json

        data = _extract_json(resp.text or "")
        if not isinstance(data, dict):
            return None
        kind = str(data.get("kind", "")).strip().lower()
        if kind not in TASK_KINDS:
            return None
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.6))))
        except (TypeError, ValueError):
            confidence = 0.6
        is_build = kind == "build"
        artifact = str(data.get("artifact", "")).strip()[:80] if is_build else ""
        verify = str(data.get("verify", "")).strip()[:200] if is_build else ""
        return TaskType(
            kind=kind, confidence=confidence, reason="model classification",
            source="model", artifact=artifact, verify=verify,
        )
    except Exception:  # noqa: BLE001 - classification must never break the caller
        return None


def classify_task(context: Any, text: str, *, use_model: bool = True) -> TaskType:
    """Classify a piece of task text (deterministic first, model tiebreak).

    ``use_model=False`` forces the offline layer only — used by hot paths
    (devon's heuristic router) where a model call per task is not worth it.
    """
    tt = _kw_classify(text)
    if tt is not None:
        return tt
    if use_model:
        tt = _model_classify(context, text)
        if tt is not None:
            return tt
    return TaskType(kind="chat", confidence=0.3, reason="no signal", source="default")


def acceptance_command(verify: str, filename: str) -> str:
    """Compile a verify hint into the REAL sandbox acceptance command.

    A build step is not "done" when the model says it is — it is done when
    this command exits 0 in the sandbox.  The model tiebreak supplies a
    runnable command ("python3 app.py --check"); keyword-classified builds
    get the default (run the artifact).  Prose that is not a command falls
    back to the default rather than being fed to the shell.
    """
    v = (verify or "").strip()
    default = f'python3 "{filename}"'
    if not v:
        return default
    # shell metacharacters / pipes / redirections: definitely a command
    if any(ch in v for ch in "|&<>;"):
        return v
    head = v.split()[0].lower().strip("\'\\")
    known_bins = {
        "python", "python3", "python3.11", "node", "deno",
        "bash", "sh", "pytest", "poetry", "pip", "npm", "npx",
        "uv", "uvx", "ruby", "perl", "make", "cargo", "go", "test",
    }
    if head in known_bins:
        return v
    # "<bin> <file.py>" shape where the first token is a relative path
    if head.endswith((".py", ".sh", ".js", ".mjs")) or "/" in head:
        return v
    return default  # prose, not a command — the default is the safe call


def artifact_filename(artifact: str, default: str = "main") -> str:
    """Turn an artifact span into a safe python filename.

    "todo cli" -> "todo-cli.py"; "" -> "main.py"; an existing filename
    ("app.py") passes through unchanged.
    """
    raw = (artifact or "").strip().lower()
    if "." in raw and "/" not in raw and " " not in raw:
        # already a filename (e.g. "app.py") — pass through unchanged
        return raw
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")[:40]
    if not slug:
        slug = default
    return f"{slug}.py"
