"""Shell-command explainer organ (ExplainShell core, 52-websites #43).

Paste a shell command; get each fragment annotated with the exact
help text for that fragment — grounded in the man pages of the
binaries actually installed on this host
(:mod:`nomorals.codews.man_corpus`), in ``--help`` output harvested
in the sandbox, or in tldr community pages. A fragment with no
grounded doc anywhere is marked ``unknown``, never filled in
silently — an improvised explanation is exactly the ungrounded
shape Devon's doctrine forbids presenting as fact.

The parser is a real bash grammar for the common grammar:
pipelines, redirects, ``&&``/``||``/``;``, quotes, escapes,
``$(...)``/backtick substitution, env assignments, ``sudo``-style
prefixes, bundled short flags. ``bashlex`` is used where the
profile runtime admits the dependency (import-guarded, like sympy
in :mod:`nomorals.codews.compute`); the hand parser is the
always-available rung and the fallback when bashlex is absent or
its walk fails. Control-flow clauses (``for``/``if``/``while``/…)
are marked ``unparsed`` honestly — the failure shape matters more
than coverage. zsh/fish syntax is out of scope, named as such.

The organ NEVER executes anything: the explained command is text.
Execution happens only in the tool layer's explicit
``explain_and_run`` action, through the sanctioned ``shell_run``
tool, after the explanation is produced.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .man_corpus import (
    BUILTIN_DOCS,
    ManCorpus,
    open_man_corpus,
    parse_help_text,
    parse_tldr_markdown,
)

__all__ = [
    "Fragment",
    "explain_command",
    "format_explanation_reply",
    "open_explanation_store",
    "parse_bash",
]

_log = get_logger(__name__)

#: A command longer than this is refused outright (runaway guard).
_MAX_COMMAND_CHARS = 4000
#: Fragment ceiling per explanation (runaway guard).
_MAX_FRAGMENTS = 200
#: Recursion ceiling for nested $(...) — deeper nests become unparsed.
_MAX_SUBST_DEPTH = 10
#: Durable explanations file under <home>/data/.
_EXPLANATIONS_FILENAME = "explanations.json"

#: Words that prefix a real command (the next word is the command).
_PREFIX_COMMANDS = {"sudo", "env", "time", "command", "nice", "nohup",
                    "doas", "stdbuf", "timeout", "xargs"}
#: Control-flow keywords: their clauses are marked unparsed honestly.
_CONTROL_KEYWORDS = {"if", "then", "else", "elif", "fi", "for", "while",
                     "until", "do", "done", "case", "esac", "select",
                     "function", "in", "coproc", "time"}
_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_LONG_FLAG_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*")
_SHORT_FLAG_RE = re.compile(r"(?<![A-Za-z0-9-])-([A-Za-z0-9])(?![A-Za-z0-9])")


# ── fragments ─────────────────────────────────────────────────────────

@dataclass
class Fragment:
    """One parsed piece of a command, with its span in the original."""

    span: tuple[int, int]
    token: str
    role: str  # command|option|argument|pipe|redirect|substitution|
    # prefix|assignment|separator|keyword|unparsed
    detail: str = ""
    doc: dict[str, Any] | None = None
    unknown: bool = False
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "span": [self.span[0], self.span[1]], "token": self.token,
            "role": self.role, "detail": self.detail,
            "doc": self.doc, "unknown": self.unknown,
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fragment:
        span = data.get("span") or [0, 0]
        return cls(
            span=(int(span[0]), int(span[1])),
            token=str(data.get("token", "")), role=str(data.get("role", "")),
            detail=str(data.get("detail", "") or ""),
            doc=data.get("doc"),
            unknown=bool(data.get("unknown", False)),
            suggestion=data.get("suggestion"))


# ── the hand parser ───────────────────────────────────────────────────

class _Tokenizer:
    """Bash word/operator tokenizer with real spans.

    Quotes, escapes, ``$(...)`` and backticks are honoured; an
    unterminated quote or substitution yields an ``unparsed`` token
    covering the remainder instead of raising.
    """

    def __init__(self, text: str, base: int = 0, depth: int = 0) -> None:
        self.text = text
        self.base = base
        self.depth = depth
        self.i = 0
        self.tokens: list[dict[str, Any]] = []

    def _emit(self, kind: str, value: str, start: int, end: int) -> None:
        self.tokens.append({"kind": kind, "value": value,
                            "start": self.base + start,
                            "end": self.base + end})

    def _read_quoted(self, quote: str) -> tuple[str, int, bool]:
        """Read a quoted span starting after the opening quote.

        Returns (text, end_index, terminated). ``$(...)`` inside
        double quotes is kept as text here; the caller re-parses it.
        """
        out: list[str] = []
        start = self.i
        while self.i < len(self.text):
            ch = self.text[self.i]
            if ch == quote:
                return "".join(out), self.i + 1, True
            if quote == '"' and ch == "\\" and self.i + 1 < len(self.text):
                nxt = self.text[self.i + 1]
                if nxt in ('"', "\\", "$", "`"):
                    out.append(nxt)
                    self.i += 2
                    continue
                out.append(ch)
                self.i += 1
                continue
            if quote == '"' and ch == "$" and self.i + 1 < len(self.text) \
                    and self.text[self.i + 1] == "(":
                depth = 0
                j = self.i
                while j < len(self.text):
                    if self.text[j] == "(":
                        depth += 1
                    elif self.text[j] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                out.append(self.text[self.i:j + 1])
                self.i = j + 1
                continue
            out.append(ch)
            self.i += 1
        return "".join(out), start, False

    def _read_substitution(self) -> tuple[str, int, bool]:
        """Read $(...) or `...` starting at self.i. Returns text, end,
        terminated. Nested parens are counted; quotes inside are
        honoured well enough for the common grammar."""
        opener = self.text[self.i]
        if opener == "`":
            j = self.i + 1
            while j < len(self.text):
                if self.text[j] == "\\":
                    j += 2
                    continue
                if self.text[j] == "`":
                    return self.text[self.i:j + 1], j + 1, True
                j += 1
            return self.text[self.i:], len(self.text), False
        depth = 0
        j = self.i
        quote: str | None = None
        while j < len(self.text):
            ch = self.text[j]
            if quote is not None:
                if ch == quote:
                    quote = None
                elif ch == "\\" and quote == '"':
                    j += 1
            elif ch in ("'", '"'):
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return self.text[self.i:j + 1], j + 1, True
            j += 1
        return self.text[self.i:], len(self.text), False

    def run(self) -> list[dict[str, Any]]:
        text = self.text
        while self.i < len(text):
            ch = text[self.i]
            if ch in " \t\r":
                self.i += 1
                continue
            if ch == "\n":
                self._emit("sep", "\n", self.i, self.i + 1)
                self.i += 1
                continue
            if ch == "#":
                # Comment to end of line.
                j = text.find("\n", self.i)
                self.i = len(text) if j < 0 else j
                continue
            if text.startswith("||", self.i) or text.startswith("&&", self.i):
                self._emit("sep", text[self.i:self.i + 2], self.i, self.i + 2)
                self.i += 2
                continue
            if text.startswith(";;", self.i):
                self._emit("sep", ";;", self.i, self.i + 2)
                self.i += 2
                continue
            if text.startswith(">>", self.i) or text.startswith("<<", self.i):
                self._emit("redir", text[self.i:self.i + 2], self.i, self.i + 2)
                self.i += 2
                continue
            if text.startswith("<<<", self.i):
                self._emit("redir", "<<<", self.i, self.i + 3)
                self.i += 3
                continue
            if ch in "|;":
                self._emit("sep", ch, self.i, self.i + 1)
                self.i += 1
                continue
            if ch == "&":
                if text.startswith("&>", self.i):
                    self._emit("redir", "&>", self.i, self.i + 2)
                    self.i += 2
                    continue
                self._emit("sep", "&", self.i, self.i + 1)
                self.i += 1
                continue
            if ch in "><":
                start = self.i
                # A leading fd digit belongs to the redirect (2>).
                if start > 0 and text[start - 1].isdigit():
                    pass
                self._emit("redir", ch, start, start + 1)
                self.i += 1
                continue
            if ch == "(":
                self._emit("lparen", "(", self.i, self.i + 1)
                self.i += 1
                continue
            if ch == ")":
                self._emit("rparen", ")", self.i, self.i + 1)
                self.i += 1
                continue
            if ch == "{":
                self._emit("lbrace", "{", self.i, self.i + 1)
                self.i += 1
                continue
            if ch == "}":
                self._emit("rbrace", "}", self.i, self.i + 1)
                self.i += 1
                continue
            # A word: quotes, escapes, substitutions, fd redirects.
            start = self.i
            parts: list[str] = []
            terminated = True
            while self.i < len(text):
                ch = text[self.i]
                if ch in " \t\r\n|;&<>(){}":
                    break
                if ch == "\\" and self.i + 1 < len(text):
                    parts.append(text[self.i + 1])
                    self.i += 2
                    continue
                if ch in ("'", '"'):
                    self.i += 1
                    quoted, end, ok = self._read_quoted(ch)
                    parts.append(quoted)
                    if not ok:
                        terminated = False
                        self.i = len(text)
                        break
                    self.i = end
                    continue
                if ch == "$" and self.i + 1 < len(text) \
                        and text[self.i + 1] == "(":
                    subst, end, ok = self._read_substitution()
                    parts.append(subst)
                    if not ok:
                        terminated = False
                        self.i = len(text)
                        break
                    self.i = end
                    continue
                if ch == "`":
                    subst, end, ok = self._read_substitution()
                    parts.append(subst)
                    if not ok:
                        terminated = False
                        self.i = len(text)
                        break
                    self.i = end
                    continue
                if ch == ">" or ch == "<":
                    break
                parts.append(ch)
                self.i += 1
            word = "".join(parts)
            # A trailing fd digit + redirect: "2>" splits as word "2"
            # then ">" — merge them into one redirect token.
            if word and word.isdigit() and self.i < len(text) \
                    and text[self.i] in "><":
                op = text[self.i]
                self.i += 1
                if self.i < len(text) and text[self.i] == op:
                    op += op
                    self.i += 1
                self._emit("redir", word + op, start, self.i)
                continue
            if not terminated:
                self._emit("unparsed", text[start:], start, len(text))
                self.i = len(text)
                continue
            if word:
                self._emit("word", word, start, self.i)
        return self.tokens


def _structure(tokens: list[dict[str, Any]], text: str,
               depth: int = 0) -> list[Fragment]:
    """Token stream → fragments with command/option/argument roles."""
    frags: list[Fragment] = []
    # Split into simple commands at separators/pipes.
    groups: list[list[dict[str, Any]]] = [[]]
    for tok in tokens:
        if tok["kind"] == "sep" and tok["value"] in ("|", "||", "&&",
                                                     ";", "&", "\n", ";;"):
            groups.append([])
        else:
            groups[-1].append(tok)
    for group in groups:
        if not group:
            continue
        # The separator between groups is recovered from the token
        # stream by the caller; here each group is one simple command.
        frags.extend(_simple_command(group, text, depth))
    # Separators and pipes as their own fragments, in span order.
    for tok in tokens:
        if tok["kind"] == "sep":
            role = "pipe" if tok["value"] == "|" else "separator"
            detail = ("pipes output to the next command" if role == "pipe"
                      else "runs the next command after this one")
            frags.append(Fragment(span=(tok["start"], tok["end"]),
                                  token=tok["value"], role=role,
                                  detail=detail))
    frags.sort(key=lambda f: (f.span[0], f.span[1]))
    return frags[:_MAX_FRAGMENTS]


def _simple_command(group: list[dict[str, Any]], text: str,
                    depth: int) -> list[Fragment]:
    frags: list[Fragment] = []
    idx = 0
    # Control-flow clause: keyword + rest is unparsed, honestly.
    if group and group[0]["kind"] == "word" \
            and group[0]["value"] in _CONTROL_KEYWORDS:
        frags.append(Fragment(
            span=(group[0]["start"], group[0]["end"]),
            token=group[0]["value"], role="keyword",
            detail="control-flow keyword — clause not expanded"))
        for tok in group[1:]:
            if tok["kind"] == "word":
                frags.append(Fragment(
                    span=(tok["start"], tok["end"]), token=tok["value"],
                    role="unparsed",
                    detail="inside a control-flow clause — not expanded"))
        return frags
    # Subshell / group: recurse into the inner tokens.
    if group and group[0]["kind"] in ("lparen", "lbrace"):
        opener = group[0]
        closer_kind = "rparen" if opener["kind"] == "lparen" else "rbrace"
        close_idx = next((j for j, t in enumerate(group)
                          if t["kind"] == closer_kind), None)
        inner = group[1:close_idx] if close_idx is not None else group[1:]
        frags.append(Fragment(
            span=(opener["start"], opener["end"]), token=opener["value"],
            role="separator",
            detail="subshell — runs in a child shell" if opener["kind"] == "lparen"
            else "command group"))
        if depth < _MAX_SUBST_DEPTH:
            frags.extend(_structure(inner, text, depth + 1))
        else:
            for tok in inner:
                if tok["kind"] == "word":
                    frags.append(Fragment(
                        span=(tok["start"], tok["end"]), token=tok["value"],
                        role="unparsed",
                        detail="nested too deep — not expanded"))
        if close_idx is not None:
            closer = group[close_idx]
            frags.append(Fragment(
                span=(closer["start"], closer["end"]),
                token=closer["value"], role="separator", detail=""))
        return frags
    command_seen = False
    expect_value_for: str | None = None
    while idx < len(group):
        tok = group[idx]
        kind = tok["kind"]
        if kind == "unparsed":
            frags.append(Fragment(
                span=(tok["start"], tok["end"]), token=tok["value"],
                role="unparsed",
                detail="unterminated quote or substitution"))
            idx += 1
            continue
        if kind == "redir":
            frags.append(Fragment(
                span=(tok["start"], tok["end"]), token=tok["value"],
                role="redirect", detail="redirects a stream"))
            # The next word is the redirect target.
            if idx + 1 < len(group) and group[idx + 1]["kind"] == "word":
                nxt = group[idx + 1]
                frags.append(Fragment(
                    span=(nxt["start"], nxt["end"]), token=nxt["value"],
                    role="argument", detail="redirect target"))
                idx += 2
                continue
            idx += 1
            continue
        if kind in ("lparen", "rparen", "lbrace", "rbrace"):
            frags.append(Fragment(
                span=(tok["start"], tok["end"]), token=tok["value"],
                role="separator", detail=""))
            idx += 1
            continue
        if kind != "word":
            idx += 1
            continue
        word = tok["value"]
        if not command_seen:
            if _ASSIGN_RE.match(word):
                frags.append(Fragment(
                    span=(tok["start"], tok["end"]), token=word,
                    role="assignment",
                    detail="sets an environment variable for this command"))
                idx += 1
                continue
            if word in _PREFIX_COMMANDS:
                frags.append(Fragment(
                    span=(tok["start"], tok["end"]), token=word,
                    role="prefix",
                    detail="runs the following command"))
                idx += 1
                continue
            if word == "!":
                frags.append(Fragment(
                    span=(tok["start"], tok["end"]), token=word,
                    role="prefix", detail="negates the exit status"))
                idx += 1
                continue
            frags.append(Fragment(
                span=(tok["start"], tok["end"]), token=word, role="command"))
            command_seen = True
            idx += 1
            continue
        # After the command: options, their values, arguments.
        if expect_value_for is not None:
            frags.append(Fragment(
                span=(tok["start"], tok["end"]), token=word,
                role="argument", detail=f"value of {expect_value_for}"))
            expect_value_for = None
            idx += 1
            continue
        if word.startswith("-") and len(word) > 1 and word != "--":
            frags.append(Fragment(
                span=(tok["start"], tok["end"]), token=word, role="option"))
            # A trailing "=value" is the option's own value; a bundled
            # flag that takes a value consumes the next word.
            if "=" in word:
                pass
            elif _option_takes_value_hint(word):
                expect_value_for = word
            idx += 1
            continue
        if word == "--":
            frags.append(Fragment(
                span=(tok["start"], tok["end"]), token=word,
                role="separator",
                detail="end of options — everything after is an argument"))
            idx += 1
            continue
        frags.append(Fragment(
            span=(tok["start"], tok["end"]), token=word, role="argument"))
        idx += 1
    # Command substitutions inside words: parse the inner command and
    # add its fragments (the outer word keeps its own span).
    extra: list[Fragment] = []
    for tok in group:
        if tok["kind"] != "word":
            continue
        extra.extend(_substitutions_in(tok["value"], tok["start"], depth))
    frags.extend(extra)
    return frags


def _option_takes_value_hint(word: str) -> bool:
    """Heuristic for bundled short flags: only used to mark the next
    word as a value when the man page is not yet consulted. The
    matcher re-derives this from the real option record, so a wrong
    hint here never invents a doc — it only shapes the detail line.
    """
    return False


_SUBST_RE = re.compile(r"\$\(([^()]*(?:\([^()]*\)[^()]*)*)\)|`([^`]+)`")


def _substitutions_in(word: str, base: int,
                      depth: int) -> list[Fragment]:
    """Fragments for $(...) / `...` inside one word."""
    out: list[Fragment] = []
    if depth >= _MAX_SUBST_DEPTH:
        return out
    for match in _SUBST_RE.finditer(word):
        inner = match.group(1) if match.group(1) is not None \
            else match.group(2)
        start = base + match.start()
        end = base + match.end()
        out.append(Fragment(
            span=(start, end), token=match.group(0),
            role="substitution",
            detail="command substitution — runs and captures its output"))
        inner_tokens = _Tokenizer(inner, base=start + 2,
                                  depth=depth + 1).run()
        out.extend(_structure(inner_tokens, inner, depth + 1))
    return out


def _parse_bashlex(command: str) -> list[Fragment] | None:
    """Parse with bashlex where the profile admits it.

    Returns None when bashlex is absent or its walk fails — the
    caller falls back to the hand parser, which is always available.
    """
    try:
        import bashlex  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — bashlex is an optional rung
        return None
    try:
        nodes = bashlex.parse(command)
    except Exception as exc:  # noqa: BLE001 — unparseable by bashlex
        _log.debug("bashlex parse failed: %s", exc)
        return None
    frags: list[Fragment] = []

    def _word_role(word: str, first: bool) -> str:
        if first:
            if _ASSIGN_RE.match(word):
                return "assignment"
            if word in _PREFIX_COMMANDS:
                return "prefix"
            return "command"
        if word.startswith("-") and len(word) > 1:
            return "option"
        return "argument"

    def _walk(node: Any, first_in_command: list[bool]) -> None:
        kind = getattr(node, "kind", "")
        if kind == "command":
            first = True
            for part in getattr(node, "parts", []):
                _walk_part(part, first)
                if getattr(part, "kind", "") == "word":
                    first = False
            return
        if kind == "operator":
            op = getattr(node, "op", "")
            pos = getattr(node, "pos", (0, 0))
            role = "pipe" if op == "|" else "separator"
            frags.append(Fragment(span=(pos[0], pos[1]), token=op,
                                  role=role))
            return
        if kind == "redirect":
            pos = getattr(node, "pos", (0, 0))
            frags.append(Fragment(span=(pos[0], pos[1]),
                                  token=getattr(node, "type", ">"),
                                  role="redirect",
                                  detail="redirects a stream"))
            return
        for child in (getattr(node, "parts", []) or []
                      + getattr(node, "list", []) or []):
            _walk(child, first_in_command)

    def _walk_part(part: Any, first: bool) -> None:
        kind = getattr(part, "kind", "")
        pos = getattr(part, "pos", (0, 0))
        if kind == "word":
            word = getattr(part, "word", "")
            frags.append(Fragment(span=(pos[0], pos[1]), token=word,
                                  role=_word_role(word, first)))
        elif kind == "redirect":
            _walk(part, [True])
        elif kind == "commandsubstitution":
            frags.append(Fragment(span=(pos[0], pos[1]),
                                  token=command[pos[0]:pos[1]],
                                  role="substitution",
                                  detail="command substitution — runs and "
                                         "captures its output"))
            inner = getattr(part, "command", None)
            if inner is not None:
                _walk(inner, [True])
        else:
            _walk(part, [True])

    try:
        for node in nodes:
            _walk(node, [True])
    except Exception as exc:  # noqa: BLE001 — walk failed, hand parser
        _log.debug("bashlex walk failed: %s", exc)
        return None
    frags.sort(key=lambda f: (f.span[0], f.span[1]))
    return frags[:_MAX_FRAGMENTS]


def parse_bash(command: str) -> list[Fragment]:
    """Parse a shell command into fragments with spans.

    bashlex where the profile admits it, else the hand parser.
    Never raises: an unparseable command yields fragments with an
    ``unparsed`` role naming the problem.
    """
    text = command or ""
    if not text.strip():
        return []
    parsed = _parse_bashlex(text)
    if parsed is not None:
        return parsed
    tokens = _Tokenizer(text).run()
    return _structure(tokens, text)


# ── the explainer ─────────────────────────────────────────────────────

def _default_home(home: str | Path | None) -> Path:
    return Path(home) if home else Path.home() / ".nomorals"


class ExplanationStore:
    """Durable explanations: <home>/data/explanations.json.

    Atomic writes (temp file + replace). ``get`` accepts a full id or
    a unique prefix, so a chat reply can cite the short id.
    """

    def __init__(self, path: str | Path = "") -> None:
        self.path = str(path) if path else str(
            _default_home(None) / "data" / _EXPLANATIONS_FILENAME)
        self._data: dict[str, Any] = {"explanations": {}}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and isinstance(
                data.get("explanations"), dict):
            self._data = data

    def _save(self) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, default=str),
                       encoding="utf-8")
        tmp.replace(path)

    def save(self, explanation: dict[str, Any]) -> None:
        self._data["explanations"][explanation["explanation_id"]] = explanation
        self._save()

    def get(self, explanation_id: str) -> dict[str, Any] | None:
        key = (explanation_id or "").strip()
        if not key:
            return None
        rows = self._data["explanations"]
        if key in rows:
            return rows[key]
        matches = [row for eid, row in rows.items() if eid.startswith(key)]
        return matches[0] if len(matches) == 1 else None

    def list(self) -> list[dict[str, Any]]:
        rows = sorted(self._data["explanations"].values(),
                      key=lambda r: r.get("created_at", ""), reverse=True)
        return [{"explanation_id": r["explanation_id"],
                 "command": r["command"],
                 "created_at": r.get("created_at", "")} for r in rows]


def open_explanation_store(home: str | Path | None = None) -> ExplanationStore:
    """Open (creating if needed) the durable explanation store."""
    return ExplanationStore(_default_home(home) / "data"
                            / _EXPLANATIONS_FILENAME)


def _split_option_token(token: str) -> list[tuple[str, tuple[int, int]]]:
    """One option token → (flag, relative span) pieces.

    ``--flag=value`` stays one piece (the value is the flag's own);
    ``-czf`` splits into ``-c``, ``-z``, ``-f`` with real sub-spans.
    """
    if token.startswith("--"):
        return [(token, (0, len(token)))]
    if token.startswith("-") and len(token) > 2 and "=" not in token:
        return [(f"-{ch}", (i, i + 1))
                for i, ch in enumerate(token[1:], start=1)]
    return [(token, (0, len(token)))]


def _match_option(flag: str, page_options: dict[str, dict[str, Any]],
                  sibling_options: dict[str, dict[str, Any]],
                  sibling_page: str,
                  help_options: dict[str, dict[str, Any]],
                  tldr_options: dict[str, dict[str, Any]],
                  page_name: str) -> tuple[dict[str, Any] | None, bool,
                                           str | None]:
    """Match one flag against man (own page) → man (sibling page) →
    --help → tldr, in that order.

    Returns (doc, unknown, suggestion). A long flag that is an
    unambiguous prefix of exactly one documented long flag matches
    it (GNU abbreviation), named as such.
    """
    pools = [("man", page_options, page_name)]
    if sibling_options:
        pools.append(("man", sibling_options, sibling_page))
    pools.extend([("help", help_options, page_name),
                  ("tldr", tldr_options, page_name)])
    for source, pool, label in pools:
        if flag in pool:
            rec = pool[flag]
            return ({"source": source,
                     "page": label,
                     "excerpt": rec.get("excerpt", "")},
                    False, None)
    if flag.startswith("--") and len(flag) > 3:
        long_names = sorted({n for n in
                             list(page_options) + list(sibling_options)
                             if n.startswith(flag)})
        if len(long_names) == 1:
            rec = (page_options.get(long_names[0])
                   or sibling_options.get(long_names[0]) or {})
            return ({"source": "man", "page": page_name,
                     "excerpt": rec.get("excerpt", ""),
                     "note": f"abbreviation of {long_names[0]}"},
                    False, None)
    suggestion = None
    candidates = (list(page_options) + list(sibling_options)
                  + list(help_options) + list(tldr_options))
    close = difflib.get_close_matches(flag, candidates, n=1, cutoff=0.6)
    if close:
        suggestion = close[0]
    return None, True, suggestion


def explain_command(home: str | Path | None, command: str, *,
                    corpus: ManCorpus | None = None,
                    store: ExplanationStore | None = None,
                    llm_fn: Any = None,
                    help_fetcher: Any = None,
                    tldr_fetcher: Any = None) -> dict[str, Any]:
    """Explain a shell command, fragment by fragment.

    Never raises and never executes the command. Returns the
    Explanation dict: ok, explanation_id, command, fragments (each
    with span/token/role/detail/doc/unknown/suggestion), summary,
    plain_summary, unknown_count, grounded_count, corpus stats,
    reply, or error. Fragments with no grounded doc are marked
    unknown — never filled in silently.
    """
    text = (command or "").strip()
    if not text:
        return {"ok": False, "command": "", "fragments": [],
                "error": "explain needs a command"}
    if len(text) > _MAX_COMMAND_CHARS:
        return {"ok": False, "command": text, "fragments": [],
                "error": f"command too long ({len(text)} chars, "
                         f"max {_MAX_COMMAND_CHARS})"}
    own_corpus = corpus is None
    if own_corpus:
        corpus = open_man_corpus(home)
    assert corpus is not None
    try:
        if corpus.status()["pages"] == 0:
            corpus.ingest()
    except Exception as exc:  # noqa: BLE001 — empty corpus, not fatal
        _log.debug("man corpus auto-ingest failed: %s", exc)
    try:
        parsed = parse_bash(text)
    except Exception as exc:  # noqa: BLE001 — parse never raises, belt
        return {"ok": False, "command": text, "fragments": [],
                "error": f"could not parse that command: {exc}"}
    fragments: list[dict[str, Any]] = []
    grounded = 0
    unknown = 0
    # Per simple command, the command fragment's page drives option
    # matching; the explainer walks fragments in span order tracking
    # the current command context.
    current_page: Any = None
    current_page_name = ""
    current_command_name = ""
    current_help: dict[str, dict[str, Any]] = {}
    current_tldr: dict[str, dict[str, Any]] = {}
    help_loaded = False
    tldr_loaded = False
    pending_value_for: str | None = None
    command_unknown = False
    sibling_cache: dict[str,
                        tuple[str | None, dict[str, Any] | None]] = {}

    def _find_sibling(flag: str) -> (
            tuple[str | None, dict[str, Any] | None]):
        """(page, record) for a flag on a sibling man page, cached
        per command. iptables' --dport lives on
        iptables-extensions(8), not iptables(8)."""
        if flag in sibling_cache:
            return sibling_cache[flag]
        hit: tuple[str | None, dict[str, Any] | None] = (None, None)
        if corpus is not None:
            hit = corpus.find_option(flag, current_command_name)
        sibling_cache[flag] = hit
        return hit

    def _load_help_tldr(flag: str) -> None:
        """Harvest --help / tldr for the current command, once each.

        Fires only for a flag the man page does not document; the
        harvests are cached per command, so repeated calls are free.
        """
        nonlocal help_loaded, tldr_loaded, current_help, current_tldr
        if not help_loaded and help_fetcher is not None \
                and current_command_name:
            help_loaded = True
            try:
                help_text = help_fetcher(current_command_name)
            except Exception:  # noqa: BLE001 — harvest is a miss
                help_text = None
            if help_text:
                current_help = parse_help_text(
                    help_text, current_command_name)
        if flag not in current_help and not tldr_loaded \
                and tldr_fetcher is not None and current_command_name:
            tldr_loaded = True
            try:
                tldr_text = tldr_fetcher(current_command_name)
            except Exception:  # noqa: BLE001 — fetch is a miss
                tldr_text = None
            if tldr_text:
                current_tldr = parse_tldr_markdown(
                    tldr_text, current_command_name)

    for frag in parsed:
        if frag.role == "command":
            pending_value_for = None
            current_command_name = frag.token
            help_loaded = False
            tldr_loaded = False
            sibling_cache = {}
            page = corpus.get(frag.token)
            if page is not None:
                current_page = page
                current_page_name = f"{page.name}({page.section})"
                frag.doc = {"source": "man", "page": current_page_name,
                            "excerpt": page.summary or
                            f"the {page.name} command"}
                frag.unknown = False
                grounded += 1
            elif frag.token in BUILTIN_DOCS:
                current_page = None
                current_page_name = frag.token
                frag.doc = {"source": "builtin", "page": "bash builtins",
                            "excerpt": BUILTIN_DOCS[frag.token]}
                frag.unknown = False
                grounded += 1
            else:
                current_page = None
                current_page_name = frag.token
                frag.doc = None
                frag.unknown = True
                command_unknown = True
                unknown += 1
            current_help = {}
            current_tldr = {}
            fragments.append(frag.to_dict())
            continue
        if frag.role == "option" and pending_value_for is not None:
            # A value that looks like an option: find -mtime -7.
            # The previous option's record says it takes a value; if
            # this token is not a documented flag anywhere, it is
            # that value, not an option.
            _load_help_tldr(frag.token)
            sib_page, sib_rec = _find_sibling(frag.token)
            page_options_now = (current_page.options
                                if current_page is not None else {})
            if (frag.token not in page_options_now
                    and frag.token not in current_help
                    and frag.token not in current_tldr
                    and sib_rec is None):
                frag.role = "argument"
        if frag.role == "option":
            page_options = (current_page.options
                            if current_page is not None else {})
            token = frag.token
            pieces = _split_option_token(token)
            if (token.startswith("-") and not token.startswith("--")
                    and len(token) > 2):
                # Single-dash multi-letter options (find -name,
                # ps -aux) are one option, not a short-flag bundle —
                # check the whole token against every rung first.
                if token not in page_options:
                    _load_help_tldr(token)
                sib_page, sib_rec = _find_sibling(token)
                if (token in page_options or token in current_help
                        or token in current_tldr or sib_rec is not None):
                    pieces = [(token, (0, len(token)))]
            for flag, (rel_start, rel_end) in pieces:
                sub = Fragment(
                    span=(frag.span[0] + rel_start,
                          frag.span[0] + rel_end),
                    token=flag, role="option")
                # A flag the man page does not document falls through
                # to the sibling man page (iptables-extensions for
                # iptables), then to the --help rung (harvested once
                # per command), then to tldr — the rungs fire per
                # missing flag, not only when the whole page is
                # option-less.
                sibling_options: dict[str, dict[str, Any]] = {}
                sibling_page = ""
                if flag not in page_options:
                    sib_page, sib_rec = _find_sibling(flag)
                    if sib_rec is not None:
                        sibling_options = {flag: sib_rec}
                        sibling_page = sib_page or ""
                if flag not in page_options and not sibling_options:
                    _load_help_tldr(flag)
                doc, is_unknown, suggestion = _match_option(
                    flag, page_options, sibling_options, sibling_page,
                    current_help, current_tldr, current_page_name)
                sub.doc = doc
                sub.unknown = is_unknown
                sub.suggestion = suggestion
                if is_unknown:
                    unknown += 1
                else:
                    grounded += 1
                    rec = (page_options.get(flag)
                           or sibling_options.get(flag)
                           or current_help.get(flag)
                           or current_tldr.get(flag) or {})
                    if rec.get("takes_value"):
                        pending_value_for = flag
                fragments.append(sub.to_dict())
            continue
        if frag.role == "argument" and pending_value_for is not None:
            frag.detail = f"value of {pending_value_for}"
            pending_value_for = None
        fragments.append(frag.to_dict())
    if own_corpus:
        with contextlib.suppress(Exception):  # noqa: BLE001 — best-effort
            corpus.close()
    if command_unknown and not fragments:
        summary = "no man page for that command on this host"
    elif unknown:
        summary = (f"{unknown} fragment(s) have no documentation on "
                   f"this host — marked unknown, not guessed")
    elif current_page_name:
        summary = f"explained from {current_page_name}'s documentation"
    else:
        summary = "explained"
    plain_summary: str | None = None
    if llm_fn is not None:
        plain_summary = _plain_summary_llm(llm_fn, text, fragments)
    explanation: dict[str, Any] = {
        "ok": True,
        "explanation_id": secrets.token_hex(6),
        "command": text,
        "fragments": fragments,
        "summary": summary,
        "plain_summary": plain_summary,
        "unknown_count": unknown,
        "grounded_count": grounded,
        "corpus": corpus.status() if not own_corpus else {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    explanation["reply"] = format_explanation_reply(explanation)
    if store is None:
        try:
            store = open_explanation_store(home)
        except Exception:  # noqa: BLE001 — store is best-effort
            store = None
    if store is not None:
        try:
            store.save(explanation)
        except Exception as exc:  # noqa: BLE001 — store never breaks explain
            _log.debug("explanation store save failed: %s", exc)
    return explanation


def _plain_summary_llm(llm_fn: Any, command: str,
                       fragments: list[dict[str, Any]]) -> str | None:
    """A one-paragraph plain-language summary from grounded fragments.

    The model only ever sees grounded docs; if its summary names a
    flag that is not among the fragments, the summary is discarded —
    the extractive reply stands on its own.
    """
    grounded_lines = []
    for frag in fragments:
        doc = frag.get("doc") or {}
        if doc.get("excerpt"):
            grounded_lines.append(
                f"{frag['token']} ({frag['role']}): {doc['excerpt']}")
    if not grounded_lines:
        return None
    prompt = (
        "In one short paragraph, in plain words, say what this shell "
        f"command does. Use ONLY the documented fragments below; "
        f"never mention a flag or option that is not listed.\n"
        f"Command: {command}\n"
        + "\n".join(grounded_lines))
    try:
        out = str(llm_fn(prompt) or "").strip()
    except Exception:  # noqa: BLE001 — LLM rung is optional
        return None
    if not out:
        return None
    known = {f["token"] for f in fragments}
    known.update({f["token"].split("=")[0] for f in fragments})
    for flag in set(_LONG_FLAG_RE.findall(out) +
                    [f"-{s}" for s in _SHORT_FLAG_RE.findall(out)]):
        if flag not in known:
            return None
    return out[:1000]


def format_explanation_reply(explanation: dict[str, Any]) -> str:
    """Render an explanation as chat text.

    The command on top, then one line per documented fragment with
    its excerpt, then the unknowns named as unknowns. Never raises.
    """
    try:
        if not explanation.get("ok"):
            return explanation.get("error") or "could not explain that"
        lines = [f"$ {explanation['command']}"]
        if explanation.get("plain_summary"):
            lines.append("")
            lines.append(explanation["plain_summary"])
        fragments = explanation.get("fragments") or []
        for frag in fragments:
            token = frag.get("token", "")
            role = frag.get("role", "")
            doc = frag.get("doc") or {}
            if frag.get("unknown"):
                note = "no documentation on this host"
                if frag.get("suggestion"):
                    note += f" — did you mean {frag['suggestion']}?"
                lines.append(f"  ?: {token} — {note}")
                continue
            if role == "command":
                lines.append(
                    f"  {token} — {doc.get('excerpt', '')}"
                    f" ({doc.get('source', '')}"
                    f"{' ' + doc['page'] if doc.get('page') else ''})"
                    .replace(" ()", ""))
                continue
            if role == "option":
                excerpt = doc.get("excerpt", "")
                note = doc.get("note", "")
                line = f"  {token} — {excerpt}"
                if note:
                    line += f" ({note})"
                lines.append(line)
                continue
            if role == "argument":
                detail = frag.get("detail", "")
                lines.append(f"  {token} — {detail or 'argument'}")
                continue
            if role == "substitution":
                lines.append(f"  {token} — runs and captures its output")
                continue
            if role in ("pipe", "redirect", "separator", "prefix",
                        "assignment", "keyword", "unparsed"):
                detail = frag.get("detail", "")
                lines.append(f"  {token} — {detail or role}")
        unknowns = [f for f in fragments if f.get("unknown")]
        if unknowns and not any("unknown" in line.lower()
                                for line in lines[1:]):
            lines.append(
                f"unknown: {', '.join(f['token'] for f in unknowns)} — "
                "no documentation on this host")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 — reply never raises
        return f"explanation failed to render: {exc}"
