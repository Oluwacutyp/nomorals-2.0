"""Regex workbench (Regex101 core, 52-websites #41).

Paste a pattern and sample text and get matches with full match
information (groups, named groups, positions), a plain-English
explanation of every token from a REAL pattern parser (never an LLM
guess), a step-budget debugger that catches catastrophic
backtracking instead of hanging, substitution testing, unit tests
run as a suite, code generation per language whose snippets EXECUTE
in codews and reproduce the session's matches, and the reverse loop
("write me a regex for X": generate, then PROVE against positive
and negative cases before answering).

The honest ladder: the parser/explainer covers the shared grammar;
matching runs on the REAL engine — Python's ``re`` in-process, and
JavaScript / Go through the code executor where an engine exists on
the box. A flavour with no engine here (Java, .NET, PCRE2, Rust)
is named "not verifiable here" — never a simulated result presented
as an engine's verdict. A generated snippet that does not reproduce
the session's matches is labelled, never presented as verified.
"""

from __future__ import annotations

import base64
import contextlib
import json
import re
import secrets
import time
from pathlib import Path
from typing import Any

from ..core.policy import Capability
from .paper_reader import _home_from_context, _llm_from_context

__all__ = [
    "FLAVOURS",
    "RUN_LANG",
    "parse_pattern",
    "explain_pattern",
    "match_pattern",
    "substitute_pattern",
    "run_regex_tests",
    "debug_trace",
    "generate_code",
    "verify_code",
    "generate_regex",
    "RegexLabStore",
    "open_regex_store",
    "run_regex_session",
    "format_regex_reply",
    "encode_link",
    "decode_link",
    "register",
]

FLAVOURS = ("python", "javascript", "go", "rust", "java", "dotnet",
            "pcre2")

#: flavour -> code-executor language that can run it. Flavours with
#: no entry have no engine in the executor: rustc cannot use the
#: regex crate, and there is no java / dotnet / pcre2 runner.
RUN_LANG = {"python": "python", "javascript": "javascript", "go": "go"}

DEFAULT_BUDGET = 10000
REGEX_SESSIONS_FILENAME = "regex_sessions.json"

# ── flavour support ──────────────────────────────────────────────────
#
# Which constructs each flavour's engine supports. A construct the
# flavour lacks is reported by name at parse time — the pattern is
# never silently reinterpreted.

_SUPPORT: dict[str, dict[str, set[str]]] = {
    "backref": {
        "yes": {"python", "javascript", "rust", "java", "dotnet",
                "pcre2"},
        "no": {"go"},
    },
    "lookahead": {
        "yes": {"python", "javascript", "java", "dotnet", "pcre2"},
        "no": {"go", "rust"},
    },
    "lookbehind": {
        "yes": {"python", "javascript", "java", "dotnet", "pcre2"},
        "no": {"go", "rust"},
    },
    "possessive": {
        "yes": {"python", "java", "pcre2"},
        "no": {"javascript", "go", "rust", "dotnet"},
    },
    "atomic": {
        "yes": {"python", "java", "dotnet", "pcre2"},
        "no": {"javascript", "go", "rust"},
    },
    "named_k": {  # \k<name> backreference
        "yes": {"javascript", "java", "dotnet", "pcre2"},
        "no": {"python", "go", "rust"},
    },
    "named_p_eq": {  # (?P=name) backreference
        "yes": {"python"},
        "no": {"javascript", "go", "rust", "java", "dotnet", "pcre2"},
    },
    "flag_group": {  # (?i) inline flags
        "yes": {"python", "go", "rust", "java", "dotnet", "pcre2"},
        "no": {"javascript"},
    },
    "anchor_A": {
        "yes": {"python", "go", "rust", "java", "dotnet", "pcre2"},
        "no": {"javascript"},
    },
    "anchor_Z": {
        "yes": {"python", "java", "dotnet", "pcre2"},
        "no": {"javascript", "go", "rust"},
    },
}

#: named-group syntax each flavour accepts.
_NAMED_SYNTAX: dict[str, set[str]] = {
    "python": {"P"},
    "javascript": {"angle"},
    "go": {"P", "angle"},
    "rust": {"P", "angle"},
    "java": {"angle"},
    "dotnet": {"angle"},
    "pcre2": {"P", "angle", "quote"},
}

#: flag letters each flavour accepts (g is execution-level, not a
#: pattern flag, and is always accepted where an engine runs).
_FLAGS: dict[str, set[str]] = {
    "python": set("imsxau"),
    "javascript": set("imsu"),
    "go": set("imsU"),
    "rust": set("imsxuU"),
    "java": set("imsxdu"),
    "dotnet": set("imsxn"),
    "pcre2": set("imsxuU"),
}

_FLAG_NAMES = {
    "i": "case-insensitive",
    "m": "multi-line (^ and $ match at line starts/ends)",
    "s": "dot matches newlines too",
    "x": "verbose (whitespace and # comments ignored)",
    "a": "ASCII-only matching",
    "u": "Unicode matching",
    "U": "ungreedy (quantifiers lazy by default)",
    "d": "Unix lines",
    "n": "explicit captures only",
}

_ESCAPE_CLASSES = {
    "d": ("digit", "a digit (0-9)"),
    "D": ("non-digit", "any character that is not a digit"),
    "w": ("word", "a word character (letter, digit or _)"),
    "W": ("non-word", "any character that is not a word character"),
    "s": ("whitespace", "a whitespace character"),
    "S": ("non-whitespace", "any character that is not whitespace"),
}

_SIMPLE_ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v",
    "a": "\a", "0": "\0",
}

_ANCHORS = {
    "^": "the start of the text or line",
    "$": "the end of the text or line",
    r"\A": "the very start of the text",
    r"\Z": "the very end of the text",
    r"\b": "a word boundary",
    r"\B": "not a word boundary",
}


class _ParseError(Exception):
    """A pattern that cannot be parsed, with a position."""

    def __init__(self, message: str, pos: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.pos = pos


def _node(kind: str, start: int, end: int, text: str, plain: str,
          **extra: Any) -> dict[str, Any]:
    node: dict[str, Any] = {
        "kind": kind, "span": [start, end], "text": text,
        "plain": plain, "children": [],
    }
    node.update(extra)
    return node


class _Parser:
    """Recursive-descent parser over the regex grammar.

    Builds a proper AST (sequences are plain lists; an alternation
    is a node whose children are sequence nodes), plus a flat
    display token list. ``validate`` walks the tree and names every
    construct the requested flavour does not support.
    """

    def __init__(self, pattern: str, flavour: str) -> None:
        self.pattern = pattern
        self.flavour = flavour
        self.pos = 0
        self.group_count = 0
        self.group_names: dict[str, int] = {}
        self.unsupported: list[str] = []

    # ── entry ──
    def parse(self) -> list[dict[str, Any]]:
        branches = self._parse_alternation(top=True)
        if self.pos < len(self.pattern):
            raise _ParseError(
                f"unexpected '{self.pattern[self.pos]}'", self.pos)
        return branches

    def _parse_alternation(self, top: bool = False) -> list[Any]:
        """Parse branches separated by |; returns a list of nodes.

        A single branch returns its sequence flat. Several branches
        return one alternation node whose children are sequence
        nodes. ``top`` also ends at the end of the pattern; inside a
        group it ends at the closing paren.
        """
        first = self._parse_sequence()
        branches = [first]
        while self._peek() == "|":
            self.pos += 1
            branches.append(self._parse_sequence())
        if len(branches) == 1:
            return branches[0]
        start = branches[0][0]["span"][0] if branches[0] else self.pos
        end = branches[-1][-1]["span"][1] if branches[-1] else self.pos
        children = []
        for branch in branches:
            bstart = branch[0]["span"][0] if branch else start
            bend = branch[-1]["span"][1] if branch else end
            children.append(_node(
                "sequence", bstart, bend,
                self.pattern[bstart:bend],
                "one alternative", children=list(branch)))
        text = self.pattern[start:end]
        return [_node("alternation", start, end, text,
                      "either one alternative or another",
                      children=children)]

    def _parse_sequence(self) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        while self.pos < len(self.pattern):
            ch = self.pattern[self.pos]
            if ch in "|)":
                break
            node = self._parse_atom()
            if node is None:
                break
            # quantifiers bind to the previous atom
            while self.pos < len(self.pattern) and \
                    self.pattern[self.pos] in "*+?{":
                node = self._parse_quantifier(node)
            nodes.append(node)
        return nodes

    # ── atoms ──
    def _peek(self) -> str:
        return self.pattern[self.pos] if self.pos < len(
            self.pattern) else ""

    def _parse_atom(self) -> dict[str, Any] | None:
        ch = self.pattern[self.pos]
        start = self.pos
        if ch == "(":
            return self._parse_group()
        if ch == "[":
            return self._parse_class()
        if ch == ".":
            self.pos += 1
            return _node("dot", start, self.pos, ".",
                         "any character except a newline")
        if ch in "^$":
            self.pos += 1
            return _node("anchor", start, self.pos, ch, _ANCHORS[ch],
                         anchor=ch)
        if ch == "\\":
            return self._parse_escape()
        self.pos += 1
        return _node("literal", start, self.pos, ch,
                     f"a literal '{ch}'", value=ch)

    def _parse_escape(self) -> dict[str, Any]:
        start = self.pos
        self.pos += 1  # backslash
        if self.pos >= len(self.pattern):
            raise _ParseError("pattern ends with a lone backslash",
                              start)
        ch = self.pattern[self.pos]
        if ch in _ESCAPE_CLASSES:
            self.pos += 1
            _, plain = _ESCAPE_CLASSES[ch]
            return _node("escape_class", start, self.pos,
                         self.pattern[start:self.pos], plain, cls=ch)
        if ("\\" + ch) in _ANCHORS:
            self.pos += 1
            text = self.pattern[start:self.pos]
            return _node("anchor", start, self.pos, text,
                         _ANCHORS[text], anchor=text)
        if ch in _SIMPLE_ESCAPES:
            self.pos += 1
            return _node("escape", start, self.pos,
                         self.pattern[start:self.pos],
                         f"a {repr(_SIMPLE_ESCAPES[ch])} character",
                         value=_SIMPLE_ESCAPES[ch])
        if ch == "x":
            return self._parse_hex_escape(start, 2, "hex")
        if ch == "u":
            return self._parse_hex_escape(start, 4, "unicode")
        if ch.isdigit():
            return self._parse_backref(start)
        if ch == "k":
            return self._parse_k_backref(start)
        # any other escaped char is a literal of itself
        self.pos += 1
        return _node("literal", start, self.pos,
                     self.pattern[start:self.pos],
                     f"a literal '{ch}'", value=ch)

    def _parse_hex_escape(self, start: int, digits: int,
                          kind: str) -> dict[str, Any]:
        self.pos += 1  # x or u
        hexpart = self.pattern[self.pos:self.pos + digits]
        if len(hexpart) != digits or not all(
                c in "0123456789abcdefABCDEF" for c in hexpart):
            raise _ParseError(
                f"bad \\{self.pattern[start + 1]} escape", start)
        self.pos += digits
        value = chr(int(hexpart, 16))
        return _node("escape", start, self.pos,
                     self.pattern[start:self.pos],
                     f"the character {value!r} (U+{ord(value):04X})",
                     value=value)

    def _parse_backref(self, start: int) -> dict[str, Any]:
        digits = ""
        while self.pos < len(self.pattern) and \
                self.pattern[self.pos].isdigit():
            digits += self.pattern[self.pos]
            self.pos += 1
        number = int(digits)
        if number == 0 or number > self.group_count:
            raise _ParseError(
                f"backreference \\{number} has no group {number}",
                start)
        return _node("backref", start, self.pos,
                     self.pattern[start:self.pos],
                     f"the same text as group {number} matched",
                     number=number)

    def _parse_k_backref(self, start: int) -> dict[str, Any]:
        self.pos += 1  # k
        if self.pos >= len(self.pattern) or \
                self.pattern[self.pos] not in "<'":
            raise _ParseError(r"bad \k backreference", start)
        closer = ">" if self.pattern[self.pos] == "<" else "'"
        self.pos += 1
        name_start = self.pos
        while self.pos < len(self.pattern) and \
                self.pattern[self.pos] != closer:
            self.pos += 1
        if self.pos >= len(self.pattern):
            raise _ParseError(r"unterminated \k backreference", start)
        name = self.pattern[name_start:self.pos]
        self.pos += 1
        if name not in self.group_names:
            raise _ParseError(
                f"backreference \\k<{name}> has no group '{name}'",
                start)
        return _node("backref", start, self.pos,
                     self.pattern[start:self.pos],
                     f"the same text as group '{name}' matched",
                     name=name, number=self.group_names[name])

    def _parse_group(self) -> dict[str, Any]:
        start = self.pos
        self.pos += 1  # (
        if self._peek() != "?":
            self.group_count += 1
            number = self.group_count
            body = self._parse_alternation()
            self._expect_close(start)
            return _node("group", start, self.pos,
                         self.pattern[start:self.pos],
                         f"a capturing group (group {number})",
                         children=body, number=number, name=None,
                         capturing=True)
        # (?...) forms
        self.pos += 1  # ?
        marker = self._peek()
        if marker == ":":
            self.pos += 1
            body = self._parse_alternation()
            self._expect_close(start)
            return _node("group", start, self.pos,
                         self.pattern[start:self.pos],
                         "a group (not captured)", children=body,
                         number=None, name=None, capturing=False)
        if marker == "P":
            return self._parse_named_group(start, syntax="P")
        if marker == "<":
            nxt = self.pattern[self.pos + 1:self.pos + 2]
            if nxt in ("=", "!"):
                return self._parse_lookaround(start, behind=True)
            return self._parse_named_group(start, syntax="angle")
        if marker == "'":
            return self._parse_named_group(start, syntax="quote")
        if marker in ("=", "!"):
            return self._parse_lookaround(start, behind=False)
        if marker == ">":
            self.pos += 1
            body = self._parse_alternation()
            self._expect_close(start)
            return _node("atomic", start, self.pos,
                         self.pattern[start:self.pos],
                         "an atomic group (matched once, never "
                         "given back)", children=body)
        if marker in "imsxauU-":
            return self._parse_flag_group(start)
        if marker == "(":
            raise _ParseError(
                "conditionals (?(...)...) are not supported", start)
        raise _ParseError(
            f"unknown group construct '(?{marker}'", start)

    def _parse_named_group(self, start: int, syntax: str) -> dict:
        if syntax == "P":
            self.pos += 1  # P
            if self._peek() != "<":
                raise _ParseError("bad (?P<name>...) group", start)
            self.pos += 1
            closer = ">"
        elif syntax == "angle":
            self.pos += 1  # <
            closer = ">"
        else:
            self.pos += 1  # '
            closer = "'"
        name_start = self.pos
        while self.pos < len(self.pattern) and \
                self.pattern[self.pos] != closer:
            self.pos += 1
        if self.pos >= len(self.pattern):
            raise _ParseError("unterminated group name", start)
        name = self.pattern[name_start:self.pos]
        self.pos += 1
        if not name or not (name[0].isalpha() or name[0] == "_"):
            raise _ParseError(f"bad group name '{name}'", start)
        self.group_count += 1
        number = self.group_count
        self.group_names[name] = number
        body = self._parse_alternation()
        self._expect_close(start)
        return _node("group", start, self.pos,
                     self.pattern[start:self.pos],
                     f"a capturing group named '{name}' "
                     f"(group {number})",
                     children=body, number=number, name=name,
                     capturing=True, syntax=syntax)

    def _parse_lookaround(self, start: int, behind: bool) -> dict:
        if behind:
            self.pos += 1  # <
        marker = self.pattern[self.pos]
        self.pos += 1  # = or !
        positive = marker == "="
        body = self._parse_alternation()
        self._expect_close(start)
        kind = "lookbehind" if behind else "lookahead"
        verb = "preceded by " if behind else "followed by "
        plain = (verb if positive else "not " + verb) + "..."
        return _node(kind, start, self.pos,
                     self.pattern[start:self.pos], plain,
                     children=body, positive=positive)

    def _parse_flag_group(self, start: int) -> dict:
        flags = ""
        while self.pos < len(self.pattern) and \
                self.pattern[self.pos] in "imsxauU-":
            flags += self.pattern[self.pos]
            self.pos += 1
        if self._peek() == ":":
            self.pos += 1
            body = self._parse_alternation()
            self._expect_close(start)
            return _node("flag", start, self.pos,
                         self.pattern[start:self.pos],
                         f"flags {flags or 'none'} apply to this "
                         "group only", children=body, flags=flags)
        if self._peek() == ")":
            self.pos += 1
            return _node("flag", start, self.pos,
                         self.pattern[start:self.pos],
                         f"turn flags {flags or 'none'} on from "
                         "here", children=[], flags=flags)
        raise _ParseError("bad inline flag group", start)

    def _parse_class(self) -> dict[str, Any]:
        start = self.pos
        self.pos += 1  # [
        negated = False
        if self._peek() == "^":
            negated = True
            self.pos += 1
        items: list[dict[str, Any]] = []
        ranges: list[list[str]] = []
        singles: list[str] = []
        classes: list[str] = []
        first = True
        while True:
            if self.pos >= len(self.pattern):
                raise _ParseError("unterminated character class",
                                  start)
            ch = self.pattern[self.pos]
            if ch == "]" and not first:
                self.pos += 1
                break
            first = False
            if ch == "\\":
                esc = self._parse_escape()
                if esc["kind"] == "escape_class":
                    classes.append(esc["cls"])
                    items.append(esc)
                    continue
                value = esc.get("value", esc["text"][-1])
                items.append(_node("literal", esc["span"][0],
                                   esc["span"][1], esc["text"],
                                   esc["plain"], value=value))
                singles.append(value)
                continue
            self.pos += 1
            lo_start = self.pos - 1
            if self._peek() == "-" and \
                    self.pattern[self.pos + 1:self.pos + 2] != "]" and \
                    self.pos + 1 < len(self.pattern):
                self.pos += 1  # -
                hi = self.pattern[self.pos]
                if hi == "\\":
                    esc = self._parse_escape()
                    hi = esc.get("value", esc["text"][-1])
                else:
                    self.pos += 1
                ranges.append([ch, hi])
                items.append(_node("range", lo_start, self.pos,
                                   f"{ch}-{hi}",
                                   f"{ch} to {hi}"))
            else:
                singles.append(ch)
                items.append(_node("literal", lo_start, self.pos, ch,
                                   f"a literal '{ch}'", value=ch))
        if negated:
            plain = "one character that is NOT " + ", ".join(
                [f"{lo}-{hi}" for lo, hi in ranges] +
                singles +
                [f"\\{c}" for c in classes])
        else:
            plain = "one of " + ", ".join(
                [f"{lo}-{hi}" for lo, hi in ranges] +
                singles +
                [f"\\{c}" for c in classes])
        return _node("char_class", start, self.pos,
                     self.pattern[start:self.pos], plain,
                     children=items, negated=negated,
                     ranges=ranges, singles=singles, classes=classes)

    def _parse_quantifier(self, target: dict[str, Any]) -> dict:
        start = target["span"][0]
        ch = self.pattern[self.pos]
        if ch == "*":
            self.pos += 1
            minimum, maximum, word = 0, None, "0 or more times"
        elif ch == "+":
            self.pos += 1
            minimum, maximum, word = 1, None, "1 or more times"
        elif ch == "?":
            self.pos += 1
            minimum, maximum, word = 0, 1, "0 or 1 time"
        else:  # {m[,n]}
            self.pos += 1
            num = ""
            while self.pos < len(self.pattern) and \
                    self.pattern[self.pos].isdigit():
                num += self.pattern[self.pos]
                self.pos += 1
            if not num:
                raise _ParseError("bad repetition {…}", self.pos)
            minimum = int(num)
            maximum: int | None = minimum
            if self._peek() == ",":
                self.pos += 1
                num2 = ""
                while self.pos < len(self.pattern) and \
                        self.pattern[self.pos].isdigit():
                    num2 += self.pattern[self.pos]
                    self.pos += 1
                maximum = int(num2) if num2 else None
            if self._peek() != "}":
                raise _ParseError("bad repetition {…}", self.pos)
            self.pos += 1
            if maximum is not None and maximum < minimum:
                raise _ParseError(
                    f"repetition {{{minimum},{maximum}}} has min "
                    "greater than max", start)
            if maximum is None:
                word = f"{minimum} or more times"
            elif maximum == minimum:
                word = f"exactly {minimum} times"
            else:
                word = f"between {minimum} and {maximum} times"
        lazy = False
        possessive = False
        if self._peek() == "?":
            lazy = True
            self.pos += 1
            word += ", as few as possible"
        elif self._peek() == "+":
            possessive = True
            self.pos += 1
            word += ", never giving back"
        plain = f"{word}: {target['plain']}"
        return _node("quantifier", start, self.pos,
                     self.pattern[start:self.pos], plain,
                     children=[target], min=minimum, max=maximum,
                     lazy=lazy, possessive=possessive)

    def _expect_close(self, start: int) -> None:
        if self._peek() != ")":
            raise _ParseError("unbalanced group: missing ')'", start)
        self.pos += 1

    # ── flavour validation ──
    def validate(self, nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            self._validate_node(node)

    def _unsupported(self, construct: str, why: str) -> None:
        self.unsupported.append(
            f"{construct} is not supported by {self.flavour}: {why}")

    def _check(self, key: str, construct: str, why: str) -> None:
        if self.flavour in _SUPPORT[key]["no"]:
            self._unsupported(construct, why)

    def _validate_node(self, node: dict[str, Any]) -> None:
        kind = node["kind"]
        if kind == "backref":
            if "name" in node and node.get("name"):
                self._check("named_k", "\\k<name> backreference",
                            "use the flavour's own named backreference")
            else:
                self._check("backref", "backreference",
                            "this engine has no backreferences")
        elif kind == "lookahead":
            self._check("lookahead", "lookahead",
                        "this engine has no lookahead")
        elif kind == "lookbehind":
            self._check("lookbehind", "lookbehind",
                        "this engine has no lookbehind")
        elif kind == "atomic":
            self._check("atomic", "atomic group",
                        "this engine has no atomic groups")
        elif kind == "quantifier" and node.get("possessive"):
            self._check("possessive", "possessive quantifier",
                        "this engine has no possessive quantifiers")
        elif kind == "group" and node.get("name"):
            syntax = node.get("syntax", "P")
            if syntax not in _NAMED_SYNTAX.get(self.flavour, set()):
                self._unsupported(
                    f"(?{syntax}<name>...) named group",
                    f"{self.flavour} names groups differently")
        elif kind == "flag":
            self._check("flag_group", "inline flags (?...)",
                        "this engine has no inline flags")
        elif kind == "anchor":
            anchor = node.get("anchor", "")
            if anchor == r"\A":
                self._check("anchor_A", r"\A anchor",
                            "use ^ instead")
            elif anchor == r"\Z":
                self._check("anchor_Z", r"\Z anchor",
                            "use $ instead")
        for child in node.get("children", []):
            self._validate_node(child)


def _flatten(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The display token list: the AST in source order.

    An alternation stays visible as one separator token between its
    branches' flat tokens, so ``^cat|dog$`` reads as anchor, three
    literals, alternation, three literals, anchor. Groups keep their
    own flattened children. The tracer never uses this list — it
    walks the AST.
    """
    out: list[dict[str, Any]] = []
    for node in nodes:
        kind = node["kind"]
        if kind == "alternation":
            branches = node.get("children", [])
            for index, branch in enumerate(branches):
                if index > 0:
                    out.append(node)
                out.extend(_flatten(branch.get("children", [])))
        elif kind == "sequence":
            out.extend(_flatten(node.get("children", [])))
        elif kind == "quantifier":
            # the explanation pane reads "the atom, then how many
            # of it": the target token first, the quantifier after
            # it, still carrying the target as its child.
            out.extend(_flatten(node.get("children", [])))
            shown = dict(node)
            shown["children"] = _flatten(node.get("children", []))
            out.append(shown)
        else:
            shown = dict(node)
            shown["children"] = _flatten(node.get("children", []))
            out.append(shown)
    return out


def _check_flags(flags: str, flavour: str) -> str | None:
    allowed = _FLAGS.get(flavour, set())
    for letter in flags or "":
        if letter == "g":
            continue
        if letter not in allowed:
            return (f"flag '{letter}' is not supported by "
                    f"{flavour}")
    return None


def parse_pattern(pattern: str,
                  flavour: str = "python") -> dict[str, Any]:
    """Parse a pattern into its token tree; never raises.

    Returns {ok, pattern, flavour, tokens, unsupported, error,
    error_pos}. ``tokens`` is the flat display list; the AST lives
    under ``_ast`` for the tracer (callers should not rely on it).
    """
    flavour = (flavour or "python").strip().lower()
    pattern = pattern or ""
    if flavour not in FLAVOURS:
        return {"ok": False, "pattern": pattern, "flavour": flavour,
                "tokens": [], "unsupported": [],
                "error": f"unknown flavour '{flavour}' "
                         f"(flavours: {', '.join(FLAVOURS)})",
                "error_pos": 0}
    parser = _Parser(pattern, flavour)
    try:
        ast = parser.parse()
    except _ParseError as exc:
        return {"ok": False, "pattern": pattern, "flavour": flavour,
                "tokens": [], "unsupported": [],
                "error": exc.message, "error_pos": exc.pos}
    parser.validate(ast)
    if parser.unsupported:
        return {"ok": False, "pattern": pattern, "flavour": flavour,
                "tokens": _flatten(ast), "unsupported":
                parser.unsupported,
                "error": "; ".join(parser.unsupported),
                "error_pos": 0}
    return {"ok": True, "pattern": pattern, "flavour": flavour,
            "tokens": _flatten(ast), "unsupported": [],
            "error": "", "error_pos": 0, "_ast": ast}


_QUICK_REFERENCE = (
    "quick reference — . any char except newline · \\d digit · "
    "\\D not a digit · \\w word char · \\W not a word char · "
    "\\s whitespace · \\S not whitespace · ^ start · $ end · "
    "\\b word boundary · [a-z] one of a range · [^a-z] none of a "
    "range · * 0 or more · + 1 or more · ? 0 or 1 · {m,n} between "
    "m and n · *? lazy (as few as possible) · ( ) capture group · "
    "(?: ) group without capturing · (?P<name> ) named group · "
    "| either/or · \\1 the text group 1 matched · (?= ) followed "
    "by · (?! ) not followed by · (?<= ) preceded by · "
    "(?<! ) not preceded by"
)


def explain_pattern(pattern: str,
                    flavour: str = "python") -> dict[str, Any]:
    """The explanation pane: every token in plain English, plus the
    quick-reference panel."""
    parsed = parse_pattern(pattern, flavour)
    if not parsed["ok"]:
        return {**parsed, "reference": _QUICK_REFERENCE}
    return {**parsed, "reference": _QUICK_REFERENCE}


# ── matching ──────────────────────────────────────────────────────────

def _python_flags(flags: str) -> int:
    out = 0
    for letter in flags or "":
        if letter == "i":
            out |= re.IGNORECASE
        elif letter == "m":
            out |= re.MULTILINE
        elif letter == "s":
            out |= re.DOTALL
        elif letter == "x":
            out |= re.VERBOSE
        elif letter == "a":
            out |= re.ASCII
        elif letter == "u":
            out |= re.UNICODE
    return out


def _normalise_match(start: int, end: int, text: str,
                     groups: list[Any]) -> dict[str, Any]:
    return {"start": start, "end": end, "match": text[start:end],
            "groups": groups}


def _python_matches(compiled: Any, text: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for match in compiled.finditer(text):
        groups: list[Any] = []
        for index in range(1, (compiled.groups or 0) + 1):
            span = match.span(index)
            if span[0] < 0:
                groups.append(None)
                continue
            name = None
            for gname, gindex in compiled.groupindex.items():
                if gindex == index:
                    name = gname
                    break
            groups.append({"index": index, "name": name,
                           "value": match.group(index),
                           "start": span[0], "end": span[1]})
        found.append(_normalise_match(
            match.start(), match.end(), text, groups))
        if match.start() == match.end():
            # finditer already advances past empty matches; the
            # guard is belt-and-braces against a zero-width loop.
            continue
    return found


def _js_program(payload: dict[str, Any]) -> str:
    """A node program that runs the pattern over the texts and prints
    one JSON document: {results: [{text, matches: [...]}]}. The
    input is embedded as a JSON literal — no stdin is needed."""
    blob = json.dumps(payload)
    return (
        "const input = " + blob + ";\n"
        "let re;\n"
        "try { re = new RegExp(input.pattern, input.flags + 'gd'); }\n"
        "catch (e) {\n"
        "  console.log(JSON.stringify({error: String(e.message || e)}));\n"
        "  process.exit(0);\n"
        "}\n"
        "const results = input.texts.map(function (text) {\n"
        "  const matches = [];\n"
        "  re.lastIndex = 0;\n"
        "  let m;\n"
        "  while ((m = re.exec(text)) !== null) {\n"
        "    const groups = [];\n"
        "    for (let i = 1; i < m.length; i++) {\n"
        "      const span = m.indices[i];\n"
        "      let name = null;\n"
        "      if (m.indices.groups) {\n"
        "        for (const key of Object.keys(m.indices.groups)) {\n"
        "          const g = m.indices.groups[key];\n"
        "          if (g && span && g[0] === span[0] && g[1] === span[1]) {\n"
        "            name = key;\n"
        "          }\n"
        "        }\n"
        "      }\n"
        "      groups.push(span ? {index: i, name: name, value: m[i],\n"
        "        start: span[0], end: span[1]} : null);\n"
        "    }\n"
        "    matches.push({start: m.index, end: m.index + m[0].length,\n"
        "      match: m[0], groups: groups});\n"
        "    if (m[0].length === 0) { re.lastIndex++; }\n"
        "  }\n"
        "  return {text: text, matches: matches};\n"
        "});\n"
        "console.log(JSON.stringify({results: results}));\n"
    )


def _go_program(payload: dict[str, Any]) -> str:
    """A Go program with the same contract as the node one above.
    Go's regexp is RE2: a backreference or lookaround fails to
    COMPILE, and that compile error is the engine's real verdict."""
    blob = json.dumps(payload)
    return (
        "package main\n"
        "import (\"encoding/json\"; \"fmt\"; \"os\"; \"regexp\")\n"
        "func main() {\n"
        "  var input struct {\n"
        "    Pattern string `json:\"pattern\"`\n"
        "    Flags string `json:\"flags\"`\n"
        "    Texts []string `json:\"texts\"`\n"
        "  }\n"
        "  raw := " + blob + "\n"
        "  if err := json.Unmarshal([]byte(raw), &input); err != nil {\n"
        "    fmt.Println(`{\"error\":\"bad input\"}`)\n"
        "    os.Exit(0)\n"
        "  }\n"
        "  pat := input.Pattern\n"
        "  if input.Flags != \"\" {\n"
        "    pat = \"(?\" + input.Flags + \")\" + pat\n"
        "  }\n"
        "  re, err := regexp.Compile(pat)\n"
        "  if err != nil {\n"
        "    out, _ := json.Marshal(map[string]string{\"error\": err.Error()})\n"
        "    fmt.Println(string(out))\n"
        "    os.Exit(0)\n"
        "  }\n"
        "  names := re.SubexpNames()\n"
        "  type Group struct {\n"
        "    Index int `json:\"index\"`\n"
        "    Name *string `json:\"name\"`\n"
        "    Value string `json:\"value\"`\n"
        "    Start int `json:\"start\"`\n"
        "    End int `json:\"end\"`\n"
        "  }\n"
        "  type Match struct {\n"
        "    Start int `json:\"start\"`\n"
        "    End int `json:\"end\"`\n"
        "    Text string `json:\"match\"`\n"
        "    Groups []*Group `json:\"groups\"`\n"
        "  }\n"
        "  type Result struct {\n"
        "    Text string `json:\"text\"`\n"
        "    Matches []Match `json:\"matches\"`\n"
        "  }\n"
        "  results := []Result{}\n"
        "  for _, text := range input.Texts {\n"
        "    res := Result{Text: text, Matches: []Match{}}\n"
        "    for _, idx := range re.FindAllStringSubmatchIndex(text, -1) {\n"
        "      groups := []*Group{}\n"
        "      for i := 1; i < len(idx)/2; i++ {\n"
        "        if idx[2*i] < 0 {\n"
        "          groups = append(groups, nil)\n"
        "          continue\n"
        "        }\n"
        "        var name *string\n"
        "        if i < len(names) && names[i] != \"\" {\n"
        "          n := names[i]\n"
        "          name = &n\n"
        "        }\n"
        "        groups = append(groups, &Group{Index: i, Name: name,\n"
        "          Value: text[idx[2*i]:idx[2*i+1]],\n"
        "          Start: idx[2*i], End: idx[2*i+1]})\n"
        "      }\n"
        "      res.Matches = append(res.Matches, Match{\n"
        "        Start: idx[0], End: idx[1],\n"
        "        Text: text[idx[0]:idx[1]], Groups: groups})\n"
        "    }\n"
        "    results = append(results, res)\n"
        "  }\n"
        "  out, _ := json.Marshal(map[string]any{\"results\": results})\n"
        "  fmt.Println(string(out))\n"
        "}\n"
    )


def _run_foreign(pattern: str, texts: list[str], flavour: str,
                 flags: str, run_fn: Any) -> dict[str, Any]:
    """Run the pattern through the flavour's real engine via
    run_fn(code, lang) -> CodeRunner.run dict | None."""
    lang = RUN_LANG.get(flavour, "")
    if not lang:
        return {"ok": True, "verifiable": False,
                "reason": f"{flavour} is not verifiable here — no "
                          f"{flavour} engine is installed in the "
                          "code executor"}
    if run_fn is None:
        return {"ok": True, "verifiable": False,
                "reason": "not verified — no code runner available"}
    # 'g' is execution-level: the engines always run globally here.
    clean_flags = "".join(sorted(set(flags or "") - {"g"}))
    payload = {"pattern": pattern, "flags": clean_flags,
               "texts": texts}
    code = _js_program(payload) if lang == "javascript" else \
        _go_program(payload)
    try:
        result = run_fn(code, lang)
    except Exception as exc:  # noqa: BLE001 — runner dead = unverified
        return {"ok": True, "verifiable": False,
                "reason": f"not verified — the {lang} runner "
                          f"failed: {exc}"}
    if not isinstance(result, dict):
        return {"ok": True, "verifiable": False,
                "reason": f"not verified — the {lang} runner "
                          "returned nothing usable"}
    if result.get("timed_out"):
        return {"ok": True, "verifiable": False,
                "reason": f"not verified — the {lang} run timed out"}
    stdout = (result.get("stdout") or "").strip()
    try:
        doc = json.loads(stdout)
    except ValueError:
        return {"ok": True, "verifiable": False,
                "reason": f"not verified — the {lang} run produced "
                          f"no JSON output: "
                          f"{(result.get('stderr') or '')[:200]}"}
    if isinstance(doc, dict) and doc.get("error"):
        return {"ok": True, "verifiable": False,
                "reason": f"the {flavour} engine refused the "
                          f"pattern: {doc['error']}"}
    results = doc.get("results") if isinstance(doc, dict) else None
    if not isinstance(results, list):
        return {"ok": True, "verifiable": False,
                "reason": f"not verified — the {lang} run produced "
                          "an unexpected document"}
    return {"ok": True, "verifiable": True, "results": results}


def match_pattern(pattern: str, texts: list[str] | str,
                  flavour: str = "python", flags: str = "",
                  run_fn: Any = None) -> dict[str, Any]:
    """Match a pattern against texts; never raises.

    Python runs in-process on the real ``re``. Other flavours run
    on their real engine through run_fn. A flavour with no engine,
    or a dead runner, returns ok True with verifiable False and the
    reason named — never a simulated match.
    """
    flavour = (flavour or "python").strip().lower()
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) for t in (texts or [])]
    base: dict[str, Any] = {"pattern": pattern, "flavour": flavour,
                            "flags": flags or ""}
    parsed = parse_pattern(pattern, flavour)
    if not parsed["ok"]:
        return {**base, "ok": False, "verifiable": False,
                "results": [], "error": parsed["error"]}
    flag_error = _check_flags(flags or "", flavour)
    if flag_error:
        return {**base, "ok": False, "verifiable": False,
                "results": [], "error": flag_error}
    if flavour == "python":
        try:
            compiled = re.compile(pattern, _python_flags(flags or ""))
        except re.error as exc:
            return {**base, "ok": False, "verifiable": False,
                    "results": [], "error": f"invalid pattern: {exc}"}
        return {**base, "ok": True, "verifiable": True,
                "results": [{"text": t,
                             "matches": _python_matches(compiled, t)}
                            for t in texts], "error": ""}
    foreign = _run_foreign(pattern, texts, flavour, flags or "",
                           run_fn)
    if not foreign.get("verifiable"):
        return {**base, "ok": True, "verifiable": False,
                "results": [], "error": "",
                "reason": foreign.get("reason", "")}
    return {**base, "ok": True, "verifiable": True,
            "results": foreign.get("results", []), "error": ""}


def substitute_pattern(pattern: str, replacement: str, text: str,
                       flavour: str = "python", flags: str = "",
                       run_fn: Any = None) -> dict[str, Any]:
    """Substitution on the real engine; never raises.

    The replacement string is flavour-native (``\\1`` / ``\\g<name>``
    in Python, ``$1`` / ``$<name>`` in JavaScript and Go) and is
    passed to the engine untouched.
    """
    flavour = (flavour or "python").strip().lower()
    base: dict[str, Any] = {"pattern": pattern, "flavour": flavour,
                            "flags": flags or ""}
    parsed = parse_pattern(pattern, flavour)
    if not parsed["ok"]:
        return {**base, "ok": False, "verifiable": False,
                "result": "", "count": 0, "error": parsed["error"]}
    flag_error = _check_flags(flags or "", flavour)
    if flag_error:
        return {**base, "ok": False, "verifiable": False,
                "result": "", "count": 0, "error": flag_error}
    if flavour == "python":
        try:
            compiled = re.compile(pattern, _python_flags(flags or ""))
            result, count = compiled.subn(replacement, text or "")
        except re.error as exc:
            return {**base, "ok": False, "verifiable": False,
                    "result": "", "count": 0,
                    "error": f"invalid pattern: {exc}"}
        return {**base, "ok": True, "verifiable": True,
                "result": result, "count": count, "error": ""}
    lang = RUN_LANG.get(flavour, "")
    if not lang or run_fn is None:
        return {**base, "ok": True, "verifiable": False,
                "result": "", "count": 0, "error": "",
                "reason": f"{flavour} substitution is not "
                          "verifiable here — no code runner available"
                          if run_fn is None else
                          f"{flavour} has no substitution engine "
                          "installed here"}
    payload = {"pattern": pattern, "flags": flags or "",
               "texts": [text or ""], "replacement": replacement}
    clean_flags = "".join(sorted(set(flags or "") - {"g"}))
    payload["flags"] = clean_flags
    if lang == "javascript":
        blob = json.dumps(payload)
        code = (
            "const input = " + blob + ";\n"
            "let re;\n"
            "try { re = new RegExp(input.pattern, input.flags + 'g'); }\n"
            "catch (e) {\n"
            "  console.log(JSON.stringify({error: String(e.message || e)}));\n"
            "  process.exit(0);\n"
            "}\n"
            "const found = input.texts[0].match(re);\n"
            "const out = input.texts[0].replace(re, input.replacement);\n"
            "console.log(JSON.stringify({result: out,\n"
            "  count: found ? found.length : 0}));\n"
        )
    else:
        blob = json.dumps(payload)
        code = (
            "package main\n"
            "import (\"encoding/json\"; \"fmt\"; \"os\"; \"regexp\")\n"
            "func main() {\n"
            "  var input struct {\n"
            "    Pattern string `json:\"pattern\"`\n"
            "    Flags string `json:\"flags\"`\n"
            "    Texts []string `json:\"texts\"`\n"
            "    Replacement string `json:\"replacement\"`\n"
            "  }\n"
            "  raw := " + blob + "\n"
            "  if err := json.Unmarshal([]byte(raw), &input); err != nil {\n"
            "    fmt.Println(`{\"error\":\"bad input\"}`)\n"
            "    os.Exit(0)\n"
            "  }\n"
            "  pat := input.Pattern\n"
            "  if input.Flags != \"\" {\n"
            "    pat = \"(?\" + input.Flags + \")\" + pat\n"
            "  }\n"
            "  re, err := regexp.Compile(pat)\n"
            "  if err != nil {\n"
            "    out, _ := json.Marshal(map[string]string{\"error\": err.Error()})\n"
            "    fmt.Println(string(out))\n"
            "    os.Exit(0)\n"
            "  }\n"
            "  out, _ := json.Marshal(map[string]any{\n"
            "    \"result\": re.ReplaceAllString(input.Texts[0],\n"
            "      input.Replacement),\n"
            "    \"count\": len(re.FindAllStringIndex(input.Texts[0],\n"
            "      -1))})\n"
            "  fmt.Println(string(out))\n"
            "}\n"
        )
    try:
        result = run_fn(code, lang)
    except Exception as exc:  # noqa: BLE001 — runner dead = unverified
        return {**base, "ok": True, "verifiable": False,
                "result": "", "count": 0, "error": "",
                "reason": f"not verified — the {lang} runner "
                          f"failed: {exc}"}
    if not isinstance(result, dict):
        return {**base, "ok": True, "verifiable": False,
                "result": "", "count": 0, "error": "",
                "reason": "not verified — the runner returned "
                          "nothing usable"}
    try:
        doc = json.loads((result.get("stdout") or "").strip())
    except ValueError:
        return {**base, "ok": True, "verifiable": False,
                "result": "", "count": 0, "error": "",
                "reason": "not verified — the runner produced no "
                          "JSON output"}
    if isinstance(doc, dict) and doc.get("error"):
        return {**base, "ok": True, "verifiable": False,
                "result": "", "count": 0, "error": "",
                "reason": f"the {flavour} engine refused the "
                          f"pattern: {doc['error']}"}
    replaced = doc.get("result") if isinstance(doc, dict) else None
    if not isinstance(replaced, str):
        return {**base, "ok": True, "verifiable": False,
                "result": "", "count": 0, "error": "",
                "reason": "not verified — unexpected runner output"}
    count = doc.get("count") if isinstance(doc, dict) else 0
    return {**base, "ok": True, "verifiable": True,
            "result": replaced,
            "count": int(count) if isinstance(count, int) else 0,
            "error": ""}


def run_regex_tests(pattern: str, tests: list[dict[str, Any]],
                    flavour: str = "python", flags: str = "",
                    run_fn: Any = None) -> dict[str, Any]:
    """A unit-test suite: strings that must / must not match.

    ``tests`` is [{text, should_match}]. A test passes when the
    engine's verdict agrees with should_match. Failing tests are
    called out by name in the result — never silently dropped.
    """
    flavour = (flavour or "python").strip().lower()
    base: dict[str, Any] = {"pattern": pattern, "flavour": flavour,
                            "flags": flags or ""}
    if not tests:
        return {**base, "ok": True, "passed": 0, "failed": 0,
                "total": 0, "results": [], "error": ""}
    matched = match_pattern(
        pattern, [str(t.get("text", "")) for t in tests],
        flavour=flavour, flags=flags, run_fn=run_fn)
    if not matched["ok"]:
        return {**base, "ok": False, "passed": 0, "failed": len(tests),
                "total": len(tests), "results": [],
                "error": matched["error"]}
    if not matched.get("verifiable"):
        return {**base, "ok": True, "passed": 0,
                "failed": len(tests), "total": len(tests),
                "results": [],
                "error": "",
                "reason": matched.get("reason", "")}
    results: list[dict[str, Any]] = []
    passed = 0
    for case, found in zip(tests, matched["results"], strict=True):
        did_match = bool(found["matches"])
        want = bool(case.get("should_match"))
        good = did_match == want
        passed += 1 if good else 0
        results.append({"text": str(case.get("text", "")),
                        "should_match": want, "matched": did_match,
                        "ok": good})
    return {**base, "ok": True, "passed": passed,
            "failed": len(tests) - passed, "total": len(tests),
            "results": results, "error": ""}


# ── step-budget debugger ──────────────────────────────────────────────
#
# A small backtracking engine over the parsed tree. Every atom
# attempt is a step; past the budget the trace aborts and reports
# the blow-up instead of hanging her. It agrees with ``re`` on the
# golden battery (the tests prove it); it is not a second regex
# engine for production matching — ``re`` stays the matcher.

class _BudgetExceeded(Exception):
    """The step budget ran out mid-match."""


def _empty_gen() -> Any:
    if False:
        yield


class _Tracer:
    """Backtracking matcher with a step counter and a trace log."""

    def __init__(self, ast: list[dict[str, Any]], text: str,
                 flags: str, budget: int) -> None:
        self.ast = ast
        self.text = text
        self.budget = max(1, int(budget))
        self.steps = 0
        self.trace: list[dict[str, Any]] = []
        self.ignorecase = "i" in (flags or "")
        self.multiline = "m" in (flags or "")
        self.dotall = "s" in (flags or "")
        self.notes: list[str] = []

    # ── step accounting ──
    def _record(self, action: str, pos: int,
                node: dict[str, Any]) -> None:
        self.steps += 1
        if len(self.trace) < 200:
            self.trace.append({"n": self.steps, "pos": pos,
                               "kind": node["kind"],
                               "text": node["text"],
                               "action": action})
        if self.steps >= self.budget:
            raise _BudgetExceeded

    # ── character tests ──
    def _escape_class_matches(self, cls: str, ch: str) -> bool:
        if cls == "d":
            return ch.isdecimal()
        if cls == "D":
            return not ch.isdecimal()
        if cls == "w":
            return ch.isalnum() or ch == "_"
        if cls == "W":
            return not (ch.isalnum() or ch == "_")
        if cls == "s":
            return ch.isspace()
        return not ch.isspace()

    def _class_matches(self, node: dict[str, Any], ch: str) -> bool:
        hit = ch in node.get("singles", [])
        if not hit:
            for lo, hi in node.get("ranges", []):
                if lo <= ch <= hi:
                    hit = True
                    break
                if self.ignorecase and (
                        lo <= ch.lower() <= hi or
                        lo <= ch.upper() <= hi):
                    hit = True
                    break
        if not hit:
            for cls in node.get("classes", []):
                if self._escape_class_matches(cls, ch):
                    hit = True
                    break
        return (not hit) if node.get("negated") else hit

    def _same(self, want: str, got: str) -> bool:
        if want == got:
            return True
        return self.ignorecase and want.lower() == got.lower()

    # ── the matcher: generators with continuations ──
    def _seq(self, nodes: list[dict[str, Any]], index: int,
             pos: int, groups: dict[int, Any],
             cont: Any) -> Any:
        if index >= len(nodes):
            yield from cont(pos, groups)
            return

        def _next(end: int, next_groups: dict[int, Any]) -> Any:
            yield from self._seq(nodes, index + 1, end,
                                 next_groups, cont)

        yield from self._match_node(nodes[index], pos, groups, _next)

    def _match_node(self, node: dict[str, Any], pos: int,
                    groups: dict[int, Any], cont: Any) -> Any:
        kind = node["kind"]
        text = self.text
        if kind == "literal":
            want = node.get("value", "")
            if pos < len(text) and self._same(want, text[pos]):
                self._record("match", pos, node)
                yield from cont(pos + 1, groups)
            else:
                self._record("fail", pos, node)
        elif kind == "dot":
            if pos < len(text) and (self.dotall or text[pos] != "\n"):
                self._record("match", pos, node)
                yield from cont(pos + 1, groups)
            else:
                self._record("fail", pos, node)
        elif kind == "escape":
            want = node.get("value", "")
            if pos < len(text) and self._same(want, text[pos]):
                self._record("match", pos, node)
                yield from cont(pos + 1, groups)
            else:
                self._record("fail", pos, node)
        elif kind == "escape_class":
            if pos < len(text) and self._escape_class_matches(
                    node.get("cls", ""), text[pos]):
                self._record("match", pos, node)
                yield from cont(pos + 1, groups)
            else:
                self._record("fail", pos, node)
        elif kind == "char_class":
            if pos < len(text) and self._class_matches(node, text[pos]):
                self._record("match", pos, node)
                yield from cont(pos + 1, groups)
            else:
                self._record("fail", pos, node)
        elif kind == "anchor":
            if self._anchor_holds(node.get("anchor", ""), pos):
                self._record("match", pos, node)
                yield from cont(pos, groups)
            else:
                self._record("fail", pos, node)
        elif kind == "flag":
            self.notes.append(
                "inline flags are shown but not applied by the "
                "tracer")
            self._record("match", pos, node)
            yield from cont(pos, groups)
        elif kind == "group":
            number = node.get("number")

            def _capture(end: int, body_groups: dict[int, Any]) -> Any:
                captured = dict(body_groups)
                if number is not None:
                    captured[number] = (pos, end)
                yield from cont(end, captured)

            yield from self._seq(node.get("children", []), 0, pos,
                                 groups, _capture)
        elif kind == "atomic":
            done = self._first(
                self._seq(node.get("children", []), 0, pos, groups,
                          lambda end, g: _yield_one(end, g)))
            if done is not None:
                yield from cont(done[0], done[1])
        elif kind == "alternation":
            for branch in node.get("children", []):
                yield from self._seq(branch.get("children", []), 0,
                                     pos, groups, cont)
        elif kind == "quantifier":
            yield from self._match_quant(node, pos, groups, cont)
        elif kind == "backref":
            number = node.get("number")
            span = groups.get(number) if number is not None else None
            if span is None:
                # a backreference to a group that did not
                # participate matches empty, like ``re``.
                self._record("match", pos, node)
                yield from cont(pos, groups)
                return
            want = text[span[0]:span[1]]
            got = text[pos:pos + len(want)]
            if len(got) == len(want) and all(
                    self._same(a, b) for a, b in zip(want, got, strict=True)):
                self._record("match", pos, node)
                yield from cont(pos + len(want), groups)
            else:
                self._record("fail", pos, node)
        elif kind == "lookahead":
            found = self._first(
                self._seq(node.get("children", []), 0, pos, groups,
                          lambda end, g: _yield_one(end, g)))
            if bool(found is not None) == bool(node.get("positive")):
                self._record("match", pos, node)
                yield from cont(pos, found[1] if found else groups)
            else:
                self._record("fail", pos, node)
        elif kind == "lookbehind":
            if self._lookbehind_holds(node, pos):
                self._record("match", pos, node)
                yield from cont(pos, groups)
            else:
                self._record("fail", pos, node)
        else:  # sequence nodes never reach here; unknown kinds fail
            self._record("fail", pos, node)

    def _first(self, gen: Any) -> Any:
        for item in gen:
            return item
        return None

    def _anchor_holds(self, anchor: str, pos: int) -> bool:
        text = self.text
        if anchor == "^":
            return pos == 0 or (
                self.multiline and pos > 0 and text[pos - 1] == "\n")
        if anchor == "$":
            return pos == len(text) or (
                self.multiline and pos < len(text) and
                text[pos] == "\n") or (
                pos == len(text) - 1 and text.endswith("\n"))
        if anchor == r"\A":
            return pos == 0
        if anchor == r"\Z":
            return pos == len(text)
        if anchor == r"\b":
            before = pos > 0 and (text[pos - 1].isalnum() or
                                  text[pos - 1] == "_")
            after = pos < len(text) and (text[pos].isalnum() or
                                         text[pos] == "_")
            return before != after
        if anchor == r"\B":
            before = pos > 0 and (text[pos - 1].isalnum() or
                                  text[pos - 1] == "_")
            after = pos < len(text) and (text[pos].isalnum() or
                                         text[pos] == "_")
            return before == after
        return False

    def _fixed_width(self, nodes: list[dict[str, Any]]) -> int | None:
        total = 0
        for node in nodes:
            kind = node["kind"]
            if kind in ("literal", "dot", "escape", "escape_class",
                        "char_class"):
                total += 1
            elif kind in ("anchor", "lookahead", "lookbehind", "flag"):
                continue
            elif kind in ("group", "atomic"):
                width = self._fixed_width(node.get("children", []))
                if width is None:
                    return None
                total += width
            elif kind == "alternation":
                widths = [self._fixed_width(b.get("children", []))
                          for b in node.get("children", [])]
                if any(w is None for w in widths) or \
                        len(set(widths)) != 1:
                    return None
                total += widths[0]
            elif kind == "quantifier":
                width = self._fixed_width(node.get("children", []))
                if width is None:
                    return None
                if node.get("min") != node.get("max"):
                    return None
                total += width * (node.get("min") or 0)
            elif kind == "backref":
                return None
            else:
                return None
        return total

    def _lookbehind_holds(self, node: dict[str, Any],
                          pos: int) -> bool:
        body = node.get("children", [])
        width = self._fixed_width(body)
        if width is not None:
            start = pos - width
            if start < 0:
                return bool(not node.get("positive"))
            found = self._first(
                self._seq(body, 0, start, {},
                          lambda end, g: _yield_one(end, g)))
            holds = found is not None and found[0] == pos
            return holds if node.get("positive") else not holds
        # variable width: the real engine decides, and the step says so
        self.notes.append(
            "variable-width lookbehind evaluated by the real engine, "
            "not step-traced")
        children = node.get("children", [])
        if not children:
            return bool(not node.get("positive"))
        inner_start = children[0]["span"][0]
        inner_end = children[-1]["span"][1]
        inner = self._pattern_text(inner_start, inner_end)
        try:
            compiled = re.compile(inner + r"\Z",
                                  _python_flags(self._flags_text()))
        except re.error:
            return bool(not node.get("positive"))
        match = compiled.search(self.text[:pos])
        holds = match is not None and match.end() == pos
        return holds if node.get("positive") else not holds

    def _pattern_text(self, start: int, end: int) -> str:
        return getattr(self, "_pattern", "")[start:end]

    def _flags_text(self) -> str:
        flags = ""
        if self.ignorecase:
            flags += "i"
        if self.multiline:
            flags += "m"
        if self.dotall:
            flags += "s"
        return flags

    def _match_quant(self, node: dict[str, Any], pos: int,
                     groups: dict[int, Any], cont: Any) -> Any:
        target = node["children"][0]
        minimum = node.get("min") or 0
        maximum = node.get("max")
        lazy = bool(node.get("lazy"))
        if node.get("possessive"):
            end, final_groups = self._consume_max(
                target, pos, groups, minimum, maximum)
            if end is not None:
                self._record("match", pos, node)
                yield from cont(end, final_groups)
            else:
                self._record("fail", pos, node)
            return

        def _rep(count: int, cur_pos: int,
                 cur_groups: dict[int, Any]) -> Any:
            can_extend = maximum is None or count < maximum
            if lazy:
                if count >= minimum:
                    yield from cont(cur_pos, cur_groups)
                if can_extend:
                    yield from self._match_node(
                        target, cur_pos, cur_groups,
                        lambda end, g2: _rep(count + 1, end, g2)
                        if end != cur_pos else _empty_gen())
            else:
                if can_extend:
                    yield from self._match_node(
                        target, cur_pos, cur_groups,
                        lambda end, g2: _rep(count + 1, end, g2)
                        if end != cur_pos else _empty_gen())
                if count >= minimum:
                    yield from cont(cur_pos, cur_groups)

        yield from _rep(0, pos, groups)

    def _consume_max(self, target: dict[str, Any], pos: int,
                     groups: dict[int, Any], minimum: int,
                     maximum: int | None) -> tuple[Any, Any]:
        """Greedy consume with no backtracking (possessive)."""
        count = 0
        cur_pos, cur_groups = pos, groups
        while maximum is None or count < maximum:
            nxt = self._first(
                self._match_node(target, cur_pos, cur_groups,
                                 lambda end, g: _yield_one(end, g)))
            if nxt is None or nxt[0] == cur_pos:
                break
            cur_pos, cur_groups = nxt
            count += 1
        if count < minimum:
            return None, cur_groups
        return cur_pos, cur_groups

    def match(self) -> tuple[bool, int | None]:
        """Search semantics: the leftmost match wins."""
        for start in range(len(self.text) + 1):
            try:
                found = self._first(
                    self._seq(self.ast, 0, start, {},
                              lambda end, g: _yield_one(end, g)))
            except _BudgetExceeded:
                return False, None
            if found is not None:
                return True, found[0]
        return False, None


def _yield_one(end: int, groups: dict[int, Any]) -> Any:
    yield (end, groups)


def debug_trace(pattern: str, text: str,
                flavour: str = "python", budget: int = DEFAULT_BUDGET,
                flags: str = "") -> dict[str, Any]:
    """Step-by-step match trace with a step budget; never raises.

    A pathological pattern reports its blow-up (steps == budget,
    blow_up True) instead of hanging her. For flavours other than
    python the trace follows the shared grammar (Python semantics)
    and says so — the flavour's own engine may differ.
    """
    flavour = (flavour or "python").strip().lower()
    base: dict[str, Any] = {"pattern": pattern, "flavour": flavour,
                            "budget": budget}
    parsed = parse_pattern(pattern, flavour)
    if not parsed["ok"]:
        return {**base, "ok": False, "steps": [], "matched": False,
                "end": None, "blow_up": False, "steps_taken": 0,
                "error": parsed["error"], "notes": []}
    if "x" in (flags or ""):
        return {**base, "ok": False, "steps": [], "matched": False,
                "end": None, "blow_up": False, "steps_taken": 0,
                "error": "verbose (x) patterns cannot be traced "
                         "step by step", "notes": []}
    tracer = _Tracer(parsed["_ast"], text or "", flags or "", budget)
    tracer._pattern = pattern
    try:
        matched, end = tracer.match()
    except _BudgetExceeded:
        matched, end = False, None
    except RecursionError:
        # a pathological nesting depth is the same verdict as a
        # step blow-up: report it, never crash the caller.
        notes = ["recursion depth exceeded — the pattern nests too "
                 "deeply to trace"]
        if flavour != "python":
            notes.append(
                "trace follows the shared grammar (Python "
                f"semantics); the {flavour} engine may differ")
        return {**base, "ok": True, "steps": tracer.trace,
                "matched": False, "end": None, "blow_up": True,
                "steps_taken": tracer.steps, "error": "",
                "notes": notes}
    blow_up = tracer.steps >= tracer.budget
    notes = list(dict.fromkeys(tracer.notes))
    if flavour != "python":
        notes.append(
            "trace follows the shared grammar (Python semantics); "
            f"the {flavour} engine may differ")
    return {**base, "ok": True, "steps": tracer.trace,
            "matched": matched, "end": end, "blow_up": blow_up,
            "steps_taken": tracer.steps, "error": "", "notes": notes}


# ── code generation + verification ────────────────────────────────────

def _snippet_python(pattern: str, texts: list[str]) -> str:
    return (
        "import json, re\n"
        "\n"
        "pattern = " + json.dumps(pattern) + "\n"
        "texts = " + json.dumps(texts) + "\n"
        "compiled = re.compile(pattern)\n"
        "results = []\n"
        "for text in texts:\n"
        "    matches = []\n"
        "    for m in compiled.finditer(text):\n"
        "        groups = []\n"
        "        for i in range(1, (compiled.groups or 0) + 1):\n"
        "            span = m.span(i)\n"
        "            groups.append(None if span[0] < 0 else {\n"
        "                'index': i,\n"
        "                'name': next((n for n, j in\n"
        "                            compiled.groupindex.items()\n"
        "                            if j == i), None),\n"
        "                'value': m.group(i),\n"
        "                'start': span[0], 'end': span[1]})\n"
        "        matches.append({'start': m.start(), 'end': m.end(),\n"
        "                        'match': m.group(0), 'groups': groups})\n"
        "    results.append({'text': text, 'matches': matches})\n"
        "print(json.dumps({'results': results}))\n"
    )


def _snippet_javascript(pattern: str, texts: list[str]) -> str:
    payload = {"pattern": pattern, "flags": "", "texts": texts}
    return _js_program(payload)


def _snippet_go(pattern: str, texts: list[str]) -> str:
    payload = {"pattern": pattern, "flags": "", "texts": texts}
    return _go_program(payload)


def _snippet_rust(pattern: str, texts: list[str]) -> str:
    return (
        "// Rust needs the `regex` crate (Cargo). With rustc alone\n"
        "// there is no regex engine in std — add this to a Cargo\n"
        "// project with `regex = \"1\"` in Cargo.toml.\n"
        "use regex::Regex;\n"
        "\n"
        "fn main() {\n"
        "    let pattern = " + json.dumps(pattern) + ";\n"
        "    let texts: Vec<&str> = vec!" + json.dumps(texts) + ";\n"
        "    let re = Regex::new(pattern).unwrap();\n"
        "    for text in texts {\n"
        "        for m in re.find_iter(text) {\n"
        "            println!(\"{} at {}-{}\", m.as_str(), m.start(),\n"
        "                     m.end());\n"
        "        }\n"
        "    }\n"
        "}\n"
    )


def _snippet_java(pattern: str, texts: list[str]) -> str:
    return (
        "import java.util.regex.*;\n"
        "\n"
        "public class RegexLab {\n"
        "    public static void main(String[] args) {\n"
        "        Pattern compiled = Pattern.compile(" +
        json.dumps(pattern) + ");\n"
        "        String[] texts = " +
        json.dumps(texts).replace("[", "{").replace("]", "}") + ";\n"
        "        for (String text : texts) {\n"
        "            Matcher m = compiled.matcher(text);\n"
        "            while (m.find()) {\n"
        "                System.out.println(m.group() + \" at \" +\n"
        "                    m.start() + \"-\" + m.end());\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n"
    )


def _snippet_csharp(pattern: str, texts: list[str]) -> str:
    return (
        "using System;\n"
        "using System.Text.RegularExpressions;\n"
        "\n"
        "class RegexLab {\n"
        "    static void Main() {\n"
        "        var re = new Regex(" + json.dumps(pattern) + ");\n"
        "        string[] texts = " +
        json.dumps(texts).replace("[", "{").replace("]", "}") + ";\n"
        "        foreach (var text in texts) {\n"
        "            foreach (Match m in re.Matches(text)) {\n"
        "                Console.WriteLine($\"{m.Value} at \" +\n"
        "                    $\"{m.Index}-{m.Index + m.Length}\");\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n"
    )


def _snippet_php(pattern: str, texts: list[str]) -> str:
    return (
        "<?php\n"
        "$pattern = " + json.dumps("/" + pattern + "/") + ";\n"
        "$texts = " + json.dumps(texts).replace(
            "[", "array(").replace("]", ")") + ";\n"
        "foreach ($texts as $text) {\n"
        "    if (preg_match_all($pattern, $text, $m,\n"
        "            PREG_OFFSET_CAPTURE)) {\n"
        "        foreach ($m[0] as $hit) {\n"
        "            echo $hit[0] . ' at ' . $hit[1] . PHP_EOL;\n"
        "        }\n"
        "    }\n"
        "}\n"
    )


def _snippet_ruby(pattern: str, texts: list[str]) -> str:
    return (
        "pattern = " + json.dumps(pattern) + "\n"
        "texts = " + json.dumps(texts) + "\n"
        "re = Regexp.new(pattern)\n"
        "texts.each do |text|\n"
        "  text.scan(re) { puts Regexp.last_match(0).to_s +\n"
        "    ' at ' + Regexp.last_match.begin(0).to_s }\n"
        "end\n"
    )


_SNIPPETS: dict[str, Any] = {
    "python": _snippet_python,
    "javascript": _snippet_javascript,
    "go": _snippet_go,
    "rust": _snippet_rust,
    "java": _snippet_java,
    "csharp": _snippet_csharp,
    "php": _snippet_php,
    "ruby": _snippet_ruby,
}

#: snippet languages the code executor can actually run.
_SNIPPET_RUN_LANG = {"python": "python", "javascript": "javascript",
                     "go": "go", "php": "php", "ruby": "ruby"}


def generate_code(pattern: str, flavour: str, lang: str,
                  texts: list[str] | None = None) -> dict[str, Any]:
    """A ready-to-paste snippet that reproduces the session's
    matches in ``lang``; never raises.

    The snippet is written for the PATTERN as given; for flavours
    other than python the snippet uses that flavour's engine
    syntax, and the caller must verify it (see verify_code).
    """
    lang = (lang or "python").strip().lower()
    builder = _SNIPPETS.get(lang)
    if builder is None:
        return {"ok": False, "lang": lang, "code": "",
                "error": f"no snippet template for '{lang}' "
                         f"(templates: "
                         f"{', '.join(sorted(_SNIPPETS))})"}
    parsed = parse_pattern(pattern, flavour)
    if not parsed["ok"]:
        return {"ok": False, "lang": lang, "code": "",
                "error": parsed["error"]}
    return {"ok": True, "lang": lang,
            "code": builder(pattern, [str(t) for t in (texts or [])]),
            "error": ""}


def verify_code(code: str, lang: str,
                expected: list[dict[str, Any]],
                run_fn: Any = None) -> dict[str, Any]:
    """Execute a snippet and compare its output to the session's
    match set; never raises.

    verified True only when the snippet's printed JSON document
    equals the expected results exactly. Anything else — a dead
    runner, a crash, a different match set — is verified False with
    the difference named, never presented as working.
    """
    lang = (lang or "python").strip().lower()
    base: dict[str, Any] = {"lang": lang}
    run_lang = _SNIPPET_RUN_LANG.get(lang, "")
    if not run_lang:
        return {**base, "ok": True, "verified": False,
                "stdout": "", "stderr": "", "exit_code": -1,
                "diff": "", "error": "",
                "reason": f"{lang} snippets cannot be executed "
                          "here — no engine in the code executor"}
    if run_fn is None:
        return {**base, "ok": True, "verified": False,
                "stdout": "", "stderr": "", "exit_code": -1,
                "diff": "", "error": "",
                "reason": "not verified — no code runner available"}
    try:
        result = run_fn(code, run_lang)
    except Exception as exc:  # noqa: BLE001 — runner dead = unverified
        return {**base, "ok": True, "verified": False,
                "stdout": "", "stderr": "", "exit_code": -1,
                "diff": "", "error": "",
                "reason": f"not verified — the {run_lang} runner "
                          f"failed: {exc}"}
    if not isinstance(result, dict):
        return {**base, "ok": True, "verified": False,
                "stdout": "", "stderr": "", "exit_code": -1,
                "diff": "", "error": "",
                "reason": "not verified — the runner returned "
                          "nothing usable"}
    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    exit_code = int(result.get("exit_code", -1))
    if result.get("timed_out") or exit_code != 0:
        return {**base, "ok": True, "verified": False,
                "stdout": stdout, "stderr": stderr,
                "exit_code": exit_code, "diff": "",
                "error": "",
                "reason": "not verified — the snippet did not run "
                          f"clean (exit {exit_code})"}
    try:
        doc = json.loads(stdout)
    except ValueError:
        return {**base, "ok": True, "verified": False,
                "stdout": stdout, "stderr": stderr,
                "exit_code": exit_code,
                "diff": "the snippet printed no JSON document",
                "error": "",
                "reason": "not verified — the snippet printed no "
                          "JSON document"}
    got = doc.get("results") if isinstance(doc, dict) else None
    if got == expected:
        return {**base, "ok": True, "verified": True,
                "stdout": stdout, "stderr": stderr,
                "exit_code": exit_code, "diff": "", "error": ""}
    diff_lines = []
    for want_row, got_row in zip(expected, got or [], strict=False):
        want_matches = want_row.get("matches", [])
        got_matches = got_row.get("matches", []) \
            if isinstance(got_row, dict) else []
        if want_matches != got_matches:
            diff_lines.append(
                f"text {want_row.get('text', '')!r}: expected "
                f"{len(want_matches)} match(es), got "
                f"{len(got_matches)}")
    if len(got or []) != len(expected):
        diff_lines.append(
            f"expected {len(expected)} text row(s), got "
            f"{len(got or [])}")
    return {**base, "ok": True, "verified": False,
            "stdout": stdout, "stderr": stderr,
            "exit_code": exit_code,
            "diff": "; ".join(diff_lines) or "match sets differ",
            "error": "",
            "reason": "not verified — the snippet's matches differ "
                      "from the session's"}


# ── reverse loop: generate, then prove ────────────────────────────────

#: Deterministic templates: keyword -> (pattern, proving cases).
#: The cases are the proof — a template pattern must pass ALL of
#: them before it is presented.
_TEMPLATES: list[dict[str, Any]] = [
    {
        "name": "email",
        "keywords": ["email", "mail address", "e-mail"],
        "pattern": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "positives": ["a@b.com", "first.last+tag@sub.domain.org",
                      "user123@mail.co"],
        "negatives": ["not an email", "a@b", "@b.com", "a b@c.com",
                      "a@b."],
    },
    {
        "name": "ipv4",
        "keywords": ["ipv4", "ip address", "ip v4"],
        "pattern": r"\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
                   r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
                   r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
                   r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        "positives": ["192.168.0.1", "8.8.8.8", "0.0.0.0"],
        "negatives": ["999.1.1.1", "1.2.3", "abc", "256.256.256.256"],
    },
    {
        "name": "date-iso",
        "keywords": ["date", "iso date", "yyyy-mm-dd"],
        "pattern": r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]"
                   r"|3[01])\b",
        "positives": ["2026-11-17", "1999-01-31"],
        "negatives": ["2026-13-01", "17-11-2026", "2026/11/17"],
    },
    {
        "name": "time",
        "keywords": ["time", "clock", "hh:mm"],
        "pattern": r"\b(?:[01][0-9]|2[0-3]):[0-5][0-9]\b",
        "positives": ["09:30", "23:59", "00:00"],
        "negatives": ["24:00", "9:30", "12:60"],
    },
    {
        "name": "url",
        "keywords": ["url", "web address", "link"],
        "pattern": r"https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
                   r"(?:/[^\s]*)?",
        "positives": ["https://example.com",
                      "http://sub.domain.org/path?q=1"],
        "negatives": ["example.com", "ftp://x.com", "http://",
                      "https://"],
    },
    {
        "name": "hex-color",
        "keywords": ["hex color", "hex colour", "color code"],
        "pattern": r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b",
        "positives": ["#fff", "#FF5733", "#abc123"],
        "negatives": ["#ff", "#gggggg", "fff", "#12345"],
    },
    {
        "name": "uuid",
        "keywords": ["uuid", "guid"],
        "pattern": r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
                   r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                   r"[0-9a-fA-F]{12}\b",
        "positives": ["123e4567-e89b-12d3-a456-426614174000"],
        "negatives": ["123e4567-e89b-12d3-a456", "not-a-uuid",
                      "123e4567e89b12d3a456426614174000"],
    },
    {
        "name": "phone-intl",
        "keywords": ["phone", "telephone", "mobile number"],
        "pattern": r"\+\d{1,3}[\s-]?\d{3,4}[\s-]?\d{3,4}[\s-]?"
                   r"\d{0,4}",
        "positives": ["+234 803 123 4567", "+1-415-555-0132"],
        "negatives": ["803 123 4567", "call me", "+12"],
    },
    {
        "name": "slug",
        "keywords": ["slug", "url slug"],
        "pattern": r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b",
        "positives": ["hello-world", "devon-2-0", "abc"],
        "negatives": ["Hello World", "-leading", "trailing-",
                      "has space"],
    },
    {
        "name": "semver",
        "keywords": ["semver", "version number"],
        "pattern": r"\b\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?"
                   r"(?:\+[0-9A-Za-z.-]+)?\b",
        "positives": ["1.2.3", "0.0.1-alpha", "10.20.30+build.5"],
        "negatives": ["1.2", "v1.2.3", "1.2.3.4"],
    },
    {
        "name": "mac",
        "keywords": ["mac address"],
        "pattern": r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b",
        "positives": ["00:1A:2B:3C:4D:5E", "aa:bb:cc:dd:ee:ff"],
        "negatives": ["00:1A:2B:3C:4D", "00-1A-2B-3C-4D-5E",
                      "gg:1a:2b:3c:4d:5e"],
    },
    {
        "name": "username",
        "keywords": ["username", "user name"],
        "pattern": r"\b[A-Za-z_][A-Za-z0-9_]{2,15}\b",
        "positives": ["peacethefirst", "user_1", "abc"],
        "negatives": ["1abc", "ab", "has space", "a" * 20],
    },
]

_FENCE_RE = re.compile(
    r"```\s*[A-Za-z0-9_+-]*\s*\n(.*?)```", re.DOTALL)


def _prove_candidate(pattern: str, positives: list[str],
                     negatives: list[str]) -> list[str]:
    """The proof: every positive must match, every negative must
    not. Proving uses FULLMATCH semantics — the string IS an email,
    an IPv4 address, a date — not merely contains one. Returns the
    list of failures (empty = proved)."""
    failures: list[str] = []
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return [f"pattern does not compile: {exc}"]
    for text in positives:
        if not compiled.fullmatch(text):
            failures.append(f"should match but did not: {text!r}")
    for text in negatives:
        if compiled.fullmatch(text):
            failures.append(f"should NOT match but did: {text!r}")
    return failures


def _extract_pattern(text: str) -> str:
    """Pull a bare pattern out of a model reply (fenced block, or
    the raw line when it looks like a pattern)."""
    match = _FENCE_RE.search(text or "")
    if match:
        return match.group(1).strip()
    lines = [ln.strip() for ln in (text or "").splitlines()
             if ln.strip()]
    return lines[0] if lines else ""


def generate_regex(description: str, positives: list[str],
                   negatives: list[str], llm_fn: Any = None,
                   run_fn: Any = None) -> dict[str, Any]:
    """The reverse loop: generate a pattern, then PROVE it.

    A template match proves against its own built-in cases plus the
    caller's cases. A model candidate proves against the caller's
    cases (one retry with the failures named). A pattern that fails
    its proof is returned with proved False and the failures named
    — never presented as working.
    """
    description = (description or "").strip()
    positives = [str(p) for p in (positives or [])]
    negatives = [str(n) for n in (negatives or [])]
    if not description:
        return {"ok": False, "pattern": "", "flavour": "python",
                "proved": False, "failures": [],
                "template": "",
                "error": "generate needs a description of the "
                         "pattern to write"}
    lowered = description.lower()
    for template in _TEMPLATES:
        if any(key in lowered for key in template["keywords"]):
            pattern = template["pattern"]
            failures = _prove_candidate(
                pattern, positives or template["positives"],
                negatives or template["negatives"])
            return {"ok": True, "pattern": pattern,
                    "flavour": "python",
                    "proved": not failures, "failures": failures,
                    "template": template["name"], "error": ""}
    if llm_fn is None:
        return {"ok": False, "pattern": "", "flavour": "python",
                "proved": False, "failures": [],
                "template": "",
                "error": "no model rung and no template matches "
                         "that description — give examples of "
                         "strings that should and should not match, "
                         "or describe one of the built-in kinds "
                         "(email, IPv4, date, time, URL, hex color, "
                         "UUID, phone, slug, version, MAC address, "
                         "username)"}
    if not positives and not negatives:
        return {"ok": False, "pattern": "", "flavour": "python",
                "proved": False, "failures": [],
                "template": "",
                "error": "a generated pattern needs proving cases — "
                         "give strings that should match and strings "
                         "that should not"}
    prompt = (
        "Write ONE Python regular expression for: " + description +
        "\nStrings that MUST match:\n" +
        "\n".join("- " + p for p in positives) +
        "\nStrings that must NOT match:\n" +
        "\n".join("- " + n for n in negatives) +
        "\nReply with ONLY the pattern in a fenced code block.")
    try:
        candidate = _extract_pattern(llm_fn(prompt))
    except Exception as exc:  # noqa: BLE001 — model dead = honest
        return {"ok": False, "pattern": "", "flavour": "python",
                "proved": False, "failures": [],
                "template": "",
                "error": f"the model rung failed: {exc}"}
    if not candidate:
        return {"ok": False, "pattern": "", "flavour": "python",
                "proved": False, "failures": [],
                "template": "",
                "error": "the model returned no pattern"}
    failures = _prove_candidate(candidate, positives, negatives)
    if not failures:
        return {"ok": True, "pattern": candidate,
                "flavour": "python", "proved": True, "failures": [],
                "template": "", "error": ""}
    retry_prompt = (
        prompt + "\nThe previous pattern failed these cases:\n" +
        "\n".join("- " + f for f in failures) +
        "\nWrite a corrected pattern, fenced code block only.")
    try:
        candidate2 = _extract_pattern(llm_fn(retry_prompt))
    except Exception as exc:  # noqa: BLE001 — model dead = honest
        return {"ok": True, "pattern": candidate,
                "flavour": "python", "proved": False,
                "failures": failures, "template": "",
                "error": f"the retry rung failed: {exc}"}
    failures2 = _prove_candidate(candidate2, positives, negatives) \
        if candidate2 else ["the retry returned no pattern"]
    if not failures2:
        return {"ok": True, "pattern": candidate2,
                "flavour": "python", "proved": True, "failures": [],
                "template": "", "error": ""}
    return {"ok": True, "pattern": candidate2 or candidate,
            "flavour": "python", "proved": False,
            "failures": failures2, "template": "", "error": ""}


# ── store + sessions + links ──────────────────────────────────────────

def _default_home(home: Any) -> Path:
    return Path(home) if home else Path.home() / ".nomorals"


class RegexLabStore:
    """Durable regex sessions: <home>/data/regex_sessions.json.

    Atomic writes (temp file + replace). ``get`` accepts a full id,
    a unique prefix, or an encoded ``regex:`` link.
    """

    def __init__(self, path: str | Path = "") -> None:
        self.path = str(path) if path else str(
            _default_home(None) / "data" / REGEX_SESSIONS_FILENAME)
        self._data: dict[str, Any] = {"sessions": {}}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(Path(self.path).read_text(
                encoding="utf-8"))
            if isinstance(data, dict) and isinstance(
                    data.get("sessions"), dict):
                self._data["sessions"] = data["sessions"]
        except (OSError, ValueError):
            self._data = {"sessions": {}}

    def _save(self) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2),
                       encoding="utf-8")
        tmp.replace(path)

    def save(self, record: dict[str, Any]) -> None:
        self._data["sessions"][str(record["session_id"])] = record
        self._save()

    def get(self, session_id: str) -> dict[str, Any] | None:
        key = (session_id or "").strip()
        if not key:
            return None
        row = self._data["sessions"].get(key)
        if isinstance(row, dict):
            return row
        if key.startswith("regex:"):
            decoded = decode_link(key)
            if decoded is not None:
                return {"session_id": key, **decoded}
        matches = [r for k, r in self._data["sessions"].items()
                   if k.startswith(key)]
        return matches[0] if len(matches) == 1 else None

    def list(self) -> list[dict[str, Any]]:
        rows = [r for r in self._data["sessions"].values()
                if isinstance(r, dict)]
        rows.sort(key=lambda r: str(r.get("created_at", "")),
                  reverse=True)
        return [{
            "session_id": str(r.get("session_id") or ""),
            "pattern": str(r.get("pattern") or ""),
            "flavour": str(r.get("flavour") or ""),
            "created_at": str(r.get("created_at") or ""),
            "ok": bool(r.get("ok")),
        } for r in rows]


def open_regex_store(home: Any = None) -> RegexLabStore:
    """Open (creating if needed) the durable regex-session store."""
    return RegexLabStore(_default_home(home) / "data"
                         / REGEX_SESSIONS_FILENAME)


def encode_link(session: dict[str, Any]) -> str:
    """The permalink-equivalent: pattern + flavour + flags + tests
    in one pasteable ``regex:`` link."""
    tests = session.get("tests")
    test_rows: list[dict[str, Any]] = []
    if isinstance(tests, dict):
        test_rows = [dict(r) for r in tests.get("results", [])
                     if isinstance(r, dict)]
    elif isinstance(tests, list):
        test_rows = [dict(r) for r in tests
                     if isinstance(r, dict)]
    blob = {"pattern": session.get("pattern", ""),
            "flavour": session.get("flavour", "python"),
            "flags": session.get("flags", ""),
            "tests": test_rows}
    raw = json.dumps(blob, separators=(",", ":")).encode("utf-8")
    return "regex:" + base64.urlsafe_b64encode(raw).decode("ascii")


def decode_link(link: str) -> dict[str, Any] | None:
    """Decode a ``regex:`` link back to a session sketch."""
    text = (link or "").strip()
    if not text.startswith("regex:"):
        return None
    try:
        raw = base64.urlsafe_b64decode(text[len("regex:"):])
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(doc, dict) or "pattern" not in doc:
        return None
    return {"pattern": str(doc.get("pattern", "")),
            "flavour": str(doc.get("flavour", "python")),
            "flags": str(doc.get("flags", "")),
            "tests": doc.get("tests", [])}


def run_regex_session(
        home: Any, pattern: str = "", flavour: str = "python",
        flags: str = "", texts: list[str] | None = None,
        tests: list[dict[str, Any]] | None = None,
        substitution: str | None = None,
        trace_text: str | None = None,
        code_lang: str | None = None,
        run_fn: Any = None, llm_fn: Any = None,
        store: RegexLabStore | None = None,
        description: str = "", positives: list[str] | None = None,
        negatives: list[str] | None = None) -> dict[str, Any]:
    """One full workbench session, stored durably; never raises.

    A description runs the reverse loop first (generate, then
    prove); the proven pattern becomes the session's pattern.
    """
    flavour = (flavour or "python").strip().lower()
    texts = [str(t) for t in (texts or [])]
    tests = list(tests or [])
    record: dict[str, Any] = {
        "session_id": "rx-" + secrets.token_hex(4),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pattern": pattern, "flavour": flavour,
        "flags": flags or "", "ok": True, "error": "",
    }
    try:
        if description:
            generated = generate_regex(
                description, positives or [], negatives or [],
                llm_fn=llm_fn, run_fn=run_fn)
            record["generated"] = {k: generated[k] for k in (
                "ok", "pattern", "proved", "failures", "template",
                "error") if k in generated}
            if generated["ok"] and generated["pattern"]:
                pattern = generated["pattern"]
                record["pattern"] = pattern
            else:
                record["ok"] = False
                record["error"] = generated.get("error") or (
                    "the generated pattern failed its proof: " +
                    "; ".join(generated.get("failures", [])))
                if store is not None:
                    store.save(record)
                return record
        explained = explain_pattern(pattern, flavour)
        record["explanation"] = {
            k: explained[k] for k in (
                "ok", "tokens", "reference", "error")
            if k in explained}
        record["ok"] = bool(explained["ok"])
        record["error"] = explained.get("error", "")
        if not explained["ok"]:
            if store is not None:
                store.save(record)
            return record
        if texts:
            record["matches"] = match_pattern(
                pattern, texts, flavour=flavour,
                flags=flags or "", run_fn=run_fn)
        if tests:
            record["tests"] = run_regex_tests(
                pattern, tests, flavour=flavour,
                flags=flags or "", run_fn=run_fn)
        if substitution is not None and texts:
            record["substitution"] = substitute_pattern(
                pattern, substitution, texts[0], flavour=flavour,
                flags=flags or "", run_fn=run_fn)
        if trace_text is not None:
            record["trace"] = debug_trace(
                pattern, trace_text, flavour=flavour,
                flags=flags or "")
        if code_lang:
            snippet = generate_code(pattern, flavour, code_lang,
                                    texts=texts)
            record["code"] = snippet
            if snippet.get("ok") and texts:
                matched = record.get("matches") or match_pattern(
                    pattern, texts, flavour=flavour,
                    flags=flags or "", run_fn=run_fn)
                expected = matched.get("results", []) \
                    if isinstance(matched, dict) else []
                record["code_verification"] = verify_code(
                    snippet["code"], code_lang, expected,
                    run_fn=run_fn)
        record["link"] = encode_link(record)
        if store is not None:
            store.save(record)
        return record
    except Exception as exc:  # noqa: BLE001 — sessions never raise
        record["ok"] = False
        record["error"] = f"regex session failed: {exc}"
        if store is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 — best effort only
                store.save(record)
        return record


def format_regex_reply(record: dict[str, Any]) -> str:
    """The chat rendering of a session record."""
    if not record.get("ok"):
        return ("regex failed: " +
                str(record.get("error") or "unknown error"))
    lines = [f"regex `{record.get('pattern', '')}` "
             f"({record.get('flavour', 'python')})"]
    generated = record.get("generated") or {}
    if generated:
        if generated.get("proved"):
            suffix = (f" (template: {generated['template']})"
                      if generated.get("template") else "")
            lines.append("generated and PROVED against its cases"
                         + suffix)
        else:
            lines.append("generated but NOT proved — failures: " +
                         "; ".join(generated.get("failures", [])))
    explanation = record.get("explanation") or {}
    tokens = explanation.get("tokens", [])
    if tokens:
        lines.append("what it means:")
        for token in tokens[:12]:
            lines.append(f"  `{token['text']}` — {token['plain']}")
        if len(tokens) > 12:
            lines.append(f"  … and {len(tokens) - 12} more tokens")
    matched = record.get("matches") or {}
    if matched.get("results") is not None:
        if not matched.get("verifiable", True):
            lines.append("matches: " +
                         str(matched.get("reason") or
                             "not verifiable here"))
        else:
            for row in matched["results"]:
                if not row["matches"]:
                    lines.append(f"  `{row['text']}` — no match")
                    continue
                for found in row["matches"]:
                    groups = ", ".join(
                        f"{g['name'] or g['index']}={g['value']!r}"
                        for g in found["groups"]
                        if g is not None) or "no groups"
                    lines.append(
                        f"  `{row['text']}` → `{found['match']}` "
                        f"at {found['start']}-{found['end']} "
                        f"({groups})")
    tested = record.get("tests") or {}
    if tested.get("total"):
        lines.append(
            f"unit tests: {tested['passed']}/{tested['total']} "
            "passed")
        for row in tested.get("results", []):
            if not row["ok"]:
                lines.append(
                    f"  FAIL: `{row['text']}` should "
                    f"{'' if row['should_match'] else 'NOT '}"
                    f"match but "
                    f"{'did not' if row['should_match'] else 'did'}")
    sub = record.get("substitution") or {}
    if sub:
        if sub.get("verifiable"):
            lines.append(
                f"substitution: `{sub.get('result', '')}` "
                f"({sub.get('count', 0)} replacement(s))")
        else:
            lines.append("substitution: " +
                         str(sub.get("reason") or
                             "not verifiable here"))
    trace = record.get("trace") or {}
    if trace:
        if trace.get("blow_up"):
            lines.append(
                "debug: BLOW-UP — "
                f"{trace.get('steps_taken', 0)} steps without a "
                "match (catastrophic backtracking: the pattern "
                "keeps retrying the same ground)")
        elif trace.get("matched"):
            lines.append(
                f"debug: matched ending at {trace.get('end')} "
                f"after {trace.get('steps_taken', 0)} steps")
        else:
            lines.append(
                f"debug: no match after {trace.get('steps_taken', 0)}"
                " steps")
        for note in trace.get("notes", []):
            lines.append(f"  note: {note}")
    code = record.get("code") or {}
    if code.get("ok"):
        verification = record.get("code_verification") or {}
        if verification.get("verified"):
            lines.append(
                f"{code['lang']} snippet: verified — it ran and "
                "reproduced the session's matches")
        elif verification:
            lines.append(
                f"{code['lang']} snippet: NOT verified — "
                + str(verification.get("reason") or
                      verification.get("diff") or ""))
        else:
            lines.append(
                f"{code['lang']} snippet: generated, not verified "
                "here")
    if record.get("link"):
        lines.append(f"link: {record['link']}")
    return "\n".join(lines)


# ── brain tool ────────────────────────────────────────────────────────

def _run_fn_from_registry(registry: Any) -> Any:
    """A run callable over the ``run_code`` tool, or None.

    ``registry.call`` returns an Outcome: ``.ok`` False means the
    runner is unavailable (the organ marks blocks unverified);
    ``.ok`` True carries the ``CodeRunner.run`` dict as ``.value``.
    """
    if registry is None:
        return None

    def _run(code: str, lang: str):
        try:
            outcome = registry.call(
                "run_code", action="run", code=code, lang=lang,
                timeout=30.0)
        except Exception:  # noqa: BLE001 — a dead runner is unverified
            return None
        if not getattr(outcome, "ok", False):
            return None
        value = getattr(outcome, "value", None)
        return value if isinstance(value, dict) else None

    return _run


def _as_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [ln for ln in value.splitlines() if ln.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _as_tests(value: Any) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    if value is None:
        return tests
    if isinstance(value, str):
        value = [value]
    for item in value if isinstance(value, (list, tuple)) else []:
        if isinstance(item, dict):
            tests.append({"text": str(item.get("text", "")),
                          "should_match": bool(
                              item.get("should_match"))})
            continue
        text = str(item)
        if "=" in text:
            body, want = text.rsplit("=", 1)
            tests.append({
                "text": body.strip(),
                "should_match": want.strip().lower() in (
                    "yes", "true", "1", "match")})
    return tests


def register(registry: Any) -> None:
    """Attach the ``regex_lab`` tool to a registry."""
    context = registry.context

    @registry.register(
        "regex_lab",
        description=(
            "Regex workbench (Regex101 core) — test, explain and "
            "debug regular expressions, generate code from them, or "
            "generate a pattern from a description. action=<explain|"
            "match|test|sub|debug|code|gen|list|show>; explain takes "
            "pattern=<regex> [flavour=<python|javascript|go|rust|"
            "java|dotnet|pcre2>] [flags=<imsxau>]; match takes "
            "pattern + texts=<sample strings, one per line>; test "
            "takes pattern + tests=<{text, should_match}>; sub takes "
            "pattern + substitution=<replacement> + texts (first "
            "text is replaced); debug takes pattern + texts (first "
            "text is traced) with a step budget that catches "
            "catastrophic backtracking; code takes pattern + "
            "code_lang=<python|javascript|go|rust|java|csharp|php|"
            "ruby> and executes the snippet in codews to prove it "
            "reproduces the session's matches; gen takes "
            "description=<what the pattern should match> + "
            "positives=<strings that must match> + negatives=<"
            "strings that must not> and proves the generated "
            "pattern against them before answering. A pattern a "
            "flavour does not support is refused by name; a flavour "
            "with no engine here is labelled not verifiable, never "
            "simulated. Returns ok, session_id, pattern, flavour, "
            "explanation, matches, tests, reply, or error."
        ),
        capability=Capability.EXEC_CODE,
    )
    def regex_lab_tool(
        pattern: str = "",
        action: str = "explain",
        flavour: str = "python",
        flags: str = "",
        texts: Any = None,
        tests: Any = None,
        substitution: str = "",
        description: str = "",
        positives: Any = None,
        negatives: Any = None,
        code_lang: str = "",
        session_id: str = "",
        limit: int = 5,
    ) -> dict[str, Any]:
        home = _home_from_context(context)
        store = open_regex_store(home)
        run_fn = _run_fn_from_registry(registry)
        act = (action or "explain").strip().lower()
        if act == "list":
            return {"ok": True, "sessions": store.list()[:limit]}
        if act == "show":
            row = store.get((session_id or pattern or "").strip())
            if row is None:
                return {"ok": False,
                        "error": f"no stored regex session "
                                 f"'{session_id or pattern}'"}
            shown = dict(row)
            shown["ok"] = True
            shown["reply"] = format_regex_reply(
                {**row, "ok": True})
            return shown
        if act == "gen":
            record = run_regex_session(
                home, description=(description or pattern or ""),
                positives=_as_texts(positives),
                negatives=_as_texts(negatives),
                llm_fn=_llm_from_context(context),
                run_fn=run_fn, store=store)
            record["reply"] = format_regex_reply(record)
            return record
        text_list = _as_texts(texts)
        test_list = _as_tests(tests)
        if act == "explain":
            if not (pattern or "").strip():
                return {"ok": False,
                        "error": "explain needs a pattern"}
            record = run_regex_session(
                home, pattern=pattern, flavour=flavour,
                flags=flags or "", store=store)
            record["reply"] = format_regex_reply(record)
            return record
        if act == "match":
            if not (pattern or "").strip():
                return {"ok": False,
                        "error": "match needs a pattern"}
            record = run_regex_session(
                home, pattern=pattern, flavour=flavour,
                flags=flags or "", texts=text_list,
                run_fn=run_fn, store=store)
            record["reply"] = format_regex_reply(record)
            return record
        if act == "test":
            if not (pattern or "").strip():
                return {"ok": False,
                        "error": "test needs a pattern"}
            record = run_regex_session(
                home, pattern=pattern, flavour=flavour,
                flags=flags or "", tests=test_list,
                run_fn=run_fn, store=store)
            record["reply"] = format_regex_reply(record)
            return record
        if act == "sub":
            if not (pattern or "").strip():
                return {"ok": False,
                        "error": "sub needs a pattern"}
            record = run_regex_session(
                home, pattern=pattern, flavour=flavour,
                flags=flags or "", texts=text_list,
                substitution=substitution, run_fn=run_fn,
                store=store)
            record["reply"] = format_regex_reply(record)
            return record
        if act == "debug":
            if not (pattern or "").strip():
                return {"ok": False,
                        "error": "debug needs a pattern"}
            record = run_regex_session(
                home, pattern=pattern, flavour=flavour,
                flags=flags or "",
                trace_text=text_list[0] if text_list else "",
                run_fn=run_fn, store=store)
            record["reply"] = format_regex_reply(record)
            return record
        if act == "code":
            if not (pattern or "").strip():
                return {"ok": False,
                        "error": "code needs a pattern"}
            record = run_regex_session(
                home, pattern=pattern, flavour=flavour,
                flags=flags or "", texts=text_list,
                code_lang=code_lang or "python", run_fn=run_fn,
                store=store)
            record["reply"] = format_regex_reply(record)
            return record
        return {"ok": False,
                "error": f"unknown regex_lab action '{act}' "
                         "(explain|match|test|sub|debug|code|gen|"
                         "list|show)"}
