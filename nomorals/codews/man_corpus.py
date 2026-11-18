"""Man-page corpus for the shell-command explainer (ExplainShell core,
52-websites #43).

The corpus is the ground truth the explainer stands on: man pages
parsed out of the host's own ``/usr/share/man`` (sections 1 and 8),
so an explanation names the documentation of the binaries actually
installed on this box — an improvement on ExplainShell's frozen
Ubuntu archive. Pages are parsed from roff source (no groff/man
binary needed), option blocks are extracted with the same
heuristics ExplainShell uses (``.TP``/``.IP`` term + description),
and the full text is indexed in the documents FTS5 store
(:class:`nomorals.storage.fts.FTSIndex` — the exact backend behind
:mod:`nomorals.documents.index`) for description search.

Two more rungs grow the corpus by discovery (open-sources law):

* ``--help`` harvesting — the output of ``<cmd> --help`` parsed for
  option lines. The harvest itself runs in the sandbox through an
  injected fetcher; the corpus module never executes anything.
* tldr community pages — fetched markdown for a command, parsed for
  its option lines. Also through an injected fetcher (the default
  one uses plain urllib against the tldr-pages GitHub raw host).

A fragment with no grounded doc anywhere is marked ``unknown`` by
the explainer — never filled in silently.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..storage.fts import FTSIndex

__all__ = [
    "BUILTIN_DOCS",
    "ManCorpus",
    "ManPage",
    "fetch_tldr_page",
    "parse_help_text",
    "parse_man_page_file",
    "parse_man_page_text",
    "parse_tldr_markdown",
]

_log = get_logger(__name__)

#: Man sections ingested (commands and system administration).
_MAN_SECTIONS = (1, 8)
#: Default man roots, in order. The host's own pages win over any
#: frozen archive: explanations match installed binaries.
_DEFAULT_MAN_ROOTS = ("/usr/share/man", "/usr/local/share/man")
#: A page larger than this is truncated before indexing (a man page
#: is never this big; the cap is a runaway guard, named here).
_MAX_MAN_TEXT_CHARS = 500_000
#: Excerpts are capped so one option never floods a reply.
_MAX_EXCERPT_CHARS = 600
#: tldr platforms tried in order for one command.
_TLDR_PLATFORMS = ("common", "linux", "osx", "sunos", "windows", "android")
_TLDR_BASE = ("https://raw.githubusercontent.com/tldr-pages/tldr"
              "/main/pages")

#: Shell builtins have no man page of their own; their documentation
#: lives in the bash manual. These one-line descriptions are the
#: curated builtin corpus (source "builtin") — short, factual, and
#: verifiable against ``help <builtin>`` on any bash.
BUILTIN_DOCS: dict[str, str] = {
    "cd": "Change the current directory.",
    "pwd": "Print the current working directory.",
    "echo": "Print its arguments to standard output.",
    "printf": "Print formatted output.",
    "export": "Mark variables for export to child processes.",
    "alias": "Define a shortcut name for a command.",
    "unalias": "Remove an alias definition.",
    "source": "Read and execute commands from a file in this shell.",
    ".": "Read and execute commands from a file in this shell.",
    "test": "Evaluate a conditional expression; true (0) or false (1).",
    "[": "Evaluate a conditional expression; true (0) or false (1).",
    "jobs": "List the jobs running in this shell.",
    "fg": "Bring a background job into the foreground.",
    "bg": "Resume a stopped job in the background.",
    "kill": "Send a signal to a process.",
    "wait": "Wait for background jobs to finish.",
    "umask": "Set the file-creation mode mask.",
    "set": "Set shell options and positional parameters.",
    "unset": "Remove a variable or function definition.",
    "shift": "Shift the positional parameters left.",
    "trap": "Run a command when the shell receives a signal.",
    "type": "Show how a name would be interpreted (alias, builtin, file).",
    "history": "Show or manage the command history.",
    "read": "Read a line of input into variables.",
    "exit": "Exit the shell.",
    "return": "Return from a function or sourced script.",
    "break": "Break out of a loop.",
    "continue": "Skip to the next iteration of a loop.",
    "eval": "Evaluate arguments as a shell command.",
    "exec": "Replace the shell with a command.",
    "getopts": "Parse options in a shell script.",
    "hash": "Remember or forget command locations.",
    "let": "Evaluate arithmetic expressions.",
    "local": "Declare a variable local to a function.",
    "readonly": "Mark variables read-only.",
    "declare": "Declare variables and their attributes.",
    "typeset": "Declare variables and their attributes.",
    "pushd": "Push a directory onto the directory stack.",
    "popd": "Pop a directory off the directory stack.",
    "dirs": "Show the directory stack.",
    "compgen": "Generate possible completions.",
    "complete": "Define how a command is completed.",
    "shopt": "Set or unset shell options.",
    "ulimit": "Set or show resource limits.",
}

# ── roff parsing ─────────────────────────────────────────────────────

_FONT_RE = re.compile(r"\\f[BIPR0-9](\([^)]+\))?")
_ESCAPES = (
    ("\\-", "-"), ("\\(hy", "-"), ("\\(em", "—"), ("\\(en", "–"),
    ("\\(bu", "•"), ("\\(lq", "“"), ("\\(rq", "”"),
    ("\\(co", "©"), ("\\(rg", "®"), ("\\e", "\\"), ("\\&", ""),
    ("\\ ", " "), ("\\0", " "), ("\\~", " "), ("\\.", "."),
    ("\\'", "'"), ('\\"', '"'),
    # Groff zero-width escapes: \^ (break point, used as \-\^--long
    # to keep a long flag breakable), \| (thin space), \/ (italic
    # correction). They carry no text — dropping them is what lets
    # _LONG_FLAG_RE see the long flag at all.
    ("\\^", ""), ("\\|", ""), ("\\/", ""),
)
_MACRO_LINE_RE = re.compile(r"^\.([A-Za-z]{1,3})\b\s?(.*)$")
_LONG_FLAG_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*")
_SHORT_FLAG_RE = re.compile(r"(?<![A-Za-z0-9-])-([A-Za-z0-9])(?![A-Za-z0-9])")
# Single-dash multi-letter options (find -name, -mtime, -type): one
# option, not a short-flag bundle. The lookbehind keeps prose
# hyphens (well-known) out.
_SINGLE_DASH_LONG_RE = re.compile(
    r"(?<![A-Za-z0-9-])-([A-Za-z][A-Za-z0-9_-]{1,15})(?![A-Za-z0-9_-])")
_SECTION_RE = re.compile(r"^[A-Z][A-Z /_-]*$")


def _strip_roff(line: str) -> str:
    """Strip roff font/escape markup from one line, keeping the text."""
    out = line
    out = _FONT_RE.sub("", out)
    for old, new in _ESCAPES:
        out = out.replace(old, new)
    out = out.replace("\\f", "")
    return out


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _flags_from_term(term: str) -> dict[str, dict[str, Any]]:
    """Flag specs from a .TP/.IP term line → {flag: option record}.

    ``-g, --listed-incremental=FILE`` yields both flags; the
    ``=FILE`` (or a following all-caps placeholder) marks the option
    as taking a value. Brackets in the term are grouping, not flags.
    """
    text = _clean(_strip_roff(term)).replace("[", " ").replace("]", " ")
    # Roff quotes its terms (".IP \"-name pattern\""); the quotes are
    # not part of any flag, but they break the placeholder match
    # (the flag no longer sits at the start of the text).
    text = text.replace('"', " ")
    options: dict[str, dict[str, Any]] = {}
    value_name: str | None = None
    # A flag in a term is -x, --long, or -name (single-dash
    # multi-letter); the placeholder follows it.
    _FLAG_IN_TERM = (r"(?:-[A-Za-z0-9][A-Za-z0-9_-]*"
                     r"|--[A-Za-z0-9][A-Za-z0-9_-]*)")
    eq = re.search(r"=\s*([A-Z][A-Z0-9_-]*)", text)
    if eq:
        value_name = eq.group(1)
    else:
        # "-e PATTERNS, --regexp=PATTERNS" style: a bare all-caps word
        # after a flag is its value placeholder.
        after = re.search(
            r"(?:^|[\s,])" + _FLAG_IN_TERM +
            r"\s+([A-Z][A-Z0-9_-]{1,15})(?:[\s,]|$)", text)
        if after:
            value_name = after.group(1)
        else:
            # Lowercase placeholder (find's "-mtime n",
            # "-name pattern"): in a TERM line a word after the flag
            # is the value placeholder, not prose — the description
            # lives on the following lines, never in the term.
            lower = re.search(
                r"(?:^|[\s,])" + _FLAG_IN_TERM +
                r"\s+([A-Za-z][A-Za-z0-9_-]{0,15})(?:[\s,]|$)", text)
            if lower:
                value_name = lower.group(1)
    for flag in _LONG_FLAG_RE.findall(text):
        options[flag] = {"excerpt": "", "takes_value": value_name is not None,
                         "value_name": value_name, "aliases": []}
    for short in _SHORT_FLAG_RE.findall(text):
        flag = f"-{short}"
        options[flag] = {"excerpt": "", "takes_value": value_name is not None,
                         "value_name": value_name, "aliases": []}
    for multi in _SINGLE_DASH_LONG_RE.findall(text):
        flag = f"-{multi}"
        options[flag] = {"excerpt": "", "takes_value": value_name is not None,
                         "value_name": value_name, "aliases": []}
    # Aliases: every flag in one term is an alias of the others.
    names = list(options)
    for flag in names:
        options[flag]["aliases"] = [n for n in names if n != flag]
    return options


@dataclass
class ManPage:
    """One parsed man page: summary + per-option excerpts."""

    name: str
    section: int = 1
    summary: str = ""
    description: str = ""
    options: dict[str, dict[str, Any]] = field(default_factory=dict)
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "section": self.section,
            "summary": self.summary, "description": self.description,
            "options": self.options, "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManPage:
        return cls(
            name=str(data.get("name", "")),
            section=int(data.get("section", 1) or 1),
            summary=str(data.get("summary", "")),
            description=str(data.get("description", "")),
            options=dict(data.get("options", {}) or {}),
            text=str(data.get("text", "")),
        )


def parse_man_page_text(text: str, *, name: str = "",
                        section: int = 1) -> ManPage:
    """Parse roff man-page source into a :class:`ManPage`.

    Never raises on malformed input: the worst case is a page with
    an empty summary and no options, which the explainer reports as
    unknown rather than inventing a doc.
    """
    page = ManPage(name=name or "", section=section)
    current_section = ""
    pending_term: str | None = None
    pending_desc: list[str] = []
    name_lines: list[str] = []
    desc_lines: list[str] = []
    syn_lines: list[str] = []

    def _flush_term() -> None:
        nonlocal pending_term, pending_desc
        if pending_term is None:
            return
        excerpt = _clean(" ".join(pending_desc))[:_MAX_EXCERPT_CHARS]
        if excerpt:
            for flag, rec in _flags_from_term(pending_term).items():
                if flag not in page.options:
                    page.options[flag] = dict(rec)
                    page.options[flag]["excerpt"] = excerpt
                elif not page.options[flag].get("excerpt"):
                    page.options[flag]["excerpt"] = excerpt
        pending_term = None
        pending_desc = []

    for raw in (text or "").splitlines():
        line = raw.rstrip("\n")
        if line.startswith('.\\"') or line.startswith(".\\\""):
            continue
        match = _MACRO_LINE_RE.match(line)
        if match:
            macro, args = match.group(1).upper(), match.group(2)
            if macro == "TH":
                parts = _strip_roff(args).split()
                if parts and not page.name:
                    page.name = parts[0].lower()
                continue
            if macro == "SH":
                _flush_term()
                current_section = _clean(
                    _strip_roff(args)).strip('"').upper()
                continue
            if macro == "SS":
                _flush_term()
                continue
            if macro == "TP":
                _flush_term()
                pending_term = ""
                continue
            if macro == "IP":
                _flush_term()
                term = _clean(_strip_roff(args))
                if term and ("-" in term or "--" in term):
                    pending_term = term
                continue
            if macro in ("PP", "LP", "P", "SP"):
                if pending_term is not None and pending_desc:
                    pending_desc.append("")
                elif current_section == "DESCRIPTION":
                    desc_lines.append("")
                continue
            if macro in ("RS", "RE", "NF", "FI", "PD", "HP", "EX", "EE"):
                continue
            if macro in ("B", "I", "BR", "BI", "RB", "IR"):
                # Inline-font macro line: the args ARE the text.
                content = _clean(_strip_roff(args))
                if pending_term == "":
                    pending_term = content
                elif pending_term is not None:
                    pending_desc.append(content)
                elif current_section == "NAME":
                    name_lines.append(content)
                elif current_section == "DESCRIPTION":
                    desc_lines.append(content)
                elif current_section == "SYNOPSIS":
                    syn_lines.append(content)
                continue
            # Any other macro: drop the macro, keep the args as text.
            content = _clean(_strip_roff(args))
            if not content:
                continue
            if pending_term == "":
                pending_term = content
            elif pending_term is not None:
                pending_desc.append(content)
            elif current_section == "NAME":
                name_lines.append(content)
            elif current_section == "DESCRIPTION":
                desc_lines.append(content)
            elif current_section == "SYNOPSIS":
                syn_lines.append(content)
            continue
        # Plain text line.
        content = _clean(_strip_roff(line))
        if not content:
            continue
        if pending_term == "":
            pending_term = content
            continue
        if pending_term is not None:
            # A wrapped term line: it ends with "," or starts with a flag.
            if (pending_term.endswith(",")
                    and (content.startswith("-")
                         or content.startswith("\\f"))):
                pending_term = pending_term + " " + content
            else:
                pending_desc.append(content)
            continue
        if current_section == "NAME":
            name_lines.append(content)
        elif current_section == "DESCRIPTION":
            desc_lines.append(content)
        elif current_section == "SYNOPSIS":
            syn_lines.append(content)
    _flush_term()

    name_text = _clean(" ".join(name_lines))
    if name_text:
        # "tar - an archiving utility" → the part after the dash.
        if " - " in name_text:
            page.summary = name_text.split(" - ", 1)[1].strip()
        elif " \\- " in name_text:
            page.summary = name_text.split(" \\- ", 1)[1].strip()
        else:
            page.summary = name_text
    page.description = _clean(" ".join(desc_lines))[:2000]
    synopsis = _clean(" ".join(syn_lines))[:1000]
    parts = [f"{page.name}({page.section})", page.summary,
             page.description, synopsis]
    for flag, rec in page.options.items():
        parts.append(f"{flag} {rec.get('excerpt', '')}")
    page.text = _clean(" ".join(p for p in parts if p))
    page.text = page.text[:_MAX_MAN_TEXT_CHARS]
    return page


def parse_man_page_file(path: str | Path) -> ManPage | None:
    """Parse one man-page file (.gz or plain) → ManPage or None.

    The name/section come from the filename (``tar.1.gz`` → tar/1);
    the ``.TH`` line is a fallback. Unreadable files return None.
    """
    path = Path(path)
    stem = path.name
    if stem.endswith(".gz"):
        stem = stem[:-3]
    match = re.match(r"^(.+?)\.(\d+)(?:\..*)?$", stem)
    name = match.group(1).lower() if match else stem.lower()
    try:
        section = int(match.group(2)) if match else 1
    except ValueError:
        section = 1
    try:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8",
                           errors="replace") as fh:
                text = fh.read()
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        _log.debug("man page unreadable %s: %s", path, exc)
        return None
    return parse_man_page_text(text, name=name, section=section)


# ── --help and tldr parsers ───────────────────────────────────────────

_HELP_OPT_RE = re.compile(
    r"^\s*(?:-([A-Za-z0-9]),\s*)?(--[A-Za-z0-9][A-Za-z0-9_-]*)"
    r"(?:[=\s]\S+)?\s*(.*)$")
_HELP_SHORT_RE = re.compile(
    r"^\s*-([A-Za-z0-9])(?:[=\s](\S+))?\s+(.+)$")


def parse_help_text(text: str, command: str = "") -> dict[str, dict[str, Any]]:
    """Option lines from ``<cmd> --help`` output → {flag: option}.

    Handles ``-w, --wiggle  description`` and short-only
    ``-q  description`` lines, with indented continuation lines
    folded into the previous option's excerpt.
    """
    options: dict[str, dict[str, Any]] = {}
    current: list[str] = []

    def _flush() -> None:
        if current:
            current.clear()

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            _flush()
            continue
        match = _HELP_OPT_RE.match(line)
        if match:
            short, long, desc = match.groups()
            flags: list[str] = []
            if short:
                flags.append(f"-{short}")
            if long:
                flags.append(long)
            takes = ("=" in line.split("--", 1)[-1].split()[0]
                     if long and line.split("--", 1)[-1].split() else False)
            value_name: str | None = None
            eq = re.search(r"=\s*([A-Z][A-Z0-9_-]*)", line)
            if eq:
                value_name = eq.group(1)
                takes = True
            excerpt = _clean(desc)[:_MAX_EXCERPT_CHARS]
            for flag in flags:
                options[flag] = {
                    "excerpt": excerpt, "takes_value": bool(takes),
                    "value_name": value_name,
                    "aliases": [f for f in flags if f != flag],
                }
            current = flags
            continue
        short_match = _HELP_SHORT_RE.match(line)
        if short_match:
            flag = f"-{short_match.group(1)}"
            options[flag] = {
                "excerpt": _clean(short_match.group(3))[:_MAX_EXCERPT_CHARS],
                "takes_value": bool(short_match.group(2)),
                "value_name": short_match.group(2),
                "aliases": [],
            }
            current = [flag]
            continue
        # Indented continuation of the previous option's description.
        if current and line.startswith((" ", "\t")):
            for flag in current:
                prev = options[flag]["excerpt"]
                joined = _clean(prev + " " + line)[:_MAX_EXCERPT_CHARS]
                options[flag]["excerpt"] = joined
            continue
        _flush()
    return options


_TLDR_BOLD_OPT_RE = re.compile(
    r"\*\*(-[A-Za-z0-9](?:,\s*--[A-Za-z0-9][A-Za-z0-9_-]*)?|"
    r"--[A-Za-z0-9][A-Za-z0-9_-]*)\*\*\s*:?\s*(.*)$")
_TLDR_CODE_RE = re.compile(r"`([^`]+)`")


def parse_tldr_markdown(text: str, command: str = "") -> dict[str, dict[str, Any]]:
    """Option lines from a tldr page → {flag: option}.

    tldr pages pair a ``- description:`` bullet with a fenced
    example command; the flags in the example are documented by
    the bullet. Bold ``**-x, --long**: desc`` lines are handled
    too. Returns {} when the page is not a tldr page at all.
    """
    options: dict[str, dict[str, Any]] = {}
    lines = (text or "").splitlines()
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("- ") and idx + 1 < len(lines):
            desc = _clean(line[2:].rstrip(":"))
            # The fenced example follows the bullet, usually after
            # a blank line — scan forward to the next non-blank line.
            nxt = ""
            for following in lines[idx + 1:]:
                if following.strip():
                    nxt = following
                    break
            code = _TLDR_CODE_RE.search(nxt)
            if code and desc:
                for flag in set(_LONG_FLAG_RE.findall(code.group(1)) +
                                [f"-{s}" for s in
                                 _SHORT_FLAG_RE.findall(code.group(1))]):
                    options.setdefault(flag, {
                        "excerpt": desc[:_MAX_EXCERPT_CHARS],
                        "takes_value": False, "value_name": None,
                        "aliases": []})
            continue
        bold = _TLDR_BOLD_OPT_RE.search(line)
        if bold:
            spec, desc = bold.groups()
            for flag in set(_LONG_FLAG_RE.findall(spec) +
                            [f"-{s}" for s in
                             _SHORT_FLAG_RE.findall(spec)]):
                options.setdefault(flag, {
                    "excerpt": _clean(desc)[:_MAX_EXCERPT_CHARS],
                    "takes_value": False, "value_name": None,
                    "aliases": []})
    return options


def fetch_tldr_page(command: str, *, timeout: float = 10.0) -> str | None:
    """Fetch a command's tldr page (markdown) or None. Never raises."""
    name = (command or "").strip().split("/")[-1].lower()
    if not name or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name):
        return None
    for platform in _TLDR_PLATFORMS:
        url = f"{_TLDR_BASE}/{platform}/{name}.md"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Devon/1.0 (tldr lookup)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    continue
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 — offline is a miss
            _log.debug("tldr fetch %s failed: %s", url, exc)
            continue
    return None


# ── the corpus store ──────────────────────────────────────────────────

_MAN_META_TABLE = "man_meta"
_MAN_PAGES_TABLE = "man_pages"
_MAN_FTS_TABLE = "man_pages_fts"


class ManCorpus:
    """Durable man-page corpus: parsed pages + FTS5 index.

    One SQLite file (default ``<home>/data/man_corpus.sqlite``):
    a ``man_pages`` meta table holding the parsed option records
    and an FTS5 table over title+text via
    :class:`nomorals.storage.fts.FTSIndex` — the same backend the
    documents index uses. Ingest is incremental: a page whose file
    mtime+size are unchanged is skipped, so a re-ingest after a
    package upgrade only re-parses what changed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = Database(self.path)
        self._db.execute(
            f'CREATE TABLE IF NOT EXISTS "{_MAN_PAGES_TABLE}" ('
            '"name" TEXT NOT NULL, "section" INTEGER NOT NULL, '
            '"summary" TEXT NOT NULL DEFAULT \'\', '
            '"options_json" TEXT NOT NULL DEFAULT \'{}\', '
            '"text" TEXT NOT NULL DEFAULT \'\', '
            '"mtime" REAL NOT NULL DEFAULT 0, '
            '"size" INTEGER NOT NULL DEFAULT 0, '
            '"source" TEXT NOT NULL DEFAULT \'man\', '
            'PRIMARY KEY ("name", "section"))'
        )
        self._db.execute(
            f'CREATE TABLE IF NOT EXISTS "{_MAN_META_TABLE}" ('
            '"key" TEXT PRIMARY KEY, "value" TEXT NOT NULL)'
        )
        self._fts = FTSIndex(self._db, _MAN_FTS_TABLE,
                             columns=["title", "text"], auto_create=True)

    def find_option(self, flag: str, command: str = "") -> (
            tuple[str | None, dict[str, Any] | None]):
        """Find which man page documents ``flag``.

        The command's own page is the first rung; this is the second:
        options that live on a sibling page (iptables' ``--dport``
        lives on iptables-extensions(8), not iptables(8)). Returns
        ``(page_label, record)`` or ``(None, None)``. Never raises.
        A page whose name starts with the command wins; otherwise
        the alphabetically first documenting page wins. Only pages
        in the command's own family are consulted — frob's --wiggle
        says nothing about frobnicate's --wiggle, so an unrelated
        page is never a hit.
        """
        if not command:
            return None, None
        try:
            rows = self._db.query(
                f'SELECT "name", "section", "options_json" FROM '
                f'"{_MAN_PAGES_TABLE}" WHERE "options_json" LIKE ?',
                (f'%"{flag}":%',))
        except Exception:  # noqa: BLE001 — lookup is a miss
            return None, None
        best: tuple[str, dict[str, Any]] | None = None
        for row in rows:
            name = str(row["name"])
            if not (name == command or name.startswith(command)):
                continue
            try:
                options = json.loads(row["options_json"] or "{}")
            except Exception:  # noqa: BLE001 — a bad row is a miss
                continue
            if flag not in options:
                continue
            label = f"{name}({row['section']})"
            if best is None or label < best[0]:
                best = (label, dict(options[flag]))
        if best is None:
            return None, None
        return best[0], best[1]

    def close(self) -> None:
        with contextlib.suppress(Exception):  # noqa: BLE001 — best-effort
            self._db.close()

    # ── ingest ────────────────────────────────────────────────────
    def ingest(self, man_roots: Any = None,
               *, force: bool = False) -> dict[str, Any]:
        """Parse man pages from ``man_roots`` into the corpus.

        Returns ``{pages, options, skipped, errors}``. Never raises:
        an unreadable page or root is counted, not fatal.
        """
        roots = list(man_roots) if man_roots else list(_DEFAULT_MAN_ROOTS)
        stats: dict[str, Any] = {"pages": 0, "options": 0, "skipped": 0,
                                 "errors": [], "roots": roots}
        with self._db.transaction():
            for root in roots:
                self._ingest_root(root, force, stats)
        self._set_meta("last_ingest", str(time.time()))
        self._set_meta("roots", json.dumps(roots))
        with contextlib.suppress(Exception):  # noqa: BLE001 — best-effort
            self._fts.optimize()
        return stats

    def _ingest_root(self, root: str | Path, force: bool,
                     stats: dict[str, Any]) -> None:
        root_path = Path(root)
        if not root_path.is_dir():
            return
        for section in _MAN_SECTIONS:
            section_dir = root_path / f"man{section}"
            if not section_dir.is_dir():
                continue
            for path in sorted(section_dir.iterdir()):
                if path.is_dir():
                    continue
                if not re.search(rf"\.{section}(\.gz)?$", path.name):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if not force:
                    row = self._db.query_one(
                        f'SELECT "mtime", "size" FROM '
                        f'"{_MAN_PAGES_TABLE}" WHERE "name" = ? '
                        f'AND "section" = ?',
                        (_page_name(path), section))
                    if (row is not None
                            and float(row["mtime"]) == stat.st_mtime
                            and int(row["size"]) == stat.st_size):
                        stats["skipped"] += 1
                        continue
                page = parse_man_page_file(path)
                if page is None:
                    stats["errors"].append(f"unreadable: {path}")
                    continue
                self._store(page, stat.st_mtime, stat.st_size)
                stats["pages"] += 1
                stats["options"] += len(page.options)

    def _store(self, page: ManPage, mtime: float, size: int) -> None:
        rowid = self._db.scalar(
            f'SELECT rowid FROM "{_MAN_PAGES_TABLE}" WHERE "name" = ? '
            f'AND "section" = ?', (page.name, page.section), default=None)
        payload = {
            "name": page.name, "section": page.section,
            "summary": page.summary,
            "options_json": json.dumps(page.options),
            "text": page.text, "mtime": mtime, "size": size,
            "source": "man",
        }
        if rowid is None:
            rowid = self._db.insert(_MAN_PAGES_TABLE, payload)
        else:
            self._db.execute(
                f'UPDATE "{_MAN_PAGES_TABLE}" SET "summary" = ?, '
                f'"options_json" = ?, "text" = ?, "mtime" = ?, '
                f'"size" = ? WHERE rowid = ?',
                (page.summary, payload["options_json"], page.text,
                 mtime, size, int(rowid)))
        self._fts.put(int(rowid), [f"{page.name}({page.section})",
                                   page.text])

    def _set_meta(self, key: str, value: str) -> None:
        self._db.execute(
            f'INSERT INTO "{_MAN_META_TABLE}" ("key", "value") '
            f'VALUES (?, ?) ON CONFLICT("key") DO UPDATE SET '
            f'"value" = excluded."value"', (key, value))

    def _get_meta(self, key: str) -> str | None:
        row = self._db.query_one(
            f'SELECT "value" FROM "{_MAN_META_TABLE}" WHERE "key" = ?',
            (key,))
        return str(row["value"]) if row else None

    # ── reads ─────────────────────────────────────────────────────
    def get(self, name: str) -> ManPage | None:
        """The man page for ``name`` (section 1 preferred), or None."""
        clean = (name or "").strip().split("/")[-1].lower()
        if not clean:
            return None
        rows = self._db.query(
            f'SELECT * FROM "{_MAN_PAGES_TABLE}" WHERE "name" = ? '
            f'ORDER BY CASE WHEN "section" = 1 THEN 0 ELSE 1 END, '
            f'"section" LIMIT 1', (clean,))
        if not rows:
            return None
        return self._row_to_page(rows[0])

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """FTS5 description search → [{name, section, summary, score}]."""
        if not query or not query.strip():
            return []
        hits = self._fts.search(query, limit=limit)
        if not hits:
            return []
        rows = {
            int(r["rowid"]): r
            for r in self._db.query(
                f'SELECT rowid, "name", "section", "summary" FROM '
                f'"{_MAN_PAGES_TABLE}" WHERE "rowid" IN '
                f'({", ".join("?" * len(hits))})',
                [h.rowid for h in hits])
        }
        out = []
        for hit in hits:
            row = rows.get(hit.rowid)
            if row is None:
                continue
            out.append({"name": row["name"], "section": int(row["section"]),
                        "summary": row["summary"], "score": hit.score})
        return out

    def status(self) -> dict[str, Any]:
        """Corpus state: pages, options, roots, last ingest."""
        pages = int(self._db.scalar(
            f'SELECT COUNT(*) FROM "{_MAN_PAGES_TABLE}"', default=0))
        options = 0
        try:
            for row in self._db.query(
                    f'SELECT "options_json" FROM "{_MAN_PAGES_TABLE}"'):
                try:
                    options += len(json.loads(row["options_json"] or "{}"))
                except (ValueError, TypeError):
                    continue
        except Exception:  # noqa: BLE001 — status never raises
            pass
        roots_raw = self._get_meta("roots")
        try:
            roots = json.loads(roots_raw) if roots_raw else []
        except ValueError:
            roots = []
        last = self._get_meta("last_ingest")
        return {
            "pages": pages, "options": options, "roots": roots,
            "path": str(self.path),
            "last_ingest": float(last) if last else None,
        }

    @staticmethod
    def _row_to_page(row: dict[str, Any]) -> ManPage:
        try:
            options = json.loads(row.get("options_json") or "{}")
        except (ValueError, TypeError):
            options = {}
        return ManPage(
            name=str(row["name"]), section=int(row["section"]),
            summary=str(row.get("summary") or ""),
            description="", options=dict(options or {}),
            text=str(row.get("text") or ""))


def _page_name(path: Path) -> str:
    stem = path.name
    if stem.endswith(".gz"):
        stem = stem[:-3]
    match = re.match(r"^(.+?)\.\d+(?:\..*)?$", stem)
    return (match.group(1).lower() if match else stem.lower())


def open_man_corpus(home: str | Path | None) -> ManCorpus:
    """Open (creating if needed) the durable man corpus for a home."""
    base = Path(home) if home else Path.home() / ".nomorals"
    return ManCorpus(base / "data" / "man_corpus.sqlite")
