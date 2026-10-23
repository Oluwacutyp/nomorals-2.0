"""Pin the declared Python floor (``requires-python >= 3.11``).

PEP 701 (Python 3.12) legalized newlines inside the expression braces
of a NON-triple-quoted f-string. On 3.11 the tokenizer ends such a
string at the newline and the file dies with "unterminated string
literal" -- this exact construct in ``apkforge/build.py`` and
``apkforge/teach.py`` failed the EC2 health layering check on every
/update (the host venv is 3.11.9) while every 3.12 machine parsed the
tree clean. ``ast.parse(feature_version=(3, 11))`` does NOT model the
restriction, so the suite enforces the floor at the tokenizer level:
any non-triple-quoted f-string whose FSTRING_START/FSTRING_END tokens
sit on different physical lines is a floor violation.

Host independence: on a pre-3.12 host there are no FSTRING_* tokens,
but there the violating file cannot even be tokenized, so a tokenize
error is recorded as a violation too -- the test is green exactly when
the package is loadable by the floor Python.
"""
from __future__ import annotations

import io
import subprocess
import token
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "nomorals"

_FSTRING_START = getattr(token, "FSTRING_START", None)
_FSTRING_END = getattr(token, "FSTRING_END", None)


def _tracked_package_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--full-name",
             "--", "nomorals"],
            capture_output=True, check=True, timeout=60,
        ).stdout.decode("utf-8", "surrogateescape")
        files = [REPO_ROOT / p for p in out.split("\0")
                 if p.endswith(".py")]
        if files:
            return sorted(files)
    except Exception:
        pass
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py")
                  if "__pycache__" not in p.parts)


def _floor_violations(path: Path) -> list[str]:
    raw = path.read_bytes()
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(raw).readline))
    except Exception as exc:
        return [f"cannot be tokenized: {exc}"]
    if _FSTRING_START is None:
        # Pre-3.12 host: tokenization above is itself the floor check.
        return []
    problems = []
    stack: list[tuple[int, bool]] = []
    for tok in tokens:
        if tok.type == _FSTRING_START:
            text = tok.string
            i = 0
            while i < len(text) and text[i] not in "\"'":
                i += 1
            stack.append((tok.start[0], text[i:i + 3] in ('"""', "'''")))
        elif tok.type == _FSTRING_END and stack:
            start_line, triple = stack.pop()
            if not triple and tok.end[0] != start_line:
                problems.append(
                    f"non-triple f-string spans lines "
                    f"{start_line}-{tok.end[0]} (PEP 701, 3.12-only)")
    return problems


def test_no_pep701_multiline_fstrings_in_package():
    offenders = []
    for path in _tracked_package_files():
        for problem in _floor_violations(path):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {problem}")
    assert not offenders, (
        "Python floor is 3.11 (requires-python >=3.11) but these files "
        "use PEP 701 multiline non-triple f-strings, which 3.11 reads "
        "as 'unterminated string literal':\n" + "\n".join(offenders))


def test_floor_check_catches_the_historical_offenders():
    """The check itself is not vacuous: feed it the pre-fix shape."""
    bad = b'x = f"({\'a\'\nif cond else \'b\'})"\n'
    good = b'x = f"({kind})"\n'
    triple = b'x = f"""\n{kind}\n"""\n'

    def spans(src: bytes) -> list[tuple[int, int]]:
        found = []
        stack = []
        for tok in tokenize.tokenize(io.BytesIO(src).readline):
            if _FSTRING_START is not None and tok.type == _FSTRING_START:
                text = tok.string
                i = 0
                while i < len(text) and text[i] not in "\"'":
                    i += 1
                stack.append((tok.start[0],
                              text[i:i + 3] in ('"""', "'''")))
            elif (_FSTRING_END is not None and tok.type == _FSTRING_END
                    and stack):
                start, is_triple = stack.pop()
                if not is_triple:
                    found.append((start, tok.end[0]))
        return found

    if _FSTRING_START is None:
        return  # pre-3.12 host: the tokenizer itself rejects the shape
    assert spans(bad) == [(1, 2)]
    assert spans(good) == [(1, 1)]
    assert spans(triple) == []
