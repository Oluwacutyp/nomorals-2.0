"""PowerShell files must be ASCII-only.

Regression: ``scripts/deploy_devon_ec2_rdp.ps1`` once contained em-dashes,
curly quotes, and box-drawing characters (UTF-8, no BOM). PowerShell 5.1
reads such a file as Windows-1252, mangling the multi-byte sequences into
mojibake (``a€"``) that breaks string parsing -- a cascade of
``Unexpected token`` / ``Missing closing ')'`` / ``ParseException`` errors
on the owner's EC2, even though the clone itself had succeeded.

ASCII-only files parse identically under every PowerShell encoding
default, so ASCII-only is the rule. This test fails if non-ASCII ever
creeps back into any ``.ps1`` in the repo.

The delimiter-balance check is a best-effort smoke check (pwsh is not
available on this Linux box, so there is no real parser here), not a
full parse. It catches gross truncation / dropped closing braces.
"""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MOJIBAKE_MARKERS = ("â", "€", "Ã", "â€”")


def _ps1_files() -> list[pathlib.Path]:
    return sorted(
        p
        for p in REPO_ROOT.rglob("*.ps1")
        if ".git" not in p.parts and "node_modules" not in p.parts
    )


def test_ps1_files_exist() -> None:
    assert _ps1_files(), "no .ps1 files found under the repo"


def test_ps1_ascii_only() -> None:
    bad: list[str] = []
    for path in _ps1_files():
        data = path.read_bytes()
        non_ascii = sorted({b for b in data if b > 127})
        if non_ascii:
            bad.append(
                f"{path.relative_to(REPO_ROOT)}: non-ASCII bytes "
                f"{[hex(b) for b in non_ascii[:10]]} -- "
                "rewrite with ASCII equivalents (em-dash -> '-', curly quotes -> straight)"
            )
    assert not bad, "non-ASCII bytes in .ps1 files:\n" + "\n".join(bad)


def test_ps1_no_mojibake() -> None:
    bad: list[str] = []
    for path in _ps1_files():
        # latin-1 never fails, so mojibake markers are visible even in a
        # file that is not valid ASCII (covered by test_ps1_ascii_only).
        text = path.read_bytes().decode("latin-1")
        for marker in MOJIBAKE_MARKERS:
            if marker in text:
                bad.append(f"{path.relative_to(REPO_ROOT)}: contains mojibake {marker!r}")
    assert not bad, "mojibake in .ps1 files:\n" + "\n".join(bad)


def test_ps1_balanced_delimiters_smoke() -> None:
    for path in _ps1_files():
        text = path.read_bytes().decode("ascii", errors="replace")
        for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
            assert text.count(opener) == text.count(closer), (
                f"{path.relative_to(REPO_ROOT)}: unbalanced {opener}{closer} "
                f"({text.count(opener)} vs {text.count(closer)}) -- "
                "best-effort smoke check, not a real parser"
            )
