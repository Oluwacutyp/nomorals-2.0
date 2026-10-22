"""ApkForge native hex patching — byte-pattern surgery inside
``.so`` files, under the same fail-closed discipline as A2's smali
patch engine (design §1.2/§4 A3).

Native code has no source to edit: an ``.so`` edit is exact bytes at
exact offsets.  A hex patch is a JSON document:

    {"name": "neutralize-integrity-check",
     "target": "lib/arm64-v8a/libapp.so",
     "match": {"pattern": "1f 20 03 d5 ?? ?? ?? ??",
               "expect_matches": 1},
     "replace": "c0 03 5f d6 1f 20 03 d5",
     "target_apk_sha256": "…",            (optional; checked)
     "note": "why this patch exists"}

Rules, all fail-closed:

* ``pattern`` is hex bytes; ``??`` (or ``xx``) is a wildcard nibble
  pair matching any byte.  ``replace`` is concrete hex of EXACTLY
  the pattern's byte length — a byte patch can never grow or shrink
  the file (offsets after the patch would silently shift).
* The occurrences found in the file must number EXACTLY
  ``expect_matches`` — zero, or one-more-than-expected, refuses the
  patch whole: bytes untouched, refusal journaled.
* The target must be an ELF file (``\\x7fELF`` magic) under the
  decoded project's ``lib/`` tree — anything else refuses by name.
* A passing patch lands through ``edit.apply_edit``: pre-edit
  snapshot, after-hash, journal row carrying every patched offset,
  and ``undo`` restores the exact pre-patch bytes — the same
  reversibility contract as every other edit in the class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .edit import apply_edit, decoded_dir
from .project import Project

__all__ = ["HexPatchError", "parse_hexpatch", "find_pattern",
           "apply_hexpatch"]


class HexPatchError(RuntimeError):
    """A designed hex-patch refusal/failure — names the cause."""


def _parse_pattern(text: str) -> tuple[bytes, bytes]:
    """Hex string → (bytes, mask).  Mask byte 0xFF = must match,
    0x00 = wildcard (from ``??``/``xx`` tokens)."""
    tokens = str(text or "").replace(",", " ").split()
    if not tokens:
        # allow continuous hex: "1f2003d5"
        compact = str(text or "").strip()
        if compact and all(c in "0123456789abcdefABCDEF"
                           for c in compact) \
                and len(compact) % 2 == 0:
            tokens = [compact[i:i + 2]
                      for i in range(0, len(compact), 2)]
    if not tokens:
        raise HexPatchError("match.pattern is empty — a hex patch "
                            "needs a byte pattern")
    pat = bytearray()
    mask = bytearray()
    for tok in tokens:
        t = tok.lower()
        if t in ("??", "xx"):
            pat.append(0)
            mask.append(0)
            continue
        if len(t) != 2 or any(c not in "0123456789abcdef"
                              for c in t):
            raise HexPatchError(
                f"pattern token {tok!r} is not a hex byte or ??")
        pat.append(int(t, 16))
        mask.append(0xFF)
    return bytes(pat), bytes(mask)


def _parse_replace(text: str, want_len: int) -> bytes:
    tokens = str(text or "").replace(",", " ").split()
    if not tokens:
        compact = str(text or "").strip()
        if compact and len(compact) % 2 == 0:
            tokens = [compact[i:i + 2]
                      for i in range(0, len(compact), 2)]
    out = bytearray()
    for tok in tokens:
        t = tok.lower()
        if t in ("??", "xx"):
            raise HexPatchError(
                "replace cannot contain wildcards — a patch writes "
                "concrete bytes")
        if len(t) != 2 or any(c not in "0123456789abcdef"
                              for c in t):
            raise HexPatchError(
                f"replace token {tok!r} is not a hex byte")
        out.append(int(t, 16))
    if len(out) != want_len:
        raise HexPatchError(
            f"replace is {len(out)} byte(s) but the pattern is "
            f"{want_len} — a byte patch must preserve file length "
            "exactly (later offsets would shift)")
    return bytes(out)


def parse_hexpatch(doc: Any) -> dict[str, Any]:
    if isinstance(doc, str):
        import json

        try:
            doc = json.loads(doc)
        except json.JSONDecodeError as exc:
            raise HexPatchError(
                f"hex patch is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise HexPatchError("hex patch must be a JSON object")
    target = str(doc.get("target") or "").replace("\\", "/")
    parts = [p for p in target.split("/") if p not in ("", ".")]
    if not parts or parts[0] != "lib" or ".." in parts \
            or not parts[-1].endswith(".so"):
        raise HexPatchError(
            f"target {doc.get('target')!r} must be a .so under the "
            "decoded project's lib/ tree")
    match = doc.get("match") or {}
    if not match.get("pattern"):
        raise HexPatchError("hex patch needs match.pattern")
    pat, mask = _parse_pattern(str(match["pattern"]))
    repl = _parse_replace(str(doc.get("replace") or ""), len(pat))
    expect = int(match.get("expect_matches",
                           doc.get("expect_matches", 1)))
    if expect < 1:
        raise HexPatchError("expect_matches must be >= 1 — a patch "
                            "that expects to match nothing is not a "
                            "patch")
    return {"name": str(doc.get("name") or "unnamed-hexpatch"),
            "target": "/".join(parts),
            "pattern": pat, "mask": mask, "replace": repl,
            "expect_matches": expect,
            "target_apk_sha256": str(doc.get("target_apk_sha256")
                                     or ""),
            "note": str(doc.get("note") or "")}


def find_pattern(data: bytes, pat: bytes, mask: bytes) -> list[int]:
    """All start offsets where ``pat`` matches under ``mask``
    (overlapping matches included)."""
    out: list[int] = []
    n, m = len(data), len(pat)
    if m == 0 or m > n:
        return out
    first = next((i for i in range(m) if mask[i]), None)
    if first is None:
        raise HexPatchError("pattern is all wildcards — it would "
                            "match everywhere; refuse to guess")
    anchor = pat[first]
    start = 0
    while True:
        idx = data.find(bytes([anchor]), start)
        if idx < 0 or idx - first + m > n:
            break
        base = idx - first
        if base >= 0 and all(
                mask[i] == 0 or data[base + i] == pat[i]
                for i in range(m)):
            out.append(base)
        start = idx + 1
    return out


def apply_hexpatch(proj: Project, doc: Any) -> dict[str, Any]:
    """Validate, locate, and apply one hex patch — or refuse it
    whole, journaled either way (smali patch engine idiom)."""
    patch = parse_hexpatch(doc)
    name = patch["name"]
    want_sha = patch["target_apk_sha256"]
    if want_sha and want_sha != proj.sha256:
        proj.journal("hexpatch_refused",
                     f"{name}: target_apk_sha256 does not match "
                     f"this project's original ({proj.sha256[:16]}…)")
        raise HexPatchError(
            f"hex patch {name!r} targets sha256 {want_sha[:16]}… "
            f"but this project's original is {proj.sha256[:16]}… — "
            "refused, nothing changed")
    rel = patch["target"]
    target = decoded_dir(proj) / rel
    if not target.exists():
        raise HexPatchError(
            f"{rel} does not exist in the decoded project — run "
            "decode first, and check the lib path")
    data = target.read_bytes()
    if data[:4] != b"\x7fELF":
        raise HexPatchError(
            f"{rel} is not an ELF file (no \\x7fELF magic) — hex "
            "patching native libraries only; refused")
    offsets = find_pattern(data, patch["pattern"], patch["mask"])
    expect = patch["expect_matches"]
    if len(offsets) != expect:
        proj.journal(
            "hexpatch_refused",
            f"{name}: expected {expect} occurrence(s) of the "
            f"pattern in {rel}, found {len(offsets)} — fail-closed, "
            "bytes untouched",
            patch=name, offsets=offsets)
        raise HexPatchError(
            f"hex patch {name!r} refused: expected exactly {expect} "
            f"occurrence(s) in {rel}, found {len(offsets)} — "
            "fail-closed, bytes untouched")
    repl = patch["replace"]

    def mutate() -> None:
        buf = bytearray(target.read_bytes())
        for off in offsets:
            buf[off:off + len(repl)] = repl
        target.write_bytes(bytes(buf))

    result = apply_edit(
        proj, "hexpatch",
        f"{name}: {len(offsets)} site(s) in {rel} at "
        + ", ".join(hex(o) for o in offsets),
        [rel], mutate, patch=name, offsets=offsets,
        note=patch["note"])
    result["patch"] = name
    result["offsets"] = offsets
    return result
