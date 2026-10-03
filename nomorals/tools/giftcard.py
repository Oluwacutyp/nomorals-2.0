"""giftcard — barcode & card-number recovery (wave 78).

The legitimate version of "find the missing code on an over-scratched
gift card": the scratched PIN is a server-verified random secret (nothing
can be generated offline that a card network would honor), but the card's
*barcode* usually survives the scrape, and it encodes the card number.

Actions:
* ``analyze`` — auto-detect + decode a barcode scanline (Code 128 /
  EAN-13 / UPC-A).  Input: bitstring, runs, JSON, or a file with any of
  those.  Returns ranked candidates with check-digit + Luhn status.
* ``recover`` — a partially legible printed number (``?`` holes, e.g.
  ``"5193?45678901"``): bounded enumeration pruned by the mod-10 check
  digit (EAN/UPC) or mod-103 check symbol (Code 128 payload).
* ``scanline`` — a scanned row with a damaged module span: decode around
  it, enumerate the missing symbols until the check symbol validates,
  cross-checked against any partially legible printed number.
* ``verify`` — Luhn + check-digit + format verification of a full number.
* ``encode`` — render a number/payload to modules (verification and
  re-printing).

Fully offline; nothing is transmitted, ever.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..core.barcode import (analyze, card_number, ean_check_digit,
                            encode_code128, encode_ean13, encode_upca,
                            luhn_valid, recover_code128, recover_ean,
                            recover_luhn, recover_scanline,
                            scanline_from_any, upc_check_digit)
from ..core.errors import ToolError
from ..core.policy import Capability

_MAX_SCANLINE = 1_000_000

#: what the vision model is asked to read off a card photo — structured so
#: the deterministic core can act on each line without guessing
_BARCODE_PROMPT = (
    "This photo shows a gift card, store card, or barcode label. "
    "Read the number area carefully and answer with EXACTLY these lines "
    "(omit a line only if truly nothing is legible for it):\n"
    "NUMBER: the printed digit(s) under/above the bars, digit by digit, "
    "spaced in the same groups as printed; use ? for any digit that is "
    "scratched off, smudged, or unreadable (never guess a digit).\n"
    "SCANNER: if you can trace the bars themselves, the run lengths of "
    "black/white bars left to right as a comma list like 1,3,1,1,4,2.\n"
    "PAYLOAD: any other legible alphanumeric code printed on the card "
    "(with ? for unreadable characters)."
)

_IMAGE_FORMATS = {"png", "jpeg", "gif", "webp", "bmp"}
_NUM_HOLE_RE = re.compile(r"\d[\d?*\s-]{9,17}[\d?]")


def _scanline(data: str) -> str:
    bits = scanline_from_any(data)
    if bits is None:
        raise ToolError(
            "could not parse a scanline from data= (expected a 0/1 "
            "bitstring, a runs list, JSON, or a path to one)")
    if len(bits) > _MAX_SCANLINE:
        raise ToolError(f"scanline too long ({len(bits)} modules)")
    return bits


def giftcard(action: str = "analyze", *,
             data: str = "", number: str = "",
             known: str = "", unknown_start: int = -1,
             unknown_end: int = -1, max_symbols: int = 3,
             alphabet: str = "", max_candidates: int = 5000) -> dict[str, Any]:
    action = (action or "analyze").strip().lower()

    if action == "verify":
        value = card_number(number or data)
        digits = re.sub(r"\D", "", value)
        report: dict[str, Any] = {"number": value, "digits": digits}
        report["luhn_ok"] = luhn_valid(value)
        if len(digits) == 13:
            report["ean13_check_ok"] = int(digits[12]) == ean_check_digit(digits[:12])
        elif len(digits) == 12 and digits[0] == "0":
            report["upca_check_ok"] = int(digits[11]) == upc_check_digit(digits[:11])
        report["valid"] = bool(report.get("luhn_ok") or report.get("ean13_check_ok")
                               or report.get("upca_check_ok"))
        return report

    if action == "analyze":
        if not data:
            raise ToolError("giftcard analyze needs data= (scanline or file)")
        _scanline(data)  # validate + size check
        return analyze(data)

    if action == "recover":
        template = re.sub(r"\s", "", number or data or "")
        if not template:
            raise ToolError("giftcard recover needs number= (e.g. '5193?45678901')")
        if re.fullmatch(r"[0-9?*]{12,13}", template):
            out = recover_ean(template, max_candidates=max_candidates)
        elif re.fullmatch(r"[0-9?*]{10,19}", template):
            out = recover_luhn(template, max_candidates=max_candidates)
        elif "?" in template or "*" in template:
            out = recover_code128(
                template,
                alphabet=alphabet or None,
                max_candidates=max_candidates)
        else:
            out = {"input": template, "candidates": [],
                   "note": "no holes in template — use action=verify for a full number"}
        return out

    if action in ("scanline", "scanline-recover", "recover-scanline"):
        if not data:
            raise ToolError("giftcard scanline needs data= (scanned row)")
        unknown = None
        if unknown_start >= 0:
            end = unknown_end if unknown_end >= 0 else min(
                unknown_start + 11 * max_symbols, _MAX_SCANLINE)
            unknown = (unknown_start, end)
        return recover_scanline(
            _scanline(data),
            unknown=unknown,
            known_number=known or None,
            max_symbols=max(1, min(int(max_symbols or 3), 3)),
            max_candidates=max_candidates)

    if action == "encode":
        text = number or data
        if not text:
            raise ToolError("giftcard encode needs number= or data=")
        digits = re.sub(r"\D", "", text)
        if len(digits) == 12:
            return encode_ean13(digits)
        if len(digits) == 11:
            return encode_upca(digits)
        return encode_code128(text)

    raise ToolError(f"unknown action {action!r} (analyze|recover|scanline|"
                    "verify|encode)")


def _summary(rep: dict[str, Any]) -> str:
    """One-line human summary for chat replies."""
    if not rep:
        return "no result"
    if "number" in rep and "luhn_ok" in rep:
        return (f"card number {rep['number']}: Luhn "
                f"{'PASS' if rep['luhn_ok'] else 'FAIL'}"
                + (f", EAN-13 check "
                   f"{'PASS' if rep.get('ean13_check_ok') else 'FAIL'}"
                   if "ean13_check_ok" in rep else "")
                + (f", UPC-A check "
                   f"{'PASS' if rep.get('upca_check_ok') else 'FAIL'}"
                   if "upca_check_ok" in rep else ""))
    if "recovered" in rep:
        return (f"recovered {rep['recovered']!r} "
                f"({len(rep.get('candidates') or [])} candidate(s), "
                f"{rep.get('holes', '?')} hole(s) searched)")
    if "candidates" in rep:
        cands = rep.get("candidates") or []
        if not cands:
            return f"no candidates ({rep.get('note', 'unrecognized')})"
        top = cands[0]
        return (f"best: {top.get('value')!r} [{top.get('symbology')}] "
                f"(check {'ok' if top.get('check_ok') else 'FAIL'}, "
                f"Luhn {'ok' if top.get('luhn_ok') else 'n/a'}, "
                f"{len(cands)} candidate(s))")
    return json.dumps(rep)[:200]


# ── wave 79: photo → recovery (the vision→barcode bridge) ──────────────────

def _normalize_number(raw: str) -> str:
    """Spaces/dashes out; digits and ?/* holes kept."""
    return re.sub(r"[\s\-]", "", raw or "")


def _extract_templates(text: str, known: str = "") -> dict[str, str]:
    """Pull every recoverable template out of what vision/OCR saw.

    Structured ``NUMBER:``/``SCANNER:``/``PAYLOAD:`` lines win; free text
    is scanned for a 10-19 digit group with holes, or an explicit
    alphanumeric code.  ``known`` (an operator-supplied template) is
    authoritative for the number slot.
    """
    text = (text or "").strip()
    number_raw = scanner_raw = payload_raw = ""
    for line in text.splitlines():
        marker, sep, rest = line.strip().partition(":")
        if not sep:
            continue
        key = marker.strip().upper()
        if key in {"NUMBER", "NUMBERS", "PRINTED NUMBER", "PRINTED"}:
            number_raw = rest.strip()
        elif key in {"SCANNER", "SCANNER RUNS", "RUNS"}:
            scanner_raw = rest.strip()
        elif key in {"PAYLOAD", "CODE 128", "CODE128", "CODE"}:
            payload_raw = rest.strip()
    if not number_raw:
        flat = text.replace("\n", " ")
        m = _NUM_HOLE_RE.search(flat)
        if m:
            number_raw = m.group(0)
    if not payload_raw and ("?" in text or "*" in text):
        m = re.search(r"[A-Z0-9?*\-]{6,40}", text, re.I)
        if m and re.search(r"[?*]", m.group(0)):
            payload_raw = m.group(0)

    number = _normalize_number(known or number_raw)
    out: dict[str, str] = {}
    if re.fullmatch(r"[0-9?*]{10,19}", number or ""):
        out["number"] = number
    if scanner_raw:
        out["scanner"] = scanner_raw
    if payload_raw:
        out["payload"] = _normalize_number(payload_raw)
    return out


def photo_recover(seen_text: str, *, known: str = "") -> dict[str, Any]:
    """Deterministic half of the photo bridge: what the vision stage read
    (free text or structured lines) becomes ranked, check-validated
    candidates — no model in this path, ever.
    """
    templates = _extract_templates(seen_text, known=known)
    if not templates:
        return {
            "recovered": "",
            "candidates": [],
            "templates": {},
            "note": ("no card number, scanline runs, or payload found in "
                     "the image reading — try a sharper photo or pass "
                     "known= with whatever you can read by hand"),
        }
    candidates: list[dict[str, Any]] = []
    recovered = ""
    notes: list[str] = []

    if "scanner" in templates:
        bits = scanline_from_any(templates["scanner"])
        if bits:
            rep = analyze(bits) if len(bits) <= _MAX_SCANLINE else {
                "ok": False, "note": "scanline too long", "candidates": []}
            for c in (rep.get("candidates") or [])[:50]:
                candidates.append(dict(c, source="scanner"))
            if not (rep.get("candidates") or []):
                notes.append(f"scanner runs unreadable ({rep.get('note', '')})")
        else:
            notes.append("scanner line could not be parsed as runs")

    if "number" in templates:
        template = templates["number"]
        complete = bool(re.fullmatch(r"[0-9]+", template))
        if complete:
            report = {"number": template, "digits": template}
            report["luhn_ok"] = luhn_valid(template)
            if len(template) == 13:
                report["ean13_check_ok"] = (
                    int(template[12]) == ean_check_digit(template[:12]))
            elif len(template) == 12 and template[0] == "0":
                report["upca_check_ok"] = (
                    int(template[11]) == upc_check_digit(template[:11]))
            if report.get("luhn_ok") or report.get("ean13_check_ok") \
                    or report.get("upca_check_ok"):
                recovered = recovered or template
                candidates.append({"value": template,
                                   "symbology": "printed number",
                                   "check_ok": True, "source": "printed"})
            else:
                notes.append("printed number fails every checksum — "
                             "some digit is misread")
        else:
            digits = re.sub(r"[?*]", "", template)
            if len(template) in (12, 13) and len(digits) >= 9:
                rep = recover_ean(template)
            elif len(digits) >= 9:
                rep = recover_luhn(template)
            else:
                rep = {"candidates": [], "note": "too few known digits"}
            for c in (rep.get("candidates") or [])[:50]:
                candidates.append(dict(c, source="printed"))
            if not (rep.get("candidates") or []):
                notes.append(rep.get("note", "no candidates from number"))

    if "payload" in templates:
        rep = recover_code128(templates["payload"])
        for c in (rep.get("candidates") or [])[:50]:
            candidates.append(dict(c, source="payload"))
        if not (rep.get("candidates") or []):
            notes.append("no candidates from payload")

    # dedupe, keep check-validated first
    seen: set[str] = set()
    ranked: list[dict[str, Any]] = []
    for c in candidates:
        key = c.get("value", "")
        if key in seen:
            continue
        seen.add(key)
        ranked.append(c)
    ranked.sort(key=lambda c: (not c.get("check_ok", c.get("luhn_ok", False)),
                               -len(c.get("value", ""))))
    if not recovered:
        validated = [c for c in ranked
                     if c.get("check_ok") or c.get("luhn_ok")]
        if len(validated) == 1:
            recovered = validated[0]["value"]
    return {
        "recovered": recovered,
        "candidates": ranked[:50],
        "templates": templates,
        "note": "; ".join(notes) if notes else "",
    }


def photo(context: Any, path: str = "", *, url: str = "",
          known: str = "") -> dict[str, Any]:
    """Photo of a scratched card → recovered number.

    The vision stage (a real vision model on the actual pixels, tesseract
    OCR as the floor) reads the number area into structured text; the
    recovery stage is the deterministic barcode core.  Returns both, so
    the operator can see exactly what was read before the math ran.
    """
    if not path and not url:
        raise ToolError("giftcard photo needs a path= or url=")
    from .filesystem import safe_path
    from .vision import describe, image_metadata

    if path:
        target = safe_path(context, path, must_exist=True)
        data = target.read_bytes()
    else:
        from ..core.http import HttpClient
        data = HttpClient(timeout=60.0).get(url).body
    meta = image_metadata(data)
    if meta.get("format") not in _IMAGE_FORMATS:
        raise ToolError(f"not a recognized image ({meta.get('format', '?')})")
    seen = describe(context, data, _BARCODE_PROMPT, ocr=True)
    vision_text = (seen.get("description") or "").strip()
    ocr_text = (seen.get("ocr_text") or "").strip()
    # the vision_text field carries the full explanation; the note just
    # flags it so the recovery result reads as one sentence
    note_prefix = "vision unavailable; " if vision_text.startswith(
        "[vision unavailable") else ""
    recovery = photo_recover(f"{vision_text}\n{ocr_text}", known=known)
    if note_prefix:
        recovery["note"] = (note_prefix + recovery.get("note", "")).strip("; ")
    return {
        "image": {k: meta.get(k) for k in
                  ("format", "width", "height", "bytes")},
        "provider": seen.get("provider", ""),
        "vision_text": vision_text[:1500],
        "ocr_text": ocr_text or None,
        "recovery": recovery,
    }


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "giftcard",
        description=(
            "Gift card / barcode number recovery. photo: a photo of a "
            "scratched card — the vision model reads the number area and "
            "the check-digit math recovers it. analyze: decode a barcode "
            "scanline (Code 128 / EAN-13 / UPC-A) from bits, runs, JSON "
            "or a file. recover: fill ? holes in a partially legible "
            "printed number using the check digit (12/13-digit numbers), "
            "Luhn (10-19 digit PANs) or mod-103 check symbol (Code 128 "
            "payloads). scanline: recover a damaged scanline span, "
            "cross-checked against a partially legible printed number "
            "(known=). verify: Luhn + check digit on a full number. "
            "encode: render a number to modules. "
        ),
        capability=Capability.NET_OUT,
    )
    # Note: giftcard also uses FS_READ (file inputs) and MODEL_CALL (vision
    # for photo action). NET_OUT is declared as the highest-privilege need.
    def _giftcard(action: str = "analyze", *,
                  data: str = "", number: str = "", known: str = "",
                  unknown_start: int = -1, unknown_end: int = -1,
                  max_symbols: int = 3, alphabet: str = "",
                  max_candidates: int = 5000,
                  url: str = "") -> dict[str, Any]:
        if action == "photo":
            return photo(context, data or "", url=url, known=known)
        return giftcard(action, data=data, number=number, known=known,
                        unknown_start=unknown_start, unknown_end=unknown_end,
                        max_symbols=max_symbols, alphabet=alphabet,
                        max_candidates=max_candidates)

