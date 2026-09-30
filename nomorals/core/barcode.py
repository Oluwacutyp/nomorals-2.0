"""Barcode decode + recovery — Code 128, EAN-13, UPC-A (wave 78).

The legitimate version of "find the missing code on an over-scratched
gift card": the scratched-off PIN is a server-verified random secret and
cannot be generated offline, but the card's *barcode* usually survives the
scrape — and barcodes encode the card/product number.  This module is the
recovery engine for that.

Ships (all fully working, no image dependency — inputs are explicit module
runs or 1-bit scanlines, from a string, JSON, or a file):

* **Code 128** — full 103-symbol data table (verified against two
  independent mature implementations), charsets A/B/C with automatic
  switching, weighted mod-103 check symbol, encoder *and* decoder.
* **EAN-13 / UPC-A** — full L/G/R parity tables, guard patterns,
  mod-10 check digit, encoder and decoder, number-system reporting.
* **Luhn** — verification + check-digit computation for card numbers.
* **Partial-damage recovery** — the over-scratched use case:
    - ``recover_ean`` — a 12/13-digit number with ``?`` holes: enumerate
      the missing digits (bounded) and keep the check-digit-valid ones.
      With a 13-digit number and one hole the check digit leaves at most
      one candidate; two holes leave at most ten.
    - ``recover_code128`` — a Code-128 payload template with ``?`` holes
      (bounded, default alphabet digits+upper): enumerate and keep the
      mod-103-check-valid payloads.
    - ``recover_scanline`` — a real scanned 1-D image row with a damaged
      span: decode everything around it, then enumerate candidate
      symbols inside the span until the check symbol validates,
      cross-checked against any partially legible printed number.
* ``analyze`` — auto-detect symbology from a scanline and rank
  candidates; ``card_number`` heuristics normalize the result.

Symbol tables
-------------
The Code 128 pattern table and the EAN L/G/R tables below were verified
entry-by-entry against python-barcode (WhyNotHugo) and barby (toretore),
two independent maintained implementations; the stop pattern 2331112
(11-module stop symbol + 2-module terminal bar) matches the ISO/IEC
15417 description (108 total symbols: 103 data + 3 start + stop/reverse).
"""
from __future__ import annotations

import itertools
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "BarcodeCandidate",
    "analyze",
    "card_number",
    "decode_code128",
    "decode_ean13",
    "decode_upca",
    "ean_check_digit",
    "encode_code128",
    "encode_ean13",
    "encode_upca",
    "luhn_check_digit",
    "luhn_valid",
    "recover_code128",
    "recover_ean",
    "recover_scanline",
    "scanline_from_any",
    "upc_check_digit",
]

# ---------------------------------------------------------------------------
# Code 128 tables
# ---------------------------------------------------------------------------

# 106 patterns: 103 data symbols (0-102) + START A/B/C (103/104/105).
# Each is 11 modules: bar,space,bar,space,bar,space.  Bar widths sum even,
# space widths sum odd (that invariant defines the 108 possible symbols;
# stop + reverse stop complete the set).
C128_PATTERNS: Tuple[str, ...] = (
    "11011001100",  # 0
    "11001101100",  # 1
    "11001100110",  # 2
    "10010011000",  # 3
    "10010001100",  # 4
    "10001001100",  # 5
    "10011001000",  # 6
    "10011000100",  # 7
    "10001100100",  # 8
    "11001001000",  # 9
    "11001000100",  # 10
    "11000100100",  # 11
    "10110011100",  # 12
    "10011011100",  # 13
    "10011001110",  # 14
    "10111001100",  # 15
    "10011101100",  # 16
    "10011100110",  # 17
    "11001110010",  # 18
    "11001011100",  # 19
    "11001001110",  # 20
    "11011100100",  # 21
    "11001110100",  # 22
    "11101101110",  # 23
    "11101001100",  # 24
    "11100101100",  # 25
    "11100100110",  # 26
    "11101100100",  # 27
    "11100110100",  # 28
    "11100110010",  # 29
    "11011011000",  # 30
    "11011000110",  # 31
    "11000110110",  # 32
    "10100011000",  # 33
    "10001011000",  # 34
    "10001000110",  # 35
    "10110001000",  # 36
    "10001101000",  # 37
    "10001100010",  # 38
    "11010001000",  # 39
    "11000101000",  # 40
    "11000100010",  # 41
    "10110111000",  # 42
    "10110001110",  # 43
    "10001101110",  # 44
    "10111011000",  # 45
    "10111000110",  # 46
    "10001110110",  # 47
    "11101110110",  # 48
    "11010001110",  # 49
    "11000101110",  # 50
    "11011101000",  # 51
    "11011100010",  # 52
    "11011101110",  # 53
    "11101011000",  # 54
    "11101000110",  # 55
    "11100010110",  # 56
    "11101101000",  # 57
    "11101100010",  # 58
    "11100011010",  # 59
    "11101111010",  # 60
    "11001000010",  # 61
    "11110001010",  # 62
    "10100110000",  # 63
    "10100001100",  # 64
    "10010110000",  # 65
    "10010000110",  # 66
    "10000101100",  # 67
    "10000100110",  # 68
    "10110010000",  # 69
    "10110000100",  # 70
    "10011010000",  # 71
    "10011000010",  # 72
    "10000110100",  # 73
    "10000110010",  # 74
    "11000010010",  # 75
    "11001010000",  # 76
    "11110111010",  # 77
    "11000010100",  # 78
    "10001111010",  # 79
    "10100111100",  # 80
    "10010111100",  # 81
    "10010011110",  # 82
    "10111100100",  # 83
    "10011110100",  # 84
    "10011110010",  # 85
    "11110100100",  # 86
    "11110010100",  # 87
    "11110010010",  # 88
    "11011011110",  # 89
    "11011110110",  # 90
    "11110110110",  # 91
    "10101111000",  # 92
    "10100011110",  # 93
    "10001011110",  # 94
    "10111101000",  # 95
    "10111100010",  # 96
    "11110101000",  # 97
    "11110100010",  # 98
    "10111011110",  # 99
    "10111101110",  # 100
    "11101011110",  # 101
    "11110101110",  # 102
    "11010000100",  # 103 = START A
    "11010010000",  # 104 = START B
    "11010011100",  # 105 = START C
)

C128_STOP = "11000111010"          # 11-module stop symbol (value 106)
C128_TERMINAL_BAR = "11"           # mandatory 2-module terminal bar
C128_STOP_FULL = C128_STOP + C128_TERMINAL_BAR  # 13 modules: 2331112
C128_REVERSE_STOP = "11010111000"  # the stop pattern read right-to-left
C128_START = {"A": 103, "B": 104, "C": 105}

# Character value tables per code set (data values 0-102).
C128_SET_A: Dict[str, int] = {
    **{ch: i for i, ch in enumerate(
        " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_")},
    **{chr(i): i + 64 for i in range(0, 32)},  # control codes
    "FNC3": 96, "FNC2": 97, "SHIFT": 98, "TO_C": 99,
    "TO_B": 100, "FNC4": 101, "FNC1": 102,
}
C128_SET_B: Dict[str, int] = {
    **{ch: i for i, ch in enumerate(
        " !\"#$%&'()*+,-./0123456789:;<=>?@"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~")},
    "\x7f": 95,
    "FNC3": 96, "FNC2": 97, "SHIFT": 98, "TO_C": 99,
    "FNC4": 100, "TO_A": 101, "FNC1": 102,
}
C128_SET_C: Dict[str, int] = {
    **{str(i).zfill(2): i for i in range(100)},  # digit pairs
    "TO_B": 100, "TO_A": 101, "FNC1": 102,
}

# Reverse lookups for decoding (per set: value -> text).
def _reverse(mapping: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in mapping.items()}


C128_REVERSE_A = _reverse(C128_SET_A)
C128_REVERSE_B = _reverse(C128_SET_B)
C128_REVERSE_C = _reverse(C128_SET_C)
C128_VALUE_TO_BITS = {i: p for i, p in enumerate(C128_PATTERNS)}
C128_BITS_TO_VALUE: Dict[str, int] = {p: i for i, p in enumerate(C128_PATTERNS)}

# ---------------------------------------------------------------------------
# EAN-13 / UPC-A tables
# ---------------------------------------------------------------------------

# L (odd parity), G (even parity), R (mirror of L).  Standard tables.
EAN_L: Tuple[str, ...] = (
    "0001101", "0011001", "0010011", "0111101", "0100011",
    "0110001", "0101111", "0111011", "0110111", "0001011",
)
EAN_G: Tuple[str, ...] = (
    "0100111", "0110011", "0011011", "0100001", "0011101",
    "0111001", "0000101", "0010001", "0001001", "0010111",
)
EAN_R: Tuple[str, ...] = (
    "1110010", "1100110", "1101100", "1000010", "1011100",
    "1001110", "1010000", "1000100", "1001000", "1110100",
)
# Left-half parity pattern for each first (system) digit: 'L' or 'G'.
EAN_LEFT_PARITY: Tuple[str, ...] = (
    "LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
    "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL",
)
EAN_GUARD = "101"
EAN_CENTER = "01010"

# Number-system (first digit) meaning, for the report.
EAN_SYSTEMS: Dict[int, str] = {
    0: "US/Canada (UPC)", 1: "US/Canada", 2: "in-store use",
    3: "France", 4: "Germany", 5: "Italy", 6: "Spain",
    7: "UK", 8: "Netherlands", 9: "Sweden",
}


# ---------------------------------------------------------------------------
# Scanline normalization
# ---------------------------------------------------------------------------

def _bits_from_sequence(seq: Sequence[Any]) -> Optional[str]:
    """Accepts a 0/1 sequence or a runs list [(dark, width), ...]."""
    flat: List[int] = []
    first = seq[0]
    if isinstance(first, (list, tuple)):
        for run in seq:
            if len(run) != 2:
                return None
            dark, width = run
            try:
                width = int(width)
            except (TypeError, ValueError):
                return None
            if width <= 0 or not (isinstance(dark, (bool, int, float))):
                return None
            flat.extend([1 if dark else 0] * width)
    else:
        for bit in seq:
            if bit in (0, 1, "0", "1", True, False):
                flat.append(1 if bit in (1, "1", True) else 0)
            else:
                return None
    return "".join(str(b) for b in flat)


def scanline_from_any(data: Any) -> Optional[str]:
    """Normalize any supported input to a '0'/'1' scanline.

    Supported:
    * bitstring: ``"11001..."`` (whitespace ignored, any separator)
    * sequence of 0/1 (list/tuple)
    * runs: ``[[1, 2], [0, 3], ...]``  (dark flag, width)
    * comma run widths: ``"1,3,1,1,4,2"`` (dark first, as read off the
      bars — the format a vision model reports)
    * JSON (any of the above, or ``{"runs": [...]}`` / ``{"bits": "..."}``)
    * path to a text file containing any of the above
    """
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        return _bits_from_sequence(data)
    if not isinstance(data, str):
        return None
    text = data.strip()
    if not text:
        return None
    # Explicit file path?  (only if it exists and isn't already bits/runs)
    if (len(text) > 2 and text[0] not in "01[{\""
            and os.path.exists(text)):
        try:
            with open(text, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read().strip()
        except OSError:
            return None
    # Bitstring?
    if re.fullmatch(r"[01 \t\r\n]*", text):
        cleaned = re.sub(r"\s+", "", text)
        return cleaned or None
    # Comma-separated run widths ("1,3,1,1,4,2") — the format a vision
    # model reports when it traces the bars.  Runs alternate dark/light
    # starting dark: a symbol row begins with a bar, and a leading quiet
    # zone is just a white margin the reader trims anyway.
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if parts and all(p.isdigit() and 1 <= int(p) <= 200 for p in parts):
            bits: List[str] = []
            dark = True
            for part in parts:
                bits.append(str(int(dark)) * int(part))
                dark = not dark
            return "".join(bits)
    # JSON?
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, (list, tuple)):
        return _bits_from_sequence(list(obj))
    if isinstance(obj, dict):
        if "bits" in obj:
            if isinstance(obj["bits"], str):
                return re.sub(r"\s+", "", obj["bits"]) or None
            if isinstance(obj["bits"], list):
                return _bits_from_sequence(obj["bits"])
        if "runs" in obj and isinstance(obj["runs"], list):
            return _bits_from_sequence(obj["runs"])
    return None


def _trim_quiet(bits: str) -> str:
    return bits.strip("0")


# ---------------------------------------------------------------------------
# Code 128 — encoding
# ---------------------------------------------------------------------------

def _c128_encode_values(text: str) -> List[Tuple[str, str, int]]:
    """Plan the symbol values for ``text``.

    Returns a list of (charset, token, value) where ``token`` is the text
    fragment that token covers (two chars for charset-C pairs).  Uses
    automatic switching: charset C for digit runs of length >= 4 (or any
    even-length run >= 2 at the tail), B for printable ASCII, A for
    control codes.
    """
    if not text:
        return []
    tokens: List[Tuple[str, str, int]] = []
    i, n = 0, len(text)

    def emit(charset: str, token: str) -> None:
        if charset == "C":
            mapping = C128_SET_C
        elif charset == "B":
            mapping = C128_SET_B
        else:
            mapping = C128_SET_A
        key = token if charset == "C" else token
        if key not in mapping:
            raise ValueError(f"character {token!r} not encodable in "
                             f"Code 128 set {charset}")
        tokens.append((charset, token, mapping[key]))

    while i < n:
        ch = text[i]
        if ch < " " or ch > "~":  # control / non-ASCII -> set A
            if ch in C128_SET_A:
                emit("A", ch)
            else:
                raise ValueError(f"character {ch!r} not in Code 128 set A")
            i += 1
            continue
        # Count the digit run starting here.
        j = i
        while j < n and text[j].isdigit():
            j += 1
        run = j - i
        # Lookahead: how many digit chars remain from i (may switch at even
        # boundaries); prefer C when the run is long, or when an even run
        # reaches the end.
        if run >= 4 or (run >= 2 and j == n):
            # Emit as many C pairs as possible.
            k = i
            while k + 2 <= n and text[k].isdigit() and text[k + 1].isdigit():
                # Stop the pair-run if switching would save symbols: if the
                # number of remaining digits is odd, leave one for B.
                remaining_digits = 0
                m = k
                while m < n and text[m].isdigit():
                    remaining_digits += 1
                    m += 1
                if remaining_digits <= 1:
                    break
                emit("C", text[k:k + 2])
                k += 2
            i = k
            continue
        emit("B", ch)
        i += 1
    return tokens


def c128_check_symbol(values: Sequence[int]) -> int:
    """Weighted mod-103 check symbol (values[0] is the start code)."""
    total = values[0]
    for idx, v in enumerate(values[1:], start=1):
        total += idx * v
    return total % 103


def encode_code128(text: str) -> Dict[str, Any]:
    """Encode ``text`` as a Code 128 bitstring (start + data + check + stop)."""
    tokens = _c128_encode_values(text)
    if not tokens:
        raise ValueError("nothing to encode")
    # Insert charset switches: track the active set.
    values: List[int] = []
    active: Optional[str] = None
    for charset, _token, value in tokens:
        if active is None:
            values.append(C128_START[charset])
            active = charset
        elif charset != active:
            switcher = f"TO_{charset}"
            mapping = (C128_SET_A if active == "A"
                       else C128_SET_B if active == "B"
                       else C128_SET_C)
            if switcher not in mapping:
                # Direct C<->B/A is fine; A<->C needs B as the bridge.
                bridge = "B" if {active, charset} == {"A", "C"} else None
                if bridge:
                    values.append(C128_SET_A.get(f"TO_B", C128_SET_B.get("TO_B")))
                    active = "B"
                    mapping = C128_SET_B
                    switcher = f"TO_{charset}"
                else:
                    raise ValueError(f"cannot switch {active}->{charset}")
            values.append(mapping[switcher])
            active = charset
        values.append(value)
    values.append(c128_check_symbol(values))
    bits = "".join(C128_PATTERNS[v] for v in values) + C128_STOP + C128_TERMINAL_BAR
    return {
        "bits": bits,
        "symbols": len(values) + 1,  # + stop
        "values": values,
        "text": text,
        "length_modules": len(bits),
    }


# ---------------------------------------------------------------------------
# Code 128 — decoding
# ---------------------------------------------------------------------------

def _c128_symbol_at(bits: str, pos: int) -> Optional[int]:
    chunk = bits[pos:pos + 11]
    if len(chunk) != 11:
        return None
    return C128_BITS_TO_VALUE.get(chunk)


def _c128_find_start(bits: str) -> int:
    """Return the offset of the start symbol, or -1."""
    for start_bit in C128_START.values():
        idx = bits.find(C128_PATTERNS[start_bit])
        if idx != -1:
            return idx
    return -1


def _c128_decode_values(bits: str) -> Tuple[Optional[int], List[int], int]:
    """Decode (start_value, symbol_values, stop_pos) from a trimmed scanline.

    ``stop_pos`` is the index just past the 13-module stop.  Symbol values
    include the start symbol at position 0 (weight 1).
    """
    start_idx = _c128_find_start(bits)
    if start_idx == -1:
        return None, [], -1
    start_value = C128_BITS_TO_VALUE.get(bits[start_idx:start_idx + 11])
    if start_value not in (103, 104, 105):
        return None, [], -1
    pos = start_idx + 11
    values = [start_value]
    # Walk symbols until we can find the stop after this one.
    while pos + 11 <= len(bits):
        v = _c128_symbol_at(bits, pos)
        if v is None or v >= 103:
            break
        values.append(v)
        pos += 11
        # Is the next thing the stop?
        if bits[pos:pos + 13] == C128_STOP_FULL:
            return start_value, values, pos + 13
        if bits[pos:pos + 11] == C128_STOP:
            # Stop without the terminal bar (lenient).
            after = pos + 11
            if bits[after:after + 2] == "11":
                return start_value, values, after + 2
            return start_value, values, after
    return start_value, values, -1


def _c128_decode_text(start: int, values: List[int]) -> Tuple[str, List[str]]:
    """Symbol values -> payload text.  Returns (text, notes)."""
    notes: List[str] = []
    out: List[str] = []
    if start == 103:
        active = "A"
    elif start == 104:
        active = "B"
    else:
        active = "C"
    i = 1
    n = len(values) - 1  # last value is the check symbol
    while i < n:
        v = values[i]
        if active == "A":
            if v == 99:
                active, i = "C", i + 1
                continue
            if v == 100:
                active, i = "B", i + 1
                continue
            if v == 98:  # SHIFT -> one char from set B
                out.append(C128_REVERSE_B.get(values[i + 1], f"\\x{values[i+1]:02x}")
                           if i + 1 < n else "")
                i += 2
                continue
            if v in (96, 97, 101, 102):
                notes.append(f"set A FNC at value {v}")
            out.append(C128_REVERSE_A.get(v, f"\\x{v:02x}"))
        elif active == "B":
            if v == 99:
                active, i = "C", i + 1
                continue
            if v == 101:
                active, i = "A", i + 1
                continue
            if v == 98:  # SHIFT -> one char from set A
                out.append(C128_REVERSE_A.get(values[i + 1], f"\\x{values[i+1]:02x}")
                           if i + 1 < n else "")
                i += 2
                continue
            if v in (96, 97, 100, 102):
                notes.append(f"set B FNC at value {v}")
            out.append(C128_REVERSE_B.get(v, f"\\x{v:02x}"))
        else:  # C
            if v == 100:
                active, i = "B", i + 1
                continue
            if v == 101:
                active, i = "A", i + 1
                continue
            if v == 102:
                notes.append("set C FNC1 (GS1-128) at value 102")
                i += 1
                continue
            out.append(f"{v:02d}")
        i += 1
    return "".join(out), notes


def decode_code128(bits: str) -> Dict[str, Any]:
    """Decode a Code 128 scanline (with or without quiet zones)."""
    trimmed = _trim_quiet(bits)
    start, values, stop_pos = _c128_decode_values(trimmed)
    result: Dict[str, Any] = {
        "symbology": "code128",
        "ok": False,
        "text": "",
        "check_ok": False,
        "note": "",
    }
    if start is None or len(values) < 2:
        result["note"] = ("no Code 128 start symbol found"
                          if start is None else "start found but no data")
        return result
    payload, notes = _c128_decode_text(start, values)
    check_ok = c128_check_symbol(values[:-1]) == values[-1]
    result.update(
        ok=check_ok and stop_pos != -1,
        text=payload,
        check_ok=check_ok,
        stop_found=stop_pos != -1,
        symbols=len(values),
        note="; ".join(notes) or
             ("" if check_ok else "check symbol mismatch (payload may still be right)"),
    )
    return result


# ---------------------------------------------------------------------------
# EAN-13 / UPC-A — encoding
# ---------------------------------------------------------------------------

def ean_check_digit(digits12: str) -> int:
    """Mod-10 check digit for 12 digits (positions 1-based: odd x1, even x3)."""
    if len(digits12) != 12 or not digits12.isdigit():
        raise ValueError("need exactly 12 digits")
    total = 0
    for i, ch in enumerate(digits12):
        d = int(ch)
        total += d * (1 if i % 2 == 0 else 3)
    return (10 - total % 10) % 10


def upc_check_digit(digits11: str) -> int:
    """UPC-A check digit (odd positions x3, even x1 — same as a leading-0 EAN)."""
    if len(digits11) != 11 or not digits11.isdigit():
        raise ValueError("need exactly 11 digits")
    return ean_check_digit("0" + digits11)


def encode_ean13(digits12: str) -> Dict[str, Any]:
    """Encode a 12-digit EAN-13 body (+ computed check digit) to modules."""
    if len(digits12) != 12 or not digits12.isdigit():
        raise ValueError("EAN-13 body must be exactly 12 digits")
    digits13 = digits12 + str(ean_check_digit(digits12))
    parity = EAN_LEFT_PARITY[int(digits13[0])]
    bits = EAN_GUARD
    for i in range(6):
        d = int(digits13[1 + i])
        bits += (EAN_L if parity[i] == "L" else EAN_G)[d]
    bits += EAN_CENTER
    for d in (int(x) for x in digits13[7:]):
        bits += EAN_R[d]
    bits += EAN_GUARD
    return {"bits": bits, "digits": digits13,
            "check_digit": int(digits13[12]),
            "length_modules": len(bits)}


def encode_upca(digits11: str) -> Dict[str, Any]:
    """Encode a 11-digit UPC-A body (+ computed check digit) to modules."""
    if len(digits11) != 11 or not digits11.isdigit():
        raise ValueError("UPC-A body must be exactly 11 digits")
    out = encode_ean13("0" + digits11)
    out["digits"] = digits11 + str(out["check_digit"])
    return out


# ---------------------------------------------------------------------------
# EAN-13 / UPC-A — decoding
# ---------------------------------------------------------------------------

def _ean_decode(bits: str, expect: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "symbology": expect,
        "ok": False,
        "digits": "",
        "check_ok": False,
        "system": "",
        "note": "",
    }
    t = _trim_quiet(bits)
    if len(t) < 95:
        result["note"] = f"too short for {expect} (need 95 modules)"
        return result
    # Try every center-guard candidate (01010 can occur inside data too).
    attempts = [idx for idx in range(0, len(t) - 4)
                if t[idx:idx + 5] == EAN_CENTER]
    if not attempts:
        result["note"] = "no center guard 01010 found"
        return result
    best: Optional[str] = None
    for center in attempts:
        rep = _ean_decode_attempt(t, center, expect)
        if rep is None:
            continue
        if rep.get("ok"):
            result.update(rep)
            return result
        if best is None or (rep.get("digits") and not best):
            best = rep.get("note") or best
        if rep.get("digits") and not result.get("digits"):
            result["digits"] = rep["digits"]
            result["check_ok"] = rep.get("check_ok", False)
            result["system"] = rep.get("system", "")
    if not result.get("digits"):
        result["note"] = best or "no valid EAN frame found"
        return result
    result["note"] = best or result.get("note", "") or "no valid EAN frame found"
    return result


def _ean_decode_attempt(t: str, center: int,
                        expect: str) -> Optional[Dict[str, Any]]:
    """Try one center-guard position.  Returns a full report dict or None
    when this position isn't a plausible EAN frame at all."""
    # Layout: guard(3) | left data(42) | center(5) | right data(42) | guard(3)
    if center - 45 < 0 or center + 5 + 45 > len(t):
        return None
    if t[center - 45:center - 42] != EAN_GUARD:
        return None
    left = t[center - 42:center]
    right = t[center + 5:center + 5 + 42]
    if t[center + 5 + 42:center + 5 + 45] != EAN_GUARD:
        return None
    left_digits: List[int] = []
    parity_seen: List[str] = []
    for i in range(6):
        chunk = left[i * 7:(i + 1) * 7]
        dl = EAN_L.index(chunk) if chunk in EAN_L else None
        dg = EAN_G.index(chunk) if chunk in EAN_G else None
        if dl is None and dg is None:
            return {**_ean_result(expect),
                    "note": f"left digit {i}: unrecognizable pattern"}
        if dl is not None and dg is not None:
            return {**_ean_result(expect),
                    "note": f"left digit {i}: ambiguous L/G pattern"}
        left_digits.append(dl if dl is not None else dg)
        parity_seen.append("L" if dl is not None else "G")
    right_digits: List[int] = []
    for i in range(6):
        chunk = right[i * 7:(i + 1) * 7]
        dr = EAN_R.index(chunk) if chunk in EAN_R else None
        if dr is None:
            return {**_ean_result(expect),
                    "note": f"right digit {i}: unrecognizable pattern"}
        right_digits.append(dr)
    # The system digit (d1) is carried by the left-half parity pattern —
    # invert it.  If no system digit matches, this isn't a valid frame.
    parity_key = "".join(parity_seen)
    if parity_key not in EAN_LEFT_PARITY:
        return {**_ean_result(expect), "note": "no valid system parity"}
    system = EAN_LEFT_PARITY.index(parity_key)
    # 13 digits: parity-derived system digit + 6 left + 6 right.
    digits13 = str(system) + "".join(str(d) for d in left_digits) \
        + "".join(str(d) for d in right_digits)
    check_ok = int(digits13[12]) == ean_check_digit(digits13[:12])
    rep = _ean_result(expect)
    rep.update(
        digits=digits13,
        check_ok=check_ok,
        system=EAN_SYSTEMS.get(system, f"system {system}"),
        ok=check_ok,
        note="" if check_ok else "check digit mismatch",
    )
    if expect == "upca":
        if system == 0:
            # Report the 12-digit UPC (implicit leading 0 dropped).
            rep["digits"] = digits13[1:]
        else:
            rep["note"] = (rep["note"] + "; " if rep["note"] else "") \
                + "system digit is not 0 (this is an EAN-13, not UPC-A)"
    return rep


def _ean_result(expect: str) -> Dict[str, Any]:
    return {
        "symbology": expect,
        "ok": False,
        "digits": "",
        "check_ok": False,
        "system": "",
        "note": "",
    }


def decode_ean13(bits: str) -> Dict[str, Any]:
    return _ean_decode(bits, "ean13")


def decode_upca(bits: str) -> Dict[str, Any]:
    return _ean_decode(bits, "upca")


# ---------------------------------------------------------------------------
# Luhn
# ---------------------------------------------------------------------------

def luhn_valid(number: str) -> bool:
    digits = re.sub(r"\D", "", number or "")
    if len(digits) < 2:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def luhn_check_digit(prefix: str) -> int:
    """The digit that makes ``prefix + d`` pass Luhn."""
    digits = re.sub(r"\D", "", prefix or "")
    if not digits:
        raise ValueError("empty prefix")
    for d in range(10):
        if luhn_valid(digits + str(d)):
            return d
    raise AssertionError("unreachable")


def recover_luhn(known: str,
                 max_candidates: int = 10000) -> Dict[str, Any]:
    """Recover a partially-known Luhn number (bank / gift-card numbers).

    ``known`` is a 10-19 digit template with ``?``/``*`` holes, e.g.
    ``"4111?1111111111"``.  Every hole combination is enumerated (bounded
    by ``max_candidates``) and the Luhn mod-10 checksum prunes it — a
    16-digit PAN with two scratched digits leaves at most ~1/10 of the
    100 combinations valid, and the survivors ARE the possible original
    numbers.  No network, no card brand tables — pure arithmetic, the
    same class of proof as the EAN check digit.

    Returns ``{"recovered", "candidates", "holes", "note"}`` in the same
    shape as :func:`recover_ean`.
    """
    template = re.sub(r"\s", "", known or "")
    if re.fullmatch(r"[0-9?*]{10,19}", template) is None:
        raise ValueError("template must be 10-19 digits with ? holes")
    if "?" not in template and "*" not in template:
        return {
            "input": template,
            "recovered": template if luhn_valid(template) else "",
            "candidates": ([{"value": template, "luhn_ok": luhn_valid(template)}]
                           if luhn_valid(template) else []),
            "holes": 0,
            "note": ("valid Luhn number" if luhn_valid(template)
                     else "fails Luhn — re-check the digits or use ? holes"),
        }
    holes = [i for i, ch in enumerate(template) if ch in "?*"]
    total = 10 ** len(holes)
    if total > max_candidates:
        raise ValueError(f"too many holes ({len(holes)}); {total} > "
                         f"max_candidates {max_candidates}")
    skeleton = list(template)
    candidates: List[Dict[str, Any]] = []
    for combo in itertools.product(range(10), repeat=len(holes)):
        for pos, d in zip(holes, combo):
            skeleton[pos] = str(d)
        value = "".join(skeleton)
        if luhn_valid(value):
            candidates.append({"value": value, "luhn_ok": True,
                               "symbology": "luhn"})
        for pos in holes:  # restore for the next combination
            skeleton[pos] = "?"
    return {
        "input": template,
        "recovered": candidates[0]["value"] if len(candidates) == 1 else "",
        "candidates": candidates,
        "holes": len(holes),
        "note": (f"{len(candidates)} of {total} combinations pass Luhn"),
    }


# ---------------------------------------------------------------------------
# Recovery — the over-scratched-card engine
# ---------------------------------------------------------------------------

@dataclass
class BarcodeCandidate:
    """One recovered candidate value with its verification status."""
    value: str
    symbology: str
    check_ok: bool
    luhn_ok: Optional[bool] = None
    score: float = 1.0
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "symbology": self.symbology,
            "check_ok": self.check_ok,
            "luhn_ok": self.luhn_ok,
            "score": round(self.score, 3),
            "note": self.note,
        }


_MAX_EAN_ENUM = 1_000_000
_MAX_C128_ENUM = 200_000


def _score_card_number(value: str, symbology: str) -> Tuple[float, Optional[bool]]:
    """Heuristic quality score for a recovered candidate (0..1)."""
    digits = re.sub(r"\D", "", value)
    luhn = luhn_valid(value)
    score = 1.0
    if len(digits) in (13, 16, 19):
        score += 0.2
    if luhn:
        score += 0.3
    # Card-looking prefixes (common gift-card issuers).
    if digits[:2] in ("51", "52", "53", "44", "45", "46", "47", "89"):
        score += 0.1
    return min(score, 1.5), luhn


def recover_ean(known: str,
                max_candidates: int = 1000) -> Dict[str, Any]:
    """Recover a partially-known EAN-13/UPC-A number.

    ``known`` is a 12- or 13-digit template with ``?`` (or ``*``) holes,
    e.g. ``"5193?234?6789"``.  Every hole is enumerated (bounded), and the
    mod-10 check digit prunes the space: with all 12 body digits known the
    check digit is determined; with one hole at most 10 survivors; with two
    holes at most 100, and so on.  Results are ranked by card-number
    heuristics (length, Luhn, issuer prefix).
    """
    template = re.sub(r"\s", "", known or "")
    if re.fullmatch(r"[0-9?*]{12,13}", template) is None:
        raise ValueError("template must be 12 or 13 digits with ? holes")
    body = template[:12]
    check_part = template[12] if len(template) == 13 else "?"
    holes = [i for i, ch in enumerate(body) if ch in "?*"]
    total = 10 ** len(holes)
    if total > _MAX_EAN_ENUM:
        raise ValueError(f"too many holes ({len(holes)}); max 6 for bounded "
                         "enumeration")
    candidates: List[BarcodeCandidate] = []
    body_digits = list(body)
    for combo in itertools.product(range(10), repeat=len(holes)):
        for pos, d in zip(holes, combo):
            body_digits[pos] = str(d)
        digits12 = "".join(body_digits)
        expected = ean_check_digit(digits12)
        if check_part in "?*":
            check = str(expected)
            check_ok = True
        else:
            check = check_part
            check_ok = int(check) == expected
        value = digits12 + check
        if not check_ok:
            continue
        score, luhn = _score_card_number(value, "ean13")
        candidates.append(BarcodeCandidate(
            value=value, symbology="ean13", check_ok=True,
            luhn_ok=luhn, score=score,
            note="check digit verified" + ("" if check_part in "?*"
                                            else " (printed check digit matches)")))
        if len(candidates) >= max_candidates:
            break
    candidates.sort(key=lambda c: -c.score)
    return {
        "input": template,
        "holes": len(holes),
        "enumerated": total,
        "candidates": [c.as_dict() for c in candidates],
        "recovered": candidates[0].value if candidates else None,
    }


def recover_code128(known: str,
                    alphabet: Optional[str] = None,
                    max_holes: int = 3,
                    max_candidates: int = 5000) -> Dict[str, Any]:
    """Recover a partially-known Code 128 payload.

    ``known`` is the payload template with ``?`` holes (max ``max_holes``).
    Each hole is tried against ``alphabet`` (default: digits + uppercase +
    ``-``) and the payload's mod-103 check symbol must validate.
    """
    template = known or ""
    holes = [i for i, ch in enumerate(template) if ch in "?*"]
    if len(holes) == 0:
        raise ValueError("template has no holes")
    if len(holes) > max_holes:
        raise ValueError(f"too many holes ({len(holes)}); max {max_holes}")
    if alphabet is None:
        alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
    candidates: List[BarcodeCandidate] = []
    chars = list(template)
    for combo in itertools.product(alphabet, repeat=len(holes)):
        if len(candidates) >= max_candidates:
            break
        for pos, ch in zip(holes, combo):
            chars[pos] = ch
        text = "".join(chars)
        try:
            enc = encode_code128(text)
        except ValueError:
            continue
        score, luhn = _score_card_number(text, "code128")
        candidates.append(BarcodeCandidate(
            value=text, symbology="code128", check_ok=True,
            luhn_ok=luhn, score=score,
            note="mod-103 check symbol validates"))
    candidates.sort(key=lambda c: -c.score)
    return {
        "input": template,
        "holes": len(holes),
        "alphabet": alphabet,
        "candidates": [c.as_dict() for c in candidates],
        "recovered": candidates[0].value if candidates else None,
    }


def recover_scanline(bits: str,
                     unknown: Optional[Tuple[int, int]] = None,
                     known_number: Optional[str] = None,
                     max_symbols: int = 3,
                     max_candidates: int = 5000) -> Dict[str, Any]:
    """Recover a Code 128 scanline with a damaged module span.

    ``bits`` — the scanned 1-D row (quiet zones fine).
    ``unknown`` — (start, end) module range of the damaged span, if known
        from the scan (e.g. where the image had a scratch).  If omitted,
        the decoder's failure point is used as the span start.
    ``known_number`` — optional partially legible printed number used to
        cross-check candidates (``?`` holes allowed; prefix/suffix match).
    Enumerates candidate symbol values inside the span (1..``max_symbols``
    symbols) until the mod-103 check symbol validates.
    """
    trimmed = _trim_quiet(bits)
    result: Dict[str, Any] = {
        "symbology": "code128",
        "candidates": [],
        "recovered": None,
        "note": "",
    }
    start_idx = _c128_find_start(trimmed)
    if start_idx == -1:
        # Maybe the damage ate the start: try the reverse stop instead.
        result["note"] = "no start symbol found — the damage may include the " \
                         "barcode start; try a re-scan or provide known_number"
        return result
    stop_idx = trimmed.find(C128_STOP_FULL)
    if stop_idx == -1:
        stop_idx = trimmed.find(C128_STOP)
    if stop_idx == -1:
        result["note"] = "no stop pattern found — damage includes the stop"
        return result
    if unknown is None:
        # First position after start that fails to decode is the span start.
        pos = start_idx + 11
        while pos + 11 <= len(trimmed) and _c128_symbol_at(trimmed, pos) is not None:
            pos += 11
        unknown = (pos, min(pos + 11 * max_symbols, stop_idx))
    span_start, span_end = unknown
    span_start = max(span_start, start_idx + 11)
    span_end = min(span_end, stop_idx)
    span_modules = span_end - span_start
    n_symbols = span_modules // 11
    remainder = span_modules % 11
    if n_symbols < 1:
        result["note"] = "damaged span smaller than one symbol"
        return result
    if n_symbols > max_symbols:
        result["note"] = (f"damaged span covers {n_symbols} symbols; max "
                          f"bounded to {max_symbols}")
        return result
    # Decode what we can: left of the span and right of it.
    left_values: List[int] = []
    pos = start_idx + 11
    while pos < span_start and pos + 11 <= len(trimmed):
        v = _c128_symbol_at(trimmed, pos)
        if v is None or v >= 103:
            break
        left_values.append(v)
        pos += 11
    # Right side: walk backwards from the stop.
    right_values: List[int] = []
    pos = stop_idx - 11
    while pos >= span_end:
        v = _c128_symbol_at(trimmed, pos)
        if v is None or v >= 103:
            break
        right_values.insert(0, v)
        pos -= 11
    start_value = C128_BITS_TO_VALUE[trimmed[start_idx:start_idx + 11]]
    base = [start_value] + left_values
    # Enumerate candidate value tuples for the damaged symbols.  Full
    # enumeration is affordable up to 3 symbols (103^3 ~= 1.09M).
    candidates: List[BarcodeCandidate] = []
    alphabet_values = list(range(103))  # 0-102 are the data values
    for combo in itertools.product(alphabet_values, repeat=n_symbols):
        values = base + list(combo) + right_values
        if len(values) > 40:  # sanity cap
            break
        check = c128_check_symbol(values[:-1]) if len(values) > 1 else 0
        # The real check symbol is the symbol right before the stop.
        real_check = _c128_symbol_at(trimmed, stop_idx - 11)
        if real_check is None:
            break
        if check != real_check:
            continue
        payload, notes = _c128_decode_text(start_value, values)
        text = payload
        if known_number:
            pattern = re.escape(known_number).replace("\\?", ".").replace("\\*", ".")
            if re.fullmatch(pattern, text) is None:
                continue
        score, luhn = _score_card_number(text, "code128")
        candidates.append(BarcodeCandidate(
            value=text, symbology="code128", check_ok=True,
            luhn_ok=luhn, score=score,
            note=f"{n_symbols} damaged symbol(s) resolved; mod-103 validates"))
        if len(candidates) >= max_candidates:
            break
    candidates.sort(key=lambda c: -c.score)
    result["candidates"] = [c.as_dict() for c in candidates]
    result["recovered"] = candidates[0].value if candidates else None
    if not candidates:
        result["note"] = (f"no candidate for the {n_symbols}-symbol span passed "
                          "the mod-103 check" + ("" if known_number is None
                                                 else " + known-number match"))
    if remainder:
        result.setdefault("note", "")
        result["note"] = (result["note"] + "; " if result["note"] else "") \
            + f"span had {remainder} extra modules (edge misalignment?)"
    return result


# ---------------------------------------------------------------------------
# Analysis entry points
# ---------------------------------------------------------------------------

def card_number(value: Any) -> str:
    """Normalize a decoded payload to a plain card/product number."""
    if value is None:
        return ""
    text = str(value).strip()
    # Strip human-readable grouping and trailing check-echo patterns.
    text = re.sub(r"\s+", "", text)
    m = re.search(r"\d{12,19}", text)
    return m.group(0) if m else text


def analyze(data: Any) -> Dict[str, Any]:
    """Auto-detect + decode any supported barcode scanline.

    Tries Code 128, EAN-13 and UPC-A, returns every candidate ranked
    (validated symbologies first, then by score).
    """
    bits = scanline_from_any(data)
    if bits is None:
        return {"ok": False, "note": "no scanline found in input",
                "candidates": []}
    out: List[Dict[str, Any]] = []
    for decoder in (decode_code128, decode_ean13, decode_upca):
        rep = decoder(bits)
        if not rep.get("ok"):
            continue
        value = card_number(rep.get("text") or rep.get("digits") or "")
        score, luhn = _score_card_number(value, rep["symbology"])
        out.append({
            "symbology": rep["symbology"],
            "value": value,
            "check_ok": rep.get("check_ok", False),
            "luhn_ok": luhn,
            "system": rep.get("system", ""),
            "note": rep.get("note", ""),
            "score": round(1.5 + score, 3),
        })
    out.sort(key=lambda c: -c["score"])
    if not out:
        # Nothing fully validated: report the closest near-misses for the
        # user (a scratched card may fail exactly one check).
        for decoder in (decode_code128, decode_ean13, decode_upca):
            rep = decoder(bits)
            if rep.get("text") or rep.get("digits"):
                value = card_number(rep.get("text") or rep.get("digits") or "")
                out.append({
                    "symbology": rep["symbology"],
                    "value": value,
                    "check_ok": False,
                    "note": rep.get("note", "unvalidated"),
                    "score": 0.4,
                })
        out.sort(key=lambda c: -c["score"])
    return {
        "ok": bool(out and out[0].get("check_ok")),
        "modules": len(_trim_quiet(bits)),
        "candidates": out,
    }
