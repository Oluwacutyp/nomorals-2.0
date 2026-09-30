"""Universal Decoder — attempt to identify and decode almost anything.

A modular, extensible analysis + decoding engine for strings and bytes:

* **Encodings** — hex / base16 / base32 / base58 / base64 (std + urlsafe) /
  binary / rot13 / atbash / reverse / URL-percent / HTML entities / unicode
  and octal escapes / morse / leet — including NESTED encoding chains
  (base64 → base64 → text) discovered automatically.
* **Hashes** — algorithm identification by shape (md5 … sha512, blake2,
  bcrypt, argon2, NTLM, CRC32, …) plus a built-in known-hash table that
  matches common secrets across md5/sha1/sha256/sha384/sha512/crc32.
* **Cookies & tokens** — Set-Cookie / Cookie header parsing with
  attribute flags (HttpOnly / Secure / SameSite) and value decoding;
  JWT decode (header + payload claims, ``alg:none`` flag, exp check);
  pattern detection for common API-token shapes (labeled, never leaves
  the machine).
* **Structure** — JSON, YAML (when available), CSV/TSV detection,
  key=value, INI, MIME (part parsing + base64 bodies), PEM (type +
  payload), gzip/zlib byte streams, file magic identification.
* **Forensics** — Shannon entropy, printable-ratio, charset detection
  (BOM / utf-8 / utf-16 / ascii), UUID detection, Luhn (card-shaped
  number) detection.

Every decoder is a small object with ``attempt(data) -> DecoderHit|None``;
``register_decoder()`` extends the engine at runtime.  The engine is
stdlib-only and fully offline — it only ever sees data YOU hand it.

    from nomorals.core.decoder import analyze
    report = analyze("cGFzc3dvcmQ=")
    report.best["chain"]       # ["base64", "text"]
    report.hash["candidates"]  # when the input is a digest
"""

from __future__ import annotations

import base64
import binascii
import codecs
import hashlib
import html
from urllib.parse import unquote as _url_unquote
import json
import math
import re
import struct
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Decoder",
    "DecoderHit",
    "DecodeReport",
    "DECODERS",
    "analyze",
    "decode_any",
    "identify_hash",
    "known_hash_lookup",
    "identify_magic",
    "shannon_entropy",
    "register_decoder",
    "TOKEN_PATTERNS",
]

# ─────────────────────────── result shapes ──────────────────────────────────


@dataclass
class DecoderHit:
    """One successful (or candidate) decode attempt."""

    decoder: str
    output: Any                      # str | bytes | dict | list
    confidence: float                # 0.0 … 1.0 (honest — see each decoder)
    note: str = ""
    data_type: str = "text"          # text | bytes | structured

    def to_dict(self) -> dict[str, Any]:
        out: Any = self.output
        if isinstance(out, (bytes, bytearray)):
            out = {
                "bytes": len(out),
                "hex_head": bytes(out)[:64].hex(),
                "magic": identify_magic(bytes(out)),
            }
        if isinstance(out, str) and len(out) > 4000:
            out = out[:4000] + f" … (+{len(out) - 4000} chars)"
        return {
            "decoder": self.decoder,
            "data_type": self.data_type,
            "confidence": round(self.confidence, 3),
            "note": self.note,
            "output": out,
        }


@dataclass
class DecodeReport:
    """Full analysis of one input."""

    best: dict[str, Any] | None
    hits: list[DecoderHit]
    forensics: dict[str, Any]
    hash: dict[str, Any] | None = None
    tokens: list[dict[str, str]] = field(default_factory=list)
    cookies: list[dict[str, Any]] = field(default_factory=list)
    jwt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        best = self.best
        if best is not None and isinstance(
                best.get("output"), (bytes, bytearray)):
            raw = bytes(best["output"])
            best = dict(best)
            best["output"] = {"bytes": len(raw),
                              "hex_head": raw[:64].hex(),
                              "magic": identify_magic(raw)}
        return {
            "best": best,
            "hits": [h.to_dict() for h in self.hits],
            "forensics": self.forensics,
            "hash": self.hash,
            "tokens": self.tokens,
            "cookies": self.cookies,
            "jwt": self.jwt,
        }


# ─────────────────────────── forensics helpers ──────────────────────────────


def shannon_entropy(data: str | bytes) -> float:
    """Shannon entropy in bits/char (8.0 = uniform-random noise)."""
    sample = data if isinstance(data, bytes) else data.encode("utf-8", "ignore")
    if not sample:
        return 0.0
    counts = Counter(sample)
    n = len(sample)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def printable_ratio(data: str | bytes) -> float:
    sample = data if isinstance(data, bytes) else data.encode("utf-8", "ignore")
    if not sample:
        return 0.0
    printable = sum(1 for b in sample if b in (9, 10, 13) or 32 <= b < 127)
    return printable / len(sample)


def looks_like_text(data: str, min_ratio: float = 0.90) -> bool:
    if not data:
        return False
    return printable_ratio(data) >= min_ratio and any(ch.isalpha() for ch in data)


def detect_charset(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-bom"
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return "utf-16"
    if len(data) >= 2 and (data[1] == 0 or (len(data) > 3 and data[2] == 0)):
        return "utf-16-le"
    try:
        data.decode("utf-8")
        return "ascii" if all(b < 128 for b in data) else "utf-8"
    except UnicodeDecodeError:
        return "latin-1/other"


_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF8", "gif"),
    (b"%PDF", "pdf"),
    (b"\x1f\x8b", "gzip"),
    (b"PK\x03\x04", "zip"),
    (b"\x7fELF", "elf"),
    (b"ID3", "mp3-id3"),
    (b"\xff\xfb", "mp3"),
    (b"RIFF", "riff/wav"),
    (b"OggS", "ogg"),
    (b"fLaC", "flac"),
    (b"WEBP", "webp"),
    (b"BM", "bmp"),
    (b"\xd0\xcf\x11\xe0", "ole2/doc"),
    (b"SQLite format 3", "sqlite"),
    (b"BZh", "bzip2"),
    (b"\xfd7zXZ", "xz"),
    (b"7z\xbc\xaf\x27\x1c", "7zip"),
    (b"\x78\x9c", "zlib"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"MZ", "pe/dos-executable"),
    (b"#!/", "script-shebang"),
)


def identify_magic(data: bytes) -> str:
    if not data:
        return ""
    for sig, label in _MAGIC:
        if data.startswith(sig):
            return label
    if len(data) >= 4:
        if struct.unpack(">I", data[:4])[0] in (
            0x00000020, 0x00000002, 0x00000003, 0x00000004, 0x00000006,
        ):
            return "webm/mkv"
    return ""


# ─────────────────────────── hash identification ────────────────────────────

_HEX = re.compile(r"^[0-9a-fA-F]+$")

#: (regex, labels) — order matters: most specific first
_HASH_SHAPES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"^\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}$"), ("bcrypt",)),
    (re.compile(r"^\$argon2(?:id|d|i)?\$v?19?\$\d+\$\d+\$\d+[A-Za-z0-9+/=]+"),
     ("argon2",)),
    (re.compile(r"^\$pbkdf2(?:-sha\d+)?\$\d+\$"), ("pbkdf2",)),
    (re.compile(r"^\$6\$\w+\$[./A-Za-z0-9]{86,}"), ("sha512crypt",)),
    (re.compile(r"^\$5\$\w+\$[./A-Za-z0-9]{43}$"), ("sha256crypt",)),
    (re.compile(r"^[0-9a-fA-F]{128}$"), ("sha512", "blake2-512", "whirlpool")),
    (re.compile(r"^[0-9a-fA-F]{96}$"), ("sha384",)),
    (re.compile(r"^[0-9a-fA-F]{64}$"), ("sha256", "blake2b-256", "keccak-256")),
    (re.compile(r"^[0-9a-fA-F]{56}$"), ("ripemd320", "tiger")),
    (re.compile(r"^[0-9a-fA-F]{40}$"), ("sha1", "ripemd160", "gost-94")),
    (re.compile(r"^[0-9a-fA-F]{32}$"), ("md5", "ntlm", "blake2s-128")),
    (re.compile(r"^[0-9a-fA-F]{24}$"), ("md4", "gost")),
    (re.compile(r"^[0-9a-fA-F]{16}$"), ("crc32", "adler32-hex")),
)


def identify_hash(candidate: str) -> list[str]:
    """Guess which hash algorithm(s) a digest string could be.

    Shape-based (length + alphabet); 32-hex is md5/NTLM, 64-hex sha256.
    Empty list = not a digest.
    """
    s = candidate.strip()
    if not s:
        return []
    for pattern, labels in _HASH_SHAPES:
        if pattern.match(s):
            return list(labels)
    if len(s) in (12, 20, 28, 48, 56, 80, 100) and _HEX.match(s):
        return ["truncated/unknown hex digest"]
    return []


#: common secrets — the built-in "crack" table: things people actually use,
#: precomputed across md5/sha1/sha256/sha384/sha512/crc32 at import time.
_COMMON_PLAINTEXTS: tuple[str, ...] = (
    "password", "password1", "password123", "123456", "1234567", "12345678",
    "123456789", "1234567890", "12345", "1234", "123456a", "admin", "admin123",
    "administrator", "root", "toor", "letmein", "welcome", "welcome1",
    "qwerty", "qwerty123", "abc123", "iloveyou", "monkey", "dragon",
    "sunshine", "master", "hello", "charlie", "donald", "shadow", "superman",
    "football", "baseball", "michelle", "jordan23", "harley", "ranger",
    "hunter", "buster", "soccer", "hockey", "starwars", "killer", "george",
    "android", "chess", "amelia", "zxcvbnm", "qazwsx", "654321", "777777",
    "696969", "123123", "111111", "000000", "a123456", "1q2w3e4r", "trustno1",
    "freedom", "login", "princess", "pass", "god", "beast", "codebeast",
    "oluwacutyp", "peace", "secret", "test", "demo", "changeme", "default",
    "guest", "user",
)


def _digests_for(plain: str) -> dict[str, str]:
    return {
        "md5": hashlib.md5(plain.encode()).hexdigest(),
        "sha1": hashlib.sha1(plain.encode()).hexdigest(),
        "sha256": hashlib.sha256(plain.encode()).hexdigest(),
        "sha384": hashlib.sha384(plain.encode()).hexdigest(),
        "sha512": hashlib.sha512(plain.encode()).hexdigest(),
        "crc32": format(zlib.crc32(plain.encode()) & 0xFFFFFFFF, "08x"),
    }


KNOWN_HASHES: dict[str, tuple[str, str]] = {}
for _plain in _COMMON_PLAINTEXTS:
    for _alg, _digest in _digests_for(_plain).items():
        KNOWN_HASHES.setdefault(_digest.lower(), (_plain, _alg))


def known_hash_lookup(candidate: str) -> dict[str, str] | None:
    """Match a digest against the built-in common-secrets table."""
    hit = KNOWN_HASHES.get(candidate.strip().lower())
    if hit is None:
        return None
    return {"plaintext": hit[0], "algorithm": hit[1]}


#: learned-hash store — every digest the system actually cracks lands here
#: (migration 25), so the next sighting is a lookup, not a crack.
_KNOWN_HASHES_DDL = (
    "CREATE TABLE IF NOT EXISTS known_hashes ("
    "digest TEXT PRIMARY KEY, plaintext TEXT NOT NULL, "
    "algorithm TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '', "
    "created_at REAL NOT NULL)")


def learn_hash(db: Any, digest: str, plaintext: str, algorithm: str = "",
               source: str = "live-crack") -> bool:
    """Persist a cracked digest into the learned known-hash store.

    Returns True when it was stored (or already was).  Fails soft: a
    missing/unavailable database never breaks the cracking path.
    """
    if db is None:
        return False
    digest = (digest or "").strip().lower()
    plaintext = (plaintext or "").strip()
    if not digest or not plaintext:
        return False
    try:
        db.execute(_KNOWN_HASHES_DDL)
        db.execute(
            "INSERT OR IGNORE INTO known_hashes "
            "(digest, plaintext, algorithm, source, created_at) "
            "VALUES (?,?,?,?,?)",
            (digest, plaintext, (algorithm or "").strip().lower(),
             source, time.time()))
        return True
    except Exception:  # noqa: BLE001
        return False


def learned_hash_lookup(db: Any, candidate: str) -> dict[str, str] | None:
    """Check the learned store (digests cracked by us in past sessions)."""
    if db is None:
        return None
    candidate = (candidate or "").strip().lower()
    if not candidate:
        return None
    try:
        db.execute(_KNOWN_HASHES_DDL)
        row = db.query_one(
            "SELECT plaintext, algorithm FROM known_hashes WHERE digest=?",
            (candidate,))
    except Exception:  # noqa: BLE001
        return None
    if row is None:
        return None
    return {"plaintext": row["plaintext"],
            "algorithm": row.get("algorithm", "")}


def known_hash_lookup_chained(db: Any, candidate: str) -> dict[str, str] | None:
    """Built-in table first, then the learned store.  The full chain:
    a digest the system has ever solved is solved instantly, forever."""
    hit = known_hash_lookup(candidate)
    if hit is not None:
        return hit
    return learned_hash_lookup(db, candidate)


# ─────────────────────────── report archive ────────────────────────────────
#: Every DecodeReport the system produces is persisted here — decoder tool,
#: investigate agent, and monitor re-decodes all land in the same archive,
# which is what makes "what did we ever decode?" a query, not a memory test.
_REPORT_DDL = (
    "CREATE TABLE IF NOT EXISTS decode_reports "
    "(id TEXT PRIMARY KEY, ts REAL NOT NULL, source TEXT NOT NULL, "
    "kind TEXT NOT NULL, input_head TEXT NOT NULL, best_name TEXT NOT NULL, "
    "summary TEXT NOT NULL, report_json TEXT NOT NULL)")


def save_report(db: Any, report: Any, *, source: str = "", kind: str = "",
                input_text: str = "") -> str | None:
    """Persist a DecodeReport (or its dict form).  Fail-soft: a missing or
    unavailable database never breaks the decode path.  Returns the report
    id, or None when it could not be stored."""
    if db is None:
        return None
    try:
        from .ids import new_short_id

        d = report if isinstance(report, dict) else \
            getattr(report, "to_dict", lambda: None)()
        if not isinstance(d, dict):
            d = getattr(report, "__dict__", {}) or {}
        best = d.get("best") or {}
        best_name = ""
        if isinstance(best, dict):
            chain = best.get("chain") or []
            best_name = str(chain[0]) if chain else ""
        rid = new_short_id("rep")
        ts = time.time()
        try:
            summary = json.dumps(
                {"best": best_name, "hits": len(d.get("hits") or []),
                 "cookies": len(d.get("cookies") or []),
                 "jwt": bool(d.get("jwt")),
                 "hash": bool(d.get("hash")),
                 "tokens": len(d.get("tokens") or [])},
                default=str)[:500]
        except (TypeError, ValueError):
            summary = ""
        db.execute(_REPORT_DDL)
        db.execute(
            "INSERT OR REPLACE INTO decode_reports "
            "(id, ts, source, kind, input_head, best_name, summary, "
            "report_json) VALUES (?,?,?,?,?,?,?,?)",
            (rid, ts, source or "", kind or "",
             (input_text or "")[:300], best_name, summary,
             json.dumps(d, default=str)[:2_000_000]))
        return rid
    except Exception:  # noqa: BLE001 — the archive is enrichment
        return None


def decode_history(db: Any, query: str = "", limit: int = 20) -> list[dict[str, Any]]:
    """Most-recent decode reports first; optional substring filter over
    source / best decoder / input head."""
    if db is None:
        return []
    try:
        db.execute(_REPORT_DDL)
        limit = max(1, min(int(limit or 20), 200))
        if query:
            like = f"%{query.strip()}%"
            rows = db.query(
                "SELECT id, ts, source, kind, input_head, best_name, summary "
                "FROM decode_reports WHERE source LIKE ? OR best_name LIKE ? "
                "OR input_head LIKE ? ORDER BY ts DESC LIMIT ?",
                (like, like, like, limit))
        else:
            rows = db.query(
                "SELECT id, ts, source, kind, input_head, best_name, summary "
                "FROM decode_reports ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]
    except Exception:  # noqa: BLE001
        return []


def get_report(db: Any, report_id: str) -> dict[str, Any] | None:
    """Fetch one archived report by id (full report_json decoded)."""
    if db is None:
        return None
    try:
        db.execute(_REPORT_DDL)
        row = db.query_one(
            "SELECT * FROM decode_reports WHERE id=?", (report_id,))
        if row is None:
            return None
        out = dict(row)
        try:
            out["report"] = json.loads(row["report_json"])
        except (TypeError, ValueError):
            out["report"] = None
        return out
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────── token / pattern detection ─────────────────────

#: (key, regex, label) — labels only; the engine never transmits tokens.
TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}(?:\.[A-Za-z0-9_\-]+)?"), "JSON Web Token"),
    ("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,255}"), "GitHub personal access / OAuth token"),
    ("github_fine", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,255}"), "GitHub fine-grained PAT"),
    ("openai", re.compile(r"\bsk-[A-Za-z0-9]{20,120}"), "OpenAI-style API key"),
    ("anthropic", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,200}"), "Anthropic API key"),
    ("google", re.compile(r"\bAIza[A-Za-z0-9_\-]{30,40}"), "Google API key"),
    ("aws_access", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}"), "AWS access key id"),
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"), "Slack token"),
    ("huggingface", re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "HuggingFace token"),
    ("telegram_bot", re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{35}"), "Telegram bot token"),
    ("stripe_live", re.compile(r"\bsk_live_[A-Za-z0-9]{10,}"), "Stripe live secret key"),
    ("npm", re.compile(r"\bnpm_[A-Za-z0-9]{30,}"), "npm token"),
    ("pypi", re.compile(r"\bpypi-A[1-zA-Z0-9_\-]{16,}"), "PyPI API token"),
    ("discord_bot", re.compile(r"\bMT[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{20,}"), "Discord bot token"),
    ("slack_webhook", re.compile(r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+"), "Slack webhook URL"),
    ("telegram_bot_url", re.compile(r"https://api\.telegram\.org/bot\d+:"), "Telegram bot API URL (token inline)"),
    ("bearer", re.compile(r"\bbearer\s+[A-Za-z0-9_\-.]{16,}"), "Authorization: Bearer value"),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"), "PEM private key block"),
    ("cert", re.compile(r"-----BEGIN CERTIFICATE-----"), "PEM certificate block"),
)

_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.I,
)


def _mask(value: str) -> str:
    if len(value) <= 8:
        return value[:2] + "…"
    return value[:4] + "…" + value[-4:]


# ─────────────────────────── decoder base + registry ───────────────────────


class Decoder:
    """Base class: one decoding strategy."""

    name: str = "base"
    description: str = ""

    def attempt(self, data: str) -> DecoderHit | None:
        raise NotImplementedError

    def try_bytes(self, data: bytes) -> DecoderHit | None:  # noqa: B027
        """Optional byte-stream path (gzip/zlib/…); default: none."""
        return None


DECODERS: list[Decoder] = []


def register_decoder(decoder: Decoder) -> Decoder:
    """Extend the engine at runtime (also used by the built-ins below)."""
    if any(d.name == decoder.name for d in DECODERS):
        return decoder
    DECODERS.append(decoder)
    return decoder


def _word_score(text: str) -> float:
    """How 'wordy' a decoded result is (common English anchors)."""
    if not text:
        return 0.0
    low = text.lower()
    anchors = (" the ", " a ", " is ", " you ", " http", "def ", "import ",
               "error", "usage", "true", "false", "null", "pass", "user",
               "name", "value", "beast")
    hits = sum(1 for a in anchors if a in low)
    return min(1.0, hits / 4.0)


_ROTWORDS = ("the", "and", "you", "for", "are", "but", "not", "all", "can",
             "her", "one", "our", "out", "day", "get", "has", "him", "how",
             "its", "was", "what", "will", "with", "code", "data", "error",
             "is", "a", "hello", "world", "that", "this", "have", "from",
             "they", "know", "want", "just", "like", "time", "very",
             "when", "make", "many", "such", "take", "come", "good",
             "well", "new", "also", "here", "there", "where", "which",
             "their", "them", "then", "only", "more", "other", "some",
             "than", "into", "over", "after", "about", "before", "text",
             "file", "name", "value", "user", "pass", "test", "check")


def _cipher_wordy(text: str) -> bool:
    words = re.findall(r"[a-z]{3,}", text.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in _ROTWORDS)
    return hits >= max(1, len(words) // 6)


#: top-20 English letters + space — the "looks like real text" gate.
#: Random base64 of ids/hashes decodes to printable garbage full of rare
#: chars (q, ], ^, {); real text is dense in common letters.
_COMMON_LETTERS = set("etaoinshrdlu cmwfpbygkv")
_COMMON_PUNCT = set(" \t.,:;!?'\"()/-_+=@#%&")


def _commonness(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.lower() in _COMMON_LETTERS
               or c in _COMMON_PUNCT) / len(text)


def _bytes_hit(decoder: str, raw: bytes, conf: float,
               note: str) -> DecoderHit | None:
    """Prefer a utf-8 reading when the bytes are real text; else bytes.

    "real text" = printable + has letters + common-letter density ≥ 0.55
    (rejects base64-of-random-ids false positives like 'q]u]u]u]u').
    The ``note`` (e.g. the recovered XOR key) is preserved in the
    utf-8/text reading so the *how* survives the *what*.
    """
    text = raw.decode("utf-8", "ignore")
    if looks_like_text(text) and len(text) >= 3 and \
            _commonness(text) >= 0.55:
        full = f"{note} ({len(raw)} bytes → utf-8)" if note \
            else f"{len(raw)} bytes → utf-8"
        return DecoderHit(decoder, text, min(conf + 0.05, 0.98), full)
    if looks_like_text(text) and len(text) >= 3:
        # printable but odd letter distribution — report as bytes with a
        # note; still decodable, just suspicious
        return DecoderHit(decoder, raw, conf * 0.5,
                          f"{note} ({len(raw)} bytes (printable, atypical))"
                          if note else
                          f"{len(raw)} bytes (printable, atypical)",
                          data_type="bytes")
    full = f"{note} ({len(raw)} bytes)" if note \
        else f"{len(raw)} bytes"
    return DecoderHit(decoder, raw, conf, full, data_type="bytes")


class _HexDecoder(Decoder):
    name = "hex"
    description = "hex / base16 (plain, 0x-, \\x-; whitespace tolerated)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if s.lower().startswith("0x"):
            s = s[2:]
        s = s.replace("\\x", " ")
        s = re.sub(r"\s+", " ", s).strip()
        if not s or len(s) < 8 or len(s) % 2:
            return None
        if not _HEX.match(s.replace(" ", "")):
            return None
        if len(set(s.replace(" ", ""))) < 3:
            return None  # "00000000" carries no information
        try:
            raw = binascii.unhexlify(s.replace(" ", ""))
        except (ValueError, binascii.Error):
            return None
        text = raw.decode("utf-8", "ignore")
        if looks_like_text(text) and len(text) >= 3 and \
                _commonness(text) >= 0.55:
            conf = 0.9 if _word_score(text) else 0.75
            return DecoderHit(self.name, text, conf,
                              f"{len(raw)} bytes as utf-8")
        if looks_like_text(text) and len(text) >= 3:
            return DecoderHit(self.name, raw, 0.3,
                              f"{len(raw)} bytes (printable, atypical)",
                              data_type="bytes")
        return DecoderHit(self.name, raw, 0.5,
                          f"{len(raw)} bytes (not utf-8)", data_type="bytes")


class _HexdumpDecoder(Decoder):
    name = "hexdump"
    description = "hexdump/xxd lines with offsets (00000000: 4d 5a …)"

    LINE = re.compile(r"^[0-9a-f]{4,16}\s*:?\s*((?:[0-9a-f]{2}\s?)+)", re.I)

    def attempt(self, data: str) -> DecoderHit | None:
        lines = [ln.strip() for ln in data.splitlines()
                 if self.LINE.match(ln.strip())]
        if len(lines) < 2:
            return None
        hexchars: list[str] = []
        for ln in lines:
            m = self.LINE.match(ln)
            if m:
                hexchars.append(
                    m.group(1).replace(" ", "").replace("\t", ""))
        blob = "".join(hexchars)
        if len(blob) < 16 or len(blob) % 2:
            return None
        try:
            raw = binascii.unhexlify(blob)
        except (ValueError, binascii.Error):
            return None
        text = raw.decode("utf-8", "ignore")
        if looks_like_text(text):
            return DecoderHit(self.name, text, 0.9,
                              f"{len(lines)} dump lines → {len(raw)} bytes")
        return DecoderHit(self.name, raw, 0.6,
                          f"{len(lines)} dump lines → {len(raw)} bytes",
                          data_type="bytes")


class _Base64Decoder(Decoder):
    name = "base64"
    description = "base64 / base64url (padding-tolerant, line wraps ok)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = re.sub(r"\s+", "", data.strip())
        if not (8 <= len(s) <= 1_000_000):
            return None
        if not re.fullmatch(r"[A-Za-z0-9+/=_\-]+", s):
            return None
        core = s.rstrip("=")
        if len(core) < 8 or len(set(core)) < 3 or "=" in core:
            return None
        padded = core + "=" * (-len(core) % 4)
        candidates: list[tuple[str, bytes]] = []
        if not any(c in core for c in "-_"):
            try:
                candidates.append(("base64", base64.b64decode(padded)))
            except (binascii.Error, ValueError):
                pass
        if re.fullmatch(r"[A-Za-z0-9_\-=]+", padded):
            try:
                candidates.append(("base64url",
                                   base64.urlsafe_b64decode(padded)))
            except (binascii.Error, ValueError):
                pass
        for label, raw in candidates:
            if not raw:
                continue
            text = raw.decode("utf-8", "ignore")
            if looks_like_text(text) and len(text) >= 3 and \
                    _commonness(text) >= 0.55:
                conf = min(0.85 + 0.1 * _word_score(text), 0.95)
                return DecoderHit(self.name, text, conf,
                                  f"{label}: {len(raw)} bytes → utf-8")
            magic = identify_magic(raw)
            if magic:
                return DecoderHit(self.name, raw, 0.9,
                                  f"{label}: decodes to {magic} "
                                  f"({len(raw)} bytes)", data_type="bytes")
            if looks_like_text(text) and len(text) >= 3:
                return DecoderHit(self.name, raw, 0.3,
                                  f"{label}: {len(raw)} bytes "
                                  f"(printable, atypical letters)",
                                  data_type="bytes")
            if printable_ratio(raw) > 0.95:
                return DecoderHit(self.name, raw, 0.55,
                                  f"{label}: {len(raw)} bytes (printable)",
                                  data_type="bytes")
        return None


class _Base32Decoder(Decoder):
    name = "base32"
    description = "RFC4648 base32 (padding-tolerant)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = re.sub(r"\s+", "", data.strip().upper())
        if not (10 <= len(s) <= 500_000) or not re.fullmatch(r"[A-Z2-7=]+", s):
            return None
        core = s.rstrip("=")
        if len(core) < 10 or len(set(core)) < 3 or "=" in core:
            return None
        try:
            raw = base64.b32decode(core + "=" * (-len(core) % 8))
        except (binascii.Error, ValueError):
            return None
        return _bytes_hit(self.name, raw, 0.8, "base32")


_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


class _Base58Decoder(Decoder):
    name = "base58"
    description = "base58 (bitcoin alphabet)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not (12 <= len(s) <= 200_000) or not all(c in _B58_ALPHABET
                                                    for c in s):
            return None
        if len(set(s)) < 3:
            return None
        n = 0
        for c in s:
            n = n * 58 + _B58_ALPHABET.index(c)
        raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
        for c in s:
            if c == "1":
                raw = b"\x00" + raw
            else:
                break
        if len(raw) < 2:
            return None
        text = raw.decode("utf-8", "ignore")
        if looks_like_text(text) and len(text) >= 4:
            return DecoderHit(self.name, text, 0.75, f"{len(raw)} bytes → utf-8")
        return None


class _BinaryDecoder(Decoder):
    name = "binary"
    description = "0/1 bit strings (spaces/commas tolerated)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = re.sub(r"[\s,]+", "", data.strip())
        if len(s) < 16 or len(s) % 8 or not set(s) <= {"0", "1"}:
            return None
        raw = bytes(int(s[i:i + 8], 2) for i in range(0, len(s), 8))
        return _bytes_hit(self.name, raw, 0.9, "binary")


class _Rot13Decoder(Decoder):
    name = "rot13"
    description = "ROT13 (suggested only when the result reads as words)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not s or len(s) > 100_000:
            return None
        if not re.fullmatch(r"[A-Za-z0-9 ,.\n\-_]+", s):
            return None
        out = codecs.decode(s, "rot_13")
        if out != s and _cipher_wordy(out):
            return DecoderHit(self.name, out, 0.5, "rot13 — verify it reads")
        return None


class _AtbashDecoder(Decoder):
    name = "atbash"
    description = "Atbash (a↔z …) — suggested only when wordy"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not s or len(s) > 100_000:
            return None
        if not re.fullmatch(r"[A-Za-z0-9 ,.\n\-_]+", s):
            return None
        out = "".join(
            chr(219 - ord(c)) if "a" <= c <= "z"
            else chr(155 - ord(c)) if "A" <= c <= "Z"
            else c
            for c in s
        )
        if out != s and _cipher_wordy(out):
            return DecoderHit(self.name, out, 0.45, "atbash — verify it reads")
        return None


class _ReverseDecoder(Decoder):
    name = "reverse"
    description = "reversed string (suggested when it opens with a word)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if len(s) < 8 or len(s) > 100_000:
            return None
        out = s[::-1]
        head = out[:12].lower()
        for w in ("the ", "hello", "http", "def ", "import", "from ",
                  "usage", "error", "note", "code"):
            if head.startswith(w):
                return DecoderHit(self.name, out, 0.5, "reversed")
        return None


class _UrlDecoder(Decoder):
    name = "url-percent"
    description = "URL percent-encoding (%20 …)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not re.search(r"%[0-9a-fA-F]{2}", s):
            return None
        out = _url_unquote(s)
        if out == s or not looks_like_text(out):
            return None
        return DecoderHit(self.name, out, 0.9, "percent-decoded")


class _HtmlEntityDecoder(Decoder):
    name = "html-entities"
    description = "HTML entities (&amp; &#39; …)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not re.search(r"&(?:\w+|#\d+|#x[0-9a-fA-F]+);", s):
            return None
        out = html.unescape(s)
        # &nbsp; unescapes to U+00A0, which the text gate treats as
        # non-printable; normalize it (standard nbsp → space).
        out = out.replace("\xa0", " ")
        if out == s or not looks_like_text(out):
            return None
        return DecoderHit(self.name, out, 0.9, "entities decoded")


class _UnicodeEscapeDecoder(Decoder):
    name = "unicode-escapes"
    description = "\\uXXXX / \\UXXXXXXXX / \\xXX escape sequences"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not re.search(
                r"\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}|\\x[0-9a-fA-F]{2}", s):
            return None
        try:
            out = s.encode("utf-8").decode("unicode_escape")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return None
        if out == s or not looks_like_text(out):
            return None
        return DecoderHit(self.name, out, 0.85, "escape-decoded")


class _OctalEscapeDecoder(Decoder):
    name = "octal-escapes"
    description = "C-style \\NNN octal escapes"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not re.search(r"\\[0-7]{3}", s):
            return None
        out = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), s)
        if out == s or not looks_like_text(out):
            return None
        return DecoderHit(self.name, out, 0.8, "octal escapes resolved")


_MORSE = {
    ".-": "a", "-...": "b", "-.-.": "c", "-..": "d", ".": "e", "..-.": "f",
    "--.": "g", "....": "h", "..": "i", ".---": "j", "-.-": "k",
    ".-..": "l", "--": "m", "-.": "n", "---": "o", ".--.": "p", "--.-": "q",
    ".-.": "r", "...": "s", "-": "t", "..-": "u", "...-": "v", ".--": "w",
    "-..-": "x", "-.--": "y", "--..": "z",
    "-----": "5", ".----": "6", "..---": "7", "...--": "8", "....-": "9",
    "-....": "1", "--...": "2", "---..": "3", "----.": "4", ".....": "0",
    ".-.-.-": ".", "--..--": ",", "-.-.--": "!", "-..-.-": "?",
    "--...--": ":", ".-...-": '"', "-...-.-": "$", "..--.-": "@",
}


class _MorseDecoder(Decoder):
    name = "morse"
    description = "international morse (./- with spaces or | separators)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not re.fullmatch(r"[.\- |/]+", s) or len(s) < 8:
            return None
        if s.count(".") + s.count("-") < 5:
            return None
        out: list[str] = []
        for word in re.split(r"\s*[\|/]\s*", s):
            letters: list[str] = []
            ok = True
            for sym in word.split():
                if sym in _MORSE:
                    letters.append(_MORSE[sym])
                else:
                    ok = False
                    break
            if ok and letters:
                out.append("".join(letters))
        text = "".join(out).strip()
        if len(re.sub(r"\s", "", text)) >= 3:
            return DecoderHit(self.name, text, 0.85, "morse decoded")
        return None


_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
         "8": "b", "9": "g", "@": "a", "$": "s"}


class _LeetDecoder(Decoder):
    name = "leet"
    description = "leet-speak substitution (p@ssw0rd → password)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not s or len(s) > 1000 or not any(c in _LEET for c in s):
            return None
        if not re.fullmatch(r"[A-Za-z0-9@$]+", s):
            return None
        out = "".join(_LEET.get(c.lower(), c) for c in s)
        if out == s or re.search(r"(.)\1{3,}", out):
            # no change — or the result is a letter-run ("cardaiiiiiiiook"),
            # which is a leet-decode of a random id, not a word
            return None
        if _cipher_wordy(out) or re.search(
                r"\b(pass|word|admin|user|test|code)\b", out, re.I) \
                or (out.isalpha() and len(out) >= 4):
            return DecoderHit(self.name, out, 0.6, "leet → plain")
        return None


class _JsonDecoder(Decoder):
    name = "json"
    description = "JSON document"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not s or s[0] not in "{[\"":
            return None
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(parsed, (dict, list, str)):
            return None
        return DecoderHit(self.name, parsed, 0.98, "parsed JSON",
                          data_type="structured")


class _YamlDecoder(Decoder):
    name = "yaml"
    description = "YAML document (needs PyYAML installed)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not s or s[0] in "{[\"" or not re.search(
                r"^[A-Za-z_][\w\-]*\s*:", s, re.M):
            return None
        try:
            import yaml  # type: ignore
        except ImportError:
            return None
        try:
            parsed = yaml.safe_load(s)
        except Exception:  # noqa: BLE001 — yaml raises many types
            return None
        if not isinstance(parsed, (dict, list)) or not parsed:
            return None
        return DecoderHit(self.name, parsed, 0.8, "parsed YAML",
                          data_type="structured")


class _KeyValueDecoder(Decoder):
    name = "key-value"
    description = "key=value / key: value lines"

    LINE = re.compile(r"^([A-Za-z0-9_.\-]{1,64})\s*[=:]\s*(.*)$")

    def attempt(self, data: str) -> DecoderHit | None:
        lines = [ln for ln in data.strip().splitlines() if ln.strip()]
        if len(lines) < 2 or len(lines) > 5000:
            return None
        parsed: dict[str, str] = {}
        for ln in lines:
            m = self.LINE.match(ln.strip())
            if not m:
                return None
            parsed[m.group(1)] = m.group(2).strip().strip("\"'")
        return DecoderHit(self.name, parsed, 0.8,
                          f"{len(parsed)} keys", data_type="structured")


class _IniDecoder(Decoder):
    name = "ini"
    description = "INI / properties (sections + key=value)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not re.search(r"^\[[^\]]+\]", s, re.M) or "=" not in s:
            return None
        import configparser

        try:
            cp = configparser.ConfigParser(strict=False, interpolation=None)
            cp.read_string(s)
        except (configparser.Error, ValueError):
            return None
        if not cp.sections():
            return None
        parsed = {sec: dict(cp.items(sec)) for sec in cp.sections()}
        return DecoderHit(self.name, parsed, 0.8,
                          f"sections: {', '.join(parsed)}",
                          data_type="structured")


class _CsvDecoder(Decoder):
    name = "csv"
    description = "CSV / TSV table (consistent column count)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if len(s) < 20:
            return None
        delim = "\t" if s.count("\t") >= s.count(",") else ","
        if s.count(delim) < 3:
            return None
        rows = [ln for ln in s.splitlines() if ln.strip()]
        if len(rows) < 2:
            return None
        counts = {ln.count(delim) + 1 for ln in rows}
        if len(counts) > 1:
            return None
        cols = rows[0].split(delim)
        if any(not c.strip() for c in cols):
            return None
        record = {
            "delimiter": "tab" if delim == "\t" else "comma",
            "columns": cols,
            "row_count": len(rows) - 1,
            "rows": [dict(zip(cols, ln.split(delim))) for ln in rows[1:21]],
            "note": f"{len(rows) - 1} rows total"
            if len(rows) - 1 > 20 else "",
        }
        return DecoderHit(self.name, record, 0.75,
                          f"{len(rows) - 1} rows × {len(cols)} cols",
                          data_type="structured")


class _MimeDecoder(Decoder):
    name = "mime"
    description = "MIME message (headers, attachments, base64 bodies)"

    def attempt(self, data: str) -> DecoderHit | None:
        if not re.search(
                r"(?im)^(content-type|mime-version|content-transfer-encoding)\s*:",
                data):
            return None
        headers = dict(re.findall(
            r"(?im)^(content-type|mime-version|content-transfer-encoding|"
            r"from|subject)\s*:\s*(.+)$", data))
        parts: list[dict[str, Any]] = []
        for m in re.finditer(
                r"(?is)content-disposition:\s*attachment[^;]*;\s*"
                r"filename=\"?([^\";]+)\"?(.*?)(?:--[\w=]{8,}|\Z)",
                data):
            body = m.group(2)
            decoded = None
            if re.search(r"(?i)base64", data[: m.start() + 200]) or \
                    "base64" in data.lower():
                core = re.sub(r"\s+", "", body)
                try:
                    raw = base64.b64decode(core)
                    decoded = raw.decode("utf-8", "ignore")
                except (binascii.Error, ValueError):
                    decoded = None
            parts.append({"filename": m.group(1), "decoded": decoded})
        if not parts and "content-transfer-encoding: base64" in data.lower():
            body = data.split("\n", 2)[-1]
            core = re.sub(r"\s+", "", body)
            try:
                raw = base64.b64decode(core)
            except (binascii.Error, ValueError):
                return None
            magic = identify_magic(raw)
            text = raw.decode("utf-8", "ignore")
            if magic or looks_like_text(text):
                parts.append({"filename": "",
                              "decoded": text if looks_like_text(text)
                              else None,
                              "magic": magic})
        if not parts:
            return None
        return DecoderHit(self.name, {"headers": headers, "parts": parts},
                          0.8, f"{len(parts)} part(s)", data_type="structured")


class _PemDecoder(Decoder):
    name = "pem"
    description = "PEM block (certificate / private key / OpenSSH)"

    BLOCK = re.compile(
        r"-----BEGIN ([A-Z0-9 ]+)-----\s*(.*?)\s*-----END \1-----", re.S
    )

    def attempt(self, data: str) -> DecoderHit | None:
        m = self.BLOCK.search(data)
        if not m:
            return None
        label, core = m.group(1), re.sub(r"\s+", "", m.group(2))
        try:
            raw = base64.b64decode(core)
        except (binascii.Error, ValueError):
            raw = b""
        note = f"PEM {label}"
        out: dict[str, Any] = {"label": label}
        if raw:
            magic = identify_magic(raw)
            if magic:
                note += f" → DER looks like {magic}"
            out["der_bytes"] = len(raw)
        return DecoderHit(self.name, out, 0.9, note, data_type="structured")


class _JwtDecoder(Decoder):
    name = "jwt"
    description = "JSON Web Token (header + payload claims; flags alg:none)"

    def attempt(self, data: str) -> DecoderHit | None:
        m = re.search(
            r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)?",
            data)
        if not m:
            return None
        parts = m.group(0).split(".")
        if len(parts) < 2:
            return None

        def _b64u(seg: str) -> bytes:
            return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))

        try:
            header = json.loads(_b64u(parts[0]))
            payload = json.loads(_b64u(parts[1]))
        except (binascii.Error, ValueError, json.JSONDecodeError):
            return None
        out: dict[str, Any] = {"header": header, "payload": payload}
        notes: list[str] = []
        alg = str(header.get("alg", ""))
        if alg.lower() == "none":
            notes.append("WARNING: alg=none — unsigned token")
        elif alg and alg not in ("HS256", "HS384", "HS512", "RS256",
                                 "RS384", "RS512", "ES256", "ES384",
                                 "ES512", "PS256"):
            notes.append(f"unusual alg: {alg}")
        exp = payload.get("exp")
        if isinstance(exp, (int, float)):
            delta = time.time() - float(exp)
            notes.append("exp in the PAST" if delta > 0
                         else f"exp in {int(delta // 3600)}h")
        signed = len(parts) == 3 and bool(parts[2])
        out["signed"] = signed
        if not signed:
            notes.append("no signature segment")
        out["warnings"] = notes
        return DecoderHit(self.name, out, 0.95,
                          "; ".join(notes) or "decoded",
                          data_type="structured")


class _CookieDecoder(Decoder):
    name = "cookies"
    description = "Cookie / Set-Cookie header (flags + value decode)"

    def attempt(self, data: str) -> DecoderHit | None:
        head = data.strip()[:60].lower()
        is_set = "set-cookie" in head
        has_header = "cookie" in head[:20]
        body = data
        if is_set:
            body = re.sub(r"(?im)^set-cookie\s*:\s*", "", body)
        else:
            body = re.sub(r"(?im)^cookie\s*:\s*", "", body)
        if "=" not in body:
            return None
        flag_keys = {"path", "domain", "expires", "max-age", "secure",
                     "httponly", "samesite", "version", "comment"}
        out: list[dict[str, Any]] = []
        # the CookieLab gives each cookie its class + service fingerprint
        # (wave 76) — the decoder's existing shape is preserved.
        lab_cookies: list[Any] = []
        try:
            from .cookies import CookieLab
            lab_cookies = CookieLab().parse(body)
        except Exception:  # noqa: BLE001 - classification is an enhancement
            lab_cookies = []
        lab_by_name: dict[str, Any] = {}
        for c in lab_cookies:
            lab_by_name.setdefault(c.name, c)
        # multi-line headers: split lines first, then ';' within a line
        # (a Set-Cookie attribute never starts a new cookie on its own
        # line, and a new cookie never continues on the next line)
        parts: list[str] = []
        for line in body.splitlines():
            parts.extend(line.split(";"))
        for part in parts:
            part = part.strip()
            if not part or "=" not in part:
                continue
            name, _, rest = part.partition("=")
            name = name.strip()
            if not name or name.lower() in flag_keys:
                continue
            cookie: dict[str, Any] = {"name": name, "value": rest.strip()}
            if re.search(r"%[0-9a-fA-F]{2}", cookie["value"]):
                unq = _url_unquote(cookie["value"])
                if unq != cookie["value"]:
                    cookie["value_decoded"] = unq
            if "eyJ" in cookie["value"]:
                jwt = _JwtDecoder().attempt(cookie["value"])
                if jwt is not None:
                    cookie["value_is_jwt"] = True
                    cookie["jwt"] = jwt.output
            labc = lab_by_name.get(name)
            if labc is not None:
                cookie["class"] = labc.kind
                if labc.service:
                    cookie["service"] = labc.service
            out.append(cookie)
        if not out:
            return None
        flags = re.findall(
            r"\b(HttpOnly|Secure|SameSite=None|SameSite=Lax|SameSite=Strict"
            r"|Max-Age=\d+|Expires=[^;\s]+|Domain=[^;\s]+|Path=[^;\s]+)",
            data, re.I)
        # a bare "abc=" is not a cookie (base64 with padding, k=v noise) —
        # require a header, several cookies, or at least one flag
        if not (has_header or len(out) >= 2 or flags):
            return None
        # no header, no flags, yet it "looks like cookies"?  Without any
        # cookie attribute, multi-line input is just structured text
        # (INI / key-value) — those decoders own it.  A bare cookie set
        # is either one line ("a=1; b=2", the classic header body) or
        # carries a header / at least one flag.
        if not has_header and not flags:
            if "\n" in body.strip() or \
                    re.search(r"^\s*\[[^\]]+\]\s*$", body, re.M):
                return None
        for f in flags:
            out.append({"flag": f})
        return DecoderHit(self.name, out, 0.9,
                          f"{len(out)} cookie(s)", data_type="structured")


class _UuidDecoder(Decoder):
    name = "uuid"
    description = "UUID v1–v5 detection"

    def attempt(self, data: str) -> DecoderHit | None:
        found = _UUID.findall(data)
        if not found:
            return None
        out = []
        for u in found:
            variant = f"v{u[14]}" if u[14] in "12345" else "v?"
            out.append({"uuid": u, "version": variant})
        return DecoderHit(self.name, out, 0.9,
                          f"{len(out)} UUID(s)", data_type="structured")


class _LuhnDecoder(Decoder):
    name = "luhn"
    description = "Luhn (mod-10) check on 12–19 digit numbers"

    def attempt(self, data: str) -> DecoderHit | None:
        runs = re.findall(r"\d(?:[ \-]?\d){11,18}", data)
        if not runs:
            return None
        runs = [re.sub(r"[ \-]", "", r) for r in runs]
        out = []
        for run in runs:
            total, dbl = 0, False
            for d in (int(c) for c in reversed(run)):
                if dbl:
                    d *= 2
                    if d > 9:
                        d -= 9
                total += d
                dbl = not dbl
            out.append({"number": _mask(run), "length": len(run),
                        "luhn_valid": total % 10 == 0})
        return DecoderHit(
            self.name, out, 0.6,
            f"{sum(1 for o in out if o['luhn_valid'])}/{len(out)} pass Luhn",
            data_type="structured")


class _GzipDecoder(Decoder):
    name = "gzip"
    description = "gzip / zlib byte streams"

    def try_bytes(self, data: bytes) -> DecoderHit | None:
        if not data or data[:2] not in (b"\x1f\x8b", b"\x78\x9c",
                                        b"\x78\x01", b"\x78\xda"):
            return None
        try:
            if data[:2] == b"\x1f\x8b":
                raw = zlib.decompress(data, 16 + zlib.MAX_WBITS)
            else:
                raw = zlib.decompress(data)
        except zlib.error:
            return None
        return _bytes_hit(self.name, raw, 0.9,
                          f"decompressed {len(data)} → {len(raw)} bytes")


class _Bz2Decoder(Decoder):
    name = "bz2"
    description = "bzip2 byte stream (BZh magic)"

    def try_bytes(self, data: bytes) -> DecoderHit | None:
        if not data or not data.startswith(b"BZh"):
            return None
        try:
            import bz2

            raw = bz2.decompress(data)
        except (OSError, ValueError, EOFError):
            return None
        return _bytes_hit(self.name, raw, 0.9,
                          f"bzip2 {len(data)} → {len(raw)} bytes")


class _XzDecoder(Decoder):
    name = "xz"
    description = "xz/lzma byte stream (\\xfd7zXZ\\0 magic)"

    def try_bytes(self, data: bytes) -> DecoderHit | None:
        # the xz magic is 6 bytes; the following byte is flags (check type)
        if not data or data[:6] != b"\xfd7zXZ\x00":
            return None
        try:
            import lzma

            raw = lzma.decompress(data)
        except (OSError, ValueError, EOFError):
            return None
        return _bytes_hit(self.name, raw, 0.9,
                          f"xz {len(data)} → {len(raw)} bytes")


class _QuotedPrintableDecoder(Decoder):
    name = "quoted-printable"
    description = "Quoted-Printable (=XX escapes, soft line breaks)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if not s or "=" not in s:
            return None
        # require a real QP signature: several =XX hex escapes, or a
        # soft line break ("=\n") — a lone '=' (key=value, base64 pad)
        # is not QP
        escapes = re.findall(r"=[0-9A-F]{2}", s, re.I)
        if len(escapes) < 3 and "=\n" not in s:
            return None
        try:
            import quopri

            raw = quopri.decodestring(s.encode("utf-8", "ignore"),
                                      header=False)
        except Exception:  # noqa: BLE001
            return None
        text = raw.decode("utf-8", "ignore")
        if not looks_like_text(text) or _commonness(text) < 0.7:
            return None
        # it must actually change the input
        if text.replace("\n", "") == s.replace("\n", "").replace("=", ""):
            return None
        conf = 0.8 if _word_score(text) else 0.6
        return DecoderHit(self.name, text, conf, "quoted-printable decoded",
                          data_type="text")


class _XorDecoder(Decoder):
    name = "xor"
    description = "single-byte XOR (key recovered by printable-score)"

    #: compression signatures — a short random blob that carries one of
    #: these is compressed data, not XOR noise, and the dedicated
    #: decompressors own it (XOR "winning" there is a false positive)
    _COMPRESSED_MAGICS = (b"\x1f\x8b", b"\x78\x9c", b"\x78\x01",
                          b"\x78\xda", b"BZh", b"\xfd7zXZ\x00")

    def try_bytes(self, data: bytes) -> DecoderHit | None:
        if not data or len(data) < 16:
            return None
        if any(data.startswith(m) for m in self._COMPRESSED_MAGICS):
            return None
        best: tuple[float, int, bytes] | None = None
        for key in range(1, 256):
            out = bytes(b ^ key for b in data)
            text = out.decode("utf-8", "ignore")
            if not looks_like_text(text):
                continue
            words = _word_score(text)
            if words <= 0:
                continue  # must contain a real English anchor
            score = _commonness(text) + words * 0.5
            if score >= 0.8 and (best is None or score > best[0]):
                best = (score, key, out)
        if best is None:
            return None
        _score, key, raw = best
        conf = min(0.95, 0.6 + 0.35 * (_score - 0.8) / 0.4)
        return _bytes_hit(self.name, raw, conf,
                          f"single-byte XOR key=0x{key:02x} "
                          f"({len(data)} bytes)")


class _CaesarDecoder(Decoder):
    name = "caesar"
    description = "Caesar/ROT-N on letters (shift recovered by word score)"

    def attempt(self, data: str) -> DecoderHit | None:
        s = data.strip()
        if len(s) < 16:
            return None
        letters = [c for c in s if c.isalpha()]
        if len(letters) < max(8, len(s) // 2):
            return None  # not letter-dense enough to be a Caesar cipher
        base_score = _cipher_wordy(s.lower())
        best: tuple[float, int, str] | None = None
        for shift in range(1, 26):
            out = []
            for c in s:
                if c.isalpha():
                    ord_a = 65 if c.isupper() else 97
                    out.append(chr((ord(c) - ord_a + shift) % 26 + ord_a))
                else:
                    out.append(c)
            cand = "".join(out)
            score = _word_score(cand) + (0.2 if _cipher_wordy(cand.lower())
                                         else 0.0)
            if best is None or score > best[0]:
                best = (score, shift, cand)
        if best is None:
            return None
        score, shift, cand = best
        words = _word_score(cand)
        # must be genuinely wordy — at least two common English anchors
        # (one anchor is too easy to hit by chance on 26 trials) — and
        # the input must not already be wordier than the candidate
        if words < 0.5 or _commonness(cand) < 0.75:
            return None
        if _cipher_wordy(s.lower()) and words <= 0.5 and \
                _word_score(s) >= words:
            return None
        return DecoderHit(self.name, cand, min(0.9, 0.5 + score),
                          f"caesar shift +{shift}", data_type="text")


class _StringsDecoder(Decoder):
    name = "strings"
    description = "embedded ASCII/UTF-16 strings in a binary blob"

    def try_bytes(self, data: bytes) -> DecoderHit | None:
        if not data or len(data) < 16:
            return None
        # ASCII runs of >= 6 printable chars
        ascii_runs = re.findall(rb"[\x20-\x7e]{6,}", data)
        # UTF-16LE runs of >= 6 printable chars
        utf16_runs = re.findall(
            rb"(?:[\x20-\x7e]\x00){6,}", data)
        found: list[str] = []
        for run in ascii_runs:
            found.append(run.decode("ascii"))
        for run in utf16_runs:
            found.append(run.decode("utf-16-le", "ignore"))
        if not found:
            return None
        # keep the interesting ones: wordy, urls, emails, key=value,
        # json-ish, hex/base64 blobs, or long
        interesting: list[str] = []
        for s in found:
            if len(s) >= 16 or _word_score(s) >= 0.25 or \
                    re.search(r"(https?://|@|=|{|{|\{|:)", s) or \
                    re.fullmatch(r"[0-9a-f]{16,}", s) or \
                    re.fullmatch(r"[A-Za-z0-9+/=]{16,}", s):
                interesting.append(s)
        if not interesting:
            return None
        interesting = interesting[:60]
        conf = 0.55 + min(0.35, len(interesting) / 40)
        return DecoderHit(self.name, interesting, min(conf, 0.9),
                          f"{len(interesting)} embedded string(s) in "
                          f"{len(data)} bytes", data_type="structured")


# ── built-in registry ──
for _d in (_HexDecoder, _HexdumpDecoder, _Base64Decoder, _Base32Decoder,
           _Base58Decoder, _BinaryDecoder, _Rot13Decoder, _AtbashDecoder,
           _ReverseDecoder, _UrlDecoder, _HtmlEntityDecoder,
           _UnicodeEscapeDecoder, _OctalEscapeDecoder, _MorseDecoder,
           _LeetDecoder, _JsonDecoder, _YamlDecoder, _KeyValueDecoder,
           _IniDecoder, _CsvDecoder, _MimeDecoder, _PemDecoder,
           _JwtDecoder, _CookieDecoder, _UuidDecoder, _LuhnDecoder,
           _GzipDecoder, _Bz2Decoder, _XzDecoder, _QuotedPrintableDecoder,
           _XorDecoder, _CaesarDecoder, _StringsDecoder):
    register_decoder(_d())


# ─────────────────────────── the engine ────────────────────────────────────

#: decode priority when two decoders both "work"
_RANKED_FIRST: tuple[str, ...] = (
    "json", "jwt", "cookies", "mime", "pem", "hexdump", "hex", "base64",
    "base32", "base58", "binary", "url-percent", "html-entities",
    "quoted-printable", "unicode-escapes", "octal-escapes", "morse",
    "leet", "reverse", "caesar", "rot13", "atbash", "key-value", "ini",
    "csv", "yaml", "uuid", "luhn",
)


def _rank(name: str) -> int:
    try:
        return _RANKED_FIRST.index(name)
    except ValueError:
        return len(_RANKED_FIRST)


def _run_decoders(data: str, min_confidence: float) -> list[DecoderHit]:
    hits: list[DecoderHit] = []
    for dec in DECODERS:
        try:
            hit = dec.attempt(data)
        except Exception:  # noqa: BLE001 — one decoder must never kill the run
            hit = None
        if hit is not None and hit.confidence >= min_confidence:
            hits.append(hit)
    hits.sort(key=lambda h: (-h.confidence, _rank(h.decoder)))
    return hits


def _run_byte_decoders(raw: bytes) -> list[DecoderHit]:
    hits: list[DecoderHit] = []
    for dec in DECODERS:
        try:
            hit = dec.try_bytes(raw)
        except Exception:  # noqa: BLE001
            hit = None
        if hit is not None:
            hits.append(hit)
    return hits


def decode_any(data: str, *, max_depth: int = 3,
               min_confidence: float = 0.4) -> dict[str, Any]:
    """Run every decoder; return ranked hits + the best nested chain.

    The chain walks the winning decode, re-decoding the result until
    nothing further decodes (or ``max_depth``), so a double-encoded paste
    lands as plain text.
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8", "ignore")
    hits = _run_decoders(data, min_confidence)

    best: dict[str, Any] | None = None
    if hits:
        top = hits[0]
        chain: list[str] = [top.decoder]
        final: Any = top.output
        conf = top.confidence
        current = top.output if isinstance(top.output, str) else ""
        for _ in range(max(0, max_depth - 1)):
            if not current or len(current) < 8:
                break
            subs: list[DecoderHit] = []
            for h in _run_decoders(current, min_confidence):
                if isinstance(h.output, str) and h.output == current:
                    continue
                if h.data_type == "bytes" and isinstance(h.output, (bytes,
                                                                    bytearray)):
                    # a nested byte hit is only worth taking if it decodes
                    # to something with a known signature (e.g. base64→PNG)
                    if not identify_magic(bytes(h.output)):
                        continue
                elif isinstance(h.output, str) and \
                        _commonness(h.output) < 0.55:
                    continue
                subs.append(h)
            if not subs:
                break
            nxt = subs[0]
            chain.append(nxt.decoder)
            conf = round(conf * (0.5 + 0.5 * nxt.confidence), 3)
            final = nxt.output
            current = nxt.output if isinstance(nxt.output, str) else ""
        out_kind = "text" if isinstance(final, str) else "bytes"
        best = {
            "chain": chain + ([out_kind] if out_kind == "text" else []),
            "confidence": conf,
            "output": final if isinstance(final, (str, dict, list))
            else final,
            "note": top.note,
        }
    return {"best": best, "hits": [h.to_dict() for h in hits],
            "count": len(hits), "_hits_raw": hits}


def analyze(data: str | bytes, *, max_depth: int = 3) -> DecodeReport:
    """Full analysis: forensics + hash id + tokens + every decoder hit."""
    raw = data if isinstance(data, bytes) else data.encode("utf-8", "ignore")
    text = data if isinstance(data, str) else raw.decode("utf-8", "ignore")

    forensics: dict[str, Any] = {
        "chars": len(text),
        "bytes": len(raw),
        "entropy": round(shannon_entropy(text), 3),
        "printable_ratio": round(printable_ratio(raw), 3),
        "charset": detect_charset(raw) if raw else "empty",
        "magic": identify_magic(raw),
        "top_chars": Counter(text).most_common(5),
        "looks_like_text": looks_like_text(text),
    }

    stripped = text.strip()
    hash_info: dict[str, Any] | None = None
    candidates = identify_hash(stripped)
    if candidates:
        hash_info = {"candidates": candidates,
                     "known": known_hash_lookup(stripped)}

    tokens = []
    for key, pattern, label in TOKEN_PATTERNS:
        for m in pattern.finditer(text):
            tokens.append({"kind": key, "label": label,
                           "sample": _mask(m.group(0))})
    tokens = tokens[:20]

    all_hits: list[DecoderHit] = []
    if text.strip():
        str_report = decode_any(text, max_depth=max_depth)
        all_hits.extend(str_report.pop("_hits_raw"))
        best = str_report["best"]
    else:
        best = None
    all_hits.extend(_run_byte_decoders(raw))

    cookies: list[dict[str, Any]] = []
    jwt_info: dict[str, Any] | None = None
    for h in all_hits:
        if h.decoder == "cookies" and isinstance(h.output, list):
            cookies = [c for c in h.output if isinstance(c, dict)]
        if h.decoder == "jwt" and isinstance(h.output, dict):
            jwt_info = h.output

    if best is None and all_hits:
        top = max(all_hits, key=lambda h: h.confidence)
        out_kind = "text" if isinstance(top.output, str) else "bytes"
        best = {"chain": [top.decoder] + ([out_kind] if out_kind == "text"
                                          else []),
                "confidence": top.confidence,
                "output": top.output, "note": top.note}

    all_hits.sort(key=lambda h: (-h.confidence, _rank(h.decoder)))
    return DecodeReport(best=best, hits=all_hits, forensics=forensics,
                        hash=hash_info, tokens=tokens, cookies=cookies,
                        jwt=jwt_info)
