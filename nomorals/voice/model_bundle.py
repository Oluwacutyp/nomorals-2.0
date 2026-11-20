"""Devon's own three-part voice-model bundle (``.dvmodel``).

Mined from storytold/fakeyou-recast's ``model-serde`` crate (census
§3B.13 — PATTERN ONLY, no licence, no code ported: the wire format,
the crypto and the framing below are all written fresh here). The
upstream shape we keep:

* one file carries a voice-conversion model as three named parts after
  a small plaintext header: (1) unencrypted lightweight details,
  (2) encrypted lightweight details, (3) the encrypted heavy payload
  (acoustic model bytes + vocoder model bytes);
* the header is readable without the secret, so a directory scan is
  fast — the upstream reads the first 500 bytes and stops;
* three read tiers: fast metadata only, metadata (decrypt the light
  parts), full (decrypt everything);
* the packager takes acoustic + vocoder files, a name, a token, an
  author and a secret, and writes one sealed file.

What Devon does differently (the port is the floor, not the ceiling):

* the upstream seals with magic-crypt — deterministic AES with no
  salt and no authentication. This format derives per-file keys with
  PBKDF2-HMAC-SHA256 over a random 16-byte salt and seals each part
  with encrypt-then-MAC (HMAC-SHA256 over the part name + ciphertext),
  so a wrong secret or a flipped byte is refused naming the part;
* every part's SHA-256 is recorded in the header and checked before
  the part is parsed, so corruption is refused naming the part;
* the upstream's errors are deliberately obscure (every failure prints
  ``"error"``). Devon's refusals name the part and the cause;
* the crypto is stdlib-only (``hashlib``/``hmac``) — no AES in the
  standard library, so the stream cipher is a SHA-256 counter-mode
  keystream, documented honestly here as at-rest sealing, not a
  vetted cipher. It runs everywhere Devon runs, including EC2 and
  Termux, with no new dependency.

File layout (all integers little-endian)::

    magic            8 bytes  "DVMODEL\\0"
    format_version   u16      1
    salt             16 bytes random per file
    u32 len + fast metadata JSON (UTF-8)
    u32 len + part 1 "details" JSON (unencrypted)
    u32 len + part 2 "encrypted_details" envelope (MAC || ciphertext)
    u32 len + part 3 "heavy" envelope (MAC || ciphertext)

Fast metadata JSON: ``format_version`` (semver, <=12 chars),
``token`` (model primary key, <=32 chars), ``name`` (<=255),
``author`` (<=20), ``parts`` (hex SHA-256 of each part's STORED
bytes — ciphertext for the encrypted parts).

Part 1 JSON: ``model_version``, ``created_at_ms``, ``updated_at_ms``
(epoch milliseconds, upstream's units).

Part 2 JSON (inside the envelope): ``expiry_s``,
``invalid_before_s`` (epoch seconds, upstream's units),
``bypass_expiry`` (may an explicit caller override an expiry),
``password`` (optional short string gating use).

Part 3 frame (inside the envelope, binary): u32 pipeline version,
u64 acoustic length + acoustic bytes, u64 vocoder length + vocoder
bytes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import hmac
import json
import os
import re
import struct
import time
from dataclasses import dataclass, field

MAGIC = b"DVMODEL\x00"
FORMAT_VERSION = 1
FORMAT_VERSION_STR = "1.0.0"
EXTENSION = ".dvmodel"

# Upstream's field caps, kept: the header is meant to stay readable in
# under 500 bytes, so the fast fields are small by construction.
_MAX_FORMAT_VERSION_LEN = 12
_MAX_TOKEN_LEN = 32
_MAX_NAME_LEN = 255
_MAX_AUTHOR_LEN = 20
_MAX_PASSWORD_LEN = 64

_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_SEMVER_CORE_RE = re.compile(r"^\d+\.\d+\.\d+$")

_PBKDF2_ITERATIONS = 200_000
_SALT_LEN = 16
_MAC_LEN = 32
_CHUNK = 1 << 20


class ModelBundleError(ValueError):
    """A bundle refusal — the message always names the part and the cause."""


@dataclass
class FastMetadata:
    """The plaintext header: readable without the secret."""

    format_version: str
    token: str
    name: str
    author: str
    parts: dict[str, str]  # part name -> hex SHA-256 of the stored bytes

    def to_dict(self) -> dict:
        return {"format_version": self.format_version, "token": self.token,
                "name": self.name, "author": self.author,
                "parts": dict(self.parts)}


@dataclass
class BundleMetadata:
    """Fast metadata + both lightweight detail parts (no model bytes)."""

    fast: FastMetadata
    model_version: str | None = None
    created_at_ms: int | None = None
    updated_at_ms: int | None = None
    expires_at_s: int | None = None
    invalid_before_s: int | None = None
    bypass_expiry: bool | None = None
    password_set: bool = False
    heavy_bytes: int = 0
    heavy_sha256: str = ""

    def to_dict(self) -> dict:
        return {"fast": self.fast.to_dict(),
                "model_version": self.model_version,
                "created_at_ms": self.created_at_ms,
                "updated_at_ms": self.updated_at_ms,
                "expires_at_s": self.expires_at_s,
                "invalid_before_s": self.invalid_before_s,
                "bypass_expiry": self.bypass_expiry,
                "password_set": self.password_set,
                "heavy_bytes": self.heavy_bytes,
                "heavy_sha256": self.heavy_sha256}


@dataclass
class VoiceModel:
    """The full decoded bundle: metadata + both model byte strings."""

    format_version: str = FORMAT_VERSION_STR
    token: str = ""
    name: str = ""
    author: str = ""
    model_version: str | None = None
    created_at_ms: int | None = None
    updated_at_ms: int | None = None
    expires_at_s: int | None = None
    invalid_before_s: int | None = None
    bypass_expiry: bool | None = None
    password: str | None = None
    pipeline_version: int = 1
    acoustic_model_bytes: bytes = b""
    vocoder_model_bytes: bytes = b""
    parts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"format_version": self.format_version, "token": self.token,
                "name": self.name, "author": self.author,
                "model_version": self.model_version,
                "created_at_ms": self.created_at_ms,
                "updated_at_ms": self.updated_at_ms,
                "expires_at_s": self.expires_at_s,
                "invalid_before_s": self.invalid_before_s,
                "bypass_expiry": self.bypass_expiry,
                "password_set": self.password is not None,
                "pipeline_version": self.pipeline_version,
                "acoustic_bytes": len(self.acoustic_model_bytes),
                "vocoder_bytes": len(self.vocoder_model_bytes),
                "parts": dict(self.parts)}


# -- crypto: stdlib-only sealing -------------------------------------
# PBKDF2-HMAC-SHA256 over a per-file salt derives two keys; each part
# is sealed with a SHA-256 counter-mode keystream (the standard
# library has no AES) and authenticated with HMAC-SHA256 over the
# part name + ciphertext. This is at-rest sealing for model files,
# not a vetted cipher — it is documented as such wherever it is used.

def _derive_keys(secret: str, salt: bytes) -> tuple[bytes, bytes]:
    material = hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"),
                                   salt, _PBKDF2_ITERATIONS, dklen=64)
    return material[:32], material[32:]


def _keystream(enc_key: bytes, part_name: str, length: int) -> bytes:
    out = bytearray()
    counter = 0
    label = part_name.encode("utf-8")
    while len(out) < length:
        out += hashlib.sha256(enc_key + label
                              + struct.pack("<Q", counter)).digest()
        counter += 1
    return bytes(out[:length])


def _seal_part(keys: tuple[bytes, bytes], part_name: str,
               plaintext: bytes) -> bytes:
    enc_key, mac_key = keys
    cipher = bytes(a ^ b for a, b in zip(
        plaintext, _keystream(enc_key, part_name, len(plaintext)),
        strict=True))
    mac = hmac.new(mac_key, part_name.encode("utf-8") + cipher,
                   hashlib.sha256).digest()
    return mac + cipher


def _open_part(keys: tuple[bytes, bytes], part_name: str,
               envelope: bytes) -> bytes:
    if len(envelope) < _MAC_LEN:
        raise ModelBundleError(
            f"{part_name}: envelope too short ({len(envelope)} bytes) — "
            "truncated bundle")
    enc_key, mac_key = keys
    mac, cipher = envelope[:_MAC_LEN], envelope[_MAC_LEN:]
    expect = hmac.new(mac_key, part_name.encode("utf-8") + cipher,
                      hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expect):
        raise ModelBundleError(
            f"{part_name}: authentication failed — wrong secret or "
            "tampered bytes")
    return bytes(a ^ b for a, b in zip(
        cipher, _keystream(enc_key, part_name, len(cipher)),
        strict=True))


# -- framing ----------------------------------------------------------

@dataclass
class _Section:
    name: str
    length: int
    sha256: str
    data: bytes | None  # None when streamed past (heavy part, metadata reads)


def _read_u32(fh, part_name: str) -> int:
    raw = fh.read(4)
    if len(raw) < 4:
        raise ModelBundleError(
            f"{part_name}: truncated bundle — missing length prefix")
    return struct.unpack("<I", raw)[0]


def _hash_stream(fh, length: int, part_name: str) -> str:
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        chunk = fh.read(min(_CHUNK, remaining))
        if not chunk:
            raise ModelBundleError(
                f"{part_name}: truncated bundle — expected {length} bytes")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_fast_fields(format_version: str, token: str, name: str,
                          author: str) -> None:
    """The header caps, enforced on write and on read (a tampered
    header is refused with the field named, not silently accepted)."""
    if len(format_version) > _MAX_FORMAT_VERSION_LEN:
        raise ModelBundleError(
            f"fast metadata: format version too long "
            f"({len(format_version)} > {_MAX_FORMAT_VERSION_LEN})")
    if len(token) > _MAX_TOKEN_LEN:
        raise ModelBundleError(
            f"fast metadata: model token too long "
            f"({len(token)} > {_MAX_TOKEN_LEN})")
    if not _TOKEN_RE.match(token):
        raise ModelBundleError(
            f"fast metadata: bad model token {token!r} — letters, digits, "
            "'-' and '_' only, must start with a letter or digit")
    if len(name) > _MAX_NAME_LEN:
        raise ModelBundleError(
            f"fast metadata: model name too long "
            f"({len(name)} > {_MAX_NAME_LEN})")
    if len(author) > _MAX_AUTHOR_LEN:
        raise ModelBundleError(
            f"fast metadata: author username too long "
            f"({len(author)} > {_MAX_AUTHOR_LEN})")


def _parse_fast(raw: bytes, path: str) -> FastMetadata:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ModelBundleError(
            f"fast metadata: unreadable header in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelBundleError(
            f"fast metadata: header is not an object in {path}")
    for key in ("format_version", "token", "name", "author", "parts"):
        if key not in data:
            raise ModelBundleError(
                f"fast metadata: header missing {key!r} in {path}")
    parts = data["parts"]
    if not isinstance(parts, dict):
        raise ModelBundleError(
            f"fast metadata: header 'parts' is not an object in {path}")
    for part in ("details", "encrypted_details", "heavy"):
        if part not in parts:
            raise ModelBundleError(
                f"fast metadata: header has no hash for part {part!r} "
                f"in {path}")
    _validate_fast_fields(str(data["format_version"]), str(data["token"]),
                          str(data["name"]), str(data["author"]))
    return FastMetadata(format_version=str(data["format_version"]),
                        token=str(data["token"]), name=str(data["name"]),
                        author=str(data["author"]),
                        parts={k: str(v) for k, v in parts.items()})


def _scan(path: str, *, retain_heavy: bool = False
          ) -> tuple[FastMetadata, dict[str, _Section], bytes]:
    """Walk a bundle once, streaming: header + per-part lengths and
    SHA-256, retaining the two small parts (and the heavy part only
    when ``retain_heavy``). The heavy part is hashed while streaming,
    never held in memory on metadata reads. A missing file raises
    FileNotFoundError; any other I/O failure is a named bundle error."""
    try:
        with open(path, "rb") as fh:
            return _scan_stream(fh, path, retain_heavy=retain_heavy)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ModelBundleError(f"cannot read bundle {path}: {exc}") from exc


def _scan_stream(fh, path: str, *, retain_heavy: bool = False
                 ) -> tuple[FastMetadata, dict[str, _Section], bytes]:
    magic = fh.read(len(MAGIC))
    if len(magic) < len(MAGIC):
        raise ModelBundleError(
            f"fast metadata: file too small for a bundle header: {path}")
    if magic != MAGIC:
        raise ModelBundleError(
            f"fast metadata: bad magic — not a Devon voice model "
            f"bundle (.dvmodel): {path}")
    raw_version = fh.read(2)
    if len(raw_version) < 2:
        raise ModelBundleError(
            "fast metadata: truncated bundle — missing format version")
    (version,) = struct.unpack("<H", raw_version)
    if version != FORMAT_VERSION:
        raise ModelBundleError(
            f"fast metadata: unsupported bundle format version "
            f"{version} (this build reads {FORMAT_VERSION})")
    salt = fh.read(_SALT_LEN)
    if len(salt) < _SALT_LEN:
        raise ModelBundleError(
            "fast metadata: truncated bundle — missing salt")
    fast_len = _read_u32(fh, "fast metadata")
    if fast_len > 64 * 1024:
        raise ModelBundleError(
            f"fast metadata: header too large ({fast_len} bytes)")
    fast_raw = fh.read(fast_len)
    if len(fast_raw) < fast_len:
        raise ModelBundleError(
            "fast metadata: truncated bundle — short header")
    fast = _parse_fast(fast_raw, path)
    parts: dict[str, _Section] = {}
    for name in ("details", "encrypted_details", "heavy"):
        length = _read_u32(fh, name)
        if name == "heavy" and not retain_heavy:
            digest = _hash_stream(fh, length, name)
            parts[name] = _Section(name, length, digest, None)
        else:
            data = fh.read(length)
            if len(data) < length:
                raise ModelBundleError(
                    f"{name}: truncated bundle — expected {length} "
                    "bytes")
            parts[name] = _Section(
                name, length, hashlib.sha256(data).hexdigest(), data)
    if fh.read(1):
        raise ModelBundleError(
            f"{path}: trailing bytes after the heavy payload — "
            "not a clean bundle")
    return fast, parts, salt


def _check_hashes(fast: FastMetadata,
                  parts: dict[str, _Section]) -> None:
    for name, section in parts.items():
        recorded = fast.parts.get(name, "")
        if section.sha256 != recorded:
            raise ModelBundleError(
                f"{name}: sha256 mismatch — stored {section.sha256} != "
                f"recorded {recorded}")


def _parse_json_part(part_name: str, raw: bytes) -> dict:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ModelBundleError(
            f"{part_name}: unreadable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelBundleError(f"{part_name}: payload is not an object")
    return data


def _encode_heavy_frame(model: VoiceModel) -> bytes:
    return (struct.pack("<I", model.pipeline_version)
            + struct.pack("<Q", len(model.acoustic_model_bytes))
            + model.acoustic_model_bytes
            + struct.pack("<Q", len(model.vocoder_model_bytes))
            + model.vocoder_model_bytes)


def _decode_heavy_frame(raw: bytes) -> tuple[int, bytes, bytes]:
    if len(raw) < 4 + 8:
        raise ModelBundleError(
            "heavy payload: frame too short for pipeline version + "
            "acoustic length")
    (pipeline_version,) = struct.unpack_from("<I", raw, 0)
    (acoustic_len,) = struct.unpack_from("<Q", raw, 4)
    pos = 12
    if len(raw) < pos + acoustic_len + 8:
        raise ModelBundleError(
            "heavy payload: frame too short for the acoustic model bytes")
    acoustic = raw[pos:pos + acoustic_len]
    pos += acoustic_len
    (vocoder_len,) = struct.unpack_from("<Q", raw, pos)
    pos += 8
    if len(raw) < pos + vocoder_len:
        raise ModelBundleError(
            "heavy payload: frame too short for the vocoder model bytes")
    vocoder = raw[pos:pos + vocoder_len]
    if len(raw) != pos + vocoder_len:
        raise ModelBundleError(
            "heavy payload: trailing bytes after the vocoder model")
    return pipeline_version, acoustic, vocoder


def _check_usable(meta: BundleMetadata, *, stored_password: str | None,
                  password: str | None, allow_expired: bool,
                  now: float | None) -> None:
    """Expiry + password gates, applied on every full read. A refusal
    names the part and the rule that fired."""
    moment = int(now if now is not None else time.time())
    if (meta.expires_at_s is not None and moment >= meta.expires_at_s
            and not (allow_expired or meta.bypass_expiry)):
        raise ModelBundleError(
            "encrypted details: model expired "
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(meta.expires_at_s))} "
            f"({meta.expires_at_s}) — pass allow_expired to use it anyway")
    if meta.invalid_before_s is not None and moment < meta.invalid_before_s:
        raise ModelBundleError(
            "encrypted details: model is not valid before "
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(meta.invalid_before_s))} "
            f"({meta.invalid_before_s})")
    if stored_password is not None:
        if password is None:
            raise ModelBundleError(
                "encrypted details: this model needs its password — "
                "pass it to use the model")
        # The stored password lives inside the authenticated envelope;
        # compare in constant time.
        if not hmac.compare_digest(password, stored_password):
            raise ModelBundleError(
                "encrypted details: password mismatch")



# -- public API -------------------------------------------------------

def validate_model(model: VoiceModel) -> None:
    """Refuse a model that cannot be packed, naming the field."""
    _validate_fast_fields(model.format_version, model.token,
                          model.name, model.author)
    if model.pipeline_version < 1:
        raise ModelBundleError(
            "heavy payload: pipeline version must be >= 1, got "
            f"{model.pipeline_version}")
    if not model.acoustic_model_bytes:
        raise ModelBundleError(
            "heavy payload: acoustic model missing — a bundle that "
            "cannot convert is not shippable")
    if not model.vocoder_model_bytes:
        raise ModelBundleError(
            "heavy payload: vocoder model missing — a bundle that "
            "cannot convert is not shippable")
    if model.password is not None and len(model.password) > _MAX_PASSWORD_LEN:
        raise ModelBundleError(
            "encrypted details: password too long "
            f"({len(model.password)} > {_MAX_PASSWORD_LEN})")


def _assemble_blob(model: VoiceModel, secret: str, salt: bytes) -> bytes:
    """Seal ``model`` into bundle bytes (no validation, no disk write —
    the single assembly point used by ``write_bundle`` and by tests
    crafting reader edge cases). Timestamps come from the model;
    ``write_bundle`` fills them in first."""
    keys = _derive_keys(secret, salt)
    details = json.dumps({
        "model_version": model.model_version,
        "created_at_ms": model.created_at_ms,
        "updated_at_ms": model.updated_at_ms,
    }, separators=(",", ":"), sort_keys=True).encode("utf-8")
    enc_details_plain = json.dumps({
        "expiry_s": model.expires_at_s,
        "invalid_before_s": model.invalid_before_s,
        "bypass_expiry": model.bypass_expiry,
        "password": model.password,
    }, separators=(",", ":"), sort_keys=True).encode("utf-8")
    enc_details = _seal_part(keys, "encrypted_details", enc_details_plain)
    heavy = _seal_part(keys, "heavy", _encode_heavy_frame(model))
    fast = json.dumps({
        "format_version": model.format_version,
        "token": model.token,
        "name": model.name,
        "author": model.author,
        "parts": {
            "details": hashlib.sha256(details).hexdigest(),
            "encrypted_details": hashlib.sha256(enc_details).hexdigest(),
            "heavy": hashlib.sha256(heavy).hexdigest(),
        },
    }, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return (MAGIC + struct.pack("<H", FORMAT_VERSION) + salt
            + struct.pack("<I", len(fast)) + fast
            + struct.pack("<I", len(details)) + details
            + struct.pack("<I", len(enc_details)) + enc_details
            + struct.pack("<I", len(heavy)) + heavy)


def _part_sizes(blob: bytes) -> dict:
    """Byte sizes of the three parts inside an assembled blob."""
    pos = len(MAGIC) + 2 + _SALT_LEN
    (fast_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4 + fast_len
    sizes = {}
    for name in ("details", "encrypted_details", "heavy"):
        (length,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        sizes[name] = length
        pos += length
    return sizes


def write_bundle(model: VoiceModel, secret: str, path: str, *,
                 replace: bool = False) -> dict:
    """Seal ``model`` into a ``.dvmodel`` file. Atomic (tmp file in the
    same directory + replace); without ``replace`` an existing file is
    refused, like the upstream packager's create-new write. Returns
    the receipt: path, token, sha256, per-part byte sizes."""
    if not secret:
        raise ModelBundleError(
            "a bundle needs a secret — pass one (never an empty string)")
    validate_model(model)
    now_ms = int(time.time() * 1000)
    created_ms = model.created_at_ms if model.created_at_ms is not None else now_ms
    updated_ms = (model.updated_at_ms if model.updated_at_ms is not None
                  else created_ms)
    # Fill the timestamps on a copy — the caller's model is not mutated.
    model = dataclasses.replace(model, created_at_ms=created_ms,
                                updated_at_ms=updated_ms)
    salt = os.urandom(_SALT_LEN)
    blob = _assemble_blob(model, secret, salt)
    target = os.path.abspath(path)
    if os.path.exists(target) and not replace:
        raise FileExistsError(
            f"bundle already exists: {target} (pass replace=True to overwrite)")
    tmp = target + ".tmp"
    try:
        with open(tmp, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return {"path": target, "token": model.token, "name": model.name,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
            "parts": _part_sizes(blob)}


def read_fast(path: str) -> FastMetadata:
    """The quick read: magic + header only, no secret, no part bytes.
    The directory-scan tier — fast enough to enumerate a folder."""
    fast, _parts, _salt = _scan(path)
    return fast


def read_details(path: str) -> dict:
    """The unencrypted details part (hash-verified, no secret)."""
    fast, parts, _salt = _scan(path)
    _check_hashes(fast, parts)
    raw = parts["details"].data
    assert raw is not None
    return _parse_json_part("details", raw)


def _hydrate_metadata(fast: FastMetadata,
                      parts: dict[str, _Section],
                      keys: tuple[bytes, bytes]) -> tuple[BundleMetadata, str | None]:
    details = _parse_json_part("details", parts["details"].data or b"")
    enc_raw = _open_part(keys, "encrypted_details",
                         parts["encrypted_details"].data or b"")
    enc = _parse_json_part("encrypted_details", enc_raw)
    stored_password = enc.get("password")
    meta = BundleMetadata(
        fast=fast,
        model_version=details.get("model_version"),
        created_at_ms=details.get("created_at_ms"),
        updated_at_ms=details.get("updated_at_ms"),
        expires_at_s=enc.get("expiry_s"),
        invalid_before_s=enc.get("invalid_before_s"),
        bypass_expiry=enc.get("bypass_expiry"),
        password_set=stored_password is not None,
        heavy_bytes=parts["heavy"].length,
        heavy_sha256=parts["heavy"].sha256)
    return meta, stored_password


def read_metadata(path: str, secret: str) -> BundleMetadata:
    """The intermediate read: fast metadata + both lightweight detail
    parts (the encrypted one decrypted). No model bytes are touched —
    the heavy part is only hash-verified while streaming past it."""
    if not secret:
        raise ModelBundleError(
            "reading bundle metadata needs the secret")
    fast, parts, salt = _scan(path)
    _check_hashes(fast, parts)
    meta, _stored = _hydrate_metadata(fast, parts,
                                      _derive_keys(secret, salt))
    return meta


def read_full(path: str, secret: str, *, password: str | None = None,
              allow_expired: bool = False,
              now: float | None = None) -> VoiceModel:
    """The full read: every part verified, decrypted and decoded.
    Expiry and password gates apply (a refusal names the rule)."""
    if not secret:
        raise ModelBundleError("reading a bundle needs the secret")
    fast, parts, salt = _scan(path, retain_heavy=True)
    _check_hashes(fast, parts)
    keys = _derive_keys(secret, salt)
    meta, stored_password = _hydrate_metadata(fast, parts, keys)
    _check_usable(meta, stored_password=stored_password,
                  password=password, allow_expired=allow_expired, now=now)
    heavy_raw = _open_part(keys, "heavy", parts["heavy"].data or b"")
    pipeline_version, acoustic, vocoder = _decode_heavy_frame(heavy_raw)
    if not acoustic:
        raise ModelBundleError("heavy payload: acoustic model missing")
    if not vocoder:
        raise ModelBundleError("heavy payload: vocoder model missing")
    return VoiceModel(
        format_version=fast.format_version, token=fast.token,
        name=fast.name, author=fast.author,
        model_version=meta.model_version,
        created_at_ms=meta.created_at_ms,
        updated_at_ms=meta.updated_at_ms,
        expires_at_s=meta.expires_at_s,
        invalid_before_s=meta.invalid_before_s,
        bypass_expiry=meta.bypass_expiry,
        password=stored_password,
        pipeline_version=pipeline_version,
        acoustic_model_bytes=acoustic,
        vocoder_model_bytes=vocoder,
        parts=dict(fast.parts))


def unpack_bundle(path: str, secret: str, out_dir: str, *,
                  password: str | None = None,
                  allow_expired: bool = False) -> dict:
    """Full read + write the two model byte strings out as
    ``<token>-acoustic.bin`` / ``<token>-vocoder.bin``. Returns the
    receipt (paths + sizes + token)."""
    model = read_full(path, secret, password=password,
                      allow_expired=allow_expired)
    slug = model.token.lower().replace("_", "-")
    os.makedirs(out_dir, exist_ok=True)
    acoustic_path = os.path.join(out_dir, f"{slug}-acoustic.bin")
    vocoder_path = os.path.join(out_dir, f"{slug}-vocoder.bin")
    with open(acoustic_path, "wb") as fh:
        fh.write(model.acoustic_model_bytes)
    with open(vocoder_path, "wb") as fh:
        fh.write(model.vocoder_model_bytes)
    return {"token": model.token, "name": model.name,
            "acoustic_path": acoustic_path,
            "acoustic_bytes": len(model.acoustic_model_bytes),
            "vocoder_path": vocoder_path,
            "vocoder_bytes": len(model.vocoder_model_bytes)}


def validate_bundle(path: str, secret: str | None = None) -> dict:
    """Health-style check: never raises for a bad bundle (a missing
    file still raises — that is a caller error, not a bundle fault).
    With a secret, both encrypted parts are also opened, so the MAC
    and the heavy frame are proven; without one, only the header,
    the part hashes and the unencrypted details are proven."""
    result: dict = {"ok": False, "path": os.path.abspath(path)}
    try:
        fast, parts, salt = _scan(path, retain_heavy=bool(secret))
        _check_hashes(fast, parts)
        details = _parse_json_part("details", parts["details"].data or b"")
        result.update({"token": fast.token, "name": fast.name,
                       "author": fast.author,
                       "format_version": fast.format_version,
                       "model_version": details.get("model_version"),
                       "parts": {name: {"bytes": s.length,
                                        "sha256": s.sha256, "ok": True}
                                 for name, s in parts.items()}})
        if secret:
            keys = _derive_keys(secret, salt)
            enc_raw = _open_part(keys, "encrypted_details",
                                 parts["encrypted_details"].data or b"")
            _parse_json_part("encrypted_details", enc_raw)
            heavy_raw = _open_part(keys, "heavy",
                                   parts["heavy"].data or b"")
            pipeline_version, acoustic, vocoder = _decode_heavy_frame(
                heavy_raw)
            result["pipeline_version"] = pipeline_version
            result["acoustic_bytes"] = len(acoustic)
            result["vocoder_bytes"] = len(vocoder)
            if not acoustic:
                raise ModelBundleError(
                    "heavy payload: acoustic model missing")
            if not vocoder:
                raise ModelBundleError(
                    "heavy payload: vocoder model missing")
        result["ok"] = True
    except ModelBundleError as exc:
        result["error"] = str(exc)
    return result


def register_model_bundle(path: str, *, services_dir: str = ""):
    """Register a bundle file as a model service, reusing the
    existing ServiceRegistry — no parallel registry. The slug comes
    from the bundle token (lowercased, ``_`` -> ``-``); the version
    from the bundle's model version when it is a semver core, else
    the file format version. The artifact format is ``dvmodel``; the
    registry's own validation proves the file (magic + header +
    sha256). No secret is needed — registration only reads the
    plaintext header."""
    from .services import ModelArtifact, ServiceDeclaration, ServiceRegistry
    target = os.path.abspath(path)
    fast = read_fast(target)
    details = read_details(target)
    version = details.get("model_version") or fast.format_version
    if not _SEMVER_CORE_RE.match(str(version)):
        version = fast.format_version
    slug = fast.token.lower().replace("_", "-")
    artifact = ModelArtifact(
        name=slug, version=str(version), task="voice-convert",
        format="dvmodel", path=target, sha256=_sha256_file(target),
        entry="bundle",
        source=f"model bundle {fast.token} by {fast.author}")
    decl = ServiceDeclaration(
        slug=slug, version=str(version), task="voice-convert",
        artifact=artifact)
    return ServiceRegistry(services_dir).register(decl)


# -- chat control ------------------------------------------------------

def control_model_bundle(tail: str, *, services_dir: str = "") -> str:
    """/voice model — the chat surface for model bundles.

    list                      registered voice-model bundles
    inspect <path>            fast metadata (no secret needed)
    register <path>           register a bundle file as a service

    Chat never carries a secret, so inspect/register only read the
    plaintext header — the sealed parts stay sealed until the CLI
    opens them with one.
    """
    parts = (tail or "").strip().split(None, 1)
    verb = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""
    if verb in ("list", "ls"):
        from .services import ServiceRegistry
        rows = [d for d in ServiceRegistry(services_dir).list()
                if d.artifact.format == "dvmodel"]
        if not rows:
            return ("no voice-model bundles registered yet — run "
                    "`nm voice model register <file.dvmodel>`")
        lines = ["voice-model bundles:"]
        for d in rows:
            lines.append(f"  {d.slug}@{d.version} "
                         f"{d.artifact.path}")
        return "\n".join(lines)
    if verb == "inspect":
        if not rest:
            return "usage: /voice model inspect <path>"
        try:
            fast = read_fast(rest)
        except (ModelBundleError, FileNotFoundError, OSError) as exc:
            return f"model inspect failed: {exc}"
        return (f"bundle {fast.token} — {fast.name} by {fast.author}\n"
                f"format {fast.format_version} · "
                f"parts: details {fast.parts['details'][:12]}…, "
                f"encrypted details "
                f"{fast.parts['encrypted_details'][:12]}…, heavy "
                f"{fast.parts['heavy'][:12]}…")
    if verb == "register":
        if not rest:
            return "usage: /voice model register <path>"
        try:
            decl = register_model_bundle(rest, services_dir=services_dir)
        except (ModelBundleError, ValueError, FileNotFoundError,
                OSError) as exc:
            return f"model register failed: {exc}"
        return (f"registered {decl.ref} [{decl.task}] "
                f"{decl.artifact.path}")
    return ("usage: /voice model list | inspect <path> | "
            "register <path>")
