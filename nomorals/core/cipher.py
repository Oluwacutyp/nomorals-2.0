"""Real cryptography, hermetic.

No third-party crypto dependency — AES (128/192/256), key derivation
(PBKDF2-HMAC-SHA256), and classic ciphers are implemented here in pure
Python so the capability works everywhere the rest of the system runs
(phones, Termux, offline boxes).

Symmetric encryption
--------------------
* **AES-CTR**  — stream mode; no padding, works on any length, safe to
  append to (each message uses its own random nonce).
* **AES-CBC**  — block mode with PKCS#7 (for compatibility with tools
  that expect padded ciphertext).
* Keys derive from a passphrase via PBKDF2-HMAC-SHA256 (200k iterations
  by default) or from a raw 16/24/32-byte key.

Wire format (self-describing, one line)::

    nmc1:v1:<ctr|cbc>:<kdf-iterations>:<salt-hex>:<nonce-or-iv-hex>:<ciphertext-hex>:<hmac-hex>

kdf-iterations = 0 means a raw key was used.  The trailing HMAC-SHA256
(first 16 bytes, hex) authenticates the ciphertext with a key derived
alongside the AES key — wrong passphrases and tampered blobs are
rejected, not silently returned as garbage.  Blobs without the tag
field are still decryptable (legacy/unauthenticated).

Classic ciphers — caesar, vigenere, atbash, xor — for the low end.

    from nomorals.core.cipher import aes_encrypt, aes_decrypt
    blob = aes_encrypt(b"secret", passphrase="hunter2")
    assert aes_decrypt(blob) == b"secret"

Everything raises :class:`CipherError` on bad input (wrong passphrase,
truncated blob, corrupt MAC) — never returns garbage silently.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
from typing import Any

__all__ = [
    "CipherError",
    "aes_encrypt", "aes_decrypt",
    "aes_encrypt_cbc", "aes_decrypt_cbc",
    "derive_key",
    "caesar", "vigenere", "atbash", "xor_cipher",
    "FORMAT", "DEFAULT_ITERATIONS",
]

DEFAULT_ITERATIONS = 200_000
FORMAT = "nmc1"


class CipherError(ValueError):
    """Raised for malformed blobs, bad keys, or failed authentication."""


# ─────────────────────────────── AES (FIPS 197) ──────────────────────────────

_SBOX = bytes.fromhex(
    "637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0"
    "b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b275"
    "09832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cf"
    "d0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2"
    "cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdb"
    "e0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08"
    "ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9e"
    "e1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16")
_INV_SBOX = bytearray(256)
for _i, _v in enumerate(_SBOX):
    _INV_SBOX[_v] = _i

_RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36)


def _xtime(a: int) -> int:
    a <<= 1
    if a & 0x100:
        a ^= 0x11b
    return a & 0xff


def _gmul(a: int, b: int) -> int:
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        a = _xtime(a)
        b >>= 1
    return p & 0xff


def _expand_key(key: bytes) -> list[list[int]]:
    """FIPS 197 key expansion → list of round keys (each 16 bytes)."""
    if len(key) not in (16, 24, 32):
        raise CipherError(f"AES key must be 16/24/32 bytes, got {len(key)}")
    nk = len(key) // 4
    nr = nk + 6
    w: list[list[int]] = [list(key[4 * i:4 * i + 4]) for i in range(nk)]
    for i in range(nk, 4 * (nr + 1)):
        temp = list(w[i - 1])
        if i % nk == 0:
            temp = temp[1:] + temp[:1]
            temp = [_SBOX[b] for b in temp]
            temp[0] ^= _RCON[i // nk - 1]
        elif nk > 6 and i % nk == 4:
            temp = [_SBOX[b] for b in temp]
        w.append([w[i - nk][j] ^ temp[j] for j in range(4)])
    return [w[4 * r:4 * r + 4] for r in range(nr + 1)]


def _add_round_key(state: list[list[int]], rk: list[list[int]]) -> None:
    for c in range(4):
        for r in range(4):
            state[r][c] ^= rk[c][r]


def _sub_bytes(state: list[list[int]], box: bytes) -> None:
    for c in range(4):
        for r in range(4):
            state[r][c] = box[state[r][c]]


def _shift_rows(state: list[list[int]]) -> None:
    state[1] = state[1][1:] + state[1][:1]
    state[2] = state[2][2:] + state[2][:2]
    state[3] = state[3][3:] + state[3][:3]


def _mix_columns(state: list[list[int]]) -> None:
    for c in range(4):
        a = [state[r][c] for r in range(4)]
        state[0][c] = _gmul(a[0], 2) ^ _gmul(a[1], 3) ^ a[2] ^ a[3]
        state[1][c] = a[0] ^ _gmul(a[1], 2) ^ _gmul(a[2], 3) ^ a[3]
        state[2][c] = a[0] ^ a[1] ^ _gmul(a[2], 2) ^ _gmul(a[3], 3)
        state[3][c] = _gmul(a[0], 3) ^ a[1] ^ a[2] ^ _gmul(a[3], 2)


def _state_from(block: bytes) -> list[list[int]]:
    # AES is column-major: state[r][c] = block[c*4 + r]
    return [[block[c * 4 + r] for c in range(4)] for r in range(4)]


def _to_bytes(state: list[list[int]]) -> bytes:
    out = bytearray(16)
    for c in range(4):
        for r in range(4):
            out[c * 4 + r] = state[r][c]
    return bytes(out)


def _encrypt_block(block: bytes, rkeys: list[list[int]]) -> bytes:
    state = _state_from(block)
    _add_round_key(state, rkeys[0])
    for rnd in range(1, len(rkeys) - 1):
        _sub_bytes(state, _SBOX)
        _shift_rows(state)
        _mix_columns(state)
        _add_round_key(state, rkeys[rnd])
    _sub_bytes(state, _SBOX)
    _shift_rows(state)
    _add_round_key(state, rkeys[-1])
    return _to_bytes(state)


def _inv_shift_rows(state: list[list[int]]) -> None:
    # inverse of left shift r: right shift r per row
    state[1] = state[1][-1:] + state[1][:-1]
    state[2] = state[2][-2:] + state[2][:-2]
    state[3] = state[3][-3:] + state[3][:-3]


def _decrypt_block(block: bytes, rkeys: list[list[int]]) -> bytes:
    state = _state_from(block)
    _add_round_key(state, rkeys[-1])
    for rnd in range(len(rkeys) - 2, 0, -1):
        _inv_shift_rows(state)
        _sub_bytes(state, _INV_SBOX)
        _add_round_key(state, rkeys[rnd])
        for c in range(4):
            a = [state[r][c] for r in range(4)]
            state[0][c] = _gmul(a[0], 14) ^ _gmul(a[1], 11) ^ _gmul(a[2], 13) ^ _gmul(a[3], 9)
            state[1][c] = _gmul(a[0], 9) ^ _gmul(a[1], 14) ^ _gmul(a[2], 11) ^ _gmul(a[3], 13)
            state[2][c] = _gmul(a[0], 13) ^ _gmul(a[1], 9) ^ _gmul(a[2], 14) ^ _gmul(a[3], 11)
            state[3][c] = _gmul(a[0], 11) ^ _gmul(a[1], 13) ^ _gmul(a[2], 9) ^ _gmul(a[3], 14)
    _inv_shift_rows(state)
    _sub_bytes(state, _INV_SBOX)
    _add_round_key(state, rkeys[0])
    return _to_bytes(state)


def _inc_nonce(nonce: bytes) -> bytes:
    # CTR: increment the last 64-bit counter (NIST SP 800-38A)
    b = bytearray(nonce)
    for i in range(len(b) - 1, -1, -1):
        b[i] = (b[i] + 1) & 0xff
        if b[i]:
            break
    return bytes(b)


# ─────────────────────────── key derivation ─────────────────────────────────

def derive_key(passphrase: str, salt: bytes, *, iterations: int,
               key_len: int = 32) -> bytes:
    if iterations < 1:
        raise CipherError("iterations must be >= 1")
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt,
                               int(iterations), key_len)


def _derive_keys(passphrase: str, salt: bytes, *, iterations: int,
                 key: bytes) -> tuple[bytes, bytes]:
    """Return (aes_key, hmac_key).

    Passphrase → PBKDF2 to 48 bytes (32 AES + 16 HMAC).  Raw key → the key
    itself for AES, and HMAC key = SHA-256(key ‖ "nmc1-hmac")[:16]."""
    if passphrase:
        material = derive_key(passphrase, salt, iterations=iterations,
                              key_len=48)
        return material[:32], material[32:]
    if len(key) not in (16, 24, 32):
        raise CipherError("raw key must be 16/24/32 bytes")
    hmac_key = hashlib.sha256(key + b"nmc1-hmac").digest()[:16]
    return key, hmac_key


def _tag(hmac_key: bytes, ciphertext: bytes) -> bytes:
    return hmac.new(hmac_key, ciphertext, hashlib.sha256).digest()[:16]


# ─────────────────────────── symmetric API ──────────────────────────────────

def _format_blob(iterations: int, salt: bytes, iv: bytes, ct: bytes,
                 mode: str = "ctr", tag: bytes = b"") -> str:
    parts = [
        f"{FORMAT}:v1",
        mode,
        str(int(iterations)),
        salt.hex(),
        iv.hex(),
        ct.hex(),
    ]
    if tag:
        parts.append(tag.hex())
    return ":".join(parts)


def _parse_blob(blob: str) -> tuple[str, int, bytes, bytes, bytes, bytes]:
    parts = (blob or "").strip().split(":")
    if len(parts) not in (7, 8) or parts[0] != FORMAT or parts[1] != "v1":
        raise CipherError("not an nmc1 cipher blob")
    mode = parts[2]
    if mode not in ("ctr", "cbc"):
        raise CipherError(f"unknown cipher mode {mode!r}")
    try:
        iterations = int(parts[3])
        salt = bytes.fromhex(parts[4])
        iv = bytes.fromhex(parts[5])
        payload = bytes.fromhex(parts[6])
        tag = bytes.fromhex(parts[7]) if len(parts) == 8 else b""
    except (ValueError, IndexError) as exc:
        raise CipherError(f"malformed cipher blob: {exc}") from exc
    if iterations < 0 or not salt or not iv:
        raise CipherError("malformed cipher blob: bad salt/iv")
    return mode, iterations, salt, iv, payload, tag


def aes_encrypt(
    data: bytes | str,
    *,
    passphrase: str = "",
    key: bytes = b"",
    iterations: int = DEFAULT_ITERATIONS,
    mode: str = "ctr",
) -> str:
    """Encrypt → self-describing ``nmc1`` blob.

    ``passphrase`` (PBKDF2-derived AES-256) or a raw ``key`` (16/24/32
    bytes).  ``mode`` is ``ctr`` (default, any length) or ``cbc``
    (PKCS#7 padded, 16-byte blocks).
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    mode = (mode or "ctr").lower()
    if mode not in ("ctr", "cbc"):
        raise CipherError("mode must be 'ctr' or 'cbc'")
    if passphrase:
        salt = os.urandom(16)
    else:
        salt = b"\x00" * 16
        iterations = 0
    aes_key, hmac_key = _derive_keys(passphrase, salt,
                                     iterations=iterations, key=key)

    rkeys = _expand_key(aes_key)
    if mode == "ctr":
        nonce = os.urandom(16)
        out = bytearray()
        for i in range(0, len(data), 16):
            ks = _ctr_keystream_block(nonce, i // 16, rkeys)
            chunk = data[i:i + 16]
            out.extend(a ^ b for a, b in zip(chunk, ks))
        ct = bytes(out)
        return _format_blob(iterations, salt, bytes(nonce), ct, mode="ctr",
                            tag=_tag(hmac_key, ct))
    # CBC
    iv = os.urandom(16)
    pad = 16 - (len(data) % 16)
    padded = data + bytes([pad]) * pad
    out = bytearray()
    prev = iv
    for i in range(0, len(padded), 16):
        block = bytes(a ^ b for a, b in zip(padded[i:i + 16], prev))
        enc = _encrypt_block(block, rkeys)
        out.extend(enc)
        prev = enc
    ct = bytes(out)
    return _format_blob(iterations, salt, iv, ct, mode="cbc",
                        tag=_tag(hmac_key, ct))


def _ctr_keystream_block(nonce: bytes, index: int,
                         rkeys: list[list[int]]) -> bytes:
    # full 128-bit counter increment for block `index`
    n = int.from_bytes(nonce, "big") + index
    counter = n.to_bytes(16, "big")
    return _encrypt_block(counter, rkeys)


def aes_decrypt(
    blob: str,
    *,
    passphrase: str = "",
    key: bytes = b"",
) -> bytes:
    """Decrypt an ``nmc1`` blob.  Raises CipherError on bad input,
    wrong key, or a failed integrity check."""
    mode, iterations, salt, iv, payload, tag = _parse_blob(blob)
    aes_key, hmac_key = _derive_keys(passphrase, salt,
                                     iterations=iterations, key=key)
    if tag:
        expected = _tag(hmac_key, payload)
        if not hmac.compare_digest(expected, tag):
            raise CipherError("integrity check failed "
                              "(wrong key or tampered blob)")
    if mode == "ctr" and not payload:
        raise CipherError("empty ciphertext")
    if mode == "cbc" and (len(payload) < 16 or len(payload) % 16):
        raise CipherError("ciphertext length invalid")

    rkeys = _expand_key(aes_key)
    if mode == "cbc":
        out = bytearray()
        prev = iv
        for i in range(0, len(payload), 16):
            enc = payload[i:i + 16]
            dec = _decrypt_block(enc, rkeys)
            out.extend(a ^ b for a, b in zip(dec, prev))
            prev = enc
        pad = out[-1]
        if not (1 <= pad <= 16) or out[-pad:] != bytes([pad]) * pad:
            raise CipherError("CBC padding invalid (wrong key?)")
        return bytes(out[:-pad])
    # CTR
    out = bytearray()
    for i in range(0, len(payload), 16):
        ks = _ctr_keystream_block(iv, i // 16, rkeys)
        out.extend(a ^ b for a, b in zip(payload[i:i + 16], ks))
    return bytes(out)


def aes_encrypt_cbc(data: bytes | str, *, passphrase: str = "",
                    key: bytes = b"",
                    iterations: int = DEFAULT_ITERATIONS) -> str:
    return aes_encrypt(data, passphrase=passphrase, key=key,
                       iterations=iterations, mode="cbc")


def aes_decrypt_cbc(blob: str, *, passphrase: str = "",
                    key: bytes = b"") -> bytes:
    return aes_decrypt(blob, passphrase=passphrase, key=key)


# ─────────────────────────── classic ciphers ────────────────────────────────

def caesar(text: str, shift: int, *, decrypt: bool = False) -> str:
    s = -int(shift) if decrypt else int(shift)
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + s) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + s) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def vigenere(text: str, keyword: str, *, decrypt: bool = False) -> str:
    keyword = re.sub(r"[^a-zA-Z]", "", keyword or "").lower()
    if not keyword:
        raise CipherError("vigenere keyword must contain letters")
    s = -1 if decrypt else 1
    ki = 0
    out = []
    for ch in text:
        if ch.isalpha():
            base = 97 if ch.islower() else 65
            shift = ord(keyword[ki % len(keyword)]) - 97
            out.append(chr((ord(ch) - base + s * shift) % 26 + base))
            ki += 1
        else:
            out.append(ch)
    return "".join(out)


def atbash(text: str) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr(122 - (ord(ch) - 97)))
        elif "A" <= ch <= "Z":
            out.append(chr(90 - (ord(ch) - 65)))
        else:
            out.append(ch)
    return "".join(out)


def xor_cipher(data: bytes | str, key: bytes | str) -> bytes:
    if isinstance(data, str):
        data = data.encode("utf-8")
    if isinstance(key, str):
        key = key.encode("utf-8")
    if not key:
        raise CipherError("xor key must not be empty")
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def b64_encode(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return base64.b64encode(data).decode("ascii")


def b64_decode(text: str) -> bytes:
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (ValueError, TypeError) as exc:
        raise CipherError(f"not valid base64: {exc}") from exc


def hmac_hex(key: bytes | str, msg: bytes | str) -> str:
    if isinstance(key, str):
        key = key.encode("utf-8")
    if isinstance(msg, str):
        msg = msg.encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


# ─────────────────────────── HKDF + seal/unseal ────────────────────────────

def hkdf(ikm: bytes, *, salt: bytes = b"", info: bytes = b"",
         length: int = 32, hash_name: str = "sha256") -> bytes:
    """HKDF key derivation (RFC 5869): extract-then-expand.

    The KDF hygiene the codebase was missing: one master secret in,
    any number of *separated* keys out. ``info`` is the domain-separation
    label — ``b"enc"`` vs ``b"mac"`` must never share derived bytes::

        master = derive_key(passphrase, salt)
        enc_key = hkdf(master, info=b"file-enc")
        mac_key = hkdf(master, info=b"file-mac")
    """
    if length <= 0 or length > 255 * hmac.new(b"", b"", hash_name).digest_size:
        raise CipherError("hkdf length out of range")
    if not salt:
        salt = b"\x00" * hmac.new(b"", b"", hash_name).digest_size
    prk = hmac.new(salt, ikm, hash_name).digest()
    okm = b""
    prev = b""
    counter = 1
    while len(okm) < length:
        prev = hmac.new(prk, prev + info + bytes([counter]), hash_name).digest()
        okm += prev
        counter += 1
    return okm[:length]


def seal(data: bytes | str, key: bytes, *, associated: bytes = b"") -> str:
    """High-level authenticated encryption with a raw 32-byte key.

    Derives separate enc/mac subkeys via HKDF (never reuse one key for
    both), binds ``associated`` data (headers, filenames) into the tag,
    and returns a self-describing ``nmc2`` blob. Inverse: :func:`unseal`.
    """
    if len(key) != 32:
        raise CipherError("seal needs a 32-byte key")
    if isinstance(data, str):
        data = data.encode("utf-8")
    enc_key = hkdf(key, info=b"nmc2-enc")
    mac_key = hkdf(key, info=b"nmc2-mac")
    nonce = os.urandom(16)
    rkeys = _expand_key(enc_key)
    out = bytearray()
    for i in range(0, len(data), 16):
        ks = _ctr_keystream_block(nonce, i // 16, rkeys)
        chunk = data[i:i + 16]
        out.extend(a ^ b for a, b in zip(chunk, ks))
    ct = bytes(out)
    tag = hmac.new(mac_key, associated + b"\x00" + nonce + ct,
                   hashlib.sha256).digest()[:16]
    import base64 as _b64

    blob = b"nmc2$" + _b64.urlsafe_b64encode(nonce + tag + ct)
    return blob.decode("ascii")


def unseal(blob: str, key: bytes, *, associated: bytes = b"") -> bytes:
    """Inverse of :func:`seal`. Raises :class:`CipherError` on tampering,
    wrong key, or mismatched associated data."""
    if len(key) != 32:
        raise CipherError("unseal needs a 32-byte key")
    import base64 as _b64

    if not blob.startswith("nmc2$"):
        raise CipherError("not an nmc2 blob")
    try:
        raw = _b64.urlsafe_b64decode(blob[5:].encode("ascii"))
    except Exception as exc:
        raise CipherError(f"malformed nmc2 blob: {exc}") from exc
    if len(raw) < 32:
        raise CipherError("nmc2 blob too short")
    nonce, tag, ct = raw[:16], raw[16:32], raw[32:]
    enc_key = hkdf(key, info=b"nmc2-enc")
    mac_key = hkdf(key, info=b"nmc2-mac")
    expected = hmac.new(mac_key, associated + b"\x00" + nonce + ct,
                        hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(expected, tag):
        raise CipherError("integrity check failed (wrong key, tampered "
                          "blob, or mismatched associated data)")
    rkeys = _expand_key(enc_key)
    out = bytearray()
    for i in range(0, len(ct), 16):
        ks = _ctr_keystream_block(nonce, i // 16, rkeys)
        out.extend(a ^ b for a, b in zip(ct[i:i + 16], ks))
    return bytes(out)
