"""Build + load the native hash library (nmhash.so), with a pure fallback.

The .so is built next to this file on first use (g++ -O2 -shared -fPIC) and
cached; if no compiler is available the loader returns None and the engine
keeps using the pure-Python hash functions. The native layer is verified
against the Python oracle on load — a mismatch fails open to Python.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

__all__ = ["available", "fast_hash_fn", "backend_name"]

_HERE = Path(__file__).parent
_SRC = _HERE / "nmhash.c"
_SONAME = "nmhash" + (".dll" if os.name == "nt" else ".so")
_lib: Any = None
_checked = False


def _candidates() -> list[Path]:
    out = [_HERE / _SONAME]
    cache = Path(os.path.expanduser("~/.nomorals/cache"))
    out.append(cache / _SONAME)
    return out


def _build_out() -> Path:
    """Where a fresh build lands: the runtime cache (outside the repo tree),
    falling back to beside the source when the cache dir is not writable."""
    try:
        cache = Path(os.path.expanduser("~/.nomorals/cache"))
        cache.mkdir(parents=True, exist_ok=True)
        return cache / _SONAME
    except OSError:
        return _HERE / _SONAME


def _load() -> Any:
    """Load a cached .so, or build it now. None when impossible."""
    global _lib
    if _lib is not None:
        return _lib
    for path in _candidates():
        if path.exists():
            try:
                _lib = ctypes.CDLL(str(path))
                return _lib
            except OSError:
                continue
    if not _SRC.exists():
        return None
    compiler = "g++" if os.popen("which g++").read().strip() else "gcc"
    out = _build_out()
    try:
        proc = subprocess.run(
            [compiler, "-O2", "-shared", "-fPIC", "-o", str(out), str(_SRC)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        _lib = ctypes.CDLL(str(out))
        return _lib
    except OSError:
        return None


def _hex(fn_name: str, out_len: int):
    def _call(data: bytes) -> str:
        fn = getattr(_lib, fn_name)
        buf = ctypes.create_string_buffer(out_len)
        fn(ctypes.c_char_p(data), ctypes.c_int(len(data)), buf)
        return buf.value.decode("ascii")
    return _call


def available() -> bool:
    """True when the native layer loaded AND matches the Python oracle."""
    global _checked, _lib
    if _checked:
        return _lib is not None
    _checked = True
    lib = _load()
    if lib is None:
        return False
    try:
        tests = [b"", b"a", b"abc", b"hello world", b"x" * 63, b"y" * 64, b"z" * 65, b"q" * 1000]
        for data in tests:
            if _hex("nm_md5_hex", 33)(data) != hashlib.md5(data).hexdigest():
                _lib = None
                return False
            if _hex("nm_sha1_hex", 41)(data) != hashlib.sha1(data).hexdigest():
                _lib = None
                return False
            if _hex("nm_sha256_hex", 65)(data) != hashlib.sha256(data).hexdigest():
                _lib = None
                return False
            # MD4 vs the repo's pure-Python MD4 (the NTLM primitive)
            from ..hashcrack import md4

            if _hex("nm_md4_hex", 33)(data) != md4(data):
                _lib = None
                return False
        # Pinned cross-check: Windows NTLM of "password" (RFC-1320 MD4 over UTF-16LE)
        if _hex("nm_md4_hex", 33)("password".encode("utf-16-le")) != "8846f7eaee8fb117ad06bdd830b7586c":
            _lib = None
            return False
    except Exception:  # noqa: BLE001
        _lib = None
        return False
    return True


#: Where the C layer actually wins. md5/sha1/sha256/sha512 are already C under
#: Python's hood (hashlib/OpenSSL), and the ctypes trampoline adds ~1us per
#: call — measured ~2x SLOWER than the pure-Python fallback. NTLM is different:
#: OpenSSL 3 removed MD4, so the pure-Python fallback is ~24x slower than C
#: (measured 692k/s vs 28k/s on this machine).
FAST_ALGOS = ("ntlm",)


def fast_hash_fn(algo: str) -> Callable[[str], str] | None:
    """A C-backed ``hash(plaintext) -> hex`` for the cracker, or None."""
    if not available():
        return None
    if algo not in FAST_ALGOS:
        return None
    if algo == "ntlm":
        md4_hex = _hex("nm_md4_hex", 33)
        return lambda p: md4_hex(p.encode("utf-16-le"))
    return None


def backend_name() -> str:
    return "c" if available() else "python"


def build_report() -> dict[str, Any]:
    """What the doctor prints about the native layer."""
    started = time.time()
    ok = available()
    return {
        "native": "c" if ok else "python",
        "verified": ok,
        "fast_algos": list(FAST_ALGOS) if ok else [],
        "seconds": round(time.time() - started, 2),
    }
