"""Native (C++) accelerators with a pure-Python fallback.

The main system's hot paths run on a phone with no numpy, so by default
they are pure Python.  This package provides the C++ replacements:

* ``vecsim.cpp``   — C ABI top-k cosine search (memory recall)
* ``mlptrain.cpp`` — C ABI batch forward/backward + SGD step for the
                     native next-token MLP trainer (phone model training)
* ``load()`` / ``load_mlp()`` — load each shared library if built
* ``build()``     — compiles both with the local C++ compiler (Termux: clang)
* ``topk()`` / ``mlp_batch()`` — native when available, pure-Python otherwise

The Python fallback is the reference implementation: every test that runs
the native path also runs this one and compares the results (loss AND
every gradient element), so a broken build or ABI mismatch cannot
silently change what the bot remembers or what it learns.
"""

from __future__ import annotations

import array
import ctypes
import heapq
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from ..core.logging_setup import get_logger

__all__ = [
    "available",
    "benchmark",
    "bpe_available",
    "bpe_encode_word",
    "bpe_lib_path",
    "bpe_pack_stream",
    "bpe_train",
    "bpe_unpack",
    "build",
    "dot",
    "find_compiler",
    "info",
    "lib_path",
    "load",
    "load_bpe",
    "load_mem",
    "load_mlp",
    "mem_available",
    "mem_heuristic",
    "mem_lib_path",
    "mlp_available",
    "mlp_batch",
    "mlp_lib_path",
    "python_dot",
    "python_topk",
    "topk",
]

_log = get_logger(__name__)

_HERE = Path(__file__).resolve().parent
_CPP_SOURCE = _HERE / "vecsim.cpp"
_MLP_SOURCE = _HERE / "mlptrain.cpp"
_BPE_SOURCE = _HERE / "bpe.cpp"
_MEM_SOURCE = _HERE / "memextract.cpp"

#: (source, library file stem) — everything build() compiles.
BUILD_TARGETS = (
    (_CPP_SOURCE, "libvecsim"),
    (_MLP_SOURCE, "libmlptrain"),
    (_BPE_SOURCE, "libbpe"),
    (_MEM_SOURCE, "libmemextract"),
)


def _lib_name(stem: str) -> str:
    return f"{stem}.dylib" if platform.system() == "Darwin" else f"{stem}.so"


def lib_path() -> Path:
    return _HERE / _lib_name("libvecsim")


def mlp_lib_path() -> Path:
    return _HERE / _lib_name("libmlptrain")


def bpe_lib_path() -> Path:
    return _HERE / _lib_name("libbpe")


def mem_lib_path() -> Path:
    return _HERE / _lib_name("libmemextract")


class _NotGiven:  # pragma: no cover - sentinel
    """Distinguishes 'not probed yet' from 'probed and absent'."""

    def __repr__(self) -> str:
        return "not-loaded"


_NOT_GIVEN = _NotGiven()
_LIBS: dict[str, ctypes.CDLL | None | _NotGiven] = {
    "vec": _NOT_GIVEN, "mlp": _NOT_GIVEN, "bpe": _NOT_GIVEN,
    "mem": _NOT_GIVEN,
}


def find_compiler() -> str | None:
    """The first working C++ compiler on PATH (c++, clang++, g++)."""
    for candidate in ("c++", "clang++", "g++", "cc"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def build(force: bool = False) -> tuple[bool, str]:
    """Compile every C++ source next to this package.  Returns (ok, message)."""
    compiler = find_compiler()
    if compiler is None:
        return False, "no C++ compiler found (need c++ / clang++ / g++ on PATH)"
    results: list[str] = []
    all_ok = True
    for source, stem in BUILD_TARGETS:
        target = _HERE / _lib_name(stem)
        if target.exists() and not force:
            results.append(f"already built: {target.name}")
            continue
        if not source.is_file():
            all_ok = False
            results.append(f"source missing: {source.name}")
            continue
        cmd = [compiler, "-O2", "-fPIC", "-shared", "-std=c++17",
               str(source), "-o", str(target)]
        started = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except Exception as exc:  # noqa: BLE001 - a broken toolchain must not crash a chat
            all_ok = False
            results.append(f"compiler failed to run for {source.name}: {exc}")
            continue
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[-400:]
            all_ok = False
            results.append(f"compilation failed for {source.name}:\n{detail}")
            continue
        if not target.exists():
            all_ok = False
            results.append(f"compiler reported success but {target.name} is missing")
            continue
        results.append(f"built {target.name} with {compiler} in {time.time() - started:.1f}s")
    if all_ok:
        # A probe that ran before this build may have cached "absent";
        # the next load() must re-probe the freshly built libraries.
        for slot in _LIBS:
            _LIBS[slot] = _NOT_GIVEN
    return all_ok, "; ".join(results)


def _load_one(slot: str) -> ctypes.CDLL | None | _NotGiven:
    current = _LIBS[slot]
    if current is not _NOT_GIVEN:
        return current
    target = _HERE / _lib_name({"vec": "libvecsim", "mlp": "libmlptrain",
                                "bpe": "libbpe", "mem": "libmemextract"}[slot])
    if not target.exists():
        value: ctypes.CDLL | None | _NotGiven = None
    else:
        try:
            value = ctypes.CDLL(str(target))
        except OSError as exc:
            _log.warning("could not load %s: %s — using pure-Python fallback",
                         target.name, exc)
            value = None
    _LIBS[slot] = value
    return value


def load() -> ctypes.CDLL | None:
    """Load the vector-search library once.  None when not built (fallback)."""
    value = _load_one("vec")
    if value is None or value is _NOT_GIVEN:
        return None
    lib = value
    lib.nm_vecsim_version.restype = ctypes.c_char_p
    lib.nm_vecsim_dot.restype = ctypes.c_float
    lib.nm_vecsim_dot.argtypes = [ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_float),
                                  ctypes.c_int32]
    lib.nm_vecsim_normalize.restype = ctypes.c_float
    lib.nm_vecsim_normalize.argtypes = [ctypes.POINTER(ctypes.c_float),
                                        ctypes.c_int32]
    lib.nm_vecsim_topk.restype = ctypes.c_int32
    lib.nm_vecsim_topk.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
                                   ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
                                   ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
                                   ctypes.POINTER(ctypes.c_int32)]
    return lib


def load_mlp() -> ctypes.CDLL | None:
    """Load the MLP training kernel once.  None when not built (fallback)."""
    value = _load_one("mlp")
    if value is None or value is _NOT_GIVEN:
        return None
    lib = value
    lib.nm_mlp_version.restype = ctypes.c_char_p
    lib.nm_mlp_batch.restype = ctypes.c_double
    lib.nm_mlp_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),
    ]
    return lib


def load_bpe() -> ctypes.CDLL | None:
    """Load the BPE kernel once.  None when not built (fallback)."""
    value = _load_one("bpe")
    if value is None or value is _NOT_GIVEN:
        return None
    lib = value
    lib.nm_bpe_version.restype = ctypes.c_char_p
    lib.nm_bpe_train.restype = ctypes.c_int32
    lib.nm_bpe_train.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
    ]
    lib.nm_bpe_encode.restype = ctypes.c_int32
    lib.nm_bpe_encode.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    return lib


def available() -> bool:
    return load() is not None


def mlp_available() -> bool:
    return load_mlp() is not None


def bpe_available() -> bool:
    return load_bpe() is not None


# ── BPE wire format ──────────────────────────────────────────────────────────
# One "blob" = [n_words:int32, packed UTF-8 bytes (4 per int32, LE, padded)].
# A merge stream = per merge: blob(left), blob(right).  An encode result =
# the final symbols as concatenated blobs.


def bpe_pack_blob(s: str) -> array.array:
    data = s.encode("utf-8")
    words = (len(data) + 3) // 4
    out = array.array("i", [0] * (1 + words))
    out[0] = words
    for i, byte in enumerate(data):
        out[1 + i // 4] |= byte << (8 * (i % 4))
    return out


def bpe_pack_stream(merges: Sequence[tuple[str, str]]) -> array.array:
    out = array.array("i")
    for left, right in merges:
        out.extend(bpe_pack_blob(left))
        out.extend(bpe_pack_blob(right))
    return out


def bpe_unpack(stream: array.array, n: int) -> list[str]:
    """Read blobs from ``stream[:n]`` until the stream is exhausted."""
    out: list[str] = []
    pos = 0
    while pos < n:
        words = stream[pos]
        pos += 1
        buf = b"".join(
            (stream[pos + i] & 0xFFFFFFFF).to_bytes(4, "little")
            for i in range(words)
        )
        out.append(buf.rstrip(b"\0").decode("utf-8"))
        pos += words
    return out


def _unpack_blobs(buf: array.array, pos: int, count: int,
                  capacity: int) -> list[str] | None:
    """Read exactly ``count`` blobs starting at ``pos``.

    Returns None when the buffer ends early (truncated kernel output) —
    the caller must then use the pure-Python reference, never a partial
    result.
    """
    out: list[str] = []
    for _ in range(count):
        if pos >= capacity:
            return None
        words = buf[pos]
        pos += 1
        if words < 0 or pos + words > capacity:
            return None
        raw = b"".join(
            (buf[pos + i] & 0xFFFFFFFF).to_bytes(4, "little")
            for i in range(words)
        )
        try:
            out.append(raw.rstrip(b"\0").decode("utf-8"))
        except UnicodeDecodeError:
            return None
        pos += words
    return out


def bpe_train(words: Sequence[str], freqs: Sequence[int], *,
              target_merges: int, min_frequency: int) -> list[tuple[str, str]] | None:
    """C++ BPE training.  Returns the merge list, or None when the kernel
    is absent OR produced a truncated result (in which case the caller
    must use the pure-Python reference — never a partial merge list)."""
    lib = load_bpe()
    if lib is None:
        return None
    n = len(words)
    codes = array.array("i")
    lens = array.array("i", (len(w) for w in words))
    for w in words:
        codes.extend(ord(c) for c in w)
    fbuf = array.array("i", (int(f) for f in freqs))
    # Capacity: each merged symbol is at most the longest word's UTF-8
    # length; 4 bytes/word-int32 plus the two length cells.
    max_bytes = max((len(w.encode("utf-8")) for w in words), default=1)
    capacity = max(16, target_merges * (2 * ((max_bytes + 3) // 4) + 2) + 8)
    out = array.array("i", [0] * capacity)
    produced = lib.nm_bpe_train(
        (ctypes.c_int32 * len(codes)).from_buffer(codes),
        (ctypes.c_int32 * n).from_buffer(lens),
        (ctypes.c_int32 * n).from_buffer(fbuf),
        n, target_merges, min_frequency,
        (ctypes.c_int32 * capacity).from_buffer(out), capacity,
    )
    if produced <= 0:
        return [] if produced == 0 else None
    # nm_bpe_train returns the MERGE count, not cells written — unpack
    # exactly produced*2 blobs. (Unpacking the whole capacity used to
    # decode the trailing zero padding as empty blobs, so the length
    # check below failed on every run and the native path never
    # delivered.)
    blobs = _unpack_blobs(out, 0, produced * 2, capacity)
    if blobs is None or len(blobs) != produced * 2:
        return None  # truncated — refuse; caller falls back to Python
    return [(blobs[2 * i], blobs[2 * i + 1]) for i in range(produced)]


def bpe_encode_word(word: str, stream: array.array) -> list[str] | None:
    """C++ single-word encode.  Returns the final symbols, or None when the
    kernel is absent (caller uses the Python path)."""
    lib = load_bpe()
    if lib is None:
        return None
    codes = array.array("i", (ord(c) for c in word))
    n = len(codes)
    capacity = max(8, n * 4 + 8)
    out = array.array("i", [0] * capacity)
    written = lib.nm_bpe_encode(
        (ctypes.c_int32 * n).from_buffer(codes), n,
        (ctypes.c_int32 * len(stream)).from_buffer(stream), len(stream),
        (ctypes.c_int32 * capacity).from_buffer(out),
    )
    if written <= 0:
        return []
    if written > capacity:  # pragma: no cover - capacity is a hard upper bound
        return None
    return bpe_unpack(out, written)


# ── memory-extraction kernel ─────────────────────────────────────────────────
# Wire format (little-endian int32 cells + raw UTF-8 bytes):
#   [count] then per memory:
#     [kind_idx, importance_pct, content_len, content bytes]
#     [tag_count] then per tag: [tag_len, tag bytes]
# kind_idx: 0 preference, 1 decision, 2 relationship, 3 fact.

_KIND_BY_IDX = ("preference", "decision", "relationship", "fact")


def load_mem() -> ctypes.CDLL | None:
    """Load the memory-extraction kernel once.  None when not built."""
    value = _load_one("mem")
    if value is None or value is _NOT_GIVEN:
        return None
    lib = value
    lib.nm_memextract_version.restype = ctypes.c_char_p
    lib.nm_memextract.restype = ctypes.c_int32
    lib.nm_memextract.argtypes = [ctypes.POINTER(ctypes.c_uint8),
                                  ctypes.c_int32,
                                  ctypes.POINTER(ctypes.c_uint8),
                                  ctypes.c_int32]
    return lib


def mem_available() -> bool:
    return load_mem() is not None


def mem_heuristic(text: str) -> list[tuple[str, float, str, list[str]]] | None:
    """C++ run of the heuristic extraction pass.

    Returns ``[(kind, importance, content, tags), ...]`` — exactly what the
    Python reference produces — or ``None`` when the kernel is absent or
    refused the buffer (truncation), in which case the caller MUST use the
    pure-Python reference.
    """
    lib = load_mem()
    if lib is None:
        return None
    raw = text.encode("utf-8")
    # worst case: 4 memories x (240 content chars x 4 bytes + 12 tags x ~12)
    capacity = 4 * (240 * 4 + 12 * 16 + 32) + 16
    buf = (ctypes.c_uint8 * capacity)()
    src = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
    written = lib.nm_memextract(src, len(raw), buf, capacity)
    if written <= 0:
        return None  # -1 truncation / -2 bad utf-8 → reference path

    data = bytes(buf[:written])

    def r32(off: int) -> int:
        v = int.from_bytes(data[off:off + 4], "little", signed=True)
        return v

    def rbytes(off: int, n: int) -> bytes:
        return data[off:off + n]

    count = r32(0)
    if count < 0 or count > 4:
        return None
    out: list[tuple[str, float, str, list[str]]] = []
    off = 4
    for _ in range(count):
        if off + 12 > written:
            return None
        kind = r32(off)
        imp = r32(off + 4)
        clen = r32(off + 8)
        off += 12
        if kind not in (0, 1, 2, 3) or imp < 0 or imp > 100 \
                or clen < 0 or off + clen > written:
            return None
        content = rbytes(off, clen).decode("utf-8")
        off += clen
        if off + 4 > written:
            return None
        tcount = r32(off)
        off += 4
        if tcount < 0:
            return None
        tags: list[str] = []
        for _t in range(tcount):
            if off + 4 > written:
                return None
            tlen = r32(off)
            off += 4
            if tlen < 0 or off + tlen > written:
                return None
            tags.append(rbytes(off, tlen).decode("utf-8"))
            off += tlen
        out.append((_KIND_BY_IDX[kind], imp / 100.0, content, tags))
    if off != written:
        return None
    return out


def _to_f32(values: Sequence[float] | bytes) -> array.array:
    if isinstance(values, (bytes, bytearray, memoryview)):
        data = array.array("f")
        data.frombytes(bytes(values)[: len(values) // 4 * 4])
        return data
    return array.array("f", (float(v) for v in values))


# ── pure-Python reference (also the fallback) ────────────────────────────────

def python_dot(a: Sequence[float], b: Sequence[float]) -> float:
    acc = 0.0
    for x, y in zip(a, b):
        acc += x * y
    return acc


def python_topk(matrix: Sequence[Sequence[float]], query: Sequence[float],
                k: int) -> list[tuple[float, int]]:
    """Top-k rows of ``matrix`` by dot with ``query``, descending.

    The reference implementation the native path is checked against.
    """
    rows = len(matrix)
    if rows == 0 or k <= 0:
        return []
    k = min(k, rows)
    scored = [(-python_dot(row, query), i) for i, row in enumerate(matrix)]
    heapq.heapify(scored)  # min-heap over -score → pop gives highest first
    out: list[tuple[float, int]] = []
    for _ in range(k):
        neg, i = heapq.heappop(scored)
        out.append((-neg, i))
    return out


# ── public native-or-fallback API ────────────────────────────────────────────

def _fptr(buf: array.array) -> Any:
    """Zero-copy float32 buffer → c_float* for ctypes."""
    return (ctypes.c_float * len(buf)).from_buffer(buf)


def dot(a: Sequence[float] | bytes, b: Sequence[float] | bytes) -> float:
    lib = load()
    buf_a, buf_b = _to_f32(a), _to_f32(b)
    n = min(len(buf_a), len(buf_b))
    if n == 0:
        return 0.0
    if lib is not None:
        return float(lib.nm_vecsim_dot(_fptr(buf_a), _fptr(buf_b), n))
    return python_dot(buf_a[:n], buf_b[:n])


def topk(matrix: Sequence[Sequence[float]] | bytes, query: Sequence[float] | bytes,
         k: int) -> list[tuple[float, int]]:
    """Top-k rows by (pre-normalized) dot product, descending.

    Native C++ when the library is built, pure Python otherwise.  When
    ``matrix`` is raw float32 bytes, pass its row count implicitly (rows =
    len(matrix)//(dim*4)); list input is always fine for both backends.
    """
    if isinstance(matrix, (bytes, bytearray, memoryview)):
        return _topk_packed(bytes(matrix), query, k)
    if isinstance(matrix, array.ArrayType):  # flat float32, dim == len(query)
        return _topk_packed(matrix.tobytes(), query, k)
    lib = load()
    if lib is not None and matrix:
        dim = len(matrix[0])
        rows = len(matrix)
        # The C++ kernel reads exactly `dim` floats from every row and
        # from the query — ragged rows or a short query would over-read
        # the heap.  Fail fast instead of returning garbage (or worse).
        for i, row in enumerate(matrix):
            if len(row) != dim:
                raise ValueError(
                    f"topk: row {i} has dim {len(row)}, expected {dim} "
                    "(ragged matrix)")
        q = _to_f32(query)
        if len(q) < dim:
            raise ValueError(
                f"topk: query dim {len(q)} < matrix dim {dim}")
        q = q[:dim]
        flat = array.array("f")
        for row in matrix:
            flat.extend((float(v) for v in row))
        scores = (ctypes.c_float * k)()
        indices = (ctypes.c_int32 * k)()
        count = lib.nm_vecsim_topk(_fptr(flat), rows, dim, _fptr(q), k,
                                   scores, indices)
        return [(float(scores[i]), int(indices[i])) for i in range(count)]
    q = list(_to_f32(query))
    return python_topk([[float(v) for v in row] for row in matrix], q, k)


def _topk_packed(flat: bytes, query: Sequence[float] | bytes,
                 k: int) -> list[tuple[float, int]]:
    """Top-k over a raw float32 buffer when dim is not known: impossible to
    infer rows without dim, so this is only used when dim == len(query)."""
    q = _to_f32(query)
    dim = len(q)
    if dim == 0:
        return []
    rows = len(flat) // (dim * 4)
    lib = load()
    if lib is not None and rows:
        buf = array.array("f")
        buf.frombytes(flat[: rows * dim * 4])
        kk = min(k, rows)
        scores = (ctypes.c_float * kk)()
        indices = (ctypes.c_int32 * kk)()
        count = lib.nm_vecsim_topk(_fptr(buf), rows, dim, _fptr(q), kk,
                                   scores, indices)
        return [(float(scores[i]), int(indices[i])) for i in range(count)]
    rows_list = [list(buf2) for buf2 in _chunk(flat, dim)]
    return python_topk(rows_list, list(q), k)


def _chunk(data: bytes, dim: int):
    arr = array.array("f")
    arr.frombytes(data[: len(data) // 4 * 4])
    for start in range(0, len(arr), dim):
        yield arr[start:start + dim]


def _kernel_block(lib, target) -> dict[str, Any]:
    block = {
        "library": str(target),
        "built": target.exists(),
        "loaded": lib is not None,
        "backend": "native-cpp" if lib is not None else "pure-python",
    }
    if lib is not None:
        for attr in ("nm_vecsim_version", "nm_mlp_version", "nm_bpe_version",
                     "nm_memextract_version"):
            fn = getattr(lib, attr, None)
            if fn is not None:
                try:
                    block["version"] = fn().decode("utf-8", "replace")
                    break
                except Exception:  # noqa: BLE001
                    pass
    return block


def info() -> dict[str, Any]:
    """Doctor-grade state: libraries, compiler, backends actually in use."""
    lib = load()
    mlp = load_mlp()
    bpe = load_bpe()
    mem = load_mem()
    info: dict[str, Any] = {
        "library": str(lib_path()),
        "built": lib_path().exists(),
        "loaded": lib is not None,
        "compiler": find_compiler(),
        "backend": "native-cpp" if lib is not None else "pure-python",
        "mlp": _kernel_block(mlp, mlp_lib_path()),
        "bpe": _kernel_block(bpe, bpe_lib_path()),
        "mem": _kernel_block(mem, mem_lib_path()),
    }
    if lib is not None:
        try:
            info["version"] = lib.nm_vecsim_version().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
    return info


def _dptr(buf: array.array) -> Any:
    """Zero-copy float64 buffer → c_double* for ctypes."""
    return (ctypes.c_double * len(buf)).from_buffer(buf)


def mlp_batch(
    seqs: array.array,
    lens: array.array,
    embedding: array.array,
    hidden_w: array.array,
    hidden_b: array.array,
    out_w: array.array,
    out_b: array.array,
    *,
    context: int,
    lr: float = 0.0,
    l2: float = 0.0,
) -> tuple[float, int, array.array]:
    """Run one training (lr > 0) or evaluation (lr == 0) batch in C++.

    Weights are ``array.array('d')`` flat buffers, row-major, updated
    IN PLACE when lr != 0 (SGD step + weight decay, l2 already includes
    lr — exactly what the Python ``_axpy`` receives).  Returns
    ``(mean_loss, count, grad_scratch)``; ``grad_scratch`` holds the
    batch-mean gradients (embed|hidden_w|hidden_b|out_w|out_b) so tests
    can compare every element against the pure-Python reference.
    """
    lib = load_mlp()
    if lib is None:
        raise RuntimeError("mlptrain library is not built — run: nm native --build")
    hidden = len(hidden_b)
    vocab = len(out_b)
    n = len(lens)
    scratch_size = vocab * hidden + hidden * hidden + hidden + hidden * vocab + vocab
    scratch = array.array("d", bytes(8 * scratch_size))
    count_box = array.array("i", [0])
    # Pointer views into one scratch buffer at each gradient's offset —
    # no copies, so the gradients written by C++ land in `scratch`.
    base = _dptr(scratch)

    def ptr_at(off: int) -> Any:
        return ctypes.cast(ctypes.addressof(base) + 8 * off,
                           ctypes.POINTER(ctypes.c_double))
    off_embed = 0
    off_hw = off_embed + vocab * hidden
    off_hb = off_hw + hidden * hidden
    off_ow = off_hb + hidden
    off_ob = off_ow + hidden * vocab
    mean_loss = lib.nm_mlp_batch(
        (ctypes.c_int32 * len(seqs)).from_buffer(seqs),
        (ctypes.c_int32 * n).from_buffer(lens),
        n,
        _dptr(embedding), _dptr(hidden_w), _dptr(hidden_b),
        _dptr(out_w), _dptr(out_b),
        vocab, hidden, int(context),
        lr, l2,
        ptr_at(off_embed), ptr_at(off_hw), ptr_at(off_hb),
        ptr_at(off_ow), ptr_at(off_ob),
        (ctypes.c_int32 * 1).from_buffer(count_box),
    )
    return float(mean_loss), int(count_box[0]), scratch


def benchmark(n: int = 2000, dim: int = 128) -> dict[str, Any]:
    """Native vs pure-Python top-10, each measured in the FORM THE STORE
    ACTUALLY FEEDS IT: pure Python over lists of floats, native over the
    pre-packed float32 buffer (VectorStore packs once at load, so the
    native path has no per-search flattening).  Deterministic seed so the
    numbers are comparable across runs."""
    import random as _random

    rng = _random.Random(1234)
    matrix = [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(n)]
    query = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    flat = array.array("f")
    for row in matrix:
        flat.extend(row)

    # warm both paths, then time
    python_topk(matrix, query, 10)
    topk(flat, query, 10)

    t0 = time.perf_counter()
    for _ in range(5):
        py_hits = python_topk(matrix, query, 10)
    py_ms = (time.perf_counter() - t0) * 1000.0 / 5

    t0 = time.perf_counter()
    for _ in range(5):
        nat_hits = topk(flat, query, 10)
    nat_ms = (time.perf_counter() - t0) * 1000.0 / 5

    return {
        "vectors": n,
        "dim": dim,
        "python_ms": round(py_ms, 3),
        "native_ms": round(nat_ms, 3),
        "speedup": round(py_ms / nat_ms, 1) if nat_ms > 0 else 0,
        "backend": "native-cpp" if available() else "pure-python",
        "native": [h[1] for h in nat_hits],
        "python": [h[1] for h in py_hits],
        "match": [h[1] for h in nat_hits] == [h[1] for h in py_hits],
    }
