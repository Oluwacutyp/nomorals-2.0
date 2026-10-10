# NATIVE Sweep — External Mining Report

Module: `nomorals/native/` (`__init__.py`, `biometric.py` + 4 C++ kernels:
`vecsim.cpp`, `mlptrain.cpp`, `bpe.cpp`, `memextract.cpp`).
Date: 2026-10-10. Written BEFORE any implementation, per sweep method.

Every significant class/function was compared against the best implementations
outside this repo. "Best" = mined for gold to merge. "Trash" = mined to learn
what NOT to do. Each section ends with the gaps this sweep will fill.

---

## 1. `vecsim.cpp` — SIMD top-k cosine search (memory recall hot path)

### How the best do it

**usearch / SimSIMD (unum-cloud, used by ClickHouse, 10x faster than FAISS):**
- Runtime dispatch, not compile-time flags. Each kernel carries its own
  `__attribute__((target("avx2")))` and is selected once per process via
  `__builtin_cpu_supports`, so one binary runs AVX2 where available and a
  scalar fallback elsewhere. (https://github.com/brentp/usearch-nim)
- **Critical anti-gold, verified:** do NOT add `-march=native`. On an AVX-512
  host, `-march=native` auto-vectorizes the *scalar* kernel into AVX-512, so
  dispatch still picks "scalar" on an older processor and then faults with
  SIGILL on exactly the machines the fallback path exists to serve.
  (https://github.com/brentp/usearch-nim — SIMD section)
- `metricImplementation()` reports which kernel is live — the operator can see
  what is actually executing, not just what was built.
  (SimSIMD runtime AVX2/AVX-512/NEON dispatch with scalar fallback:
  https://github.com/orhaugh/clink/commit/a8afbb09510cb9fd1ada4b0e7cd4cc6a610ec406)
- Hardware-agnostic quantization (f16/i8) and OpenMP threading for throughput.
  (https://github.com/shinyflvre/mate-engine/blob/HEAD/Assets/LLMUnity/Runtime/RAG/usearch/README.md)

**FAISS / hnswlib (facebookresearch/faiss):** `__builtin_prefetch` ahead of the
row stream in brute-force loops; blocked accumulation over row batches so the
query vector stays in registers while many rows stream past.

**Cache-blocking literature:** blocked matmul keeps the working set in L2 and
reduces cache misses ~5x at scale — the principle applies to our row stream
too: accumulate several rows per query pass instead of re-reading the query.
(https://medium.com/@mahmuuudtolba/cache-aware-matrix-multiplication-why-your-simd-code-is-slower-than-you-think-e3f730e4624d)

### What we have
- Correct C ABI, deterministic tie-break by index, min-heap top-k, ragged-row
  fail-fast on the Python side. Solid.
- The dot loop is a plain scalar accumulation relying on compiler
  auto-vectorization at `-O2` — no runtime dispatch, no prefetch, no row
  blocking, no visibility into which kernel runs.
- `nm_vecsim_normalize` exists in C++ but has NO Python wrapper; callers
  (`storage/vectors.py:normalize`) normalize in pure Python on every vector.

### Trash mined (what NOT to do)
- `-march=native` on the whole binary (SIGILL hazard above — usearch's
  explicit warning).
- OpenMP thread-parallel top-k: our searches are small (hundreds of rows) and
  latency-sensitive; thread spawn cost dominates. FAISS/usearch thread the
  *batch*, not the single query.
- Quantized i8/f16 recall: our vectors are float32 blobs consumed directly by
  Python (`array('f')`); quantization would change scores beyond the parity
  tolerance (tests assert `assertAlmostEqual(places=4)`).

### Gaps → plan
1. Add `nm_vecsim_simd_level()` C ABI: reports the *dispatched* kernel
   ("avx2"/"sse2"/"neon"/"scalar") — usearch's `metricImplementation()` gold.
2. Runtime-dispatched dot on x86-64: AVX2 variant via
   `__attribute__((target("avx2")))`, chosen once per process via
   `__builtin_cpu_supports`; portable `-O2 -fPIC` build flags (never
   `-march=native`). On AARCH64 (the phone) the single path auto-vectorizes to
   NEON — reported as "neon".
3. Row-blocked top-k (4 rows per pass) with `__builtin_prefetch` ahead of the
   stream. **Exactness constraint:** the blocked loop keeps the per-row
   accumulation order over `dim` identical, so per-row scores are bit-identical
   to the scalar loop; the SIMD dot changes rounding by ~1e-7, inside the
   `places=4` parity contract.
4. Python: `normalize()` public wrapper (fills the `nm_vecsim_normalize` gap),
   `info()["simd"]` reporting, `topk_flat(data, dim, query, k)` explicit-dim
   packed API (the private `_topk_packed` infers dim from the query, fragile),
   thread-safe `_load_one` (ctypes lazy init races under threads).

---

## 2. `bpe.cpp` — BPE trainer / encoder kernel

### How the best do it

**Hugging Face `tokenizers` (Rust, the production standard):** the trainer keeps
a priority queue keyed by `(count, pair)` with tie-break by pair identity and
updates pair counts *incrementally*: after each merge only the words containing
the merged pair are re-processed (subtract their old pair contributions, apply
the merge, add back). Heap entries are lazily invalidated instead of rebuilt.
This turns training from O(merges × corpus) into near O(corpus + total changed
symbols). Tie-break rule (Rust `Merge Ord`):
https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/models/bpe/trainer.rs
(described in https://github.com/vspcoderz/aicl/blob/HEAD/research/tokenizer-bpe.md)

**GPT-2 / byte-level BPE (the reference algorithm):** byte-level base
vocabulary so any byte sequence decomposes — our kernel already does this
(codepoint → UTF-8 wire format).

### What we have
- Correct, byte-level, exact parity with the Python reference (list-equal merge
  lists, enforced by `NativeBpeParityTest` incl. a randomized differential
  battery). Already carries a dense-epoch-stamp counting trick (one allocation,
  no per-merge zeroing) and an inverted symbol→words index.
- Still re-scans the **entire corpus per merge** (O(target_merges × words ×
  word_len)). The dense trick speeds counting but not the asymptotic shape —
  exactly the gap HF's incremental trainer closes.

### Trash mined (what NOT to do)
- Switching tie-break to "pair identity" (HF's rule): our Python reference
  uses "lexically LARGER (left, right) string pair" — parity is the contract,
  the heap must replicate OUR rule, not HF's.
- Multi-threaded merge selection: merges are strictly sequential (each merge
  changes the counts the next selection reads). Parallelism here is a
  correctness bug, not a speedup.

### Gaps → plan
1. Incremental pair counting + max-heap with lazy deletion (HF gold),
   comparator = exact replica of Python's `max(counts.items(), key=(count,
   (l, r)))`: higher count wins, tie → lexicographically larger `(l, r)`
   string pair. int64 exact counts, so parity with the reference stays
   **list-equal**, not just approximate.
2. Bump `nm_bpe_version` to "2.0.0". Validate with the existing randomized
   parity battery + new differential tests.

---

## 3. `mlptrain.cpp` — native next-token MLP batch kernel

### How the best do it

**ggml (llama.cpp):** runtime CPU detection (`ggml_cpu_init`), blocked GEMM
kernels tuned for L2, thread-pool batch parallelism, type-specific
micro-kernels.

### What we have
- Exact op-order parity with `NativeTrainer._batch_gradients`, enforced
  element-wise (loss AND every gradient element) — the strictest contract in
  this module.

### Decision: NO math changes — and why (mined, not lazy)
- ggml's blocking/OpenMP target GEMMs 100× larger than ours (vocab ~ hundreds,
  hidden ~ tens, batch ~ dozens). Blocking gains appear when the working set
  exceeds L2; ours fits in L1.
- The element-wise parity contract **forbids** reordering accumulation or
  parallelizing the sequential SGD updates (each sample's update is read by the
  next sample). OpenMP over samples would silently change training dynamics —
  the exact failure the parity tests exist to catch.
- The kernel is already C++ at `-O2`; there is no interpreter overhead left to
  remove. The honest improvement is elsewhere (Python side, batching).

**Trash avoided:** `-fopenmp` on the SGD loop (breaks sequential-update
semantics), `-ffast-math` (breaks the exactness contract — reassociates the
accumulations the tests compare element-wise).

### Gaps → plan
None in the kernel. The Python-side ergonomics (structured build report) come
under §5.

---

## 4. `memextract.cpp` — heuristic memory-extraction kernel

### How the best do it

Production memory systems (Mem0, Zep) extract with LLMs. Ours is deliberately
the *always-on, offline, zero-cost* half of `MemoryExtractor._heuristic_pass`:
five fixed English patterns compiled to a small NFA with Python-`re`-compatible
word semantics, copied verbatim from `nomorals/memory/extract.py`, tested
element-for-element against the Python path.

### Decision: NO logic changes
The patterns belong to `memory/extract.py` (another worker's module — do not
touch). The kernel is an exact port; any "smarter" heuristic here would
diverge from the reference by design. Verified structure is good: bounded
output, `-1` truncation refusal, UTF-8 strictness.

---

## 5. Python layer (`native/__init__.py`)

### Mined gold
- **usearch nim wrapper build notes:** build the C++ adapter with the plain
  local compiler, no pinned flags, graceful scalar fallback for validation.
- **vsearch:** moved back to `-O3` after fixing a strict-aliasing miscompile at
  the root (https://github.com/torognes/vsearch/releases/tag/v2.32.0) — our
  kernels are strict-aliasing-clean (plain float/double/int arrays, no type
  punning), so `-O3` for the compute kernels is safe; keep `-O2` only if a
  miscompile is ever observed.
- **ctypes hygiene (general best practice):** lazy loader singletons need a
  lock — two threads racing `load()` can both `CDLL()` the same path.

### Gaps → plan
1. `build()`: compile the 4 targets in parallel (ThreadPoolExecutor),
   per-target `-O3` for the compute kernels, portable flags only (document the
   no-`-march=native` rule with the usearch SIGILL citation), per-target
   structured results; keep the `(bool, str)` signature (`doctor.py` uses it).
2. Thread-safe `_load_one` via a module lock.
3. `normalize()` public wrapper around `nm_vecsim_normalize`.
4. `topk_flat(data, dim, query, k)` — explicit-dim packed API.
5. `info()` gains `simd` (from `nm_vecsim_simd_level()`); `benchmark()` reports
   it too.

---

## 6. `biometric.py` — Termux fingerprint approval

### How the best do it

**Android CDD §7.3.10 (fingerprint sensor, quoted in the compat docs):**
devices "MUST rate limit attempts for at least 30 seconds after 5 false
trials", matching must happen in the TEE, false-accept rate ≤ 0.002%.
(https://9to5google.com/2015/10/20/google-android-oem-requirements-for-full-disk-encryption-fingerprint-sensors-doze-mode-more/?extended-comments=1)
**BiometricPrompt:** the OS surfaces `ERROR_LOCKOUT` after too many failures
and apps are expected to handle it rather than re-prompting blindly; Android
15's Failed Authentication Lock locks the device after 5 failed attempts in an
app's prompt.
(https://www.androidauthority.com/android-15-failed-authentication-lock-3492044/)

### What we have
- Clean Termux-only design: fail-closed availability probe, never raises,
  debug-only logging, explicit-call-only contract. Good bones.
- Missing: any notion of attempt rate-limiting (a script looping
  `request_biometric` can hammer the dialog indefinitely — the OS will lock out
  the *sensor*, but we give the caller no structured signal), no guard against
  two threads opening two dialogs at once (the docstring *warns* about
  background threads but provides no mechanism), and only a bare bool —
  callers can't distinguish "user denied" from "sensor locked out".

### Gaps → plan
1. Attempt ledger mirroring the CDD policy: 5 consecutive prompt denials
   (within a 10-minute window) → 30s cooldown; `biometric_available()` reports
   `(False, "locked out …")` during cooldown instead of letting the dialog
   spam.
2. In-flight guard: a non-blocking lock — a second concurrent call gets
   `busy`, never a stacked dialog (mechanism for the docstring's warning).
3. Rich result API: `request_biometric_ex()` → `BiometricResult(status,
   reason)` with statuses `approved/denied/unavailable/locked_out/timeout/
   busy`; `request_biometric()` keeps its exact bool contract (used by
   `core/policy.py` and the existing test suite).
4. Denials only count when a dialog was actually shown (timeout/failed auth);
   missing binary / unavailable never counts toward lockout.

---

## Implementation order
1. `vecsim.cpp` v2 (SIMD dispatch + reporting + blocked topk + prefetch),
   rebuild `.so` locally, verify parity battery green.
2. `bpe.cpp` v2 (incremental counts + heap, exact tie-break), rebuild, verify
   randomized parity battery green.
3. `__init__.py` (build/threads/normalize/topk_flat/info/benchmark).
4. `biometric.py` (ledger + cooldown + in-flight guard + rich result).
5. `tests/test_native_sweep.py` — new tests for all changed behavior; run +
   pre-existing `test_native_fallback.py`, `test_biometric.py`,
   `test_audit_r23.py` native classes.
6. Commit + push only `nomorals/native/**`, `tests/test_native_sweep.py`,
   `NATIVE_SWEEP_MINING.md`.
