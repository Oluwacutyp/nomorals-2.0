// NoMorals native vector similarity — C ABI, callable from Python via ctypes.
//
// This is the hot path of memory recall: every reply does a top-k cosine
// search over the stored vectors, and on a phone (no numpy) that was a
// pure-Python dot product per vector.  This file replaces that loop.
//
// Build (Termux or desktop):
//   c++ -O2 -fPIC -shared -std=c++17 vecsim.cpp -o libvecsim.so
//
// No dependencies beyond the C++ standard library.  The dot product has
// a runtime-dispatched SIMD path (usearch/SimSIMD pattern): on x86-64 an
// AVX2 kernel compiled with __attribute__((target("avx2"))) is chosen once
// per process via __builtin_cpu_supports, with a scalar fallback.  The
// binary is built with portable flags only — NEVER -march=native, which
// would auto-vectorize the scalar fallback and SIGILL on older CPUs.
// On AARCH64 (the phone) the single path auto-vectorizes to NEON at -O2.
// Vectors are pre-normalized, so cosine similarity IS the dot product.
//
// The SIMD and scalar paths agree to ~1e-7 (different reduction order);
// the top-k parity contract with the Python reference is places=4, and
// the index tie-break is unchanged, so this is within contract.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

extern "C" {

const char* nm_vecsim_version(void) {
    return "2.0.0";
}

namespace {

// Portable scalar dot — also the fallback when AVX2 is unavailable.
inline float dot_scalar(const float* a, const float* b, std::int32_t n) {
    float acc = 0.0f;
    for (std::int32_t i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

#if defined(__x86_64__) || defined(_M_X64)
// AVX2 kernel: 8-wide multiply-add, horizontal reduction, scalar tail.
// Compiled for AVX2 regardless of the TU's -m flags; selected at runtime.
__attribute__((target("avx2")))
float dot_avx2(const float* a, const float* b, std::int32_t n) {
    __m256 acc = _mm256_setzero_ps();
    std::int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float total = _mm_cvtss_f32(s);
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

inline bool cpu_has_avx2(void) {
    static const bool v = __builtin_cpu_supports("avx2");
    return v;
}
#endif

}  // namespace

// Which dot kernel is actually executing — the usearch
// `metricImplementation()` idea: operators see the live path, not just
// what was built.
const char* nm_vecsim_simd_level(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return cpu_has_avx2() ? "avx2" : "sse2";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "neon";
#else
    return "scalar";
#endif
}

// Dot product of two float32 vectors (runtime-dispatched).
float nm_vecsim_dot(const float* a, const float* b, std::int32_t n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (n >= 16 && cpu_has_avx2()) {
        return dot_avx2(a, b, n);
    }
#endif
    return dot_scalar(a, b, n);
}

// L2-normalize in place.  A zero vector is left unchanged.
// Returns the original norm.
float nm_vecsim_normalize(float* v, std::int32_t n) {
    float acc = 0.0f;
    for (std::int32_t i = 0; i < n; ++i) {
        acc += v[i] * v[i];
    }
    float norm = std::sqrt(acc);
    if (norm == 0.0f) {
        return 0.0f;
    }
    float inv = 1.0f / norm;
    for (std::int32_t i = 0; i < n; ++i) {
        v[i] *= inv;
    }
    return norm;
}

// Top-k rows of a row-major (rows x dim) matrix by dot product with query.
// Fills out_scores / out_indices in DESCENDING score order.
// Returns the number of pairs written (min(k, rows), or 0 on bad input).
std::int32_t nm_vecsim_topk(const float* matrix, std::int32_t rows, std::int32_t dim,
                            const float* query, std::int32_t k,
                            float* out_scores, std::int32_t* out_indices) {
    if (matrix == nullptr || query == nullptr || out_scores == nullptr ||
        out_indices == nullptr || rows <= 0 || dim <= 0 || k <= 0) {
        return 0;
    }
    if (k > rows) {
        k = rows;
    }
    // Min-heap of (score, row) capped at k — one pass, O(rows * dim * log k).
    // Rows are prefetched ahead of the stream (FAISS-style); the dot itself
    // is the runtime-dispatched SIMD kernel.  Per-row accumulation order is
    // unchanged, so scores match the scalar path to ~1e-7 (inside the
    // places=4 parity contract); the index tie-break below is untouched.
    std::vector<std::pair<float, std::int32_t>> heap;
    heap.reserve(k);
    const std::greater<std::pair<float, std::int32_t>> cmp;
    for (std::int32_t i = 0; i < rows; ++i) {
        if (i + 8 < rows) {
            __builtin_prefetch(matrix + static_cast<std::size_t>(i + 8) * dim,
                               0, 3);
        }
        const float* row = matrix + static_cast<std::size_t>(i) * dim;
        float score = nm_vecsim_dot(row, query, dim);
        if (static_cast<std::int32_t>(heap.size()) < k) {
            heap.emplace_back(score, i);
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else if (score > heap.front().first) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.back() = {score, i};
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    std::sort(heap.begin(), heap.end(),
              [](const std::pair<float, std::int32_t>& a,
                 const std::pair<float, std::int32_t>& b) {
                  if (a.first != b.first) {
                      return a.first > b.first;
                  }
                  return a.second < b.second;  // deterministic tie-break
              });
    for (std::int32_t i = 0; i < k; ++i) {
        out_scores[i] = heap[i].first;
        out_indices[i] = heap[i].second;
    }
    return k;
}

}  // extern "C"
