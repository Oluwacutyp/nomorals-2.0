// NoMorals native vector similarity — C ABI, callable from Python via ctypes.
//
// This is the hot path of memory recall: every reply does a top-k cosine
// search over the stored vectors, and on a phone (no numpy) that was a
// pure-Python dot product per vector.  This file replaces that loop.
//
// Build (Termux or desktop):
//   c++ -O2 -fPIC -shared -std=c++17 vecsim.cpp -o libvecsim.so
//
// No dependencies beyond the C++ standard library.  The inner loops are
// straight accumulation over float32, which auto-vectorizes to NEON on
// the phone's ARM cores and to SSE/AVX on x86-64 at -O2.  Vectors are
// pre-normalized, so cosine similarity IS the dot product.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

extern "C" {

const char* nm_vecsim_version(void) {
    return "1.0.0";
}

// Dot product of two float32 vectors.
float nm_vecsim_dot(const float* a, const float* b, std::int32_t n) {
    float acc = 0.0f;
    for (std::int32_t i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
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
    std::vector<std::pair<float, std::int32_t>> heap;
    heap.reserve(k);
    const std::greater<std::pair<float, std::int32_t>> cmp;
    for (std::int32_t i = 0; i < rows; ++i) {
        const float* row = matrix + static_cast<std::size_t>(i) * dim;
        float score = 0.0f;
        for (std::int32_t d = 0; d < dim; ++d) {
            score += row[d] * query[d];
        }
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
