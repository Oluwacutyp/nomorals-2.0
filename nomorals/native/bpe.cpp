// bpe.cpp — C ABI kernel for the byte-level BPE tokenizer.
//
// The Python tokenizer (nomorals/training/tokenize.py) is deterministic
// but slow: every merge rescans every word of the corpus in pure Python.
// This kernel runs the identical algorithm — same counting, same
// tie-breaks (highest count, then the LEXICALLY LARGER (left, right)
// string pair), same single left-to-right merge pass per word — and the
// encode side applies merges in rank order with the first minimum-rank
// index winning.  Python remains the reference: tests compare merge
// lists and encodings exactly.
//
// Build: c++ -O2 -fPIC -shared -std=c++17 bpe.cpp -o libbpe.so

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using SymTable = std::vector<std::string>;

// utf8-decode a Python codepoint array into std::strings
std::vector<std::string> decode_words(const int32_t* words, const int32_t* lens,
                                      int32_t nwords) {
    std::vector<std::string> out;
    out.reserve(nwords);
    const int32_t* cursor = words;
    for (int32_t w = 0; w < nwords; w++) {
        std::string s;
        for (int32_t i = 0; i < lens[w]; i++, cursor++) {
            uint32_t cp = static_cast<uint32_t>(*cursor);
            // encode codepoint as UTF-8
            if (cp < 0x80) {
                s += static_cast<char>(cp);
            } else if (cp < 0x800) {
                s += static_cast<char>(0xC0 | (cp >> 6));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                s += static_cast<char>(0xE0 | (cp >> 12));
                s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                s += static_cast<char>(0xF0 | (cp >> 18));
                s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            }
        }
        out.push_back(std::move(s));
    }
    return out;
}

} // namespace

extern "C" {

const char* nm_bpe_version() { return "1.0.0"; }

// Train BPE merges.
//
//   words/lens/freqs — nwords pre-tokenized words (packed codepoints)
//   target_merges    — stop after this many merges
//   min_frequency    — stop when the best pair count is below this
//   merges_out       — capacity max_merges * 2 int32: per merge the UTF-8
//                      byte length, bytes, of the LEFT symbol, then the
//                      RIGHT symbol (u16-ish: we use int32 lengths)
//   max_merges       — capacity
//
// Returns the number of merges produced.
//
// Wire per merge (5 int32 + variable): [len_l, bytes_l..., len_r, bytes_r...]
// — written into merges_out as a flat int32 stream (UTF-8 bytes packed
// four per int32, little-endian, final group zero-padded).
int32_t nm_bpe_train(const int32_t* words, const int32_t* lens, const int32_t* freqs,
                     int32_t nwords, int32_t target_merges, int32_t min_frequency,
                     int32_t* merges_out, int32_t max_merges_capacity) {
    auto word_strs = decode_words(words, lens, nwords);

    // Symbol table: id 0..n-1 → string.  Characters interned first.
    SymTable table;
    table.reserve(4096);
    std::unordered_map<std::string, int32_t> char_to_id;

    std::vector<std::vector<int32_t>> splits(nwords);
    for (int32_t w = 0; w < nwords; w++) {
        const std::string& s = word_strs[w];
        std::vector<int32_t> syms;
        syms.reserve(s.size());
        for (size_t i = 0; i < s.size(); i++) {
            std::string ch(1, s[i]);
            auto it = char_to_id.find(ch);
            int32_t id;
            if (it == char_to_id.end()) {
                id = static_cast<int32_t>(table.size());
                table.push_back(ch);
                char_to_id[ch] = id;
            } else {
                id = it->second;
            }
            syms.push_back(id);
        }
        splits[w] = std::move(syms);
    }

    // Inverted index symbol → words containing it (kept in sync with the
    // table: one slot per symbol, appended whenever the table grows).
    std::vector<std::vector<int32_t>> word_sets;
    {
        std::unordered_set<int32_t> distinct;
        for (auto& syms : splits)
            for (int32_t s : syms) distinct.insert(s);
        word_sets.resize(table.size());
        for (int32_t w = 0; w < nwords; w++) {
            std::unordered_set<int32_t> seen_syms;
            for (int32_t s : splits[w]) {
                if (seen_syms.insert(s).second) word_sets[s].push_back(w);
            }
        }
    }

    struct Merge { int32_t l, r, merged; };
    std::vector<Merge> merges;
    merges.reserve(target_merges > 0 ? target_merges : 1);

    int64_t n0_base = static_cast<int64_t>(word_sets.size());

    // Pair counting.  CPython's dict-of-tuples is the reference speed; to
    // actually beat it we index a DENSE vector by l*N0 + r with N0 a FIXED
    // upper bound on the symbol count (distinct initial characters + all
    // planned merges), so keys stay valid for the whole run and the array
    // is allocated ONCE.  An epoch stamp replaces per-merge zeroing.
    // Above the memory limit we fall back to a hash map (still exact).
    {
        std::unordered_set<uint8_t> byte_set;
        for (const auto& s : word_strs)
            for (unsigned char c : s) byte_set.insert(c);
        n0_base = static_cast<int64_t>(byte_set.size());
    }
    int64_t n0 = n0_base + target_merges + 2;
    const int64_t DENSE_LIMIT = 20 * 1024 * 1024;  // cells (160 MB at int64)
    bool use_dense = n0 * n0 <= DENSE_LIMIT;
    std::vector<int64_t> dense;
    std::vector<int32_t> seen;
    std::vector<int32_t> touched;
    std::unordered_map<uint64_t, int64_t> sparse;
    int32_t epoch = 0;
    if (use_dense) {
        dense.assign(static_cast<size_t>(n0) * static_cast<size_t>(n0), 0);
        seen.assign(static_cast<size_t>(n0) * static_cast<size_t>(n0), 0);
    }

    int32_t produced = 0;
    while (static_cast<int32_t>(merges.size()) < target_merges) {
        touched.clear();
        int64_t best_count = -1;
        int32_t best_l = -1, best_r = -1;
        if (use_dense) {
            epoch += 1;
            for (int32_t w = 0; w < nwords; w++) {
                const auto& syms = splits[w];
                int64_t f = freqs[w];
                for (size_t i = 0; i + 1 < syms.size(); i++) {
                    int64_t cell = static_cast<int64_t>(syms[i]) * n0 + syms[i + 1];
                    if (seen[cell] != epoch) {
                        seen[cell] = epoch;
                        dense[cell] = 0;
                        touched.push_back(static_cast<int32_t>(cell));
                    }
                    dense[cell] += f;
                }
            }
        } else {
            sparse.clear();
            for (int32_t w = 0; w < nwords; w++) {
                const auto& syms = splits[w];
                int64_t f = freqs[w];
                for (size_t i = 0; i + 1 < syms.size(); i++) {
                    uint64_t k = (static_cast<uint64_t>(static_cast<uint32_t>(syms[i])) << 32) |
                                 static_cast<uint32_t>(syms[i + 1]);
                    sparse[k] += f;
                }
            }
        }

        // Highest count wins; tie → the lexicographically LARGER
        // (left, right) string pair (python: max by (count, (l, r))).
        auto consider = [&](int32_t l, int32_t r, int64_t count) {
            if (count > best_count) {
                best_count = count;
                best_l = l;
                best_r = r;
            } else if (count == best_count) {
                const std::string& cl = table[l];
                const std::string& cr = table[r];
                const std::string& sl = table[best_l];
                const std::string& sr = table[best_r];
                if (cl > sl || (cl == sl && cr > sr)) {
                    best_l = l;
                    best_r = r;
                }
            }
        };
        if (use_dense) {
            for (int32_t cell : touched) {
                consider(static_cast<int32_t>(cell / n0),
                         static_cast<int32_t>(cell % n0), dense[cell]);
            }
        } else {
            for (const auto& kv : sparse) {
                consider(static_cast<int32_t>(kv.first >> 32),
                         static_cast<int32_t>(kv.first & 0xFFFFFFFF), kv.second);
            }
        }
        if (best_l < 0 || best_count < min_frequency) break;

        int32_t merged_id = static_cast<int32_t>(table.size());
        table.push_back(table[best_l] + table[best_r]);
        word_sets.emplace_back();
        merges.push_back(Merge{best_l, best_r, merged_id});
        produced += 1;

        // Only words containing the LEFT symbol are touched — and within
        // them, only the actual (l, r) adjacencies are merged.  Python
        // scans + rewrites far more on every merge.
        for (int32_t w : word_sets[best_l]) {
            const auto& syms = splits[w];
            std::vector<int32_t> out;
            out.reserve(syms.size());
            bool merged_here = false;
            size_t i = 0;
            while (i < syms.size()) {
                if (i + 1 < syms.size() && syms[i] == best_l && syms[i + 1] == best_r) {
                    out.push_back(merged_id);
                    i += 2;
                    merged_here = true;
                } else {
                    out.push_back(syms[i]);
                    i += 1;
                }
            }
            splits[w] = std::move(out);
            if (merged_here) word_sets[merged_id].push_back(w);
        }
        word_sets[best_l].clear();
        word_sets[best_r].clear();
    }

    // Serialize: per merge [len_l, packed bytes_l, len_r, packed bytes_r].
    int32_t pos = 0;
    const int32_t cap = max_merges_capacity;
    auto pack = [&](const std::string& s) -> std::vector<int32_t> {
        int32_t n = static_cast<int32_t>(s.size());
        int32_t words_needed = (n + 3) / 4;
        std::vector<int32_t> out(words_needed, 0);
        for (int32_t i = 0; i < n; i++) {
            out[i / 4] |= static_cast<uint32_t>(static_cast<uint8_t>(s[i])) << (8 * (i % 4));
        }
        return out;
    };
    for (const auto& m : merges) {
        std::vector<int32_t> bl = pack(table[m.l]);
        std::vector<int32_t> br = pack(table[m.r]);
        int32_t need = 1 + static_cast<int32_t>(bl.size()) + 1 + static_cast<int32_t>(br.size());
        if (pos + need > cap) break;
        merges_out[pos++] = static_cast<int32_t>(bl.size());
        for (int32_t v : bl) merges_out[pos++] = v;
        merges_out[pos++] = static_cast<int32_t>(br.size());
        for (int32_t v : br) merges_out[pos++] = v;
    }
    return produced;
}

// Encode one word by applying merges in rank order.
//
//   word — packed codepoints of ONE pre-tokenized word
//   merge_bytes — flat int32 stream exactly as nm_bpe_train emits
//   nmerge_ints — its length
//
// Returns the number of int32 written to out; out holds the final
// symbols the same way merges do: [len, packed bytes] per symbol,
// concatenated.
int32_t nm_bpe_encode(const int32_t* word, int32_t len, const int32_t* merge_stream,
                      int32_t nmerge_ints, int32_t* out) {
    // 1) Rebuild the symbol table the train pass would have built:
    //    characters first (in first-seen order does NOT matter for
    //    encode — ids are arbitrary; what matters is merge order).
    SymTable table;
    table.reserve(1024);
    std::unordered_map<std::string, int32_t> char_to_id;
    std::vector<int32_t> syms;
    syms.reserve(len);
    for (int32_t i = 0; i < len; i++) {
        // encode codepoint → utf8 string (same as decode_words)
        uint32_t cp = static_cast<uint32_t>(word[i]);
        std::string ch;
        if (cp < 0x80) ch += static_cast<char>(cp);
        else if (cp < 0x800) { ch += static_cast<char>(0xC0 | (cp >> 6)); ch += static_cast<char>(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { ch += static_cast<char>(0xE0 | (cp >> 12)); ch += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); ch += static_cast<char>(0x80 | (cp & 0x3F)); }
        else { ch += static_cast<char>(0xF0 | (cp >> 18)); ch += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); ch += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); ch += static_cast<char>(0x80 | (cp & 0x3F)); }
        auto it = char_to_id.find(ch);
        if (it == char_to_id.end()) {
            int32_t id = static_cast<int32_t>(table.size());
            table.push_back(ch);
            char_to_id[ch] = id;
        }
        syms.push_back(it != char_to_id.end() ? it->second : static_cast<int32_t>(table.size() - 1));
    }

    auto key = [](int32_t l, int32_t r) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(l)) << 32) |
               static_cast<uint32_t>(r);
    };

    // 2) Parse the merge stream; create the merged symbols + rank map.
    struct Merge { int32_t l, r, merged; };
    std::vector<Merge> merges;
    std::unordered_map<uint64_t, int32_t> rank;
    rank.reserve(nmerge_ints / 8);
    int32_t p = 0;
    auto read_blob = [&](int32_t pos, int32_t* out_len) -> std::string {
        int32_t n_words = merge_stream[pos];
        std::string s;
        s.reserve(n_words * 4);
        for (int32_t wi = 0; wi < n_words; wi++) {
            uint32_t v = static_cast<uint32_t>(merge_stream[pos + 1 + wi]);
            for (int b = 0; b < 4; b++) s += static_cast<char>((v >> (8 * b)) & 0xFF);
        }
        // drop zero padding
        while (!s.empty() && s.back() == '\0') s.pop_back();
        *out_len = n_words;
        return s;
    };
    while (p < nmerge_ints) {
        int32_t ln;
        std::string l = read_blob(p, &ln);
        p += 1 + ln;
        std::string r = read_blob(p, &ln);
        p += 1 + ln;
        int32_t li, ri;
        // find-or-create is unnecessary: the stream was produced by the
        // train pass whose table already contains these — but encode may
        // run on a word with NEW characters; merged symbols reference
        // symbols by their table position, so look them up by content.
        auto find_or_create = [&](const std::string& s) {
            for (size_t i = 0; i < table.size(); i++)
                if (table[i] == s) return static_cast<int32_t>(i);
            int32_t id = static_cast<int32_t>(table.size());
            table.push_back(s);
            return id;
        };
        li = find_or_create(l);
        ri = find_or_create(r);
        int32_t mid = static_cast<int32_t>(table.size());
        table.push_back(l + r);
        merges.push_back(Merge{li, ri, mid});
        rank[key(li, ri)] = static_cast<int32_t>(merges.size() - 1);
    }

    // 3) Apply merges: repeatedly merge the lowest-rank adjacent pair
    //    (first index wins on ties — mirrors the python scan).
    while (syms.size() > 1) {
        int32_t best_rank = -1, best_idx = -1;
        for (size_t i = 0; i + 1 < syms.size(); i++) {
            auto it = rank.find(key(syms[i], syms[i + 1]));
            if (it != rank.end() && (best_rank < 0 || it->second < best_rank)) {
                best_rank = it->second;
                best_idx = static_cast<int32_t>(i);
            }
        }
        if (best_idx < 0) break;
        int32_t mid = merges[best_rank].merged;
        std::vector<int32_t> next;
        next.reserve(syms.size() - 1);
        for (size_t i = 0; i < syms.size(); i++) {
            if (static_cast<int32_t>(i) == best_idx) continue;
            if (static_cast<int32_t>(i) == best_idx + 1) { next.push_back(mid); continue; }
            next.push_back(syms[i]);
        }
        syms = std::move(next);
    }

    // 4) Serialize the final symbols: [len, packed] per symbol.
    int32_t pos = 0;
    for (int32_t id : syms) {
        const std::string& s = table[id];
        int32_t n = static_cast<int32_t>(s.size());
        int32_t words_needed = (n + 3) / 4;
        out[pos++] = words_needed;
        for (int32_t wi = 0; wi < words_needed; wi++) {
            uint32_t v = 0;
            for (int b = 0; b < 4 && wi * 4 + b < n; b++) {
                v |= static_cast<uint32_t>(static_cast<uint8_t>(s[wi * 4 + b])) << (8 * b);
            }
            out[pos++] = static_cast<int32_t>(v);
        }
    }
    return pos;
}

} // extern "C"
