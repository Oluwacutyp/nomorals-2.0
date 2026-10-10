// bpe.cpp — C ABI kernel for the BPE tokenizer.
//
// The Python tokenizer (nomorals/training/tokenize.py) is deterministic
// but slow: every merge rescans every word of the corpus in pure Python.
// This kernel runs the identical algorithm — same counting, same
// tie-breaks (highest count, then the LEXICALLY LARGER (left, right)
// string pair), same single left-to-right merge pass per word — with the
// Hugging Face `tokenizers` trainer's optimization: pair counts are kept
// in a max-heap with lazy deletion and updated incrementally (only words
// containing the merged pair are re-processed).  Symbols are Unicode
// characters, exactly like the Python reference's list(w) split; the wire
// format carries them as UTF-8 bytes.  Python remains the reference:
// tests compare merge lists and encodings exactly.
//
// Build: c++ -O2 -fPIC -shared -std=c++17 bpe.cpp -o libbpe.so

#include <algorithm>
#include <cstdint>
#include <queue>
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

const char* nm_bpe_version() { return "2.0.0"; }

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
        // One symbol per CHARACTER (Unicode codepoint), not per UTF-8
        // byte: the Python reference splits words with list(w), i.e. by
        // codepoint, and parity with it is the contract.  (A byte split
        // made multi-byte characters unmergeable and diverged from the
        // reference on any non-ASCII corpus.)
        for (size_t i = 0; i < s.size();) {
            unsigned char lead = static_cast<unsigned char>(s[i]);
            size_t seqlen = 1;
            if ((lead >> 5) == 0x6) seqlen = 2;
            else if ((lead >> 4) == 0xE) seqlen = 3;
            else if ((lead >> 3) == 0x1E) seqlen = 4;
            std::string ch = s.substr(i, seqlen);
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
            i += seqlen;
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

    // Pair counting — the Hugging Face `tokenizers` trainer pattern: a
    // max-heap keyed by (count, pair) with LAZY deletion, and counts
    // updated INCREMENTALLY.  After each merge only the words containing
    // the merged pair are re-processed (their overlapping pair counts are
    // decremented, the merge is applied, the new overlapping pairs are
    // incremented).  Training goes from O(merges x corpus) to
    // O(corpus + total changed symbols).
    //
    // Exactness: counts are exact int64 sums; pairs whose count reaches
    // zero are ERASED, mirroring Python's per-merge recount (a pair that
    // no longer occurs simply has no entry); and the heap comparator
    // replicates Python's max(counts.items(), key=(count, (l, r))) —
    // higher count wins, ties go to the lexicographically LARGER
    // (left, right) string pair.  A popped entry whose count no longer
    // matches `counts` is stale and discarded.  The merge list stays
    // list-equal to the Python reference.
    auto pair_key = [](int32_t l, int32_t r) -> uint64_t {
        return (static_cast<uint64_t>(static_cast<uint32_t>(l)) << 32) |
               static_cast<uint32_t>(r);
    };
    struct HeapEntry {
        int64_t count;
        int32_t l, r;
    };
    struct HeapCmp {
        const SymTable* table;
        // true -> a has LOWER priority (priority_queue pops the "largest").
        bool operator()(const HeapEntry& a, const HeapEntry& b) const {
            if (a.count != b.count) return a.count < b.count;
            const std::string& al = (*table)[a.l];
            const std::string& ar = (*table)[a.r];
            const std::string& bl = (*table)[b.l];
            const std::string& br = (*table)[b.r];
            if (al != bl) return al < bl;
            return ar < br;
        }
    };
    std::unordered_map<uint64_t, int64_t> counts;
    counts.reserve(8192);
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp>
        heap((HeapCmp{&table}));
    {
        std::unordered_set<uint64_t> pushed;
        for (int32_t w = 0; w < nwords; w++) {
            const auto& syms = splits[w];
            int64_t f = freqs[w];
            for (size_t i = 0; i + 1 < syms.size(); i++) {
                uint64_t k = pair_key(syms[i], syms[i + 1]);
                int64_t c = (counts[k] += f);
                if (pushed.insert(k).second) {
                    heap.push(HeapEntry{c, syms[i], syms[i + 1]});
                }
            }
        }
        // Entries pushed mid-loop carry partial counts; push the finals so
        // the heap top is usually valid on the first pop (stale entries are
        // still discarded by the pop loop — this is only a depth trim).
        for (uint64_t k : pushed) {
            heap.push(HeapEntry{counts[k],
                                static_cast<int32_t>(k >> 32),
                                static_cast<int32_t>(k & 0xFFFFFFFF)});
        }
    }

    // One pair's count changes; increments push a fresh heap entry (the old
    // one goes stale and is discarded on pop).  Zero-count pairs are erased
    // to mirror Python's recount.
    // One pair's count changes.  Increments push a fresh heap entry (the
    // old lower-count entry goes stale).  Decrements do NOT push: the pop
    // loop lazily refreshes a stale entry with the pair's current count
    // instead of discarding it, so the heap stays shallow.  Zero-count
    // pairs are erased to mirror Python's recount.
    auto dec_pair = [&](int32_t l, int32_t r, int64_t f) {
        uint64_t k = pair_key(l, r);
        auto it = counts.find(k);
        if (it == counts.end()) return;  // defensive; cannot happen
        it->second -= f;
        if (it->second <= 0) counts.erase(it);
    };
    auto inc_pair = [&](int32_t l, int32_t r, int64_t f) {
        uint64_t k = pair_key(l, r);
        int64_t c = (counts[k] += f);
        heap.push(HeapEntry{c, l, r});
    };

    int32_t produced = 0;
    while (static_cast<int32_t>(merges.size()) < target_merges) {
        // Best VALID heap entry wins (lazy deletion + lazy refresh): a
        // popped entry whose count no longer matches `counts` is either
        // gone (discard) or changed — reinsert it with the current count
        // and keep looking.  This is exactly Python's
        // max(counts.items(), key=(count, (l, r))) — higher count wins,
        // ties go to the lexicographically LARGER string pair.
        int64_t best_count = -1;
        int32_t best_l = -1, best_r = -1;
        for (;;) {
            if (heap.empty()) break;
            HeapEntry e = heap.top();
            heap.pop();
            auto it = counts.find(pair_key(e.l, e.r));
            if (it == counts.end()) continue;  // pair no longer occurs
            if (it->second != e.count) {
                heap.push(HeapEntry{it->second, e.l, e.r});  // refresh
                continue;
            }
            best_count = e.count;
            best_l = e.l;
            best_r = e.r;
            break;
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
        // The inverted index must stay exact: a word that still contains
        // best_l / best_r after the rewrite (in a non-merged position)
        // has to remain listed, or the next iteration's rewrite loop will
        // skip it and the same pair will win again forever.  Only touched
        // words can gain or lose these three symbols, so the three lists
        // are rebuilt from the touched set alone.
        std::vector<int32_t> touched_words = word_sets[best_l];
        // Untouched words can still hold best_r (only the (l,r) adjacency
        // triggers a rewrite, and that needs best_l): their entries must
        // survive the clear below.
        std::vector<int32_t> r_holders =
            (best_r == best_l) ? std::vector<int32_t>() : word_sets[best_r];
        // Incremental count maintenance: only words containing an actual
        // (best_l, best_r) adjacency change any pair count.  For each merge
        // site at i, the pairs (i-1,i), (i,i+1), (i+1,i+2) are decremented;
        // after the rewrite, the pairs around each new merged symbol are
        // incremented.  Every other pair's count is untouched and stays
        // exact.  (Python's `pair[0] not in syms` skip is the `affected`
        // check below — such words rewrite to themselves.)
        for (int32_t w : touched_words) {
            const auto& syms = splits[w];
            int64_t f = freqs[w];
            bool affected = false;
            for (size_t i = 0; i + 1 < syms.size(); i++) {
                if (syms[i] == best_l && syms[i + 1] == best_r) {
                    affected = true;
                    break;
                }
            }
            if (!affected) continue;
            // Mark every symbol position consumed by a merge site, then
            // decrement each adjacent pair touching a consumed position
            // exactly once.  (A pair sitting between two sites is the
            // right-overlap of the left site AND the left-overlap of the
            // right site — decrementing per-site would count it twice.)
            std::vector<char> consumed(syms.size(), 0);
            for (size_t i = 0; i + 1 < syms.size(); i++) {
                if (syms[i] == best_l && syms[i + 1] == best_r) {
                    consumed[i] = 1;
                    consumed[i + 1] = 1;
                    i++;  // sites are non-overlapping, left to right
                }
            }
            for (size_t i = 0; i + 1 < syms.size(); i++) {
                if (consumed[i] || consumed[i + 1]) {
                    dec_pair(syms[i], syms[i + 1], f);
                }
            }
            // Apply the merge, tracking where merged symbols land.
            std::vector<int32_t> out;
            out.reserve(syms.size());
            std::vector<char> is_merged;
            is_merged.reserve(syms.size());
            size_t i = 0;
            while (i < syms.size()) {
                if (i + 1 < syms.size() && syms[i] == best_l && syms[i + 1] == best_r) {
                    out.push_back(merged_id);
                    is_merged.push_back(1);
                    i += 2;
                } else {
                    out.push_back(syms[i]);
                    is_merged.push_back(0);
                    i += 1;
                }
            }
            // Increment each new adjacent pair touching a merged symbol.
            for (size_t j = 0; j + 1 < out.size(); j++) {
                if (is_merged[j] || is_merged[j + 1]) {
                    inc_pair(out[j], out[j + 1], f);
                }
            }
            splits[w] = std::move(out);
        }
        word_sets[best_l].clear();
        word_sets[best_r].clear();
        word_sets[merged_id].clear();
        std::vector<int32_t> touched_sorted = touched_words;
        std::sort(touched_sorted.begin(), touched_sorted.end());
        auto is_touched = [&](int32_t w) {
            return std::binary_search(touched_sorted.begin(),
                                      touched_sorted.end(), w);
        };
        for (int32_t w : touched_words) {
            bool has_l = false, has_r = false, has_m = false;
            for (int32_t s : splits[w]) {
                if (s == best_l) has_l = true;
                else if (s == best_r) has_r = true;
                else if (s == merged_id) has_m = true;
            }
            if (has_l) word_sets[best_l].push_back(w);
            if (has_r) word_sets[best_r].push_back(w);
            if (has_m) word_sets[merged_id].push_back(w);
        }
        for (int32_t w : r_holders) {
            if (!is_touched(w)) word_sets[best_r].push_back(w);
        }
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
