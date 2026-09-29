// C ABI kernel for the per-message memory-extraction heuristic pass.
//
// This is the ALWAYS-ON half of `MemoryExtractor._heuristic_pass`:
//   split a message into candidate sentences, classify each against the five
//   fixed patterns (preference / decision / relationship / fact-about-him /
//   fact-about-self), clean + tag + dedupe, and cap at four per turn.
//
// Correctness model: the five patterns are copied VERBATIM from
// nomorals/memory/extract.py and run through a small NFA that implements the
// exact regex subset they use (literals, |, ( ), ? * +, \b, \w, \s) with
// Python-compatible word/space semantics.  The Python path is the reference:
// tests run both and compare every (kind, content, importance, tags) element
// for element, so a divergence is a red test, never a silent behaviour change.
//
// The kernel returns -1 when the output buffer is too small; the caller must
// then use the pure-Python reference (never a truncated result).

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

namespace {

// ── UTF-8 ────────────────────────────────────────────────────────────────────

bool utf8_decode(const uint8_t* p, size_t len, std::vector<uint32_t>* out) {
    out->clear();
    size_t i = 0;
    while (i < len) {
        uint8_t b = p[i];
        uint32_t cp = 0;
        size_t extra = 0;
        if (b < 0x80) { cp = b; extra = 0; }
        else if ((b >> 5) == 0x6) { cp = b & 0x1F; extra = 1; }
        else if ((b >> 4) == 0xE) { cp = b & 0x0F; extra = 2; }
        else if ((b >> 3) == 0x1E) { cp = b & 0x07; extra = 3; }
        else return false;  // invalid lead byte
        if (i + extra >= len) return false;
        for (size_t k = 1; k <= extra; ++k) {
            if ((p[i + k] >> 6) != 0x2) return false;
            cp = (cp << 6) | (p[i + k] & 0x3F);
        }
        if (extra == 1 && cp < 0x80) return false;
        if (extra == 2 && cp < 0x800) return false;
        if (extra == 3 && (cp < 0x10000 || cp > 0x10FFFF)) return false;
        if (cp >= 0xD800 && cp <= 0xDFFF) return false;
        out->push_back(cp);
        i += 1 + extra;
    }
    return true;
}

void utf8_encode(uint32_t cp, std::string* out) {
    if (cp < 0x80) out->push_back(char(cp));
    else if (cp < 0x800) {
        out->push_back(char(0xC0 | (cp >> 6)));
        out->push_back(char(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out->push_back(char(0xE0 | (cp >> 12)));
        out->push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(char(0x80 | (cp & 0x3F)));
    } else {
        out->push_back(char(0xF0 | (cp >> 18)));
        out->push_back(char(0x80 | ((cp >> 12) & 0x3F)));
        out->push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(char(0x80 | (cp & 0x3F)));
    }
}

uint32_t lowercase(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp - 'A' + 'a';
    return cp;
}

// Python re treats a "word" as [a-zA-Z0-9_] plus Unicode letters/numbers.
// The five patterns are English phrases, so the ASCII part is what matters;
// the Unicode letter/number ranges below keep \b honest for accented text,
// CJK, etc.  (Verified against the Python reference on a Unicode battery.)
bool is_word(uint32_t cp) {
    if (cp == '_') return true;
    if (cp < 0x80)
        return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') ||
               (cp >= '0' && cp <= '9');
    if (cp >= 0x00C0 && cp <= 0x02AF) return true;   // Latin-1 + Ext A/B
    if (cp >= 0x0370 && cp <= 0x03FF) return true;   // Greek
    if (cp >= 0x0400 && cp <= 0x04FF) return true;   // Cyrillic
    if (cp >= 0x0500 && cp <= 0x052F) return true;   // Cyrillic Supplement
    if (cp >= 0x1E00 && cp <= 0x1EFF) return true;   // Latin Extended Add.
    if (cp >= 0x1F00 && cp <= 0x1FFF) return true;   // Greek Extended
    if (cp >= 0x2C00 && cp <= 0x2CFF) return true;   // Glagolitic
    if (cp >= 0x3040 && cp <= 0x30FF) return true;   // Kana
    if (cp >= 0x31F0 && cp <= 0x31FF) return true;   // Katakana Ext
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;   // CJK
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;   // Hangul
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;   // CJK Compat
    if (cp >= 0xFF21 && cp <= 0xFF3A) return true;   // Fullwidth A-Z
    if (cp >= 0xFF41 && cp <= 0xFF5A) return true;   // Fullwidth a-z
    if (cp >= 0x0660 && cp <= 0x0669) return true;   // Arabic-Indic digits
    if (cp >= 0x06F0 && cp <= 0x06F9) return true;   // Extended Arabic digits
    if (cp >= 0x0966 && cp <= 0x096F) return true;   // Devanagari digits
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;   // Fullwidth 0-9
    return false;
}

bool is_space(uint32_t cp) {
    // Python's \s (str mode) for the characters that occur in real chat text.
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
        cp == '\f' || cp == '\v')
        return true;
    if (cp == 0x85 || cp == 0xA0) return true;       // NEL, NBSP
    if (cp >= 0x1680 && cp <= 0x169A) return true;   // various spaces
    if (cp == 0x2000 || cp == 0x200A) return true;
    if (cp >= 0x2028 && cp <= 0x2029) return true;   // line/para separator
    if (cp == 0x202F || cp == 0x205F) return true;
    if (cp == 0x3000) return true;                   // ideographic space
    return false;
}

// ── tiny NFA for the fixed pattern subset ───────────────────────────────────

enum class StKind { Match, Split, Jump, Char, Word, Space, Boundary };

struct State {
    StKind kind;
    uint32_t cp = 0;   // Char: the code point
    int a = -1;        // Split.a / Jump.target / atom next
    int b = -1;        // Split.b
};

struct NfaBuilder {
    const std::string& p;
    size_t i = 0;
    std::vector<State> st;
    struct Frag { int entry; int exit; };

    explicit NfaBuilder(const std::string& pat) : p(pat) {}

    int fresh(StKind k) { st.push_back(State{k}); return (int)st.size() - 1; }
    bool eof() const { return i >= p.size(); }
    char peek() const { return eof() ? '\0' : p[i]; }
    void next() { ++i; }

    // one atom (input-consuming or \b), with a trailing Jump standing for the
    // fragment end.  Handles \w+ / \s+ / \s* here since they are the only
    // starred constructs in the five patterns.
    Frag atom() {
        char c = peek();
        if (c == '\\') {
            next();
            char e = peek(); next();
            if (e == 'b') {
                int s = fresh(StKind::Boundary);
                int x = fresh(StKind::Jump); st[s].a = x;
                return Frag{s, x};
            }
            if (e == 'w' || e == 's') {
                int cls = fresh(e == 'w' ? StKind::Word : StKind::Space);
                if (peek() == '+' || peek() == '*') {
                    next();
                    return star(cls);
                }
                int x = fresh(StKind::Jump); st[cls].a = x;
                return Frag{cls, x};
            }
            throw 0;
        }
        if (c == '^') throw 0;  // question regex is handled without the NFA
        // literal (possibly multi-byte UTF-8)
        uint8_t lead = (uint8_t)c;
        size_t extra = lead < 0x80 ? 0
            : ((lead >> 5) == 0x6 ? 1
            : ((lead >> 4) == 0xE ? 2
            : (((lead >> 3) == 0x1E) ? 3 : 0)));
        if (extra == 0 && lead >= 0x80) throw 0;
        uint32_t cp = lead;
        for (size_t k = 1; k <= extra; ++k) {
            if (i + k >= p.size() || (p[i + k] >> 6) != 0x2) throw 0;
            cp = (cp << 6) | ((uint8_t)p[i + k] & 0x3F);
        }
        i += 1 + extra;
        int s = fresh(StKind::Char); st[s].cp = cp;
        int x = fresh(StKind::Jump); st[s].a = x;
        return Frag{s, x};
    }

    Frag star(int cls) {
        int sp = fresh(StKind::Split);
        int out = fresh(StKind::Jump);
        int loop = fresh(StKind::Jump);
        st[sp].a = cls; st[sp].b = out;
        st[cls].a = loop; st[loop].a = sp;
        return Frag{sp, out};
    }

    Frag quant(Frag f) {
        if (peek() == '?') {
            next();
            int sp = fresh(StKind::Split);
            int out = fresh(StKind::Jump);
            st[sp].a = f.entry; st[sp].b = out;
            st[f.exit].a = out;
            return Frag{sp, out};
        }
        if (peek() == '*') {
            next();
            int sp = fresh(StKind::Split);
            int out = fresh(StKind::Jump);
            int loop = fresh(StKind::Jump);
            st[sp].a = f.entry; st[sp].b = out;
            st[f.exit].a = loop; st[loop].a = sp;
            return Frag{sp, out};
        }
        if (peek() == '+') {
            next();
            int sp = fresh(StKind::Split);
            int out = fresh(StKind::Jump);
            int loop = fresh(StKind::Jump);
            st[sp].a = f.entry; st[sp].b = out;
            st[f.exit].a = loop; st[loop].a = sp;
            return Frag{sp, out};
        }
        return f;
    }

    Frag term() {
        std::vector<Frag> parts;
        while (!eof() && peek() != '|' && peek() != ')') {
            Frag f;
            if (peek() == '(') {
                next();
                if (peek() == '?' && i + 1 < p.size() && p[i + 1] == ':') {
                    next(); next();
                }
                f = expr(false);
                if (peek() != ')') throw 0;
                next();
            } else {
                f = atom();
            }
            parts.push_back(quant(f));
        }
        if (parts.empty()) {
            int j = fresh(StKind::Jump);
            return Frag{j, j};
        }
        for (size_t k = 0; k + 1 < parts.size(); ++k)
            st[parts[k].exit].a = parts[k + 1].entry;
        return Frag{parts.front().entry, parts.back().exit};
    }

    Frag expr(bool top) {
        Frag left = term();
        while (!top && peek() == '|') {
            next();
            Frag right = term();
            int sp = fresh(StKind::Split);
            int merged = fresh(StKind::Jump);
            st[sp].a = left.entry; st[sp].b = right.entry;
            st[left.exit].a = merged; st[right.exit].a = merged;
            left = Frag{sp, merged};
        }
        return left;
    }

    int compile() {
        Frag f = expr(true);
        int m = fresh(StKind::Match);
        st[f.exit].a = m;
        return m;
    }
};

bool word_boundary(const std::vector<uint32_t>& s, size_t i) {
    bool left = i > 0 && is_word(s[i - 1]);
    bool right = i < s.size() && is_word(s[i]);
    return left != right;
}

inline bool in_list(const std::vector<int>& v, int x) {
    for (int y : v) if (y == x) return true;
    return false;
}

// epsilon successors of state `cur` at input position `pos`, invoked for
// each (Split.a, Split.b / Jump.a / Boundary.a-if-boundary-holds)
template <typename Emit>
inline void nfa_eps(const State& cs, const std::vector<uint32_t>& s,
                    size_t pos, Emit emit) {
    if (cs.kind == StKind::Split) {
        if (cs.a >= 0) emit(cs.a);
        if (cs.b >= 0) emit(cs.b);
    } else if (cs.kind == StKind::Jump) {
        if (cs.a >= 0) emit(cs.a);
    } else if (cs.kind == StKind::Boundary) {
        if (cs.a >= 0 && word_boundary(s, pos)) emit(cs.a);
    }
}

// Does the NFA match ANYWHERE in s?  (Python re.search semantics; only
// existence is needed for classification.)
//
// Invariant: at the end of every iteration, every state in `active` is
// bit-marked.  The re-seed closure is always computed in FULL (into a
// fresh list) — a state already in `active` does not make its fresh path
// redundant, because boundary states on that path are evaluated at the
// NEW position.
bool nfa_search(const std::vector<State>& st, const std::vector<uint32_t>& s) {
    int match = -1;
    for (size_t k = 0; k < st.size(); ++k)
        if (st[k].kind == StKind::Match) match = (int)k;

    size_t words = (st.size() + 63) / 8;
    std::vector<uint64_t> bits(words, 0);
    auto in_bits = [&](int x) -> bool {
        return (bits[x >> 6] >> (x & 63)) & 1;
    };
    auto add_bits = [&](int x) { bits[x >> 6] |= (1ULL << (x & 63)); };
    auto clear_bits = [&]() {
        for (size_t k = 0; k < words; ++k) bits[k] = 0;
    };

    size_t n = s.size();
    std::vector<int> active, seeded, nextset, stack;

    // expand epsilon successors of `stack` into `active` (bits-dedup)
    auto expand = [&](size_t pos) {
        while (!stack.empty()) {
            int cur = stack.back(); stack.pop_back();
            const State& cs = st[cur];
            nfa_eps(cs, s, pos, [&](int y) {
                if (!in_bits(y)) { add_bits(y); active.push_back(y); stack.push_back(y); }
            });
        }
    };

    for (size_t pos = 0; pos <= n; ++pos) {
        // (a) the pattern may START here: full re-seed closure at pos
        //     (fresh list, linear dedup — the seed set is small)
        seeded.clear();
        seeded.push_back(0);
        stack = {0};
        while (!stack.empty()) {
            int cur = stack.back(); stack.pop_back();
            const State& cs = st[cur];
            nfa_eps(cs, s, pos, [&](int y) {
                if (!in_list(seeded, y)) { seeded.push_back(y); stack.push_back(y); }
            });
        }
        // (b) merge into active; expand the fresh states' epsilons at pos
        for (int x : seeded)
            if (!in_bits(x)) { add_bits(x); active.push_back(x); stack.push_back(x); }
        expand(pos);
        for (int a : active)
            if (a == match) return true;
        if (pos == n) break;
        // (c) consume one input char
        uint32_t c = s[pos];
        clear_bits();
        nextset.clear();
        for (int a : active) {
            const State& sa = st[a];
            bool ok = false;
            if (sa.kind == StKind::Char) ok = (sa.cp == c);
            else if (sa.kind == StKind::Word) ok = is_word(c);
            else if (sa.kind == StKind::Space) ok = is_space(c);
            if (ok && sa.a >= 0 && !in_bits(sa.a)) { add_bits(sa.a); nextset.push_back(sa.a); }
        }
        // (d) active := epsilon closure of nextset at pos+1
        active.clear();
        active.insert(active.end(), nextset.begin(), nextset.end());
        stack = nextset;
        expand(pos + 1);
    }
    return false;
}

// ── the five patterns (verbatim from extract.py) ─────────────────────────────

static const char* P_PREFERENCE =
    "\\b(?:i prefer|i'?d rather|i would rather|don'?t (?:call|text|message|post|send|ask) me"
    "|never (?:send|post|message|call)|always (?:send|post|message)|call me "
    "|address me as|i want you to (?:always|never|stop|keep)|stop (?:sending|asking)"
    "|i (?:like|love|hate) (?:it when|when) (?:you|you')|keep (?:it|things) (?:short|simple|casual))\\b";
static const char* P_DECISION =
    "\\b(?:let'?s (?:do|use|try|go|start|make|pick|stick with)"
    "|i (?:just )?decided (?:to|on|that)|we (?:should|will|'ll|can) (?:do|use|go|start|pick)"
    "|going with|i'?m going to (?:start|begin|try|use)|final (?:answer|decision):?)\\b";
static const char* P_RELATIONSHIP =
    "\\b(?:you'?re my|you are my|i (?:really )?(?:like|love|miss|appreciate) (?:you|that|when you)"
    "|best friend|you make me|i'?m glad (?:you|that you|to have you|we)"
    "|how (?:do|did) you (?:feel|think) about (?:us|me|yourself|what we)"
    "|i (?:need|want) (?:you|us) to)\\b";
static const char* P_FACT_ABOUT_HIM =
    "\\b(?:you (?:live|work|study|are from|come from)|your name is"
    "|you (?:like|love|prefer|don'?t like|hate) (?:it when|when)? )\\b";
static const char* P_FACT_SELF =
    "\\b(?:i (?:live|work|study|lived|worked|studied) (?:in|as|at|for|on)"
    "|(?:i'?m|i am) (?:from|a|an|about|in \\w+)"
    "|i (?:just |still |already |finally |going to )?(?:got|have|had|bought|sold|named) "
    "|i (?:just |still |already |finally )?(?:went|started|finished|moved|joined))\\b";

// compiled once, on first use
struct CompiledPattern {
    std::vector<State> st;
    bool ok = false;
    bool built = false;
    void build(const char* pat) {
        if (built) return;
        built = true;
        try {
            std::string p(pat);  // must outlive the builder (it holds a ref)
            NfaBuilder b(p);
            b.compile();
            st = std::move(b.st);
            ok = true;
        } catch (...) { ok = false; }
    }
    bool search(const std::string& hay) {
        if (!ok) return false;
        std::vector<uint32_t> cps;
        if (!utf8_decode((const uint8_t*)hay.data(), hay.size(), &cps))
            return false;
        return nfa_search(st, cps);
    }
    // the hot path: the caller already decoded the sentence once.
    bool search_cp(const std::vector<uint32_t>& cps) const {
        if (!ok) return false;
        return nfa_search(st, cps);
    }
};

static CompiledPattern CP_PREFERENCE, CP_DECISION, CP_RELATIONSHIP,
    CP_FACT_ABOUT_HIM, CP_FACT_SELF;

// ── tag lexicon (verbatim from extract.py) ───────────────────────────────────

struct Tag { const char* key; const char* tag; };
static const Tag TAGS[] = {
    {"coffee", "coffee"}, {"tea", "coffee"}, {"espresso", "coffee"},
    {"sleep", "sleep"}, {"insomnia", "sleep"}, {"tired", "sleep"}, {"bedtime", "sleep"},
    {"gym", "fitness"}, {"workout", "fitness"}, {"running", "fitness"}, {"protein", "fitness"},
    {"money", "money"}, {"salary", "money"}, {"broke", "money"}, {"budget", "money"}, {"pay", "money"},
    {"work", "work"}, {"job", "work"}, {"boss", "work"}, {"office", "work"}, {"deadline", "work"},
    {"family", "family"}, {"mum", "family"}, {"mom", "family"}, {"dad", "family"}, {"sister", "family"},
    {"brother", "family"}, {"wife", "family"}, {"husband", "family"},
    {"food", "food"}, {"eat", "food"}, {"eating", "food"}, {"cook", "food"}, {"chef", "food"},
    {"music", "music"}, {"song", "music"}, {"playlist", "music"},
    {"travel", "travel"}, {"flight", "travel"}, {"trip", "travel"}, {"holiday", "travel"},
    {"tech", "tech"}, {"code", "tech"}, {"coding", "tech"}, {"script", "tech"}, {"server", "tech"},
    {"health", "health"}, {"sick", "health"}, {"illness", "health"}, {"doctor", "health"},
    {"lagos", "lagos"}, {"nigeria", "nigeria"}, {"abuja", "abuja"},
};
static const int N_TAGS = (int)(sizeof(TAGS) / sizeof(TAGS[0]));

}  // namespace

// ── C ABI ────────────────────────────────────────────────────────────────────

extern "C" {

const char* nm_memextract_version() { return "memextract-1"; }

// out[] receives:
//   int32 count
//   per memory:
//     int32 kind_idx       (0 preference, 1 decision, 2 relationship, 3 fact)
//     int32 importance_pct (0..100)
//     int32 content_len    (UTF-8 bytes)
//     bytes  content
//     int32 tag_count
//     per tag: int32 tag_len (bytes), bytes tag
// Returns bytes written, or -1 if out_cap is too small, -2 on bad UTF-8.
int32_t nm_memextract(const uint8_t* text, int32_t len, uint32_t* out,
                      int32_t out_cap) {
    std::vector<uint32_t> textcps;
    if (!utf8_decode(text, (size_t)len, &textcps)) return -2;

    CP_PREFERENCE.build(P_PREFERENCE);
    CP_DECISION.build(P_DECISION);
    CP_RELATIONSHIP.build(P_RELATIONSHIP);
    CP_FACT_ABOUT_HIM.build(P_FACT_ABOUT_HIM);
    CP_FACT_SELF.build(P_FACT_SELF);

    const size_t MIN_LEN = 4, MAX_LEN = 240;
    const int MAX_PER_TURN = 4;

    // 1. split into candidate sentences
    //    (Python: re.split(r"(?<=[.!?])\s+|\n+", text))
    std::vector<std::vector<uint32_t>> parts;
    {
        std::vector<uint32_t> cur;
        auto flush = [&]() {
            if (!cur.empty()) { parts.push_back(cur); cur.clear(); }
        };
        for (size_t i = 0; i < textcps.size(); ++i) {
            uint32_t c = textcps[i];
            if (!is_space(c)) { cur.push_back(c); continue; }
            bool prev_punct = (i > 0) &&
                (textcps[i - 1] == '.' || textcps[i - 1] == '!' ||
                 textcps[i - 1] == '?');
            size_t j = i;
            bool contains_nl = false;
            while (j < textcps.size() && is_space(textcps[j])) {
                if (textcps[j] == '\n') contains_nl = true;
                ++j;
            }
            if (prev_punct || contains_nl) {
                flush();
                i = j - 1;  // the loop's ++i lands on j
            } else {
                cur.push_back(c);
            }
        }
        flush();
    }

    struct Mem {
        int kind;
        int imp_pct;
        std::string content;
        std::vector<std::string> tags;
    };
    std::vector<Mem> found;
    std::vector<std::string> seen;

    for (const auto& raw : parts) {
        // normalize: collapse whitespace runs to single spaces, trim
        std::vector<uint32_t> norm;
        bool pending = false;
        for (uint32_t c : raw) {
            if (is_space(c)) { pending = true; continue; }
            if (pending && !norm.empty()) norm.push_back(' ');
            pending = false;
            norm.push_back(c);
        }
        size_t ncp = norm.size();
        if (ncp < MIN_LEN || ncp > MAX_LEN) continue;
        size_t e = ncp;
        while (e > 0) {
            uint32_t c = norm[e - 1];
            if (c == '.' || c == '!' || c == '?') --e;
            else break;
        }
        if (e == 0) continue;  // s.rstrip(".!?") == ""

        // lowercased code points for the case-insensitive patterns
        // (decoded once, searched five times)
        std::vector<uint32_t> lowcp(norm.size());
        for (size_t t = 0; t < ncp; ++t) lowcp[t] = lowercase(norm[t]);
        std::string low;
        for (uint32_t c : lowcp) utf8_encode(c, &low);

        // question test (Python: s.strip().endswith("?") — only whitespace
        // is stripped, never punctuation — or a leading what/why/how/when/
        // where/who/do/does/did followed by a word boundary).  `norm` has
        // no leading/trailing whitespace, so the raw last char decides.
        bool question = (ncp > 0 && norm[ncp - 1] == '?');
        if (!question) {
            size_t k = 0;
            while (k < ncp && is_space(norm[k])) ++k;
            size_t w0 = k;
            while (k < ncp && is_word(norm[k])) ++k;
            if (k > w0) {
                std::string lw;
                for (size_t t = w0; t < k; ++t)
                    utf8_encode(lowercase(norm[t]), &lw);
                static const char* QS[] = {"what", "why", "how", "when",
                                           "where", "who", "do", "does", "did"};
                for (const char* q : QS)
                    if (lw == q) { question = true; break; }
            }
        }

        // classify in Python's priority order
        int kind = -1;
        if (CP_PREFERENCE.search_cp(lowcp)) kind = 0;
        else if (CP_DECISION.search_cp(lowcp)) kind = 1;
        else if (CP_RELATIONSHIP.search_cp(lowcp)) kind = 2;
        else if (CP_FACT_ABOUT_HIM.search_cp(lowcp)) kind = 3;
        else if (!question && CP_FACT_SELF.search_cp(lowcp)) kind = 3;
        if (kind < 0) continue;

        // clean: strip leading [,;:-\s]+ (en dash included), cap at 240
        std::vector<uint32_t> content = norm;
        size_t st2 = 0;
        while (st2 < content.size()) {
            uint32_t c = content[st2];
            if (c == ',' || c == ';' || c == ':' || c == '-' || c == 0x2013 ||
                is_space(c)) ++st2;
            else break;
        }
        content.erase(content.begin(), content.begin() + st2);
        if (content.size() > MAX_LEN) content.resize(MAX_LEN);
        if (content.empty()) continue;

        std::string content_utf8;
        std::string key;
        for (uint32_t c : content) {
            utf8_encode(c, &content_utf8);
            utf8_encode(lowercase(c), &key);
        }
        while (!key.empty()) {
            char last = key.back();
            if (last == '.' || last == '!' || last == '?') key.pop_back();
            else break;
        }
        if (key.empty()) continue;
        bool dup = false;
        for (const auto& k : seen) if (k == key) { dup = true; break; }
        if (dup) continue;
        seen.push_back(key);

        // tags: substring match over lowercased content, sorted, unique
        std::string lowc;
        for (uint32_t c : content) utf8_encode(lowercase(c), &lowc);
        std::vector<std::string> tags;
        for (int t = 0; t < N_TAGS; ++t) {
            if (lowc.find(TAGS[t].key) != std::string::npos) {
                bool have = false;
                for (const auto& x : tags)
                    if (x == TAGS[t].tag) { have = true; break; }
                if (!have) tags.push_back(TAGS[t].tag);
            }
        }
        std::sort(tags.begin(), tags.end());

        int imp = (kind == 0) ? 80 : (kind == 1) ? 70 : (kind == 2) ? 65 : 60;
        found.push_back(Mem{kind, imp, content_utf8, std::move(tags)});
        if ((int)found.size() >= MAX_PER_TURN) break;
    }

    // serialize
    size_t need = 4;
    for (const auto& m : found) {
        need += 4 + 4 + 4 + m.content.size() + 4;
        for (const auto& t : m.tags) need += 4 + t.size();
    }
    if (need > (size_t)out_cap) return -1;

    // plain little-endian byte stream (int32 cells + raw bytes, no padding)
    uint8_t* B = reinterpret_cast<uint8_t*>(out);
    size_t ob = 0;
    auto w32 = [&](int32_t v) {
        B[ob] = (uint8_t)(v & 0xFF);
        B[ob + 1] = (uint8_t)((v >> 8) & 0xFF);
        B[ob + 2] = (uint8_t)((v >> 16) & 0xFF);
        B[ob + 3] = (uint8_t)((v >> 24) & 0xFF);
        ob += 4;
    };
    w32((int32_t)found.size());
    for (const auto& m : found) {
        w32(m.kind);
        w32(m.imp_pct);
        w32((int32_t)m.content.size());
        std::memcpy(B + ob, m.content.data(), m.content.size());
        ob += m.content.size();
        w32((int32_t)m.tags.size());
        for (const auto& t : m.tags) {
            w32((int32_t)t.size());
            std::memcpy(B + ob, t.data(), t.size());
            ob += t.size();
        }
    }
    return (int32_t)ob;
}

}  // extern "C"
