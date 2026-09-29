/* nmhash.c — native fast path for the hash cracker's inner loop.
 *
 * This is the ONE place the project uses a compiled language, and it is
 * deliberate: the cracker hashes millions of candidates per second, and a
 * Python function call per candidate is the bottleneck. Everything else in
 * the codebase stays Python.
 *
 * Implemented from the public specs (RFC 1320 MD4, RFC 1321 MD5, FIPS
 * 180-1/2000 SHA-1, FIPS 180-4 SHA-256). NTLM is MD4 over UTF-16LE bytes —
 * the UTF-16LE encoding stays in Python (where it is unambiguous), so the C
 * side exposes plain MD4 and Python composes NTLM on top.
 *
 * Build:  g++ -O2 -shared -fPIC -o nmhash.so nmhash.c
 * Load:   ctypes (see loader.py) — pure-Python fallback when unavailable.
 */
#include <stdint.h>
#include <string.h>

#define ROL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

/* ── MD5 (RFC 1321) ──────────────────────────────────────────────────────── */

static const uint32_t M5_K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};

static const int M5_S[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

static void md5_block(uint32_t h[4], const uint8_t p[64]) {
    uint32_t x[16];
    for (int i = 0; i < 16; i++)
        x[i] = (uint32_t)p[4*i] | ((uint32_t)p[4*i+1] << 8) |
               ((uint32_t)p[4*i+2] << 16) | ((uint32_t)p[4*i+3] << 24);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    for (int i = 0; i < 64; i++) {
        uint32_t f; int g;
        if (i < 16)      { f = (b & c) | (~b & d);      g = i; }
        else if (i < 32) { f = (d & b) | (~d & c);      g = (5*i + 1) & 15; }
        else if (i < 48) { f = b ^ c ^ d;               g = (3*i + 5) & 15; }
        else             { f = c ^ (b | ~d);            g = (7*i) & 15; }
        f = f + a + M5_K[i] + x[g];
        a = d; d = c; c = b;
        b = b + ROL32(f, M5_S[i]);   /* RFC 1321 macro: A' = ROTL(A+F+K+M, s) + B */
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
}

/* Shared padding driver: full 64-byte blocks run straight over `in` (no copy);
 * only the tail block (≤ 128 bytes) is buffered, so any message length is safe. */
static void nm_digest(const uint8_t *in, int len, uint32_t *h,
                      void (*block)(uint32_t *, const uint8_t *), int be_length) {
    int rem = len & 63;
    int pad = (rem < 56) ? (56 - rem) : (120 - rem);
    for (int off = 0; off + 64 <= len; off += 64) block(h, in + off);
    int tail = len & ~63, n = len - tail;
    uint8_t buf[128];
    for (int i = 0; i < n; i++) buf[i] = in[tail + i];
    buf[n] = 0x80;
    for (int i = n + 1; i < n + pad; i++) buf[i] = 0;
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++)
        buf[be_length ? (n + pad + 7 - i) : (n + pad + i)] = (uint8_t)(bits >> (8*i));
    for (int off = 0; off < n + pad + 8; off += 64) block(h, buf + off);
}

static void nm_md5(const uint8_t *in, int len, uint8_t out[16]) {
    uint32_t h[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    nm_digest(in, len, h, md5_block, 0);
    for (int i = 0; i < 4; i++) {
        out[4*i]   = (uint8_t)(h[i]);
        out[4*i+1] = (uint8_t)(h[i] >> 8);
        out[4*i+2] = (uint8_t)(h[i] >> 16);
        out[4*i+3] = (uint8_t)(h[i] >> 24);
    }
}

/* ── MD4 (RFC 1320) — the primitive NTLM is built from ───────────────────── */

static void md4_block(uint32_t h[4], const uint8_t p[64]) {
    uint32_t x[16];
    for (int i = 0; i < 16; i++)
        x[i] = (uint32_t)p[4*i] | ((uint32_t)p[4*i+1] << 8) |
               ((uint32_t)p[4*i+2] << 16) | ((uint32_t)p[4*i+3] << 24);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    /* RFC 1320 App. A tables (identical to the pure-Python md4 in tools/hashcrack.py) */
    static const int R2[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    static const int R3[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
    static const int S1[4] = {3, 7, 11, 19};
    static const int S2[4] = {3, 5, 9, 13};
    static const int S3[4] = {3, 9, 11, 15};
    for (int i = 0; i < 48; i++) {
        uint32_t f, k;
        int s, g;
        if (i < 16)      { f = (b & c) | (~b & d);          k = 0;          s = S1[i & 3];        g = i; }
        else if (i < 32) { f = (b & c) | (b & d) | (c & d); k = 0x5a827999; s = S2[(i - 16) & 3]; g = R2[i - 16]; }
        else             { f = b ^ c ^ d;                   k = 0x6ed9eba1; s = S3[(i - 32) & 3]; g = R3[i - 32]; }
        uint32_t t = ROL32(a + f + x[g] + k, s);
        uint32_t na = d, nb = t, nc = b, nd = c;            /* (A,B,C,D) <- (D, t, B, C) */
        a = na; b = nb; c = nc; d = nd;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
}

static void nm_md4(const uint8_t *in, int len, uint8_t out[16]) {
    uint32_t h[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    nm_digest(in, len, h, md4_block, 0);
    for (int i = 0; i < 4; i++) {
        out[4*i]   = (uint8_t)(h[i]);
        out[4*i+1] = (uint8_t)(h[i] >> 8);
        out[4*i+2] = (uint8_t)(h[i] >> 16);
        out[4*i+3] = (uint8_t)(h[i] >> 24);
    }
}

/* ── SHA-1 (FIPS 180-1) ──────────────────────────────────────────────────── */

static void sha1_block(uint32_t h[5], const uint8_t p[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) |
               ((uint32_t)p[4*i+2] << 8) | ((uint32_t)p[4*i+3]);
    for (int i = 16; i < 80; i++)
        w[i] = ROL32(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20)      { f = (b & c) | (~b & d);          k = 0x5a827999; }
        else if (i < 40) { f = b ^ c ^ d;                   k = 0x6ed9eba1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdc; }
        else             { f = b ^ c ^ d;                   k = 0xca62c1d6; }
        uint32_t t = ROL32(a, 5) + f + e + k + w[i];
        e = d; d = c; c = ROL32(b, 30); b = a; a = t;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
}

static void nm_sha1(const uint8_t *in, int len, uint8_t out[20]) {
    uint32_t h[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};
    nm_digest(in, len, h, sha1_block, 1);
    for (int i = 0; i < 5; i++) {
        out[4*i]   = (uint8_t)(h[i] >> 24);
        out[4*i+1] = (uint8_t)(h[i] >> 16);
        out[4*i+2] = (uint8_t)(h[i] >> 8);
        out[4*i+3] = (uint8_t)(h[i]);
    }
}

/* ── SHA-256 (FIPS 180-4) ────────────────────────────────────────────────── */

static const uint32_t S6_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static void sha256_block(uint32_t h[8], const uint8_t p[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) |
               ((uint32_t)p[4*i+2] << 8) | ((uint32_t)p[4*i+3]);
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROTR32(w[i-15], 7) ^ ROTR32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROTR32(w[i-2], 17) ^ ROTR32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = hh + S1 + ch + S6_K[i] + w[i];
        uint32_t S0 = ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

static void nm_sha256(const uint8_t *in, int len, uint8_t out[32]) {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    nm_digest(in, len, h, sha256_block, 1);
    for (int i = 0; i < 8; i++) {
        out[4*i]   = (uint8_t)(h[i] >> 24);
        out[4*i+1] = (uint8_t)(h[i] >> 16);
        out[4*i+2] = (uint8_t)(h[i] >> 8);
        out[4*i+3] = (uint8_t)(h[i]);
    }
}

/* ── hex helpers for the ctypes side ─────────────────────────────────────── */

static const char HEXC[] = "0123456789abcdef";

static void to_hex(const uint8_t *in, int len, char *out) {
    for (int i = 0; i < len; i++) {
        out[2*i] = HEXC[in[i] >> 4];
        out[2*i+1] = HEXC[in[i] & 0x0f];
    }
    out[2*len] = '\0';
}

/* Compiled with g++ — the exports need C linkage or ctypes can't find them. */
#ifdef __cplusplus
extern "C" {
#endif

void nm_md5_hex(const uint8_t *in, int len, char out[33]) {
    uint8_t d[16]; nm_md5(in, len, d); to_hex(d, 16, out);
}
void nm_md4_hex(const uint8_t *in, int len, char out[33]) {
    uint8_t d[16]; nm_md4(in, len, d); to_hex(d, 16, out);
}
void nm_sha1_hex(const uint8_t *in, int len, char out[41]) {
    uint8_t d[20]; nm_sha1(in, len, d); to_hex(d, 20, out);
}
void nm_sha256_hex(const uint8_t *in, int len, char out[65]) {
    uint8_t d[32]; nm_sha256(in, len, d); to_hex(d, 32, out);
}

#ifdef __cplusplus
}
#endif
