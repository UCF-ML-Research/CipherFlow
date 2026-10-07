// CipherFlow OpenFHE HE backend: transformer operators on CKKS, with timing, level accounting
// and error against plaintext.
//
//   he_ops <fast|precise> <op>[:k=v,...] ...
//     operators: add gelu gelu_wide layernorm layernorm_wide softmax softmax_wide qkv out_proj ffn1 ffn2 qk av; bootstrap microbench
//     self-checks: keycheck, bootcheck[:n=4,amp=1], racecheck[:rounds=3,only=name+name]
//
// Parameters (128-bit, checked by OpenFHE): N = 2^16, SPARSE_ENCAPSULATED secret, scale 2^50,
// first modulus 57 bits, level budget {3,3}. Bootstrapping leaves 8 levels (fast, one pass)
// or 7 levels (precise, two passes): a bootstrapped ciphertext still carries a pending rescale.
//
// Layouts (shared by all operators, so no hidden re-layout cost):
//   token tensor [128, H]: column-major blocks: slot = col * 128 + token, 256 columns per
//     ciphertext (H = 768 -> 3 ciphertexts, 3072 -> 12).
//   attention scores [heads, 128, 128]: "diagonal" layout: per ciphertext 2 heads x 128
//     diagonals r, slot = (head_local * 128 + r) * 128 + t holds score[head][t][(t + r) % 128]
//     (12 heads -> 6 ciphertexts). Softmax rows are sums over r.
//
// Each operator gets inputs with the levels left after a bootstrap (L) and bootstraps
// internally when it runs out; it reports levels used before its first internal bootstrap,
// the level of its output, its internal bootstraps, wall time and max |error|.
#include "openfhe.h"

#include "gelu_wide_coeffs.h"
#include "softmax_wide_coeffs.h"
#include "layernorm_wide_coeffs.h"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace lbcrypto;
using CT = Ciphertext<DCRTPoly>;
using Vec = std::vector<double>;
using Clock = std::chrono::steady_clock;

static constexpr uint32_t T = 128;            // tokens per block (sequence length)
// CF_OUTER_PARALLEL=0 runs the per-diagonal / per-block task loops serially (OpenFHE still
// parallelises inside each operation)
static bool outer_parallel() { static const bool on = !getenv("CF_OUTER_PARALLEL") || std::string(getenv("CF_OUTER_PARALLEL")) != "0"; return on; }
static double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Env {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> kp;
    uint32_t slots = 0, depth = 0, L = 0;
    bool precise = false;
    int boots = 0;
    double boot_ms = 0;
    int first_boot_used = -1;        // levels used before the first internal bootstrap
    uint32_t start_levels = 0;

    uint32_t left(const CT &c) const { return depth - c->GetLevel() - (uint32_t(c->GetNoiseScaleDeg()) - 1); }

    CT refresh(const CT &c) {
        // a ciphertext about to be bootstrapped still needs one level for its pending rescale
        if (left(c) < 1) throw std::runtime_error("bootstrapping needs at least one level left; spend fewer levels before it");
        if (first_boot_used < 0) first_boot_used = int(start_levels) - int(left(c));
        auto t = Clock::now();
        CT r = precise ? cc->EvalBootstrap(c, 2, 13) : cc->EvalBootstrap(c);
        boot_ms += ms(t);
        ++boots;
        return r;
    }
    CT need(const CT &c, uint32_t k) { return left(c) >= k ? c : refresh(c); }

    Plaintext pt(const Vec &v, const CT &like) const {
        return cc->MakeCKKSPackedPlaintext(v, 1, like->GetLevel(), nullptr, slots);
    }
    CT enc(const Vec &v) const {         // fresh ciphertext with L levels, as after a bootstrap
        auto p = cc->MakeCKKSPackedPlaintext(v, 1, depth - L, nullptr, slots);
        return cc->Encrypt(kp.publicKey, p);
    }
    Vec dec(const CT &c) const {
        Plaintext p;
        cc->Decrypt(kp.secretKey, c, &p);
        p->SetLength(slots);
        return p->GetRealPackedValue();
    }
    CT rot(const CT &c, int k) const { return k == 0 ? c : cc->EvalRotate(c, k); }
};

static std::vector<int32_t> rotation_indices() {
    std::vector<int32_t> idx;
    for (int k = 1; k < 256; k <<= 1) { idx.push_back(128 * k); idx.push_back(-128 * k); }
    for (int k = 3; k < 8; ++k) if (k & (k - 1)) { idx.push_back(128 * k); idx.push_back(-128 * k); }   // softmax packing
    for (int j = 1; j < 16; ++j) idx.push_back(128 * j);        // linear: baby steps (column blocks)
    for (int g = 1; g < 16; ++g) idx.push_back(2048 * g);       // linear: giant steps
    for (int b = 1; b < 16; ++b) idx.push_back(b);              // attention: rotation inside token blocks
    for (int a = 1; a < 8; ++a) { idx.push_back(16 * a); idx.push_back(16 * a - 128); }
    idx.push_back(-128);
    for (int b = 1; b < 16; ++b) idx.push_back(-128 * b);       // attention: place diagonal r at block r
    for (int a = 1; a < 8; ++a) idx.push_back(-2048 * a);
    std::sort(idx.begin(), idx.end());
    idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
    return idx;
}

static Env make_env(bool precise) {
    std::vector<uint32_t> budget = {3, 3};
    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecretKeyDist(SPARSE_ENCAPSULATED);
    p.SetSecurityLevel(HEStd_128_classic);
    p.SetRingDim(1 << 16);
    p.SetScalingModSize(50);
    p.SetFirstModSize(57);
    p.SetScalingTechnique(FLEXIBLEAUTO);
    uint32_t depth = 8 + FHECKKSRNS::GetBootstrapDepth(budget, SPARSE_ENCAPSULATED);
    p.SetMultiplicativeDepth(depth);
    Env e;
    e.cc = GenCryptoContext(p);
    for (auto f : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE}) e.cc->Enable(f);
    e.slots = e.cc->GetRingDimension() / 2;
    e.depth = depth;
    e.precise = precise;
    e.cc->EvalBootstrapSetup(budget, {0, 0}, e.slots);
    e.kp = e.cc->KeyGen();
    e.cc->EvalMultKeyGen(e.kp.secretKey);
    e.cc->EvalBootstrapKeyGen(e.kp.secretKey, e.slots);
    e.cc->EvalRotateKeyGen(e.kp.secretKey, rotation_indices());
    // levels left after a bootstrap, in this mode
    auto probe = e.cc->Encrypt(e.kp.publicKey, e.cc->MakeCKKSPackedPlaintext(Vec(e.slots, 0.1), 1, depth - 1, nullptr, e.slots));
    auto b = precise ? e.cc->EvalBootstrap(probe, 2, 13) : e.cc->EvalBootstrap(probe);
    e.L = e.depth - b->GetLevel() - (uint32_t(b->GetNoiseScaleDeg()) - 1);
    return e;
}

// ---------------------------------------------------------------- helpers

// sum over all 256 column blocks of a ciphertext, replicated in every block
static CT sum_blocks(Env &e, CT c) {
    for (int k = 1; k < 256; k <<= 1) c = e.cc->EvalAdd(c, e.rot(c, 128 * k));
    return c;
}
// per 128-block group (head): sum of blocks [g*128, g*128+127] at block g*128 (other blocks: partial sums)
static CT sum_head_forward(Env &e, CT c) {
    for (int k = 1; k < 128; k <<= 1) c = e.cc->EvalAdd(c, e.rot(c, 128 * k));
    return c;
}
// replicate block g*128 over blocks g*128 .. g*128+127 (other blocks must be zero)
static CT broadcast_head(Env &e, CT c) {
    for (int k = 1; k < 128; k <<= 1) c = e.cc->EvalAdd(c, e.rot(c, -128 * k));
    return c;
}
static Vec block_mask(Env &e, const std::vector<int> &blocks) {
    Vec m(e.slots, 0.0);
    for (int b : blocks) for (uint32_t t = 0; t < T; ++t) m[size_t(b) * T + t] = 1.0;
    return m;
}

// ---------------------------------------------------------------- operators

struct Result {
    std::string op, shape;
    int cts = 0;
    double total_ms = 0, err = 0;
    int levels_used = 0;     // levels consumed before the first internal bootstrap (or overall)
    int level_out = 0;       // levels left on the output
    int boots = 0;
    double boot_ms = 0;
    std::string approx;
};

static std::mt19937_64 rng(11);
static Vec rand_vec(size_t n, double lo, double hi) {
    std::uniform_real_distribution<double> d(lo, hi);
    Vec v(n);
    for (auto &x : v) x = d(rng);
    return v;
}
static double gelu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }

static double max_err(Env &e, const std::vector<CT> &out, const std::vector<Vec> &ref) {
    double m = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        Vec v;
        try { v = e.dec(out[i]); } catch (const std::exception &) { return -1; }   // not decodable: wrong result
        for (size_t j = 0; j < ref[i].size(); ++j) m = std::max(m, std::fabs(v[j] - ref[i][j]));
    }
    return m;
}

static Result run_add(Env &e, int H) {
    int n = H / 256;
    std::vector<Vec> a, b, ref;
    std::vector<CT> ca, cb, out;
    for (int i = 0; i < n; ++i) {
        a.push_back(rand_vec(e.slots, -4, 4)); b.push_back(rand_vec(e.slots, -4, 4));
        ref.push_back(Vec(e.slots)); for (uint32_t j = 0; j < e.slots; ++j) ref[i][j] = a[i][j] + b[i][j];
        ca.push_back(e.enc(a[i])); cb.push_back(e.enc(b[i]));
    }
    auto t = Clock::now();
    for (int i = 0; i < n; ++i) out.push_back(e.cc->EvalAdd(ca[i], cb[i]));
    Result r{"Residual_Add", "[1,128," + std::to_string(H) + "]", n, ms(t)};
    r.err = max_err(e, out, ref);
    r.level_out = int(e.left(out[0]));
    return r;
}

static Result run_gelu(Env &e, int F, double bound, int degree) {
    int n = F / 256;
    std::vector<Vec> x, ref;
    std::vector<CT> cx, out;
    for (int i = 0; i < n; ++i) {
        x.push_back(rand_vec(e.slots, -bound * 0.75, bound * 0.75));
        ref.push_back(Vec(e.slots)); for (uint32_t j = 0; j < e.slots; ++j) ref[i][j] = gelu(x[i][j]);
        cx.push_back(e.enc(x[i]));
    }
    auto t = Clock::now();
    for (int i = 0; i < n; ++i) out.push_back(e.cc->EvalChebyshevFunction(gelu, cx[i], -bound, bound, degree));
    Result r{"GeLU", "[1,128," + std::to_string(F) + "]", n, ms(t)};
    r.err = max_err(e, out, ref);
    r.level_out = int(e.left(out[0]));
    r.approx = "Chebyshev degree " + std::to_string(degree) + " on [" + std::to_string(-bound) + "," + std::to_string(bound) + "]";
    return r;
}

// GeLU on [-B, B] (NONLINEAR.md): input y = x / B (the scaling folds into the producer: the
// previous linear layer's weights or the MPC -> HE conversion's output scale); out = y * (B h)(f1(y))
// with f1 (degree 59) and h (degree 27) Chebyshev series on [-1, 1] and one bootstrap between them.
static Result run_gelu_wide(Env &e, int F, int B) {
    auto it = GELU_WIDE.find(B);
    if (it == GELU_WIDE.end()) throw std::runtime_error("gelu_wide: no coefficients for B = " + std::to_string(B));
    std::vector<double> hB = it->second.h;
    for (auto &c : hB) c *= B;
    int n = F / 256;
    std::vector<Vec> x, ref;
    std::vector<CT> cy, out;
    for (int i = 0; i < n; ++i) {
        x.push_back(rand_vec(e.slots, -B, B));                  // the whole range: worst-case error
        ref.push_back(Vec(e.slots)); for (uint32_t j = 0; j < e.slots; ++j) ref[i][j] = gelu(x[i][j]);
        Vec y(e.slots); for (uint32_t j = 0; j < e.slots; ++j) y[j] = x[i][j] / B;
        cy.push_back(e.enc(y));
    }
    e.boots = 0; e.boot_ms = 0; e.first_boot_used = -1; e.start_levels = e.L;
    auto t = Clock::now();
    for (int i = 0; i < n; ++i) {
        CT v = e.cc->EvalChebyshevSeries(cy[i], it->second.f1, -1.0, 1.0);
        v = e.refresh(v);
        CT g = e.cc->EvalChebyshevSeries(v, hB, -1.0, 1.0);
        out.push_back(e.cc->EvalMult(cy[i], g));
    }
    Result r{"GeLU_wide", "[1,128," + std::to_string(F) + "]", n, ms(t)};
    r.err = max_err(e, out, ref);
    r.levels_used = e.first_boot_used;
    r.level_out = int(e.left(out[0]));
    r.boots = e.boots; r.boot_ms = e.boot_ms;
    r.approx = "x h(f1(x/B)), f1 degree " + std::to_string(Degree(it->second.f1)) + ", h degree " +
               std::to_string(Degree(it->second.h)) + ", B " + std::to_string(B) + ", one bootstrap between";
    return r;
}

// LayerNorm on wide variance ranges (NONLINEAR.md): variances in [vlo, vhi] (calibrated), w = (var +
// eps) / vhi in [e0, 1]. The variances (one ciphertext, every block) give two copies: t in [-1, 1]
// (bootstrapped; y0 = P(t) / s ~ 1 / sqrt(s^2 w)) and 0.5 s^2 w, kept unbootstrapped for the Newton
// steps y <- y (1.5 - 0.5 s^2 w y^2), so the result is exact up to their count. The kernel needs 6
// levels on its input (the exact variances keep 3 for the Newton steps); its output is c * y, i.e.
// the normalised values times sqrt(vhi) / s, a constant that folds into gamma (the next layer).
static Result run_layernorm_wide(Env &e, int H, double vlo, double vhi, double eps, bool debug = false) {
    auto dbg = [&](const std::string &what, const CT &c) {
        if (!debug) return;
        Vec v;
        try { v = e.dec(c); } catch (const std::exception &) { std::cerr << "  [debug] " << what << ": not decodable\n"; return; }
        double lo = 1e300, hi = -1e300;
        for (double z : v) { lo = std::min(lo, z); hi = std::max(hi, z); }
        std::cerr << "  [debug] " << what << ": levels left " << e.left(c) << ", values [" << lo << ", " << hi << "]\n";
    };
    const LnWide *fit = nullptr;
    for (const auto &f : LN_WIDE) if (f.e0 <= vlo / vhi && (!fit || f.e0 > fit->e0)) fit = &f;
    if (!fit) throw std::runtime_error("layernorm_wide: variance range wider than the fits (vhi / vlo > 1e4)");
    const double e0 = fit->e0, sc = fit->s;
    const int n = H / 256;
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> mu(T), sd(T);
    for (uint32_t t = 0; t < T; ++t) {                     // variances log-uniform inside [vlo, vhi], with room for
        mu[t] = -2 + 4 * U(rng);                           // the sample variance's spread (beyond vhi, P diverges)
        sd[t] = std::sqrt(1.2 * vlo * std::pow(vhi / vlo / 1.44, U(rng)));
    }
    std::vector<Vec> x(n, Vec(e.slots)), ref(n, Vec(e.slots));
    for (int i = 0; i < n; ++i) for (uint32_t c = 0; c < 256; ++c) for (uint32_t t = 0; t < T; ++t)
        x[i][c * T + t] = mu[t] + sd[t] * nd(rng);
    const double outscale = std::sqrt(vhi) / sc;              // the kernel's output is the reference times this
    for (uint32_t t = 0; t < T; ++t) {
        double m = 0, v = 0;
        for (int i = 0; i < n; ++i) for (uint32_t c = 0; c < 256; ++c) m += x[i][c * T + t];
        m /= H;
        for (int i = 0; i < n; ++i) for (uint32_t c = 0; c < 256; ++c) v += std::pow(x[i][c * T + t] - m, 2);
        v /= H;
        for (int i = 0; i < n; ++i) for (uint32_t c = 0; c < 256; ++c)
            ref[i][c * T + t] = (x[i][c * T + t] - m) / std::sqrt(v + eps) * outscale;
    }
    std::vector<CT> cx;
    for (auto &v : x) cx.push_back(e.enc(v));
    Vec yexact(T);                                            // 1 / sqrt(s^2 w) per token
    for (uint32_t t = 0; t < T; ++t) {
        double m = 0, v = 0;
        for (int i = 0; i < n; ++i) for (uint32_t c = 0; c < 256; ++c) m += x[i][c * T + t];
        m /= H;
        for (int i = 0; i < n; ++i) for (uint32_t c = 0; c < 256; ++c) v += std::pow(x[i][c * T + t] - m, 2);
        yexact[t] = 1.0 / (sc * std::sqrt((v / H + eps) / vhi));
    }
    auto ycheck = [&](const std::string &what, const CT &c, int blocks) {
        if (!debug) return;
        Vec v = e.dec(c);
        double m = 0, rel = 0;
        for (int b = 0; b < blocks; ++b) for (uint32_t t = 0; t < T; ++t) {
            m = std::max(m, std::fabs(v[size_t(b) * T + t] - yexact[t]));
            rel = std::max(rel, std::fabs(v[size_t(b) * T + t] / yexact[t] - 1));
        }
        std::cerr << "  [debug] " << what << " vs exact (" << blocks << " blocks): max abs " << m << ", max rel " << rel << "\n";
    };
    e.boots = 0; e.boot_ms = 0; e.first_boot_used = -1; e.start_levels = e.L;
    auto t0 = Clock::now();
    for (auto &c : cx) c = e.need(c, 6);
    CT s = cx[0];
    for (int i = 1; i < n; ++i) s = e.cc->EvalAdd(s, cx[i]);
    CT mean = e.cc->EvalMult(sum_blocks(e, s), 1.0 / H);                          // 1 level
    std::vector<CT> c(n);
    for (int i = 0; i < n; ++i) c[i] = e.cc->EvalSub(cx[i], mean);
    CT q = e.cc->EvalSquare(c[0]);
    for (int i = 1; i < n; ++i) q = e.cc->EvalAdd(q, e.cc->EvalSquare(c[i]));
    CT Q = sum_blocks(e, q);                                                       // H var, every block
    // the statistics are the same in every 128-slot block; bootstrapping a vector repeated like this
    // works on few large coefficients (he_ops bootlevel: its error grows from |x| ~ 4, not 32), so
    // only block 0 keeps them (the mask folds into the constants) and the result is broadcast at the end
    // fast bootstrapping's noise is absolute: t in [-1, 1] uses 1/32 of the flat range, and at the
    // smallest variances the noise would put the initial guess outside Newton's basin (vhi / vlo ~ 1e4
    // on a 9.4-bit build). t is bootstrapped 16 times larger and scaled back (one level)
    const double amp = e.precise ? 1.0 : 16.0;
    Vec m0 = block_mask(e, {0}), mw(e.slots, 0.0), mt(e.slots, 0.0), ow(e.slots, 0.0), ot(e.slots, 0.0);
    for (uint32_t q0 = 0; q0 < T; ++q0) {
        mw[q0] = 0.5 * sc * sc / (H * vhi);  ow[q0] = 0.5 * sc * sc * eps / vhi;
        mt[q0] = amp * 2.0 / (H * vhi * (1 - e0)); ot[q0] = amp * ((2 * eps / vhi - 2 * e0) / (1 - e0) - 1);
    }
    CT w2 = e.cc->EvalMult(Q, e.pt(mw, Q));                                       // 0.5 s^2 w in block 0
    if (eps / vlo > 1e-9) {                     // (BERT's eps, 1e-12, is below the arithmetic's precision)
        Plaintext pow_ = e.pt(ow, w2);
        w2 = e.cc->EvalAdd(w2, pow_);
    }
    CT tt = e.cc->EvalMult(Q, e.pt(mt, Q));                                       // t in block 0, 0 elsewhere
    Plaintext pot = e.pt(ot, tt);
    tt = e.cc->EvalAdd(tt, pot);
    dbg("Q (H var)", Q); dbg("0.5 s^2 w", w2); dbg("t", tt);
    tt = e.need(tt, amp > 1 ? 8 : 7);
    if (amp > 1) tt = e.cc->EvalMult(tt, 1.0 / amp);
    dbg("t bootstrapped", tt);
    std::vector<double> p = fit->coef;
    for (auto &v : p) v /= sc;
    CT y = e.cc->EvalChebyshevSeries(tt, p, -1.0, 1.0);                           // ~ 1 / sqrt(s^2 w)
    dbg("y0", y);
    ycheck("y0", y, 1);
    // bootstrapping is much less precise for vectors with a large mean (he_ops bootlevel: a constant
    // 2.9 outside one block costs 1e-2): before the first Newton step the unused blocks hold P(0) / s,
    // which is subtracted for the bootstrap; the steps' 1.5 lives in block 0 only, so they zero the
    // unused blocks and later bootstraps see small means
    double p0 = p[0] / 2;
    for (size_t k = 2; k < p.size(); k += 2) p0 += (k % 4 == 0 ? 1.0 : -1.0) * p[k];
    Vec half(e.slots, 0.0);
    for (uint32_t q0 = 0; q0 < T; ++q0) half[q0] = 1.5;
    for (int r = 0; r < fit->steps; ++r) {
        if (r == 0 && e.left(y) < 4) y = e.cc->EvalAdd(e.refresh(e.cc->EvalSub(y, p0)), p0);
        else y = e.need(y, 4);
        CT q2 = e.cc->EvalMult(w2, e.cc->EvalSquare(y));
        Plaintext ph = e.pt(half, q2);
        y = e.cc->EvalMult(y, e.cc->EvalAdd(e.cc->EvalNegate(q2), ph));
        dbg("y after Newton step " + std::to_string(r + 1), y);
        ycheck("y after Newton step " + std::to_string(r + 1), y, 1);
    }
    int lowest = e.L;
    for (int i = 0; i < n; ++i) lowest = std::min(lowest, int(e.left(c[i])));
    y = e.need(y, std::max(3, std::min(int(e.L), lowest + 2)));                  // refresh one ciphertext, not n
    ycheck("y refreshed", y, 1);
    y = e.cc->EvalMult(y, e.pt(m0, y));          // the refresh's noise in the unused blocks would add up below
    y = sum_blocks(e, y);                                                         // block 0 -> every block
    dbg("y refreshed", y);
    ycheck("y broadcast", y, 256);
    dbg("c[0]", c[0]);
    std::vector<CT> out(n);
    for (int i = 0; i < n; ++i) out[i] = e.cc->EvalMult(c[i], y);
    Result r{"LayerNorm_wide", "[1,128," + std::to_string(H) + "]", n, ms(t0)};
    r.err = max_err(e, out, ref);
    if (r.err > 0) r.err /= outscale;                                              // in the normalised values' units
    r.levels_used = 5;                     // needs 6 on its input (openfhe_record adds one for internal bootstraps)
    r.level_out = int(e.left(out[0]));
    r.boots = e.boots; r.boot_ms = e.boot_ms;
    r.approx = "variances [" + std::to_string(vlo) + "," + std::to_string(vhi) + "]: e0 " + std::to_string(e0) +
               ", 1/sqrt degree 59, Newton " + std::to_string(fit->steps);
    return r;
}

// LayerNorm over H per token (no affine parameters; they fold into the next linear layer)
static Result run_layernorm(Env &e, int H, double vlo, double vhi, int degree) {
    int n = H / 256;
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> mu(T), sd(T);
    for (uint32_t t = 0; t < T; ++t) { mu[t] = rand_vec(1, -2, 2)[0]; sd[t] = rand_vec(1, std::sqrt(vlo) * 1.3, std::sqrt(vhi) * 0.7)[0]; }
    std::vector<Vec> x(n, Vec(e.slots)), ref(n, Vec(e.slots));
    for (int i = 0; i < n; ++i) for (uint32_t c = 0; c < 256; ++c) for (uint32_t t = 0; t < T; ++t)
        x[i][c * T + t] = mu[t] + sd[t] * nd(rng);
    for (uint32_t t = 0; t < T; ++t) {
        double m = 0, v = 0;
        for (int i = 0; i < n; ++i) for (uint32_t c = 0; c < 256; ++c) m += x[i][c * T + t];
        m /= H;
        for (int i = 0; i < n; ++i) for (uint32_t c = 0; c < 256; ++c) v += std::pow(x[i][c * T + t] - m, 2);
        v /= H;
        for (int i = 0; i < n; ++i) for (uint32_t c = 0; c < 256; ++c) ref[i][c * T + t] = (x[i][c * T + t] - m) / std::sqrt(v + 1e-5);
    }
    std::vector<CT> cx;
    for (auto &v : x) cx.push_back(e.enc(v));
    e.boots = 0; e.boot_ms = 0; e.first_boot_used = -1; e.start_levels = e.L;
    auto t0 = Clock::now();
    CT s = cx[0];
    for (int i = 1; i < n; ++i) s = e.cc->EvalAdd(s, cx[i]);
    CT mean = e.cc->EvalMult(sum_blocks(e, s), 1.0 / H);                          // 1 level
    std::vector<CT> c(n);
    for (int i = 0; i < n; ++i) c[i] = e.cc->EvalSub(cx[i], mean);
    CT q = e.cc->EvalSquare(c[0]);
    for (int i = 1; i < n; ++i) q = e.cc->EvalAdd(q, e.cc->EvalSquare(c[i]));
    // variance / vhi in (0, 1], replicated in every block: one ciphertext, small enough to bootstrap
    CT var = e.cc->EvalMult(sum_blocks(e, q), 1.0 / (H * vhi));
    var = e.need(var, uint32_t(std::ceil(std::log2(degree + 1))) + 1);
    CT inv = e.cc->EvalChebyshevFunction([vhi](double v) { return 1.0 / std::sqrt(v * vhi + 1e-5); }, var,
                                         vlo / vhi, 1.0, degree);
    inv = e.need(inv, 1);
    std::vector<CT> out(n);
    for (int i = 0; i < n; ++i) out[i] = e.cc->EvalMult(e.need(c[i], 1), inv);
    Result r{"LayerNorm", "[1,128," + std::to_string(H) + "]", n, ms(t0)};
    r.err = max_err(e, out, ref);
    r.levels_used = e.first_boot_used >= 0 ? e.first_boot_used : int(e.L) - int(e.left(out[0]));
    r.level_out = int(e.left(out[0]));
    r.boots = e.boots; r.boot_ms = e.boot_ms;
    r.approx = "1/sqrt: Chebyshev degree " + std::to_string(degree) + " on [" + std::to_string(vlo) + "," + std::to_string(vhi) + "]";
    return r;
}

// Softmax over keys on wide score ranges (NONLINEAR.md), scores in the diagonal layout. Calibrated
// bounds: scores in [lo, hi], every row's max >= rmin. Input u = affine(x) in [-1, 1] (the map folds
// into the producer: QK^T's weights, or the MPC -> HE conversion's output scale). y0 = exp((x - c) / 2^k);
// normalise y0 (rough inverse P1), then k rounds of square-and-normalise (P2 + Newton steps); the row
// sums of all heads are inverted in one packed ciphertext; rows are kept scaled by sqrt(AMP) so that
// every bootstrapped value is in the bootstrapping's flat range.
static Result run_softmax_wide(Env &e, int heads, double lo, double hi, double rmin, int newton_last,
                               bool debug = false) {
    auto dbg = [&](const std::string &what, const CT &c, const std::vector<int> &blocks) {
        if (!debug) return;
        Vec v;
        try { v = e.dec(c); } catch (const std::exception &) { std::cerr << "  [debug] " << what << ": not decodable\n"; return; }
        double amin = 1e300, amax = -1e300, all = 0;
        for (size_t q = 0; q < v.size(); ++q) all = std::max(all, std::fabs(v[q]));
        for (int blk : blocks) for (uint32_t t = 0; t < T; ++t) { amin = std::min(amin, v[size_t(blk) * T + t]); amax = std::max(amax, v[size_t(blk) * T + t]); }
        std::cerr << "  [debug] " << what << ": levels left " << e.left(c) << ", in blocks [" << amin << ", " << amax
                  << "], max |slot| " << all << "\n";
    };
    const double span = hi - lo, c = hi + 0.125 * span, lo2 = lo - 0.125 * span, rmin2 = rmin - 0.125 * span;
    const int k = std::max(1, int(std::ceil(std::log2((c - rmin2) / SMW_TARGET))));
    const double W = (c - lo2) / std::ldexp(1.0, k), sq = std::sqrt(SMW_AMP);
    const int n = heads / 2;
    std::vector<Vec> x(n, Vec(e.slots)), ref(n, Vec(e.slots)), u(n, Vec(e.slots));
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int i = 0; i < n; ++i) for (int h = 0; h < 2; ++h) for (uint32_t t = 0; t < T; ++t) {
        const double r = rmin + U(rng) * (hi - rmin);               // this row's max
        const uint32_t at = uint32_t(U(rng) * T) % T;
        double z = 0;
        for (uint32_t d = 0; d < T; ++d) {
            const double v = d == at ? r : lo + U(rng) * (r - lo);
            x[i][(h * T + d) * T + t] = v;
            z += std::exp(v - r);
        }
        for (uint32_t d = 0; d < T; ++d) {
            const size_t q = (h * T + d) * T + t;
            ref[i][q] = std::exp(x[i][q] - r) / z;
            u[i][q] = 2 * (x[i][q] - lo2) / (c - lo2) - 1;
        }
    }
    std::vector<CT> y;
    for (auto &v : u) y.push_back(e.enc(v));
    e.boots = 0; e.boot_ms = 0; e.first_boot_used = -1; e.start_levels = e.L;
    auto t0 = Clock::now();
    for (int i = 0; i < n; ++i)
        y[i] = e.cc->EvalChebyshevFunction([W](double v) { return std::exp(W * (v - 1) / 2); }, y[i], -1.0, 1.0, SMW_EXP_DEGREE);
    dbg("exp", y[0], {0, 1, 128});
    std::vector<double> p2z = SMW_P2;
    for (auto &v : p2z) v /= SMW_ZETA;                               // ~ 1 / (zeta S) <= AMP
    for (int j = 0; j <= k; ++j) {
        const bool last = j == k;
        const double a = j == 0 ? SMW_A1 : SMW_A2, b = j == 0 ? SMW_B1 : SMW_B2;
        std::vector<CT> s(n);
        // the last round keeps 3 levels on its packed sums: its Newton steps use them unbootstrapped
        for (int i = 0; i < n; ++i) s[i] = e.need(j == 0 ? y[i] : e.cc->EvalSquare(e.need(y[i], 2)), last ? 4 : 2);
        // row sums -> one ciphertext, mapped to [-1, 1] (the map folds into the mask); in the last
        // round also zeta S (for the Newton steps)
        Vec mask = block_mask(e, {0, 128}), maskz = block_mask(e, {0, 128});
        for (auto &v : mask) v *= 2.0 / (b - a);
        for (auto &v : maskz) v *= SMW_ZETA;
        CT packed, packz;
        for (int i = 0; i < n; ++i) {
            CT sum = sum_head_forward(e, s[i]);
            CT m = e.cc->EvalMult(sum, e.pt(mask, sum));
            packed = i == 0 ? m : e.cc->EvalAdd(packed, e.rot(m, -128 * i));
            if (last) {
                CT mz = e.cc->EvalMult(sum, e.pt(maskz, sum));
                packz = i == 0 ? mz : e.cc->EvalAdd(packz, e.rot(mz, -128 * i));
            }
        }
        // the offset only where sums are: other slots stay 0 (the interval's middle), where the inverse
        // and its Newton steps stay bounded (large values anywhere would spoil bootstrapping everywhere)
        Vec offset(e.slots, 0.0);
        for (int i = 0; i < n; ++i) for (int blk : {i, 128 + i}) for (uint32_t t = 0; t < T; ++t)
            offset[size_t(blk) * T + t] = -(a + b) / (b - a);
        Plaintext po = e.pt(offset, packed);
        packed = e.cc->EvalAdd(packed, po);
        dbg("round " + std::to_string(j) + " sums (normalised)", packed, {0, 128});
        CT st = e.need(packed, 8);
        CT z = e.cc->EvalChebyshevSeries(st, j == 0 ? SMW_P1 : (last ? p2z : SMW_P2), -1.0, 1.0);
        dbg("round " + std::to_string(j) + " inverse", z, {0, 128});
        double scale = j == 0 ? sq : (last ? SMW_ZETA : sq);
        if (last) {
            for (int r = 0; r < newton_last; ++r) {                  // z <- z (2 - zeta S z) -> 1 / (zeta S)
                z = e.need(z, 3);
                packz = e.need(packz, 3);
                CT q = e.cc->EvalMult(packz, z);
                z = e.cc->EvalMult(z, e.cc->EvalSub(2.0, q));
            }
        }
        // the rows' product takes the lower of the two levels: refresh the inverse (one ciphertext)
        // rather than let it pull the rows (n ciphertexts) down to its level
        int lowest = e.L;
        for (int i = 0; i < n; ++i) lowest = std::min(lowest, int(e.left(s[i])));
        z = e.need(z, std::max(3, std::min(int(e.L), lowest + 1)));
        dbg("round " + std::to_string(j) + " inverse refined", z, {0, 128});
        Vec first = block_mask(e, {0, 128});
        for (auto &v : first) v *= scale;
        for (int i = 0; i < n; ++i) {
            CT inv = broadcast_head(e, e.cc->EvalMult(e.rot(z, 128 * i), e.pt(first, z)));
            y[i] = e.cc->EvalMult(s[i], inv);
        }
        dbg("round " + std::to_string(j) + " rows", y[0], {0, 1, 128});
    }
    Result r{"Softmax_wide", "[1," + std::to_string(heads) + ",128,128]", n, ms(t0)};
    r.err = max_err(e, y, ref);
    r.levels_used = e.first_boot_used >= 0 ? e.first_boot_used : int(e.L) - int(e.left(y[0]));
    r.level_out = int(e.left(y[0]));
    r.boots = e.boots; r.boot_ms = e.boot_ms;
    r.approx = "scores [" + std::to_string(lo) + "," + std::to_string(hi) + "], row max >= " + std::to_string(rmin) +
               ": k " + std::to_string(k) + ", exp degree " + std::to_string(SMW_EXP_DEGREE) + " on [-" + std::to_string(W) +
               ",0], inverses degree 59 (first) / 27 (rounds), Newton " + std::to_string(newton_last) + " (last round)";
    return r;
}

// Softmax over keys, scores in the diagonal layout (12 heads -> 6 ciphertexts)
static Result run_softmax(Env &e, int heads, double bound, int exp_degree, double umin, int inv_degree) {
    int n = heads / 2;
    std::vector<Vec> x(n), ref(n, Vec(e.slots));
    for (int i = 0; i < n; ++i) x[i] = rand_vec(e.slots, -bound, bound);
    for (int i = 0; i < n; ++i) for (int h = 0; h < 2; ++h) for (uint32_t t = 0; t < T; ++t) {
        double z = 0;
        for (uint32_t r = 0; r < T; ++r) z += std::exp(x[i][(h * T + r) * T + t]);
        for (uint32_t r = 0; r < T; ++r) ref[i][(h * T + r) * T + t] = std::exp(x[i][(h * T + r) * T + t]) / z;
    }
    std::vector<CT> cx;
    for (auto &v : x) cx.push_back(e.enc(v));
    e.boots = 0; e.boot_ms = 0; e.first_boot_used = -1; e.start_levels = e.L;
    auto t0 = Clock::now();
    // exp(x - bound) in (0, 1]
    std::vector<CT> ex(n), m(n);
    for (int i = 0; i < n; ++i)
        ex[i] = e.cc->EvalChebyshevFunction([bound](double v) { return std::exp(v - bound); }, cx[i], -bound, bound, exp_degree);
    // per (head, token): u = (sum over the 128 diagonals) / 128, kept at each head's first block
    Vec first_over_T = block_mask(e, {0, 128});
    for (auto &v : first_over_T) v /= T;
    for (int i = 0; i < n; ++i) {
        CT sum = sum_head_forward(e, ex[i]);
        sum = e.need(sum, 2);                                    // the mask below + one level to bootstrap
        m[i] = e.cc->EvalMult(sum, e.pt(first_over_T, sum));                          // 1 level
    }
    // pack the 2 * n head sums into one ciphertext: ciphertext i -> blocks i and 128 + i
    CT packed = m[0];
    for (int i = 1; i < n; ++i) packed = e.cc->EvalAdd(packed, e.rot(m[i], -128 * i));
    // g(u) = umin / u in [umin, 1] (small enough to bootstrap); out = ex * g / (128 * umin)
    packed = e.need(packed, uint32_t(std::ceil(std::log2(inv_degree + 1))) + 2);
    CT g = e.cc->EvalChebyshevFunction([umin](double u) { return umin / u; }, packed, umin, 1.0, inv_degree);
    g = e.need(g, 2);
    Vec first_scaled = block_mask(e, {0, 128});
    for (auto &v : first_scaled) v /= (T * umin);
    std::vector<CT> out(n);
    for (int i = 0; i < n; ++i) {
        CT a = e.rot(g, 128 * i);                                                      // blocks i, 128+i -> 0, 128
        a = broadcast_head(e, e.cc->EvalMult(a, e.pt(first_scaled, a)));               // 1 level
        CT xi = e.need(ex[i], 1);
        a = e.need(a, 1);
        out[i] = e.cc->EvalMult(xi, a);
    }
    Result r{"Softmax", "[1," + std::to_string(heads) + ",128,128]", n, ms(t0)};
    r.err = max_err(e, out, ref);
    r.levels_used = e.first_boot_used >= 0 ? e.first_boot_used : int(e.L) - int(e.left(out[0]));
    r.level_out = int(e.left(out[0]));
    r.boots = e.boots; r.boot_ms = e.boot_ms;
    r.approx = "exp: Chebyshev degree " + std::to_string(exp_degree) + " on [" + std::to_string(-bound) + "," +
               std::to_string(bound) + "]; 1/u: Chebyshev degree " + std::to_string(inv_degree) + " on [" +
               std::to_string(umin) + ",1]";
    return r;
}

// Linear layer Y = X W + b, X encrypted [128, Hin] (column-major), W, b plaintext (server).
// Matrix-vector product over column blocks per (input, output) ciphertext pair, diagonal
// method with baby-step/giant-step (16 x 16); baby steps hoisted. Weight diagonals are
// encoded on the fly (precomputing them all would need ~4.7 MB per diagonal).
static Result run_linear(Env &e, const std::string &name, int Hin, int Hout) {
    const int nin = Hin / 256, nout = Hout / 256, B1 = 16, G = 256 / B1;
    std::normal_distribution<double> nd(0.0, 1.0);
    Vec X(size_t(T) * Hin), W(size_t(Hin) * Hout), bias(Hout);
    for (auto &v : X) v = rand_vec(1, -1, 1)[0];
    for (auto &w : W) w = nd(rng) / std::sqrt(double(Hin));
    for (auto &v : bias) v = 0.1 * nd(rng);
    std::vector<CT> cx(nin);
    for (int a = 0; a < nin; ++a) {
        Vec v(e.slots);
        for (int u = 0; u < 256; ++u) for (uint32_t t = 0; t < T; ++t) v[u * T + t] = X[t * Hin + 256 * a + u];
        cx[a] = e.enc(v);
    }
    std::vector<Vec> ref(nout, Vec(e.slots));
    for (uint32_t t = 0; t < T; ++t) for (int jcol = 0; jcol < Hout; ++jcol) {
        double y = bias[jcol];
        for (int k = 0; k < Hin; ++k) y += X[t * Hin + k] * W[size_t(k) * Hout + jcol];
        ref[jcol / 256][(jcol % 256) * T + t] = y;
    }
    double t_mul = 0, t_rot = 0;
    auto t0 = Clock::now();
    const uint32_t M = e.cc->GetCyclotomicOrder();
    std::vector<std::vector<CT>> baby(nin, std::vector<CT>(B1));
    auto tr = Clock::now();
    for (int a = 0; a < nin; ++a) {
        auto digits = e.cc->EvalFastRotationPrecompute(cx[a]);
        baby[a][0] = cx[a];
        for (int j = 1; j < B1; ++j) baby[a][j] = e.cc->EvalFastRotation(cx[a], 128 * j, M, digits);
    }
    t_rot += ms(tr);
    // inner products for every (output ciphertext, giant step) in parallel: each task encodes
    // its diagonals and accumulates nin * 16 plaintext products (OpenFHE's own threads only
    // split one operation over the RNS limbs, which leaves most cores idle here)
    std::vector<CT> inner(size_t(nout) * G);
    { Plaintext warm = e.pt(Vec(e.slots, 0.0), baby[0][0]); (void)warm; }   // encoder tables at this level
    auto tm = Clock::now();
#pragma omp parallel for schedule(dynamic) if (outer_parallel())
    for (int task = 0; task < nout * G; ++task) {
        const int b = task / G, g = task % G;
        Vec diag(e.slots);
        CT accum;
        for (int a = 0; a < nin; ++a) for (int j = 0; j < B1; ++j) {
            const int i = B1 * g + j;
            for (int v = 0; v < 256; ++v) {              // diag_i rotated right by B1*g blocks
                const int vs = ((v - B1 * g) % 256 + 256) % 256;
                const double w = W[size_t(256 * a + (vs + i) % 256) * Hout + 256 * b + vs];
                std::fill(diag.begin() + size_t(v) * T, diag.begin() + size_t(v + 1) * T, w);
            }
            CT x = baby[a][j]->Clone();               // private copy: tasks share the baby steps
            Plaintext pd = e.pt(diag, x);
            CT prod = e.cc->EvalMult(x, pd);
            accum = accum ? e.cc->EvalAdd(accum, prod) : prod;
        }
        inner[task] = accum;
    }
    t_mul += ms(tm);
    std::vector<CT> out(nout);
    auto tg = Clock::now();
#pragma omp parallel for schedule(dynamic) if (outer_parallel())
    for (int b = 0; b < nout; ++b) {
        CT acc = inner[size_t(b) * G];
        for (int g = 1; g < G; ++g) acc = e.cc->EvalAdd(acc, e.rot(inner[size_t(b) * G + g]->Clone(), 128 * B1 * g));
        Vec bb(e.slots);
        for (int v = 0; v < 256; ++v) std::fill(bb.begin() + size_t(v) * T, bb.begin() + size_t(v + 1) * T, bias[256 * b + v]);
        Plaintext pb = e.pt(bb, acc);
        out[b] = e.cc->EvalAdd(acc, pb);
    }
    t_rot += ms(tg);
    Result r{name, "[1,128," + std::to_string(Hin) + "]", nin, ms(t0)};
    r.err = max_err(e, out, ref);
    r.level_out = int(e.left(out[0]));
    r.levels_used = int(e.L) - r.level_out;
    std::ostringstream os;
    os << "diagonal BSGS 16x16 over column blocks, " << nin * nout * 256 << " plaintext mults; "
       << "encode+multiply+add (parallel) " << t_mul / 1e3 << " s, rotate+bias " << t_rot / 1e3 << " s";
    r.approx = os.str();
    return r;
}

// ---------------------------------------------------------------- attention

// token layout (4 heads x 64 columns per ciphertext) -> attention layout (2 heads per
// ciphertext, head h_local in blocks h_local*128 .. +63)
static std::vector<CT> to_att(Env &e, const std::vector<CT> &tok) {
    std::vector<CT> att;
    Vec m0(e.slots, 0.0), m1(e.slots, 0.0), m2(e.slots, 0.0), m3(e.slots, 0.0);
    for (uint32_t s = 0; s < e.slots; ++s) {
        const uint32_t blk = s / T;
        (blk < 64 ? m0 : blk < 128 ? m1 : blk < 192 ? m2 : m3)[s] = 1.0;
    }
    for (const CT &c : tok) {
        Plaintext p0 = e.pt(m0, c), p1 = e.pt(m1, c), p2 = e.pt(m2, c), p3 = e.pt(m3, c);
        CT a = e.cc->EvalAdd(e.cc->EvalMult(c, p0), e.rot(e.cc->EvalMult(c, p1), -128 * 64));   // heads 0,1
        CT b = e.cc->EvalAdd(e.rot(e.cc->EvalMult(c, p2), 128 * 128), e.rot(e.cc->EvalMult(c, p3), 128 * 64));  // heads 2,3
        att.push_back(a);
        att.push_back(b);
    }
    return att;
}
static std::vector<CT> from_att(Env &e, const std::vector<CT> &att) {
    std::vector<CT> tok;
    Vec lo(e.slots, 0.0), hi(e.slots, 0.0);
    for (uint32_t s = 0; s < e.slots; ++s) { const uint32_t blk = s / T; if (blk < 64) lo[s] = 1; else if (blk >= 128 && blk < 192) hi[s] = 1; }
    for (size_t i = 0; i + 1 < att.size(); i += 2) {
        const CT &a = att[i], &b = att[i + 1];
        Plaintext la = e.pt(lo, a), ha = e.pt(hi, a), lb = e.pt(lo, b), hb = e.pt(hi, b);
        CT c = e.cc->EvalAdd(e.cc->EvalMult(a, la), e.rot(e.cc->EvalMult(a, ha), 128 * 64));
        c = e.cc->EvalAdd(c, e.rot(e.cc->EvalMult(b, lb), -128 * 128));
        c = e.cc->EvalAdd(c, e.rot(e.cc->EvalMult(b, hb), -128 * 64));
        tok.push_back(c);
    }
    return tok;
}
// rotate every 128-token block by r (token t <- token (t + r) % 128), from hoisted rotations by 0..15
static CT inblock_rot(Env &e, const std::vector<CT> &baby, int r) {
    const int a = r / 16, b = r % 16;
    Vec lo(e.slots), hi(e.slots);
    for (uint32_t s = 0; s < e.slots; ++s) { const bool keep = int(s % T) < int(T) - r; lo[s] = keep; hi[s] = !keep; }
    CT x = baby[b]->Clone();                      // private copy: tasks share the hoisted rotations
    CT main = e.rot(x, 16 * a), wrap = e.rot(x, 16 * a - 128);
    Plaintext pl = e.pt(lo, main), ph = e.pt(hi, wrap);
    return e.cc->EvalAdd(e.cc->EvalMult(main, pl), e.cc->EvalMult(wrap, ph));
}
static std::vector<CT> hoisted(Env &e, const CT &c, int step) {
    std::vector<CT> v(16);
    auto digits = e.cc->EvalFastRotationPrecompute(c);
    v[0] = c;
    for (int b = 1; b < 16; ++b) v[b] = e.cc->EvalFastRotation(c, uint32_t(step * b), e.cc->GetCyclotomicOrder(), digits);
    return v;
}
static void fill_tok(Env &e, const Vec &X, int H, std::vector<Vec> &out) {    // [128, H] -> token layout
    out.assign(H / 256, Vec(e.slots));
    for (int a = 0; a < H / 256; ++a) for (int u = 0; u < 256; ++u) for (uint32_t t = 0; t < T; ++t)
        out[a][u * T + t] = X[t * H + 256 * a + u];
}

// scores = Q K^T per head (scaling by 1/sqrt(64) folds into the Q projection), output in the
// diagonal layout used by Softmax
static Result run_qk(Env &e, int heads) {
    const int H = heads * 64;
    Vec Q(size_t(T) * H), K(size_t(T) * H);
    for (auto &v : Q) v = rand_vec(1, -1, 1)[0];
    for (auto &v : K) v = rand_vec(1, -1, 1)[0];
    std::vector<Vec> q, k;
    fill_tok(e, Q, H, q); fill_tok(e, K, H, k);
    std::vector<Vec> ref(heads / 2, Vec(e.slots));
    for (int h = 0; h < heads; ++h) for (uint32_t t = 0; t < T; ++t) for (uint32_t r = 0; r < T; ++r) {
        double z = 0;
        const uint32_t s2 = (t + r) % T;
        for (int d = 0; d < 64; ++d) z += Q[t * H + 64 * h + d] * K[s2 * H + 64 * h + d];
        ref[h / 2][((h % 2) * T + r) * T + t] = z;
    }
    std::vector<CT> cq, ck;
    for (auto &v : q) cq.push_back(e.enc(v));
    for (auto &v : k) ck.push_back(e.enc(v));
    auto t0 = Clock::now();
    auto qa = to_att(e, cq), ka = to_att(e, ck);
    Vec first(e.slots, 0.0);
    for (uint32_t t = 0; t < T; ++t) { first[t] = 1; first[128 * T + t] = 1; }
    std::vector<CT> out(heads / 2);
    for (int i = 0; i < heads / 2; ++i) {
        auto kb = hoisted(e, ka[i], 1);
        std::vector<CT> diag(T);
#pragma omp parallel for schedule(dynamic) if (outer_parallel())
        for (int r = 0; r < int(T); ++r) {
            CT prod = e.cc->EvalMult(qa[i]->Clone(), inblock_rot(e, kb, r));
            for (int s = 1; s < 64; s <<= 1) prod = e.cc->EvalAdd(prod, e.rot(prod, 128 * s));   // sum over 64 dims
            Plaintext pf = e.pt(first, prod);
            diag[r] = e.cc->EvalMult(prod, pf);                                                  // heads at blocks 0, 128
        }
        // out = sum_r rot(diag_r, -128 r) = sum_a rot(sum_b rot(diag_{16a+b}, -128 b), -2048 a)
        std::vector<CT> part(8);
#pragma omp parallel for if (outer_parallel())
        for (int a = 0; a < 8; ++a) {
            CT acc = diag[16 * a]->Clone();
            for (int b = 1; b < 16; ++b) acc = e.cc->EvalAdd(acc, e.rot(diag[16 * a + b]->Clone(), -128 * b));
            part[a] = e.rot(acc, -2048 * a);
        }
        CT o = part[0];
        for (int a = 1; a < 8; ++a) o = e.cc->EvalAdd(o, part[a]);
        out[i] = o;
    }
    Result r{"Attention_QK_MatMul", "[3,1,128," + std::to_string(H) + "]", int(cq.size()), ms(t0)};
    r.err = max_err(e, out, ref);
    r.level_out = int(e.left(out[0]));
    r.levels_used = int(e.L) - r.level_out;
    r.approx = "diagonal method over 128 token offsets, 2 heads per ciphertext";
    return r;
}

// out = P V per head, P in the diagonal layout, V and the output in the token layout
static Result run_av(Env &e, int heads) {
    const int H = heads * 64;
    Vec V(size_t(T) * H);
    for (auto &v : V) v = rand_vec(1, -1, 1)[0];
    std::vector<Vec> pv(heads / 2, Vec(e.slots)), vv;
    std::vector<double> P(size_t(heads) * T * T);                    // P[h][t][s], rows sum to 1
    for (int h = 0; h < heads; ++h) for (uint32_t t = 0; t < T; ++t) {
        double z = 0;
        for (uint32_t s2 = 0; s2 < T; ++s2) z += (P[(size_t(h) * T + t) * T + s2] = rand_vec(1, 0, 1)[0]);
        for (uint32_t s2 = 0; s2 < T; ++s2) P[(size_t(h) * T + t) * T + s2] /= z;
    }
    for (int h = 0; h < heads; ++h) for (uint32_t t = 0; t < T; ++t) for (uint32_t r = 0; r < T; ++r)
        pv[h / 2][((h % 2) * T + r) * T + t] = P[(size_t(h) * T + t) * T + (t + r) % T];
    fill_tok(e, V, H, vv);
    Vec O(size_t(T) * H, 0.0);
    for (int h = 0; h < heads; ++h) for (uint32_t t = 0; t < T; ++t) for (int d = 0; d < 64; ++d) {
        double z = 0;
        for (uint32_t s2 = 0; s2 < T; ++s2) z += P[(size_t(h) * T + t) * T + s2] * V[s2 * H + 64 * h + d];
        O[t * H + 64 * h + d] = z;
    }
    std::vector<Vec> ref;
    fill_tok(e, O, H, ref);
    std::vector<CT> cp, cv;
    for (auto &v : pv) cp.push_back(e.enc(v));
    for (auto &v : vv) cv.push_back(e.enc(v));
    auto t0 = Clock::now();
    auto va = to_att(e, cv);
    Vec first(e.slots, 0.0);
    for (uint32_t t = 0; t < T; ++t) { first[t] = 1; first[128 * T + t] = 1; }
    std::vector<CT> att(heads / 2);
    for (int i = 0; i < heads / 2; ++i) {
        auto vb = hoisted(e, va[i], 1);
        auto pb = hoisted(e, cp[i], 128);
        std::vector<CT> terms(T);
#pragma omp parallel for schedule(dynamic) if (outer_parallel())
        for (int r = 0; r < int(T); ++r) {
            CT pr = e.rot(pb[r % 16]->Clone(), 2048 * (r / 16));        // diagonal r of both heads -> blocks 0, 128
            Plaintext pf = e.pt(first, pr);
            pr = e.cc->EvalMult(pr, pf);
            for (int s = 1; s < 64; s <<= 1) pr = e.cc->EvalAdd(pr, e.rot(pr, -128 * s));       // over the 64 dims
            terms[r] = e.cc->EvalMult(pr, inblock_rot(e, vb, r));
        }
        CT acc = terms[0];
        for (int r = 1; r < int(T); ++r) acc = e.cc->EvalAdd(acc, terms[r]);
        att[i] = acc;
    }
    auto out = from_att(e, att);
    Result r{"Attention_V_MatMul", "[1," + std::to_string(heads) + ",128,128]", int(cp.size()), ms(t0)};
    r.err = max_err(e, out, ref);
    r.level_out = int(e.left(out[0]));
    r.levels_used = int(e.L) - r.level_out;
    r.approx = "diagonal method over 128 token offsets, 2 heads per ciphertext";
    return r;
}

// ---------------------------------------------------------------- self-checks
//
//   keycheck   every rotation key, the relinearisation key and bootstrapping, against plaintext
//   racecheck  the operations the kernels run inside parallel tasks, run in parallel and compared
//              bit for bit with a serial run: they are deterministic, so any difference is a race
//              (or a memory fault)
// A result that does not decode on a machine that passes these checks points at the hardware:
// run a memory test (memtester, memtest86+).

static void keycheck(Env &e) {
    const Vec v = rand_vec(e.slots, -1, 1);
    const CT c = e.enc(v);
    const int s = int(e.slots);
    int bad = 0, n = 0;
    double worst = 0;
    std::string failed;
    auto t = Clock::now();
    auto check = [&](const std::string &what, const CT &r, const std::function<double(int)> &want, double tol) {
        double err = 0;
        try { Vec d = e.dec(r); for (int j = 0; j < s; ++j) err = std::max(err, std::fabs(d[j] - want(j))); }
        catch (const std::exception &) { err = -1; }
        ++n;
        if (err < 0 || err > tol) { ++bad; failed += (failed.empty() ? "" : " ") + what; }
        if (err > worst) worst = err;
    };
    for (int32_t k : rotation_indices())
        check("rot" + std::to_string(k), e.cc->EvalRotate(c, k), [&, k](int j) { return v[((j + k) % s + s) % s]; }, 1e-6);
    check("relin", e.cc->EvalMult(c, c), [&](int j) { return v[j] * v[j]; }, 1e-6);
    CT low = e.cc->Encrypt(e.kp.publicKey, e.cc->MakeCKKSPackedPlaintext(v, 1, e.depth - 1, nullptr, e.slots));
    check("bootstrap", e.precise ? e.cc->EvalBootstrap(low, 2, 13) : e.cc->EvalBootstrap(low), [&](int j) { return v[j]; }, 1e-2);
    std::cout << "{\"check\":\"keys\",\"keys\":" << n << ",\"failed\":" << bad << ",\"which\":\"" << failed
              << "\",\"worst_err_of_passing\":" << worst << ",\"ms\":" << ms(t) << "}" << std::endl;
}

// n independent bootstraps (fresh data and encryption each), each run twice on the same input:
// a failure that repeats on the same input comes from the input (or the keys); one that does
// not is nondeterminism inside bootstrapping
// What bootstrapping accepts: levels left on its input, and the amplitude of the values (plain
// bootstrapping, then scaled: x / s, bootstrap, * s, which needs one more level).
static void bootlevel(Env &e) {
    auto run = [&](const char *what, double amp, uint32_t left, double s) {
        Vec v = rand_vec(e.slots, -amp, amp);
        CT c = e.cc->Encrypt(e.kp.publicKey, e.cc->MakeCKKSPackedPlaintext(v, 1, e.depth - left, nullptr, e.slots));
        std::string status = "ok";
        double err = -1;
        try {
            CT x = s > 1 ? e.cc->EvalMult(c, 1.0 / s) : c;
            CT b = e.precise ? e.cc->EvalBootstrap(x, 2, 13) : e.cc->EvalBootstrap(x);
            if (s > 1) b = e.cc->EvalMult(b, s);
            Vec d = e.dec(b);
            err = 0;
            for (uint32_t j = 0; j < e.slots; ++j) err = std::max(err, std::fabs(d[j] - v[j]));
        } catch (const std::exception &ex) { status = std::string("exception: ") + ex.what(); }
        std::cout << "{\"check\":\"bootlevel\",\"case\":\"" << what << "\",\"amp\":" << amp << ",\"levels_left\":" << left
                  << ",\"scale\":" << s << ",\"status\":\"" << status.substr(0, 120) << "\",\"max_abs_err\":" << err << "}" << std::endl;
    };
    for (uint32_t left : {0u, 1u, 2u}) run("levels", 1.0, left, 1);
    // more levels left, values up to 18; and the product of a ciphertext with a constant (pending rescale)
    for (uint32_t left : {3u, 4u, 5u, 6u, 7u}) run("levels", 18.0, left, 1);
    // values repeated in every 128-slot block (as LayerNorm's per-token statistics): their energy sits
    // in few coefficients, which bootstrapping sees ~sqrt(256) times larger
    for (double amp : {0.5, 1.0, 2.0, 4.0, 8.0, 18.0}) {
        Vec base = rand_vec(T, -amp, amp), v(e.slots);
        for (uint32_t j = 0; j < e.slots; ++j) v[j] = base[j % T];
        CT c = e.cc->Encrypt(e.kp.publicKey, e.cc->MakeCKKSPackedPlaintext(v, 1, e.depth - 2, nullptr, e.slots));
        std::string status = "ok";
        double err = -1;
        try {
            Vec d = e.dec(e.precise ? e.cc->EvalBootstrap(c, 2, 13) : e.cc->EvalBootstrap(c));
            err = 0;
            for (uint32_t j = 0; j < e.slots; ++j) err = std::max(err, std::fabs(d[j] - v[j]));
        } catch (const std::exception &ex) { status = std::string("exception: ") + ex.what(); }
        std::cout << "{\"check\":\"bootlevel\",\"case\":\"repeated every 128 slots\",\"amp\":" << amp
                  << ",\"levels_left\":2,\"scale\":1,\"status\":\"" << status.substr(0, 120) << "\",\"max_abs_err\":" << err << "}" << std::endl;
    }
    // values in one 128-slot block, zeros elsewhere (as per-token statistics kept in block 0)
    for (double amp : {1.0, 4.0, 8.0, 16.0, 32.0}) {
        Vec v(e.slots, 0.0), blk = rand_vec(T, -amp, amp);
        for (uint32_t j = 0; j < T; ++j) v[j] = blk[j];
        CT c = e.cc->Encrypt(e.kp.publicKey, e.cc->MakeCKKSPackedPlaintext(v, 1, e.depth - 2, nullptr, e.slots));
        std::string status = "ok";
        double err = -1;
        try {
            Vec d = e.dec(e.precise ? e.cc->EvalBootstrap(c, 2, 13) : e.cc->EvalBootstrap(c));
            err = 0;
            for (uint32_t j = 0; j < e.slots; ++j) err = std::max(err, std::fabs(d[j] - v[j]));
        } catch (const std::exception &ex) { status = std::string("exception: ") + ex.what(); }
        std::cout << "{\"check\":\"bootlevel\",\"case\":\"one block, zeros elsewhere\",\"amp\":" << amp
                  << ",\"levels_left\":2,\"scale\":1,\"status\":\"" << status.substr(0, 120) << "\",\"max_abs_err\":" << err << "}" << std::endl;
    }
    // a constant in every slot but one block (as an iterate whose other blocks are unused)
    for (double cst : {0.0, 1.0, 2.9, 10.0}) {
        Vec v(e.slots, cst), blk = rand_vec(T, 1.5, 19.0);
        for (uint32_t j = 0; j < T; ++j) v[j] = blk[j];
        CT c = e.cc->Encrypt(e.kp.publicKey, e.cc->MakeCKKSPackedPlaintext(v, 1, e.depth - 2, nullptr, e.slots));
        std::string status = "ok";
        double err = -1;
        try {
            Vec d = e.dec(e.precise ? e.cc->EvalBootstrap(c, 2, 13) : e.cc->EvalBootstrap(c));
            err = 0;
            for (uint32_t j = 0; j < e.slots; ++j) err = std::max(err, std::fabs(d[j] - v[j]));
        } catch (const std::exception &ex) { status = std::string("exception: ") + ex.what(); }
        std::cout << "{\"check\":\"bootlevel\",\"case\":\"block 0 in [1.5, 19], constant elsewhere\",\"amp\":" << cst
                  << ",\"levels_left\":2,\"scale\":1,\"status\":\"" << status.substr(0, 120) << "\",\"max_abs_err\":" << err << "}" << std::endl;
    }
    for (uint32_t left : {2u, 3u, 5u, 7u}) {
        Vec v = rand_vec(e.slots, -18.0, 18.0);
        CT c = e.cc->Encrypt(e.kp.publicKey, e.cc->MakeCKKSPackedPlaintext(v, 1, e.depth - left, nullptr, e.slots));
        CT m = e.cc->EvalMult(c, 0.5);
        std::string status = "ok";
        double err = -1;
        try {
            Vec d = e.dec(e.precise ? e.cc->EvalBootstrap(m, 2, 13) : e.cc->EvalBootstrap(m));
            err = 0;
            for (uint32_t j = 0; j < e.slots; ++j) err = std::max(err, std::fabs(d[j] - 0.5 * v[j]));
        } catch (const std::exception &ex) { status = std::string("exception: ") + ex.what(); }
        std::cout << "{\"check\":\"bootlevel\",\"case\":\"after a constant product\",\"amp\":9,\"levels_left\":" << e.left(m)
                  << ",\"scale\":1,\"status\":\"" << status.substr(0, 120) << "\",\"max_abs_err\":" << err << "}" << std::endl;
    }
    for (double amp : {32.0, 60.0, 64.0, 100.0, 256.0, 1100.0}) run("amplitude", amp, 1, 1);
    for (double amp : {256.0, 1100.0}) run("scaled", amp, 2, amp <= 256 ? 8 : 32);
}

// bootstrapping of a vector with values in one 128-slot block and zeros elsewhere, nothing else
static void bootblock(Env &e, double amp, int n, uint32_t left) {
    for (int k = 0; k < n; ++k) {
        Vec v(e.slots, 0.0), blk = rand_vec(T, -amp, amp);
        for (uint32_t j = 0; j < T; ++j) v[j] = blk[j];
        CT c = e.cc->Encrypt(e.kp.publicKey, e.cc->MakeCKKSPackedPlaintext(v, 1, e.depth - left, nullptr, e.slots));
        std::string status = "ok";
        double err = -1;
        try {
            Vec d = e.dec(e.precise ? e.cc->EvalBootstrap(c, 2, 13) : e.cc->EvalBootstrap(c));
            err = 0;
            for (uint32_t j = 0; j < e.slots; ++j) err = std::max(err, std::fabs(d[j] - v[j]));
        } catch (const std::exception &ex) { status = std::string("exception: ") + ex.what(); }
        if (status != "ok") std::cerr << status << "\n";
        std::cout << "{\"check\":\"bootblock\",\"amp\":" << amp << ",\"levels_left\":" << left << ",\"status\":\""
                  << status.substr(0, 60) << "\",\"max_abs_err\":" << err << "}" << std::endl;
    }
}

static void bootcheck(Env &e, int n, double amp) {
    auto run = [&](const CT &in, const Vec &v, double &err, int &over, double &logerr) {
        auto t = Clock::now();
        CT b = e.precise ? e.cc->EvalBootstrap(in->Clone(), 2, 13) : e.cc->EvalBootstrap(in->Clone());
        const double took = ms(t);
        err = 0; over = 0; logerr = 0;
        try {
            Plaintext p;
            e.cc->Decrypt(e.kp.secretKey, b, &p);
            p->SetLength(e.slots);
            logerr = p->GetLogError();
            const auto d = p->GetRealPackedValue();
            for (uint32_t j = 0; j < e.slots; ++j) {
                const double x = std::fabs(d[j] - v[j]);
                err = std::max(err, x);
                over += x > 1e-2;
            }
        } catch (const std::exception &) { err = -1; }
        return std::make_pair(b, took);
    };
    for (int i = 0; i < n; ++i) {
        const Vec v = rand_vec(e.slots, -amp, amp);
        const CT low = e.cc->Encrypt(e.kp.publicKey, e.cc->MakeCKKSPackedPlaintext(v, 1, e.depth - 1, nullptr, e.slots));
        double err1, err2, le1, le2;
        int over1, over2;
        auto r1 = run(low, v, err1, over1, le1);
        auto r2 = run(low, v, err2, over2, le2);
        const bool same = r1.first->GetElements() == r2.first->GetElements();
        std::cout << "{\"check\":\"bootstrap\",\"i\":" << i << ",\"err\":[" << err1 << "," << err2 << "],\"slots_over_1e-2\":["
                  << over1 << "," << over2 << "],\"log_error\":[" << le1 << "," << le2 << "],\"identical\":"
                  << (same ? "true" : "false") << ",\"ms\":[" << r1.second << "," << r2.second << "]}" << std::endl;
    }
}

// a ciphertext's or plaintext's polynomials, for bit-exact comparison
struct Snap {
    std::vector<DCRTPoly> el;
    size_t level = 0, deg = 0;
    bool operator==(const Snap &o) const { return level == o.level && deg == o.deg && el == o.el; }
};
static Snap snap(const CT &c) { return {c->GetElements(), c->GetLevel(), c->GetNoiseScaleDeg()}; }
static Snap snap(const Plaintext &p) { return {{p->GetElement<DCRTPoly>()}, p->GetLevel(), p->GetNoiseScaleDeg()}; }

static void racecheck(Env &e, int rounds, const std::string &only) {
    const Vec v = rand_vec(e.slots, -1, 1);
    const CT c1 = e.enc(rand_vec(e.slots, -1, 1)), c2 = e.enc(rand_vec(e.slots, -1, 1));
    const CT d2 = e.cc->EvalMult(c1, c2);                       // pending rescale (noise degree 2)
    const CT low = e.cc->Encrypt(e.kp.publicKey, e.cc->MakeCKKSPackedPlaintext(v, 1, e.depth - 1, nullptr, e.slots));
    const uint32_t M = e.cc->GetCyclotomicOrder();
    auto digits = e.cc->EvalFastRotationPrecompute(c1);
    const auto kb = hoisted(e, c2, 1);                          // as in QK^T: rotations inside token blocks
    const auto pb = hoisted(e, c1, 128);                        // as in Attn*V and the linear layers
    Vec first(e.slots, 0.0);
    for (uint32_t t = 0; t < T; ++t) { first[t] = 1; first[128 * T + t] = 1; }
    auto diag = [&](int k, int j) { Vec d(e.slots); for (uint32_t i = 0; i < e.slots; ++i) d[i] = std::sin(1e-3 * i + k + 0.1 * j); return d; };
    struct Check { std::string name; int n; bool outer; bool ref_after; std::function<Snap(int)> f; };
    const std::vector<Check> checks = {
        {"encode", 64, true, false, [&](int) { return snap(e.pt(v, c1)); }},
        {"encode_every_level_first_use", int(e.depth) * 2, true, true, [&](int k) {
             return snap(e.cc->MakeCKKSPackedPlaintext(v, 1, uint32_t(k / 2), nullptr, e.slots)); }},
        {"mult_plain", 64, true, false, [&](int) { return snap(e.cc->EvalMult(c1->Clone(), e.pt(v, c1))); }},
        {"mult_plain_shared_input", 64, true, false, [&](int) { return snap(e.cc->EvalMult(c1, e.pt(v, c1))); }},
        {"mult_plain_pending_rescale", 64, true, false, [&](int) { return snap(e.cc->EvalMult(d2->Clone(), e.pt(v, d2))); }},
        {"add_pending_rescale", 64, true, false, [&](int) { return snap(e.cc->EvalAdd(c1->Clone(), d2->Clone())); }},
        {"rotate", 64, true, false, [&](int k) { return snap(e.rot(c1->Clone(), 128 * (1 + k % 15))); }},
        {"rotate_shared_input", 64, true, false, [&](int k) { return snap(e.rot(c1, 128 * (1 + k % 15))); }},
        {"rotate_pending_rescale", 64, true, false, [&](int k) { return snap(e.rot(d2->Clone(), 128 * (1 + k % 15))); }},
        {"fast_rotate_shared_digits", 64, true, false, [&](int k) {
             return snap(e.cc->EvalFastRotation(c1, 128 * (1 + k % 15), M, digits)); }},
        {"mult_cipher", 32, true, false, [&](int) { return snap(e.cc->EvalMult(c1->Clone(), c2->Clone())); }},
        {"linear_task", 32, true, false, [&](int k) {            // run_linear's task body
             CT acc;
             for (int j = 0; j < 16; ++j) {
                 CT x = pb[j]->Clone();
                 Plaintext pd = e.pt(diag(k, j), x);
                 CT prod = e.cc->EvalMult(x, pd);
                 acc = acc ? e.cc->EvalAdd(acc, prod) : prod;
             }
             return snap(acc); }},
        {"qk_task", 32, true, false, [&](int k) {                // run_qk's task body
             const int r = (k * 37) % int(T);
             CT prod = e.cc->EvalMult(c1->Clone(), inblock_rot(e, kb, r));
             for (int s = 1; s < 64; s <<= 1) prod = e.cc->EvalAdd(prod, e.rot(prod, 128 * s));
             Plaintext pf = e.pt(first, prod);
             return snap(e.cc->EvalMult(prod, pf)); }},
        {"av_task", 32, true, false, [&](int k) {                // run_av's task body
             const int r = (k * 37) % int(T);
             CT pr = e.rot(pb[r % 16]->Clone(), 2048 * (r / 16));
             Plaintext pf = e.pt(first, pr);
             pr = e.cc->EvalMult(pr, pf);
             for (int s = 1; s < 64; s <<= 1) pr = e.cc->EvalAdd(pr, e.rot(pr, -128 * s));
             return snap(e.cc->EvalMult(pr, inblock_rot(e, kb, r))); }},
        {"chebyshev_serial", 2, false, false, [&](int) {
             return snap(e.cc->EvalChebyshevFunction(gelu, c1, -8, 8, 27)); }},
        {"bootstrap_serial", 1, false, false, [&](int) {
             return snap(e.precise ? e.cc->EvalBootstrap(low->Clone(), 2, 13) : e.cc->EvalBootstrap(low->Clone())); }},
    };
    for (const auto &ck : checks) {
        if (!only.empty() && only.find(ck.name) == std::string::npos) continue;
        auto t = Clock::now();
        std::vector<Snap> ref(ck.n);
        auto reference = [&] { for (int k = 0; k < ck.n; ++k) ref[k] = ck.f(k); };
        if (!ck.ref_after) reference();
        int bad = 0, runs = 0;
        for (int round = 0; round < rounds; ++round) {
            std::vector<Snap> out(ck.n);
#pragma omp parallel for schedule(dynamic) if (ck.outer)
            for (int k = 0; k < ck.n; ++k) out[k] = ck.f(k);
            if (ck.ref_after && round == 0) reference();
            for (int k = 0; k < ck.n; ++k) { ++runs; if (!(out[k] == ref[k])) ++bad; }
        }
        std::cout << "{\"check\":\"" << ck.name << "\",\"outer_parallel\":" << (ck.outer ? "true" : "false")
                  << ",\"runs\":" << runs << ",\"mismatches\":" << bad << ",\"ms\":" << ms(t) << "}" << std::endl;
    }
}

// ---------------------------------------------------------------- driver

static std::map<std::string, double> kv(const std::string &spec) {
    std::map<std::string, double> m;
    auto c = spec.find(':');
    if (c == std::string::npos) return m;
    std::stringstream ss(spec.substr(c + 1));
    std::string item;
    while (std::getline(ss, item, ',')) {
        auto eq = item.find('=');
        if (eq == std::string::npos) continue;
        try { m[item.substr(0, eq)] = std::stod(item.substr(eq + 1)); } catch (const std::exception &) {}   // text values: see text()
    }
    return m;
}

static std::string text(const std::string &spec, const std::string &key) {      // e.g. only=qk_task+av_task
    auto c = spec.find(key + "=");
    if (c == std::string::npos) return "";
    c += key.size() + 1;
    return spec.substr(c, spec.find(',', c) == std::string::npos ? std::string::npos : spec.find(',', c) - c);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: he_ops <fast|precise> <op>[:k=v,...] ...\n";
        return 2;
    }
    bool precise = std::string(argv[1]) == "precise";
    auto t = Clock::now();
    Env e = make_env(precise);
    std::cerr << "setup " << ms(t) / 1e3 << " s, depth " << e.depth << ", levels after bootstrap " << e.L
              << ", threads " << omp_get_max_threads() << ", openfhe " << GetOPENFHEVersion() << "\n";
    for (int a = 2; a < argc; ++a) {
        std::string spec = argv[a], op = spec.substr(0, spec.find(':'));
        auto p = kv(spec);
        auto get = [&](const char *k, double d) { return p.count(k) ? p[k] : d; };
        Result r;
        if (op == "keycheck") { keycheck(e); continue; }
        if (op == "bootlevel") { bootlevel(e); continue; }
        if (op == "bootblock") { bootblock(e, get("amp", 1.0), int(get("n", 3)), uint32_t(get("left", 2))); continue; }
        if (op == "bootcheck") { bootcheck(e, int(get("n", 4)), get("amp", 1.0)); continue; }
        if (op == "racecheck") { racecheck(e, int(get("rounds", 3)), text(spec, "only")); continue; }
        if (op == "add") r = run_add(e, int(get("H", 768)));
        else if (op == "gelu") r = run_gelu(e, int(get("F", 3072)), get("bound", 8), int(get("degree", 59)));
        else if (op == "gelu_wide") r = run_gelu_wide(e, int(get("F", 3072)), int(get("B", 64)));
        else if (op == "softmax_wide") r = run_softmax_wide(e, int(get("heads", 12)), get("lo", -28), get("hi", 29),
                                                            get("rmin", 0), int(get("newton", 3)), get("debug", 0) > 0);
        else if (op == "layernorm_wide") r = run_layernorm_wide(e, int(get("H", 768)), get("vlo", 0.024), get("vhi", 2.1),
                                                                get("eps", 1e-12), get("debug", 0) > 0);
        else if (op == "layernorm") r = run_layernorm(e, int(get("H", 768)), get("vlo", 0.25), get("vhi", 25), int(get("degree", 59)));
        else if (op == "microbench") {               // per-operation costs at the post-bootstrap level
            CT c = e.enc(rand_vec(e.slots, -1, 1));
            Vec v = rand_vec(e.slots, -1, 1);
            const int k = 40;
            auto t1 = Clock::now(); Plaintext pp; for (int i = 0; i < k; ++i) pp = e.pt(v, c); double enc = ms(t1) / k;
            t1 = Clock::now(); CT pr; for (int i = 0; i < k; ++i) pr = e.cc->EvalMult(c, pp); double mul = ms(t1) / k;
            t1 = Clock::now(); for (int i = 0; i < k; ++i) pr = e.cc->EvalAdd(pr, c); double add = ms(t1) / k;
            t1 = Clock::now(); for (int i = 0; i < k / 4; ++i) pr = e.rot(c, 128); double rotm = ms(t1) / (k / 4);
            t1 = Clock::now(); CT sq; for (int i = 0; i < k / 4; ++i) sq = e.cc->EvalMult(c, c); double ctmul = ms(t1) / (k / 4);
            Vec par_enc(32);
            t1 = Clock::now();
#pragma omp parallel for
            for (int i = 0; i < 64; ++i) { Plaintext q = e.pt(v, c); CT z = e.cc->EvalMult(c, q); (void)z; }
            double par = ms(t1) / 64;
            auto digits = e.cc->EvalFastRotationPrecompute(c);
            CT fr = e.cc->EvalFastRotation(c, 128, e.cc->GetCyclotomicOrder(), digits);
            t1 = Clock::now(); for (int i = 0; i < k; ++i) pr = e.cc->EvalMult(fr, pp); double mul_fr = ms(t1) / k;
            std::cerr << "fresh ct: format " << int(c->GetElements()[0].GetFormat()) << ", towers " << c->GetElements()[0].GetNumOfElements()
                      << ", noiseScaleDeg " << c->GetNoiseScaleDeg() << "; fast-rotated: format " << int(fr->GetElements()[0].GetFormat())
                      << ", towers " << fr->GetElements()[0].GetNumOfElements() << ", noiseScaleDeg " << fr->GetNoiseScaleDeg() << "\n";
            std::cerr << "ct-pt mult on fast-rotated ct: " << mul_fr << " ms\n";
            std::cout << "{\"op\":\"microbench\",\"limbs_left\":" << e.left(c) << ",\"encode_ms\":" << enc
                      << ",\"ct_pt_mult_ms\":" << mul << ",\"add_ms\":" << add << ",\"rotate_ms\":" << rotm
                      << ",\"ct_ct_mult_ms\":" << ctmul << ",\"parallel_encode_mult_ms_per_op\":" << par << "}" << std::endl;
            continue;
        }
        else if (op == "bootstrap") {                // one ciphertext, from 1 level left (the cheapest input)
            const double amp = get("amp", 1.0);
            Vec v = rand_vec(e.slots, -amp, amp);         // random (not smooth) values: the hard case
            CT c = e.cc->Encrypt(e.kp.publicKey, e.cc->MakeCKKSPackedPlaintext(v, 1, e.depth - 1, nullptr, e.slots));
            e.start_levels = 1; e.first_boot_used = 0;
            double best = 1e300; CT b;
            for (int k = 0; k < int(get("repeats", 2)); ++k) { auto t1 = Clock::now(); b = e.refresh(c); best = std::min(best, ms(t1)); }
            r = Result{"Bootstrap", "[32768]", 1, best};
            try { r.err = max_err(e, {b}, {v}); } catch (const std::exception &) { r.err = -1; }   // -1: not decodable
            r.approx += ", random values in [-" + std::to_string(amp) + "," + std::to_string(amp) + "]";
            r.level_out = int(e.left(b));
            r.approx = std::string(precise ? "two passes (iterative)" : "one pass");
        }
        else if (op == "qk") r = run_qk(e, int(get("heads", 12)));
        else if (op == "av") r = run_av(e, int(get("heads", 12)));
        else if (op == "out_proj") r = run_linear(e, "Out_Projection", 768, 768);
        else if (op == "qkv") {                      // Q, K, V in the token layout: 3 x H/256 ciphertexts
            const int H = int(get("H", 768));
            r = run_linear(e, "Linear_QKV", H, 3 * H);
            r.shape = "[1,128," + std::to_string(H) + "]";
        }
        else if (op == "ffn1") r = run_linear(e, "FFN_Linear_1", int(get("H", 768)), int(get("F", 3072)));
        else if (op == "ffn2") r = run_linear(e, "FFN_Linear_2", int(get("F", 3072)), int(get("H", 768)));
        else if (op == "softmax") r = run_softmax(e, int(get("heads", 12)), get("bound", 6), int(get("exp_degree", e.L >= 8 ? 27 : 13)),
                                                 get("umin", 0.02), int(get("inv_degree", 27)));
        else { std::cerr << "unknown op " << op << "\n"; return 2; }
        if (op == "add" || op == "gelu") r.levels_used = int(e.L) - r.level_out;
        std::cout << "{\"op\":\"" << r.op << "\",\"shape\":" << r.shape << ",\"mode\":\"" << argv[1]
                  << "\",\"ciphertexts\":" << r.cts << ",\"ms\":" << r.total_ms << ",\"levels_after_bootstrap\":" << e.L
                  << ",\"levels_used\":" << r.levels_used << ",\"level_out\":" << r.level_out
                  << ",\"internal_bootstraps\":" << r.boots << ",\"bootstrap_ms\":" << r.boot_ms
                  << ",\"max_abs_err\":" << r.err << ",\"approx\":\"" << r.approx << "\"}" << std::endl;
    }
    return 0;
}
