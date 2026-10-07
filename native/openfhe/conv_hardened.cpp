// Hardened HE <-> MPC conversion at the OpenFHE backend's parameters (same as he_ops.cpp).
//
//   conv_hardened <numel> [repeats] [ell] [frac] [levels] [bound_bits] [kappa] [log2_values] [delta] [degree]
//
// HE -> MPC. The server adds to every complex slot (real and imaginary part) a mask drawn
// uniformly from a range 2^M times larger than the values, M = bound_bits + log2_values + kappa,
// so that the client's view of all (value + noise) is within statistical distance 2^-kappa of
// uniform over every conversion; re-randomises with a fresh encryption of zero; keeps the towers
// the masked values need and sends. The client decrypts exactly (CRT to big integers) and
// decodes in quadruple precision (doubles cannot hold M + frac bits), and rounds the real parts
// to its share in Z_2^ell; the server's share is minus the integer part of its mask. The real
// mask also carries a uniform offset within +-1/2 fixed-point step, so the rounding does not reveal
// the low bits of the noise. Needs (M + log2(scale) + 3 - 57) / 50 levels left on the ciphertext.
//
// MPC -> HE. Shares in Z_2^ell are uniform: adding them in HE would leave a wrap of 2^ell. Both
// parties reduce their shares modulo 2^ell' (ell' = bound_bits + frac + delta, exact for the
// value); the client encrypts c' / 2^ell' (in [0, 1)) with enough levels; the server adds
// s' / 2^ell', so t = X / 2^ell' + w with w in {0, 1, 2} and |X / 2^ell'| < 2^-delta, and removes
// w in two steps: y = sin(2 pi t) by Chebyshev interpolation on [0, 2] (two periods, low degree),
// then g = arcsin(y) / (2 pi) by its odd series to y^7 (exact near integers up to (2 pi 2^-delta)^9);
// and scales g by 2^(ell' - frac). Defaults: bound_bits 12 (fine-tuned BERT-base reaches |x| ~ 1060 in
// its residual stream), delta 6 (series error ~1e-6, CKKS noise amplified by 2^(bound_bits + delta)).
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

#include <quadmath.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;
using f128 = __float128;
using i128 = __int128;
static double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// ---------------------------------------------------------------- quadruple-precision CKKS FFT
// Port of OpenFHE's DiscreteFourierTransform::FFTSpecial / FFTSpecialInv (same slot order).
struct cq {
    f128 re = 0, im = 0;
};
static inline cq operator+(cq a, cq b) { return {a.re + b.re, a.im + b.im}; }
static inline cq operator-(cq a, cq b) { return {a.re - b.re, a.im - b.im}; }
static inline cq operator*(cq a, cq b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }

struct SpecialFFT {
    uint32_t M = 0, n = 0;
    std::vector<uint32_t> rot;
    std::vector<cq> ksi;
    explicit SpecialFFT(uint32_t ringDim) : M(2 * ringDim), n(ringDim / 2) {
        rot.resize(n);
        uint64_t p = 1;
        for (uint32_t i = 0; i < n; ++i) { rot[i] = uint32_t(p); p = (p * 5) % M; }
        ksi.resize(M + 1);
        for (uint32_t j = 0; j <= M; ++j) {
            f128 a = 2 * M_PIq * f128(j) / f128(M);
            ksi[j] = {cosq(a), sinq(a)};
        }
    }
    static void bitReverse(std::vector<cq> &v) {
        size_t size = v.size();
        for (size_t i = 1, j = 0; i < size; ++i) {
            size_t bit = size >> 1;
            for (; j >= bit; bit >>= 1) j -= bit;
            j += bit;
            if (i < j) std::swap(v[i], v[j]);
        }
    }
    void inv(std::vector<cq> &vals) const {      // slots -> "complex coefficients"
        const size_t size = vals.size();
        for (size_t len = size; len >= 1; len >>= 1) {
            for (size_t i = 0; i < size; i += len) {
                size_t lenh = len >> 1, lenq = len << 2, gap = M / lenq;
                for (size_t j = 0; j < lenh; ++j) {
                    size_t idx = (lenq - (rot[j] % lenq)) * gap;
                    cq u = vals[i + j] + vals[i + j + lenh];
                    cq v = (vals[i + j] - vals[i + j + lenh]) * ksi[idx];
                    vals[i + j] = u;
                    vals[i + j + lenh] = v;
                }
            }
        }
        bitReverse(vals);
        for (auto &v : vals) { v.re /= f128(size); v.im /= f128(size); }
    }
    void fwd(std::vector<cq> &vals) const {      // "complex coefficients" -> slots
        bitReverse(vals);
        const size_t size = vals.size();
        for (size_t len = 2; len <= size; len <<= 1) {
            size_t lenh = len >> 1, lenq = len << 2, gap = M / lenq;
            for (size_t i = 0; i < size; i += len) {
                for (size_t j = 0; j < lenh; ++j) {
                    size_t idx = (rot[j] % lenq) * gap;
                    cq u = vals[i + j], v = vals[i + j + lenh] * ksi[idx];
                    vals[i + j] = u + v;
                    vals[i + j + lenh] = u - v;
                }
            }
        }
    }
    // slots -> integer coefficients at `scale` (OpenFHE layout: real parts, then imaginary parts)
    std::vector<i128> encode(std::vector<cq> z, f128 scale) const {
        inv(z);
        std::vector<i128> c(2 * n);
        for (uint32_t i = 0; i < n; ++i) {
            c[i] = i128(roundq(z[i].re * scale));
            c[i + n] = i128(roundq(z[i].im * scale));
        }
        return c;
    }
    std::vector<cq> decode(const std::vector<i128> &c, f128 scale) const {
        std::vector<cq> w(n);
        for (uint32_t i = 0; i < n; ++i) w[i] = {f128(c[i]) / scale, f128(c[i + n]) / scale};
        fwd(w);
        return w;
    }
};

// ---------------------------------------------------------------- exact polynomial access

static i128 parse_i128(const std::string &s) {
    i128 r = 0;
    for (char ch : s) r = r * 10 + (ch - '0');
    return r;
}

// m = c0 + c1 * s over the ciphertext's towers, as centred big integers (must fit 127 bits)
static std::vector<i128> decrypt_exact(const PrivateKey<DCRTPoly> &sk, const Ciphertext<DCRTPoly> &ct) {
    const auto &el = ct->GetElements();
    DCRTPoly s = sk->GetPrivateElement();
    s.DropLastElements(s.GetNumOfElements() - el[0].GetNumOfElements());
    DCRTPoly m = el[0] + el[1] * s;
    m.SetFormat(Format::COEFFICIENT);
    auto big = m.CRTInterpolate();
    const BigInteger Q = big.GetModulus(), half = Q >> 1;
    std::vector<i128> out(big.GetLength());
    for (size_t i = 0; i < out.size(); ++i) {
        BigInteger v = big[i];
        const bool neg = v > half;
        if (neg) v = Q - v;
        const i128 r = parse_i128(v.ToString());
        out[i] = neg ? -r : r;
    }
    return out;
}

static DCRTPoly from_coeffs(const std::vector<i128> &c, const std::shared_ptr<DCRTPoly::Params> &params) {
    DCRTPoly out(params, Format::COEFFICIENT, true);
    for (size_t t = 0; t < out.GetNumOfElements(); ++t) {
        const auto &tp = params->GetParams()[t];
        const uint64_t q = tp->GetModulus().ConvertToInt<uint64_t>();
        NativeVector v(c.size(), tp->GetModulus());
        for (size_t i = 0; i < c.size(); ++i) {
            i128 r = c[i] % i128(q);
            if (r < 0) r += q;
            v[i] = NativeInteger(uint64_t(r));
        }
        NativePoly p(tp, Format::COEFFICIENT, true);
        p.SetValues(std::move(v), Format::COEFFICIENT);
        out.SetElementAtIndex(t, std::move(p));
    }
    out.SetFormat(Format::EVALUATION);
    return out;
}

static f128 scale_of(const Ciphertext<DCRTPoly> &ct) {
    return powq(f128(ct->GetScalingFactor()), f128(ct->GetNoiseScaleDeg()));
}

// ---------------------------------------------------------------- driver

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: conv_hardened <numel> [repeats] [ell] [frac] [levels] [bound_bits] [kappa] "
                     "[log2_values] [delta] [degree]\n";
        return 2;
    }
    const size_t numel = std::stoull(argv[1]);
    const int repeats = argc > 2 ? std::stoi(argv[2]) : 2;
    const int ell = argc > 3 ? std::stoi(argv[3]) : 37, frac = argc > 4 ? std::stoi(argv[4]) : 12;
    const uint32_t L = argc > 5 ? std::stoi(argv[5]) : 8;
    const int bbits = argc > 6 ? std::stoi(argv[6]) : 12, kappa = argc > 7 ? std::stoi(argv[7]) : 40;
    const int logn = argc > 8 ? std::stoi(argv[8]) : 22, delta = argc > 9 ? std::stoi(argv[9]) : 6;
    const uint32_t degree = argc > 10 ? std::stoi(argv[10]) : 31;
    const int Mbits = bbits + logn + kappa;

    std::vector<uint32_t> budget = {3, 3};
    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecretKeyDist(SPARSE_ENCAPSULATED);
    p.SetSecurityLevel(HEStd_128_classic);
    p.SetRingDim(1 << 16);
    p.SetScalingModSize(50);
    p.SetFirstModSize(57);
    p.SetScalingTechnique(FLEXIBLEAUTO);
    const uint32_t depth = 8 + FHECKKSRNS::GetBootstrapDepth(budget, SPARSE_ENCAPSULATED);
    p.SetMultiplicativeDepth(depth);
    auto cc = GenCryptoContext(p);
    for (auto f : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE}) cc->Enable(f);
    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);
    const uint32_t N = cc->GetRingDimension(), slots = N / 2;
    const size_t cts = (numel + slots - 1) / slots;
    SpecialFFT fft(N);
    std::mt19937_64 rng(7);
    const uint64_t mask = ell == 64 ? ~uint64_t(0) : ((uint64_t(1) << ell) - 1);

    // -------- 0. the quadruple-precision codec reproduces OpenFHE's
    {
        std::uniform_real_distribution<double> d(-4, 4);
        std::vector<double> v(slots);
        for (auto &x : v) x = d(rng);
        auto pt = cc->MakeCKKSPackedPlaintext(v, 1, depth - L, nullptr, slots);
        auto ct = cc->Encrypt(kp.publicKey, pt);
        auto z = fft.decode(decrypt_exact(kp.secretKey, ct), scale_of(ct));
        double e = 0;
        for (uint32_t i = 0; i < slots; ++i) e = std::max(e, std::fabs(double(z[i].re) - v[i]));
        std::vector<cq> zz(slots);
        for (uint32_t i = 0; i < slots; ++i) zz[i] = {f128(v[i]), 0};
        auto c = fft.encode(zz, scale_of(ct));
        ct->GetElements()[0] = ct->GetElements()[0] + from_coeffs(c, ct->GetElements()[0].GetParams());  // + v again
        Plaintext out;
        cc->Decrypt(kp.secretKey, ct, &out);
        out->SetLength(slots);
        auto w = out->GetRealPackedValue();
        double e2 = 0;
        for (uint32_t i = 0; i < slots; ++i) e2 = std::max(e2, std::fabs(w[i] - 2 * v[i]));
        std::cerr << "codec check: exact decode vs values " << e << ", quad encode added via OpenFHE decode " << e2 << "\n";
        if (e > 1e-6 || e2 > 1e-6) { std::cerr << "codec mismatch\n"; return 1; }
    }

    // -------- 1. HE -> MPC
    // keep the towers the masked values need: log2 Q >= M + log2(scale) + 3
    const double logScale = 50.0;
    int towers = 1;
    while (57.0 + 50.0 * (towers - 1) < Mbits + frac * 0 + logScale + 3) ++towers;
    if (uint32_t(towers - 1) > L) { std::cerr << "not enough levels for the mask\n"; return 1; }
    std::uniform_real_distribution<double> xd(-std::ldexp(1.0, bbits) * 0.9, std::ldexp(1.0, bbits) * 0.9);
    std::vector<std::vector<double>> x(cts, std::vector<double>(slots, 0.0));
    for (size_t c = 0; c < cts; ++c) for (size_t i = 0; i < slots && c * slots + i < numel; ++i) x[c][i] = xd(rng);
    std::vector<Ciphertext<DCRTPoly>> cx(cts);
    for (size_t c = 0; c < cts; ++c)
        cx[c] = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(x[c], 1, depth - L, nullptr, slots));

    auto rand_i128 = [&](int bits) {
        i128 r = (i128(rng()) << 64) | i128(rng());
        if (r < 0) r = -r;
        return bits >= 127 ? r : (r & ((i128(1) << bits) - 1));
    };
    auto rand_unit = [&]() { return f128(rng() >> 11) / f128(uint64_t(1) << 53); };   // [0, 1)

    double h_srv = 1e300, h_cli = 1e300, h_err = 0;
    size_t h_bytes = 0;
    std::vector<std::vector<uint64_t>> share_c(cts, std::vector<uint64_t>(slots)), share_s(cts, std::vector<uint64_t>(slots));
    for (int rep = 0; rep < repeats; ++rep) {
        std::stringstream wire;
        auto t = Clock::now();
        for (size_t c = 0; c < cts; ++c) {
            auto ct = cc->Compress(cx[c], towers);
            std::vector<cq> r(slots);
            for (uint32_t i = 0; i < slots; ++i) {
                const i128 Rint = rand_i128(Mbits + frac);
                share_s[c][i] = uint64_t(-Rint) & mask;
                r[i].re = (f128(Rint) + rand_unit() - f128(0.5)) / f128(uint64_t(1) << frac);   // +- half a step
                r[i].im = (2 * rand_unit() - 1) * powq(2, Mbits);
            }
            auto &el = ct->GetElements();
            DCRTPoly mpoly = from_coeffs(fft.encode(r, scale_of(ct)), el[0].GetParams());
            auto zero = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, 0.0), 1,
                                                                               ct->GetLevel(), nullptr, slots));
            auto &ez = zero->GetElements();
            if (ez[0].GetNumOfElements() != el[0].GetNumOfElements()) { std::cerr << "tower mismatch\n"; return 1; }
            el[0] = el[0] + mpoly + ez[0];
            el[1] = el[1] + ez[1];
            Serial::Serialize(ct, wire, SerType::BINARY);
        }
        const double ts = ms(t);
        const size_t hb = wire.str().size();
        t = Clock::now();
        for (size_t c = 0; c < cts; ++c) {
            Ciphertext<DCRTPoly> ct;
            Serial::Deserialize(ct, wire, SerType::BINARY);
            auto z = fft.decode(decrypt_exact(kp.secretKey, ct), scale_of(ct));
            for (uint32_t i = 0; i < slots; ++i)
                share_c[c][i] = uint64_t(i128(roundq(z[i].re * f128(uint64_t(1) << frac)))) & mask;
        }
        const double tc = ms(t);
        double err = 0;
        for (size_t c = 0; c < cts; ++c) for (size_t i = 0; i < slots && c * slots + i < numel; ++i) {
            uint64_t u = (share_c[c][i] + share_s[c][i]) & mask;
            int64_t sgn = ell == 64 ? int64_t(u) : ((u >> (ell - 1)) & 1 ? int64_t(u) - (int64_t(1) << ell) : int64_t(u));
            err = std::max(err, std::fabs(double(sgn) / std::ldexp(1.0, frac) - x[c][i]));
        }
        h_srv = std::min(h_srv, ts); h_cli = std::min(h_cli, tc); h_bytes = hb; h_err = err;
    }
    std::cout << "{\"direction\":\"HE->MPC\",\"protocol\":\"hardened\",\"numel\":" << numel << ",\"ciphertexts\":" << cts
              << ",\"server_ms\":" << h_srv << ",\"client_ms\":" << h_cli << ",\"bytes\":" << h_bytes
              << ",\"rounds\":1,\"max_abs_err\":" << h_err << ",\"towers_sent\":" << towers
              << ",\"min_levels_left\":" << towers - 1 << ",\"mask_bits\":" << Mbits << ",\"kappa\":" << kappa
              << ",\"bound_bits\":" << bbits << ",\"log2_values\":" << logn << "}" << std::endl;

    // -------- 2. MPC -> HE: uniform shares of X = x * 2^frac in Z_2^ell
    const int ellp = bbits + frac + delta;
    const uint64_t maskp = (uint64_t(1) << ellp) - 1;
    std::vector<std::vector<uint64_t>> cs(cts, std::vector<uint64_t>(slots)), ss(cts, std::vector<uint64_t>(slots));
    for (size_t c = 0; c < cts; ++c) for (uint32_t i = 0; i < slots; ++i) {
        const uint64_t X = uint64_t(int64_t(std::llround(x[c][i] * std::ldexp(1.0, frac)))) & mask;
        cs[c][i] = rng() & mask;
        ss[c][i] = (X - cs[c][i]) & mask;
    }
    const double amp = std::ldexp(1.0, ellp - frac);          // g -> x
    // arcsin(y) / (2 pi) = (y + y^3/6 + 3 y^5/40 + 5 y^7/112) / (2 pi), times 2^(ell' - frac)
    const double k = amp / (2 * M_PI);
    const std::vector<double> series = {0, k, 0, k / 6, 0, k * 3 / 40, 0, k * 5 / 112};
    auto sine = [](double t) { return std::sin(2 * M_PI * t); };
    auto wrap_removal = [&](Ciphertext<DCRTPoly> c) {
        c = cc->EvalChebyshevFunction(sine, c, 0.0, 2.0, degree);
        return cc->EvalPoly(c, series);
    };
    uint32_t extra = 1;                                        // levels the wrap removal uses (found below)
    {
        auto probe = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, 0.5), 1,
                                                                            depth - L - 14, nullptr, slots));
        auto o = wrap_removal(probe);
        extra = (o->GetLevel() + (o->GetNoiseScaleDeg() - 1)) - probe->GetLevel() + 1;   // + one to spare
    }
    double m_cli = 1e300, m_srv = 1e300, m_err = 0;
    size_t m_bytes = 0;
    uint32_t m_left = 0;
    double m_rms = 0;
    for (int rep = 0; rep < repeats; ++rep) {
        std::stringstream wire;
        auto t = Clock::now();
        for (size_t c = 0; c < cts; ++c) {
            std::vector<double> v(slots);
            for (uint32_t i = 0; i < slots; ++i) v[i] = double(cs[c][i] & maskp) / std::ldexp(1.0, ellp);
            Serial::Serialize(cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(v, 1, depth - L - extra, nullptr, slots)),
                              wire, SerType::BINARY);
        }
        const double tc = ms(t);
        const size_t mb = wire.str().size();
        t = Clock::now();
        std::vector<Ciphertext<DCRTPoly>> out(cts);
        for (size_t c = 0; c < cts; ++c) {
            Serial::Deserialize(out[c], wire, SerType::BINARY);
            std::vector<double> v(slots);
            for (uint32_t i = 0; i < slots; ++i) v[i] = double(ss[c][i] & maskp) / std::ldexp(1.0, ellp);
            Plaintext ps = cc->MakeCKKSPackedPlaintext(v, 1, out[c]->GetLevel(), nullptr, slots);
            out[c] = cc->EvalAdd(out[c], ps);
            out[c] = wrap_removal(out[c]);
        }
        const double ts = ms(t);
        double err = 0, sq = 0;
        size_t cnt = 0;
        for (size_t c = 0; c < cts; ++c) {
            Plaintext pt;
            cc->Decrypt(kp.secretKey, out[c], &pt);
            pt->SetLength(slots);
            auto v = pt->GetRealPackedValue();
            for (size_t i = 0; i < slots && c * slots + i < numel; ++i) {
                err = std::max(err, std::fabs(v[i] - x[c][i]));
                // the conversion's own noise: against the fixed-point value the shares hold
                const double d = v[i] - std::ldexp(double(std::llround(x[c][i] * std::ldexp(1.0, frac))), -frac);
                sq += d * d; ++cnt;
            }
        }
        m_rms = std::sqrt(sq / double(cnt));
        m_left = depth - out[0]->GetLevel() - (out[0]->GetNoiseScaleDeg() - 1);
        m_cli = std::min(m_cli, tc); m_srv = std::min(m_srv, ts); m_bytes = mb; m_err = err;
    }
    std::cout << "{\"direction\":\"MPC->HE\",\"protocol\":\"hardened\",\"numel\":" << numel << ",\"ciphertexts\":" << cts
              << ",\"server_ms\":" << m_srv << ",\"client_ms\":" << m_cli << ",\"bytes\":" << m_bytes
              << ",\"rounds\":1,\"max_abs_err\":" << m_err << ",\"noise_rms\":" << m_rms << ",\"levels_left\":" << m_left << ",\"wrap_levels\":" << extra
              << ",\"reduced_ring_bits\":" << ellp << ",\"delta\":" << delta << ",\"degree\":" << degree << "}" << std::endl;
    return 0;
}
