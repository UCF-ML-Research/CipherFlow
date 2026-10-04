// HE <-> MPC conversion at the OpenFHE backend's CKKS parameters (same as he_ops.cpp).
//
//   conv_bench <numel> [repeats] [ell] [frac] [levels_left]
//
// Same message flow as native/conversion (SEAL): HE->MPC masks the ciphertext, re-randomises it
// with a fresh encryption of zero, drops it to the last level and sends it; the client decrypts
// to its share. MPC->HE: the client encrypts its share at the post-bootstrap level and sends it;
// the server adds its share. The mask is added in the slot domain: the costs (encode, encrypt,
// serialise, decrypt) are those of the protocol, but a hardened version masks in the
// coefficient domain with noise flooding.
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int main(int argc, char **argv) {
    if (argc < 2) { std::cerr << "usage: conv_bench <numel> [repeats] [ell] [frac] [levels_left]\n"; return 2; }
    const size_t numel = std::stoull(argv[1]);
    const int repeats = argc > 2 ? std::stoi(argv[2]) : 3;
    const int ell = argc > 3 ? std::stoi(argv[3]) : 37, frac = argc > 4 ? std::stoi(argv[4]) : 12;
    const uint32_t L = argc > 5 ? std::stoi(argv[5]) : 8;

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
    for (auto f : {PKE, KEYSWITCH, LEVELEDSHE}) cc->Enable(f);
    auto kp = cc->KeyGen();
    const size_t slots = cc->GetRingDimension() / 2, cts = (numel + slots - 1) / slots;

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> xd(-4, 4), rd(-64, 64);
    const uint64_t mask = ell == 64 ? ~uint64_t(0) : ((uint64_t(1) << ell) - 1);
    auto fx = [&](double v) { return uint64_t(int64_t(std::llround(v * std::ldexp(1.0, frac)))) & mask; };
    auto unfx = [&](uint64_t u) {
        int64_t s = ell == 64 ? int64_t(u) : ((u >> (ell - 1)) & 1 ? int64_t(u) - (int64_t(1) << ell) : int64_t(u));
        return double(s) / std::ldexp(1.0, frac);
    };
    std::vector<std::vector<double>> x(cts, std::vector<double>(slots, 0.0));
    for (size_t c = 0; c < cts; ++c) for (size_t i = 0; i < slots && c * slots + i < numel; ++i) x[c][i] = xd(rng);
    std::vector<Ciphertext<DCRTPoly>> cx(cts);
    for (size_t c = 0; c < cts; ++c)
        cx[c] = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(x[c], 1, depth - L, nullptr, slots));

    double h_srv = 1e300, h_cli = 1e300, m_cli = 1e300, m_srv = 1e300, h_err = 0, m_err = 0;
    size_t h_bytes = 0, m_bytes = 0;
    for (int rep = 0; rep < repeats; ++rep) {
        // ---------------- HE -> MPC
        std::stringstream wire;
        std::vector<std::vector<uint64_t>> ss(cts, std::vector<uint64_t>(slots)), sc(cts, std::vector<uint64_t>(slots));
        auto t = Clock::now();
        for (size_t c = 0; c < cts; ++c) {
            std::vector<double> r(slots);
            for (auto &v : r) v = rd(rng);
            Plaintext pr = cc->MakeCKKSPackedPlaintext(r, 1, cx[c]->GetLevel(), nullptr, slots);
            auto ct = cc->EvalSub(cx[c], pr);
            auto zero = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, 0.0), 1, ct->GetLevel(), nullptr, slots));
            ct = cc->EvalAdd(ct, zero);
            ct = cc->Compress(ct, 1);                                  // keep one modulus before sending
            Serial::Serialize(ct, wire, SerType::BINARY);
            for (size_t i = 0; i < slots; ++i) ss[c][i] = fx(r[i]);
        }
        double ts = ms(t);
        const size_t hb = wire.str().size();
        t = Clock::now();
        for (size_t c = 0; c < cts; ++c) {
            Ciphertext<DCRTPoly> ct;
            Serial::Deserialize(ct, wire, SerType::BINARY);
            Plaintext pt;
            cc->Decrypt(kp.secretKey, ct, &pt);
            pt->SetLength(slots);
            auto v = pt->GetRealPackedValue();
            for (size_t i = 0; i < slots; ++i) sc[c][i] = fx(v[i]);
        }
        double tc = ms(t);
        double err = 0;
        for (size_t c = 0; c < cts; ++c) for (size_t i = 0; i < slots && c * slots + i < numel; ++i)
            err = std::max(err, std::fabs(unfx((sc[c][i] + ss[c][i]) & mask) - x[c][i]));
        h_srv = std::min(h_srv, ts); h_cli = std::min(h_cli, tc); h_bytes = hb; h_err = err;

        // ---------------- MPC -> HE
        std::stringstream wire2;
        t = Clock::now();
        for (size_t c = 0; c < cts; ++c) {
            std::vector<double> v(slots);
            for (size_t i = 0; i < slots; ++i) v[i] = unfx(sc[c][i]);
            Serial::Serialize(cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(v, 1, depth - L, nullptr, slots)),
                              wire2, SerType::BINARY);
        }
        tc = ms(t);
        const size_t mb = wire2.str().size();
        t = Clock::now();
        std::vector<Ciphertext<DCRTPoly>> out(cts);
        for (size_t c = 0; c < cts; ++c) {
            Serial::Deserialize(out[c], wire2, SerType::BINARY);
            std::vector<double> v(slots);
            for (size_t i = 0; i < slots; ++i) v[i] = unfx(ss[c][i]);
            Plaintext ps = cc->MakeCKKSPackedPlaintext(v, 1, out[c]->GetLevel(), nullptr, slots);
            out[c] = cc->EvalAdd(out[c], ps);
        }
        ts = ms(t);
        err = 0;
        const double wrap = std::ldexp(1.0, ell - frac);
        for (size_t c = 0; c < cts; ++c) {
            Plaintext pt;
            cc->Decrypt(kp.secretKey, out[c], &pt);
            pt->SetLength(slots);
            auto v = pt->GetRealPackedValue();
            for (size_t i = 0; i < slots && c * slots + i < numel; ++i) {
                double d = std::fmod(std::fabs(v[i] - x[c][i]), wrap);
                err = std::max(err, std::min(d, wrap - d));
            }
        }
        m_cli = std::min(m_cli, tc); m_srv = std::min(m_srv, ts); m_bytes = mb; m_err = err;
    }
    std::cout << "{\"direction\":\"HE->MPC\",\"numel\":" << numel << ",\"ciphertexts\":" << cts << ",\"server_ms\":" << h_srv
              << ",\"client_ms\":" << h_cli << ",\"bytes\":" << h_bytes << ",\"rounds\":1,\"max_abs_err\":" << h_err << "}\n";
    std::cout << "{\"direction\":\"MPC->HE\",\"numel\":" << numel << ",\"ciphertexts\":" << cts << ",\"server_ms\":" << m_srv
              << ",\"client_ms\":" << m_cli << ",\"bytes\":" << m_bytes << ",\"rounds\":1,\"max_abs_err\":" << m_err
              << ",\"levels_left\":" << L << "}\n";
    return 0;
}
