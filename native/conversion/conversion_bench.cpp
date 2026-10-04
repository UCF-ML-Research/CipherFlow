// HE <-> MPC conversion benchmark: real CKKS (Microsoft SEAL 4.1) at the NEXUS parameters.
//
// Parties: the client owns the secret key; the server holds the public key and a ciphertext
// of the activation x (HE domain) or an additive share of x (MPC domain).
//
// HE -> MPC (mask and decrypt), one message server -> client:
//   server: r <- random mask; ct' = ct(x) - Encode(r) + Enc_pk(0)   (re-randomise)
//           mod-switch ct' to the lowest level, serialize (zstd)
//   client: deserialize, decrypt, decode -> x - r; fixed-point encode -> share_c
//   server: share_s = fixed-point encode(r)            (share_c + share_s = x mod 2^ell)
// MPC -> HE (share and encrypt), one message client -> server:
//   client: decode share_c -> real; Encode; symmetric encryption at the top level (seeded,
//           so only half of the ciphertext is sent); serialize (zstd)
//   server: deserialize; ct(x) = ct + Encode(decode(share_s))   (fresh ciphertext, top level)
//
// Prints one JSON line per direction with each party's compute time, the bytes sent, the
// number of ciphertexts, and the reconstruction error.
//
//   conversion_bench <numel> [repeats] [ell] [frac_bits]
#include <seal/seal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using namespace seal;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: conversion_bench <numel> [repeats] [ell] [frac_bits]\n";
        return 2;
    }
    const size_t numel = std::stoull(argv[1]);
    const int repeats = argc > 2 ? std::stoi(argv[2]) : 3;
    const int ell = argc > 3 ? std::stoi(argv[3]) : 37;     // SCI ring bits (CrypTen: 64)
    const int frac = argc > 4 ? std::stoi(argv[4]) : 12;    // fixed-point fraction bits

    // NEXUS main.cpp parameters (non-MatMul operators): N = 2^16, {58, 40 x 18, 58}, scale 2^40
    const size_t N = size_t(1) << 16;
    std::vector<int> bits = {58};
    for (int i = 0; i < 18; ++i) bits.push_back(40);
    bits.push_back(58);
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, bits));
    SEALContext ctx(parms, true, sec_level_type::none);   // 836-bit modulus; same as NEXUS
    const double scale = std::pow(2.0, 40);
    const size_t slots = N / 2;
    const size_t cts = (numel + slots - 1) / slots;

    KeyGenerator keygen(ctx);
    SecretKey sk = keygen.secret_key();
    PublicKey pk;
    keygen.create_public_key(pk);
    CKKSEncoder encoder(ctx);
    Encryptor enc_pk(ctx, pk);
    Encryptor enc_sk(ctx, sk);
    Decryptor dec(ctx, sk);
    Evaluator ev(ctx);

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> xdist(-4.0, 4.0), rdist(-64.0, 64.0);
    const uint64_t mask = (ell == 64) ? ~uint64_t(0) : ((uint64_t(1) << ell) - 1);
    auto fx = [&](double v) -> uint64_t { return uint64_t(int64_t(std::llround(v * std::ldexp(1.0, frac)))) & mask; };
    auto unfx = [&](uint64_t u) -> double {                 // signed (centered) decoding
        int64_t s = (ell == 64) ? int64_t(u) : ((u >> (ell - 1)) & 1 ? int64_t(u) - (int64_t(1) << ell) : int64_t(u));
        return double(s) / std::ldexp(1.0, frac);
    };

    // inputs: x, and the server's ciphertexts of x at the top level
    std::vector<std::vector<double>> x(cts, std::vector<double>(slots, 0.0));
    for (size_t c = 0; c < cts; ++c)
        for (size_t i = 0; i < slots && c * slots + i < numel; ++i) x[c][i] = xdist(rng);
    std::vector<Ciphertext> ctx_x(cts);
    for (size_t c = 0; c < cts; ++c) {
        Plaintext p;
        encoder.encode(x[c], scale, p);
        enc_pk.encrypt(p, ctx_x[c]);
    }

    double h2m_srv = 1e300, h2m_cli = 1e300, m2h_cli = 1e300, m2h_srv = 1e300;
    size_t h2m_bytes = 0, m2h_bytes = 0;
    double h2m_err = 0, m2h_err = 0;
    for (int rep = 0; rep < repeats; ++rep) {
        // ---------------- HE -> MPC
        std::vector<std::vector<double>> r(cts, std::vector<double>(slots));
        for (auto &v : r) for (auto &e : v) e = rdist(rng);
        std::stringstream wire;
        auto t0 = Clock::now();
        size_t bytes = 0;
        std::vector<std::vector<uint64_t>> share_s(cts, std::vector<uint64_t>(slots));
        for (size_t c = 0; c < cts; ++c) {
            Ciphertext ct = ctx_x[c];
            Plaintext pr;
            encoder.encode(r[c], ct.parms_id(), ct.scale(), pr);
            ev.sub_plain_inplace(ct, pr);
            Ciphertext zero;
            enc_pk.encrypt_zero(ct.parms_id(), zero);
            zero.scale() = ct.scale();
            ev.add_inplace(ct, zero);
            ev.mod_switch_to_inplace(ct, ctx.last_parms_id());
            bytes += ct.save(wire, compr_mode_type::zstd);
            for (size_t i = 0; i < slots; ++i) share_s[c][i] = fx(r[c][i]);
        }
        double t_srv = ms_since(t0);
        t0 = Clock::now();
        std::vector<std::vector<uint64_t>> share_c(cts, std::vector<uint64_t>(slots));
        for (size_t c = 0; c < cts; ++c) {
            Ciphertext ct;
            ct.load(ctx, wire);
            Plaintext p;
            dec.decrypt(ct, p);
            std::vector<double> v;
            encoder.decode(p, v);
            for (size_t i = 0; i < slots; ++i) share_c[c][i] = fx(v[i]);
        }
        double t_cli = ms_since(t0);
        double err = 0;
        for (size_t c = 0; c < cts; ++c)
            for (size_t i = 0; i < slots && c * slots + i < numel; ++i)
                err = std::max(err, std::fabs(unfx((share_c[c][i] + share_s[c][i]) & mask) - x[c][i]));
        h2m_srv = std::min(h2m_srv, t_srv); h2m_cli = std::min(h2m_cli, t_cli);
        h2m_bytes = bytes; h2m_err = err;

        // ---------------- MPC -> HE (shares from above)
        std::stringstream wire2;
        t0 = Clock::now();
        bytes = 0;
        for (size_t c = 0; c < cts; ++c) {
            std::vector<double> v(slots);
            for (size_t i = 0; i < slots; ++i) v[i] = unfx(share_c[c][i]);
            Plaintext p;
            encoder.encode(v, scale, p);
            bytes += enc_sk.encrypt_symmetric(p).save(wire2, compr_mode_type::zstd);
        }
        t_cli = ms_since(t0);
        t0 = Clock::now();
        std::vector<Ciphertext> out(cts);
        for (size_t c = 0; c < cts; ++c) {
            out[c].load(ctx, wire2);
            std::vector<double> v(slots);
            for (size_t i = 0; i < slots; ++i) v[i] = unfx(share_s[c][i]);
            Plaintext p;
            encoder.encode(v, out[c].parms_id(), out[c].scale(), p);
            ev.add_plain_inplace(out[c], p);
        }
        t_srv = ms_since(t0);
        err = 0;                                           // check (not timed): decrypt ct(x)
        const double wrap = std::ldexp(1.0, ell - frac);
        for (size_t c = 0; c < cts; ++c) {
            Plaintext p;
            dec.decrypt(out[c], p);
            std::vector<double> v;
            encoder.decode(p, v);
            for (size_t i = 0; i < slots && c * slots + i < numel; ++i) {
                double d = std::fmod(std::fabs(v[i] - x[c][i]), wrap);   // shares sum mod 2^ell
                err = std::max(err, std::min(d, wrap - d));
            }
        }
        m2h_cli = std::min(m2h_cli, t_cli); m2h_srv = std::min(m2h_srv, t_srv);
        m2h_bytes = bytes; m2h_err = err;
    }
    std::cout << "{\"direction\":\"HE->MPC\",\"numel\":" << numel << ",\"ciphertexts\":" << cts
              << ",\"server_ms\":" << h2m_srv << ",\"client_ms\":" << h2m_cli << ",\"bytes\":" << h2m_bytes
              << ",\"rounds\":1,\"max_abs_err\":" << h2m_err << "}\n";
    std::cout << "{\"direction\":\"MPC->HE\",\"numel\":" << numel << ",\"ciphertexts\":" << cts
              << ",\"server_ms\":" << m2h_srv << ",\"client_ms\":" << m2h_cli << ",\"bytes\":" << m2h_bytes
              << ",\"rounds\":1,\"max_abs_err\":" << m2h_err << "}\n";
    return 0;
}
