// Feasibility check: CKKS bootstrapping in OpenFHE (128-bit security), timing and precision.
//   boot_spike <levels_after> <scale_bits> <budget0> <budget1> <logN> <slots|0=full> <tech: auto|ext> <input_amplitude>
#include "openfhe.h"
#include <omp.h>
#include <chrono>
#include <iostream>
using namespace lbcrypto;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int main(int argc, char **argv) {
    uint32_t levels_after = argc > 1 ? std::stoi(argv[1]) : 10;
    uint32_t scale_bits = argc > 2 ? std::stoi(argv[2]) : 59;
    std::vector<uint32_t> budget = {uint32_t(argc > 3 ? std::stoi(argv[3]) : 4), uint32_t(argc > 4 ? std::stoi(argv[4]) : 4)};
    uint32_t logN = argc > 5 ? std::stoi(argv[5]) : 17;
    uint32_t want_slots = argc > 6 ? std::stoi(argv[6]) : 0;
    std::string tech = argc > 7 ? argv[7] : "auto";
    double amp = argc > 8 ? std::stod(argv[8]) : 1.0;
    uint32_t corr = argc > 9 ? std::stoi(argv[9]) : 0;     // 0 = OpenFHE default
    uint32_t first = argc > 10 ? std::stoi(argv[10]) : 60;
    std::string sec = argc > 11 ? argv[11] : "128";      // "none" only for debugging
    uint32_t nonzero = argc > 12 ? std::stoi(argv[12]) : 0; // 0 = all slots
    std::string skn = argc > 13 ? argv[13] : "uniform";
    SecretKeyDist skd = skn == "sparse" ? SPARSE_TERNARY : skn == "encap" ? SPARSE_ENCAPSULATED : UNIFORM_TERNARY;
    uint32_t iters = argc > 14 ? std::stoi(argv[14]) : 1;   // iterative bootstrapping
    uint32_t iter_prec = argc > 15 ? std::stoi(argv[15]) : 12; // precision of one pass, for iterations

    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecretKeyDist(skd);
    p.SetSecurityLevel(sec == "none" ? HEStd_NotSet : HEStd_128_classic);
    p.SetRingDim(1u << logN);
    p.SetScalingModSize(scale_bits);
    p.SetFirstModSize(first);
    p.SetScalingTechnique(tech == "ext" ? FLEXIBLEAUTOEXT : tech == "composite" ? COMPOSITESCALINGAUTO : FLEXIBLEAUTO);
    if (tech == "composite") p.SetRegisterWordSize(64);
    uint32_t depth = levels_after + FHECKKSRNS::GetBootstrapDepth(budget, skd);
    p.SetMultiplicativeDepth(depth);

    auto t = Clock::now();
    auto cc = GenCryptoContext(p);
    for (auto f : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE}) cc->Enable(f);
    uint32_t N = cc->GetRingDimension();
    uint32_t slots = want_slots ? want_slots : N / 2;
    cc->EvalBootstrapSetup(budget, {0, 0}, slots, corr);
    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);
    cc->EvalBootstrapKeyGen(kp.secretKey, slots);
    double t_setup = ms(t);

    std::vector<double> x(slots);
    for (uint32_t i = 0; i < slots; ++i) x[i] = (nonzero && i >= nonzero) ? 0.0 : amp * std::sin(0.001 * i);
    auto pt = cc->MakeCKKSPackedPlaintext(x, 1, depth - 1, nullptr, slots);
    pt->SetLength(slots);
    auto ct = cc->Encrypt(kp.publicKey, pt);

    Plaintext chk;                                               // input decrypts?
    cc->Decrypt(kp.secretKey, ct, &chk);
    std::cerr << "input: level " << ct->GetLevel() << "/" << depth << ", log precision " << chk->GetLogPrecision() << "\n";

    t = Clock::now();
    auto bt = iters > 1 ? cc->EvalBootstrap(ct, iters, iter_prec) : cc->EvalBootstrap(ct);
    double t_boot = ms(t);
    std::cerr << "bootstrapped: level " << bt->GetLevel() << "/" << depth << "\n";

    double err = -1, logprec = -1;
    try {
        Plaintext out;
        cc->Decrypt(kp.secretKey, bt, &out);
        out->SetLength(slots);
        auto v = out->GetRealPackedValue();
        err = 0;
        for (uint32_t i = 0; i < slots; ++i) err = std::max(err, std::fabs(v[i] - x[i]));
        logprec = out->GetLogPrecision();
    } catch (const std::exception &e) {
        std::cerr << "decrypt failed: " << e.what() << "\n";
    }
    std::cout << "{\"N\":" << N << ",\"slots\":" << slots << ",\"depth\":" << depth
              << ",\"levels_after_bootstrap\":" << depth - bt->GetLevel() << ",\"scale_bits\":" << scale_bits
              << ",\"budget\":[" << budget[0] << "," << budget[1] << "],\"log2_q\":" << cc->GetModulus().GetMSB()
              << ",\"setup_keygen_ms\":" << t_setup << ",\"bootstrap_ms\":" << t_boot
              << ",\"max_abs_err\":" << err << ",\"log_precision\":" << logprec
              << ",\"tech\":\"" << tech << "\",\"amplitude\":" << amp << ",\"correction\":" << corr << ",\"first_mod\":" << first << ",\"secret\":\"" << skn << "\",\"iterations\":" << iters << ",\"_\":\"" << "\",\"threads\":" << omp_get_max_threads() << "}\n";
}
