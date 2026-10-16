// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rostislavs Rostovskis, Aleksandrs Platonenko
//
// zgemm.cpp - complex example: C = alpha * A^H * B + beta * C on the GPU,
// compared against the native oneMKL ZGEMM.

#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ozaki2_gpu.hpp"

using cd = std::complex<double>;

int main(int argc, char** argv) {
    const int64_t m = argc > 1 ? std::atoll(argv[1]) : 1024;
    const int64_t n = argc > 2 ? std::atoll(argv[2]) : m;
    const int64_t k = argc > 3 ? std::atoll(argv[3]) : m;
    const int s = argc > 4 ? std::atoi(argv[4]) : 16;

    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<cd> hA(k * m), hB(k * n), hC(m * n);
    for (auto& v : hA) v = {U(rng), U(rng)};
    for (auto& v : hB) v = {U(rng), U(rng)};
    for (auto& v : hC) v = {U(rng), U(rng)};

    cd* dA = sycl::malloc_device<cd>(k * m, q);
    cd* dB = sycl::malloc_device<cd>(k * n, q);
    cd* dC = sycl::malloc_device<cd>(m * n, q);
    cd* dR = sycl::malloc_device<cd>(m * n, q);
    q.memcpy(dA, hA.data(), k * m * sizeof(cd));
    q.memcpy(dB, hB.data(), k * n * sizeof(cd));
    q.memcpy(dC, hC.data(), m * n * sizeof(cd));
    q.memcpy(dR, hC.data(), m * n * sizeof(cd));

    const cd alpha(1.0, 0.5), beta(0.25, -1.0);

    oz2::Ozaki2 oz(q);
    std::printf("m=%lld n=%lld k=%lld, moduli s=%d, bits per element tk=%d\n", (long long)m, (long long)n,
                (long long)k, s, oz2::Ozaki2::precision_bits_complex(k, s));

    oz.zgemm(oz2::Trans::C, oz2::Trans::N, m, n, k, alpha, dA, k, dB, k, beta, dC, m, s).wait();

    oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::nontrans,
                                          m, n, k, alpha, dA, k, dB, k, beta, dR, m).wait();

    std::vector<cd> hR(m * n);
    q.memcpy(hC.data(), dC, m * n * sizeof(cd));
    q.memcpy(hR.data(), dR, m * n * sizeof(cd)).wait();

    double max_diff = 0, max_ref = 0;
    for (int64_t i = 0; i < m * n; ++i) {
        max_diff = std::max(max_diff, std::abs(hC[i] - hR[i]));
        max_ref = std::max(max_ref, std::abs(hR[i]));
    }
    std::printf("max |C_oz2 - C_zgemm| / max |C_zgemm| = %.3e\n", max_diff / max_ref);

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    sycl::free(dR, q);
    return 0;
}
