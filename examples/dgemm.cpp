// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rostislavs Rostovskis, Aleksandrs Platonenko
//
// dgemm.cpp - minimal DGEMM example: C = alpha * A * B + beta * C on the GPU,
// compared against the native oneMKL DGEMM.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ozaki2_gpu.hpp"

int main(int argc, char** argv) {
    const int64_t m = argc > 1 ? std::atoll(argv[1]) : 2048;
    const int64_t n = argc > 2 ? std::atoll(argv[2]) : m;
    const int64_t k = argc > 3 ? std::atoll(argv[3]) : m;

    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> hA(m * k), hB(k * n);
    for (auto& v : hA) v = U(rng);
    for (auto& v : hB) v = U(rng);

    double* dA = sycl::malloc_device<double>(m * k, q);
    double* dB = sycl::malloc_device<double>(k * n, q);
    double* dC = sycl::malloc_device<double>(m * n, q);
    double* dR = sycl::malloc_device<double>(m * n, q);
    q.memcpy(dA, hA.data(), m * k * sizeof(double));
    q.memcpy(dB, hB.data(), k * n * sizeof(double));

    oz2::Ozaki2 oz(q);
    oz.set_resident_hint(sizeof(double) * (m * k + k * n + 2 * m * n));

    const int s = oz2::Ozaki2::moduli_for_bits(k, 53);
    std::printf("m=%lld n=%lld k=%lld, moduli s=%d, bits per element tk=%d\n", (long long)m, (long long)n,
                (long long)k, s, oz2::Ozaki2::precision_bits(k, s));

    oz.gemm(oz2::Trans::N, oz2::Trans::N, m, n, k, 1.0, dA, m, dB, k, 0.0, dC, m, s).wait();

    oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                                          m, n, k, 1.0, dA, m, dB, k, 0.0, dR, m).wait();

    std::vector<double> hC(m * n), hR(m * n);
    q.memcpy(hC.data(), dC, m * n * sizeof(double));
    q.memcpy(hR.data(), dR, m * n * sizeof(double)).wait();

    double max_diff = 0, max_ref = 0;
    for (int64_t i = 0; i < m * n; ++i) {
        max_diff = std::max(max_diff, std::abs(hC[i] - hR[i]));
        max_ref = std::max(max_ref, std::abs(hR[i]));
    }
    std::printf("max |C_oz2 - C_dgemm| / max |C_dgemm| = %.3e\n", max_diff / max_ref);

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    sycl::free(dR, q);
    return 0;
}
