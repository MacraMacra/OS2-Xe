// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rostislavs Rostovskis, Aleksandrs Platonenko
//
// profiling.cpp - accuracy/speed trade-off: runs one DGEMM for several numbers
// of moduli and prints the per-phase timing, tiling and achieved TFLOP/s.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "ozaki2_gpu.hpp"

int main(int argc, char** argv) {
    const int64_t N = argc > 1 ? std::atoll(argv[1]) : 8192;

    sycl::queue q(sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    std::printf("device: %s, N=%lld\n", q.get_device().get_info<sycl::info::device::name>().c_str(), (long long)N);

    double* dA = sycl::malloc_device<double>(N * N, q);
    double* dB = sycl::malloc_device<double>(N * N, q);
    double* dC = sycl::malloc_device<double>(N * N, q);
    q.parallel_for(sycl::range<1>(static_cast<size_t>(N * N)), [=](sycl::id<1> i) {
        dA[i] = sycl::sin(static_cast<double>(i));
        dB[i] = sycl::cos(static_cast<double>(i));
    }).wait();

    oz2::Ozaki2 oz(q);
    oz.set_resident_hint(3 * sizeof(double) * N * N);

    std::printf("%4s %4s %10s %10s %10s %10s %10s %10s %8s\n", "s", "tk", "scale_ms", "split_ms", "gemm_ms",
                "crt_ms", "wall_ms", "TFLOP/s", "tiles");
    for (int s : {8, 12, 14, 16, 20}) {
        oz.gemm(oz2::Trans::N, oz2::Trans::N, N, N, N, 1.0, dA, N, dB, N, 0.0, dC, N, s).wait();

        const auto t0 = std::chrono::steady_clock::now();
        oz.gemm(oz2::Trans::N, oz2::Trans::N, N, N, N, 1.0, dA, N, dB, N, 0.0, dC, N, s).wait();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        const oz2::StageTimes st = oz.stage_times();
        const oz2::TileInfo& ti = oz.last_tiling();
        std::printf("%4d %4d %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %4lldx%lld\n", s,
                    oz2::Ozaki2::precision_bits(N, s), st.scale_ms, st.split_ms, st.gemm_ms, st.crt_ms, ms,
                    2.0 * N * N * N / (ms * 1e9), (long long)ti.pm, (long long)ti.pn);
    }

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    return 0;
}
