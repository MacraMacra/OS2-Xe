// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rostislavs Rostovskis, Aleksandrs Platonenko
//
// bench_ozaki2.cpp - throughput and per-phase timing of the emulated DGEMM.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <mkl.h>

#include "ozaki2_gpu.hpp"

namespace mkl = oneapi::mkl;
using clk = std::chrono::steady_clock;

struct Size { int64_t m, k, n; };

static std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(d, p);
        if (q == std::string::npos) q = s.size();
        if (q > p) out.push_back(s.substr(p, q - p));
        p = q + 1;
    }
    return out;
}

static void fill(sycl::queue& q, double* X, int64_t rows, int64_t cols, bool isA, double phi, uint64_t seed) {
    q.parallel_for(sycl::range<2>(cols, rows), [=](sycl::id<2> id) {
        const int64_t c = id[0], r = id[1];
        double v;
        if (phi < 0) {
            v = isA ? sycl::sin(static_cast<double>(r + c)) : sycl::cos(static_cast<double>(r) * static_cast<double>(c));
        } else {
            auto mix = [](uint64_t z) {
                z += 0x9E3779B97F4A7C15ull;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                return z ^ (z >> 31);
            };
            const uint64_t base = (static_cast<uint64_t>(c) * rows + r) * 3 + seed;
            const double u1 = ((mix(base) >> 11) + 0.5) * 0x1.0p-53;
            const double u2 = ((mix(base + 1) >> 11) + 0.5) * 0x1.0p-53;
            const double u3 = ((mix(base + 2) >> 11) + 0.5) * 0x1.0p-53 - 0.5;
            const double g = sycl::sqrt(-2.0 * sycl::log(u1)) * sycl::cos(6.283185307179586 * u2);
            v = u3 * sycl::exp(phi * g);
        }
        X[r + c * rows] = v;
    }).wait();
}

int main(int argc, char** argv) {
    std::vector<Size> sizes = {{2048, 2048, 2048}, {4096, 4096, 4096}, {4000, 8000, 4000}, {8192, 8192, 8192},
                               {16384, 16384, 16384}, {20480, 20480, 20480}, {24576, 24576, 24576}};
    std::vector<int> svals = {8, 12, 16};
    int reps = 5;
    bool do_dgemm = false, do_cpu = false, do_check = false;
    double phi = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--sizes") {
            sizes.clear();
            for (auto& t : split(next(), ',')) {
                auto d = split(t, 'x');
                if (d.size() == 1) sizes.push_back({std::stoll(d[0]), std::stoll(d[0]), std::stoll(d[0])});
                else sizes.push_back({std::stoll(d[0]), std::stoll(d[1]), std::stoll(d[2])});
            }
        } else if (a == "--s") {
            svals.clear();
            for (auto& t : split(next(), ',')) svals.push_back(std::stoi(t));
        } else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--dgemm") do_dgemm = true;
        else if (a == "--cpu") do_cpu = true;
        else if (a == "--check") do_check = true;
        else if (a == "--dist") {
            std::string d = next();
            phi = d.rfind("phi", 0) == 0 ? std::stod(d.substr(3)) : -1;
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }

    sycl::queue q(sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    oz2::Ozaki2 oz(q);

    {
        double* w = sycl::malloc_device<double>(3 * 256 * 256, q);
        q.memset(w, 0, 3 * 256 * 256 * 8).wait();
        oz.gemm(oz2::Trans::N, oz2::Trans::N, 256, 256, 256, 1.0, w, 256, w + 65536, 256, 0.0, w + 131072, 256, 8).wait();
        sycl::free(w, q);
        oz.release_workspace();
    }

    struct Row { Size sz; int s; int tk; double cold, best, med, dgemm, cpu, acc_oz, acc_dg; oz2::StageTimes st; oz2::TileInfo ti; };
    std::vector<Row> rows;

    for (const auto& sz : sizes) {
        const int64_t m = sz.m, k = sz.k, n = sz.n;
        const double flops = 2.0 * m * n * k;
        double* dA = sycl::malloc_device<double>(m * k, q);
        double* dB = sycl::malloc_device<double>(k * n, q);
        double* dC = sycl::malloc_device<double>(m * n, q);
        if (!dA || !dB || !dC) { std::printf("alloc failed for %lldx%lldx%lld\n", (long long)m, (long long)k, (long long)n); return 1; }
        fill(q, dA, m, k, true, phi, 1);
        fill(q, dB, k, n, false, phi, 12345);

        double t_dgemm = 0;
        double* dRef = nullptr;
        if (do_dgemm || do_check) {
            dRef = sycl::malloc_device<double>(m * n, q);
            if (dRef) {
                auto run = [&] { mkl::blas::column_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::nontrans,
                                                               m, n, k, 1.0, dA, m, dB, k, 0.0, dRef, m).wait(); };
                run();
                t_dgemm = 1e30;
                const int dr = flops > 2e12 ? 1 : 3;
                for (int r = 0; r < dr; ++r) {
                    auto t0 = clk::now(); run();
                    t_dgemm = std::min(t_dgemm, std::chrono::duration<double>(clk::now() - t0).count());
                }
            }
        }
        double t_cpu = 0;
        if (do_cpu) {
            std::vector<double> A(m * k), B(k * n), C(m * n);
            q.memcpy(A.data(), dA, A.size() * 8);
            q.memcpy(B.data(), dB, B.size() * 8).wait();
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A.data(), m, B.data(), k, 0.0, C.data(), m);
            auto t0 = clk::now();
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A.data(), m, B.data(), k, 0.0, C.data(), m);
            t_cpu = std::chrono::duration<double>(clk::now() - t0).count();
        }

        double* dAbsAB = nullptr;
        std::vector<double> absAB, ref;
        if (do_check && dRef) {
            double* dAa = sycl::malloc_device<double>(m * k, q);
            double* dBa = sycl::malloc_device<double>(k * n, q);
            dAbsAB = sycl::malloc_device<double>(m * n, q);
            if (dAa && dBa && dAbsAB) {
                q.parallel_for(sycl::range<1>(m * k), [=](sycl::id<1> i) { dAa[i] = sycl::fabs(dA[i]); });
                q.parallel_for(sycl::range<1>(k * n), [=](sycl::id<1> i) { dBa[i] = sycl::fabs(dB[i]); }).wait();
                mkl::blas::column_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::nontrans,
                                              m, n, k, 1.0, dAa, m, dBa, k, 0.0, dAbsAB, m).wait();
                absAB.resize(m * n); ref.resize(m * n);
                q.memcpy(absAB.data(), dAbsAB, m * n * 8);
                q.memcpy(ref.data(), dRef, m * n * 8).wait();
            }
            sycl::free(dAa, q); sycl::free(dBa, q); sycl::free(dAbsAB, q);
        }
        sycl::free(dRef, q);

        for (int s : svals) {
            Row row{};
            row.sz = sz; row.s = s;
            try {
                row.tk = oz2::Ozaki2::precision_bits(k, s);
                oz.release_workspace();
                auto t0 = clk::now();
                oz.gemm(oz2::Trans::N, oz2::Trans::N, m, n, k, 1.0, dA, m, dB, k, 0.0, dC, m, s).wait();
                row.cold = std::chrono::duration<double>(clk::now() - t0).count();
                std::vector<double> ts;
                for (int r = 0; r < reps; ++r) {
                    t0 = clk::now();
                    oz.gemm(oz2::Trans::N, oz2::Trans::N, m, n, k, 1.0, dA, m, dB, k, 0.0, dC, m, s).wait();
                    ts.push_back(std::chrono::duration<double>(clk::now() - t0).count());
                }
                row.st = oz.stage_times();
                row.ti = oz.last_tiling();
                std::sort(ts.begin(), ts.end());
                row.best = ts.front();
                row.med = ts[ts.size() / 2];
                row.dgemm = t_dgemm;
                row.cpu = t_cpu;
                if (!absAB.empty()) {
                    std::vector<double> C(m * n);
                    q.memcpy(C.data(), dC, m * n * 8).wait();
                    double e_oz = 0;
                    for (int64_t i = 0; i < m * n; ++i)
                        if (absAB[i] > 0) e_oz = std::max(e_oz, std::fabs(C[i] - ref[i]) / absAB[i]);
                    row.acc_oz = e_oz;
                }
            } catch (const std::exception& e) {
                std::printf("  %lldx%lldx%lld s=%d FAILED: %s\n", (long long)m, (long long)k, (long long)n, s, e.what());
                continue;
            }
            std::printf("%6lldx%6lldx%6lld s=%2d tk=%2d | %7.3f ms  %7.3f TFLOPS (median %.3f) | cold %.3f s | "
                        "scale %.2f split %.2f gemm %.2f (%.0f TOPS) crt %.2f ms | tiles %lldx%lld ws %.2f GB",
                        (long long)m, (long long)k, (long long)n, s, row.tk, row.best * 1e3, flops / row.best / 1e12,
                        flops / row.med / 1e12, row.cold, row.st.scale_ms, row.st.split_ms, row.st.gemm_ms,
                        row.st.gemm_ms > 0 ? flops * s / (row.st.gemm_ms * 1e-3) / 1e12 : 0.0, row.st.crt_ms,
                        (long long)row.ti.pm, (long long)row.ti.pn, row.ti.workspace_bytes / 1e9);
            if (row.dgemm > 0) std::printf(" | dgemm %.3f TFLOPS (x%.1f)", flops / row.dgemm / 1e12, row.dgemm / row.best);
            if (row.cpu > 0) std::printf(" | cpu %.3f TFLOPS (x%.1f)", flops / row.cpu / 1e12, row.cpu / row.best);
            if (row.acc_oz > 0 || !absAB.empty()) std::printf(" | max|C-C_dgemm|/(|A||B|) %.2e", row.acc_oz);
            std::printf("\n");
            std::fflush(stdout);
            rows.push_back(row);
        }
        oz.release_workspace();
        sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
    }

    std::printf("\n%-22s %3s %3s %10s %10s %12s %10s\n", "size (MxKxN)", "s", "tk", "TFLOPS", "ms", "vs GPU DGEMM", "vs CPU");
    for (const auto& r : rows) {
        const double flops = 2.0 * r.sz.m * r.sz.n * r.sz.k;
        char sz[64];
        std::snprintf(sz, sizeof sz, "%lldx%lldx%lld", (long long)r.sz.m, (long long)r.sz.k, (long long)r.sz.n);
        std::printf("%-22s %3d %3d %10.3f %10.2f", sz, r.s, r.tk, flops / r.best / 1e12, r.best * 1e3);
        if (r.dgemm > 0) std::printf(" %11.1fx", r.dgemm / r.best); else std::printf(" %12s", "-");
        if (r.cpu > 0) std::printf(" %9.1fx", r.cpu / r.best); else std::printf(" %10s", "-");
        std::printf("\n");
    }
    return 0;
}
