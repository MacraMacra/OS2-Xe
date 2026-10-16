// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rostislavs Rostovskis, Aleksandrs Platonenko
//
// bench_accuracy.cpp - accuracy of the emulated DGEMM/ZGEMM against a double-double reference.

#include <cinttypes>
#include <complex>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <mkl.h>

#include "ozaki1_gpu.hpp"

namespace mkl = oneapi::mkl;

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
static std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> v;
    for (auto& t : split(s, ',')) {
        auto r = split(t, '-');
        if (r.size() == 2) for (int i = std::stoi(r[0]); i <= std::stoi(r[1]); ++i) v.push_back(i);
        else v.push_back(std::stoi(t));
    }
    return v;
}

static void gen(std::vector<double>& X, int64_t rows, int64_t cols, const std::string& dist, uint64_t seed, bool isA) {
    X.resize(rows * cols);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> N(0.0, 1.0);
    const double phi = dist.rfind("phi", 0) == 0 ? std::stod(dist.substr(3)) : 0.0;
    for (int64_t c = 0; c < cols; ++c)
        for (int64_t r = 0; r < rows; ++r) {
            double v;
            if (dist == "urand" || dist == "spike" || dist == "cancel") v = 2.0 * U(rng) - 1.0;
            else if (dist == "sincos") v = isA ? std::sin(double(r + c)) : std::cos(double(r) * double(c));
            else v = (U(rng) - 0.5) * std::exp(phi * N(rng));
            X[r + c * rows] = v;
        }
    if (dist == "spike" && isA)
        for (int64_t r = 0; r < rows; ++r) X[r + ((r * 7919) % cols) * rows] *= std::ldexp(1.0, 40);
    if (dist == "cancel") {
        const int64_t h = isA ? cols / 2 : rows / 2;
        if (isA) {
            for (int64_t c = h; c < 2 * h; ++c)
                for (int64_t r = 0; r < rows; ++r) X[r + c * rows] = X[r + (c - h) * rows];
        } else {
            for (int64_t c = 0; c < cols; ++c)
                for (int64_t r = h; r < 2 * h; ++r) X[r + c * rows] = -X[(r - h) + c * rows] + std::ldexp(X[r + c * rows], -30);
        }
    }
}

struct Metrics { double ulp_max = 0, ulp_mean = 0, cr_frac = 0, ab_max = 0, fro_rel = 0; };

static Metrics measure(const std::vector<double>& C, const std::vector<double>& hi, const std::vector<double>& lo,
                       const std::vector<double>& absAB) {
    Metrics M;
    double s_ulp = 0, s_d2 = 0, s_r2 = 0;
    size_t cr = 0;
    const size_t n = C.size();
    for (size_t i = 0; i < n; ++i) {
        const double ref_rn = hi[i] + lo[i];
        const double err = std::fabs((C[i] - hi[i]) - lo[i]);
        const double ulp = hi[i] == 0 ? std::numeric_limits<double>::denorm_min()
                                      : std::nextafter(std::fabs(hi[i]), INFINITY) - std::fabs(hi[i]);
        const double eu = err / ulp;
        M.ulp_max = std::max(M.ulp_max, eu);
        s_ulp += eu;
        if (C[i] == ref_rn) ++cr;
        if (absAB[i] > 0) M.ab_max = std::max(M.ab_max, err / absAB[i]);
        s_d2 += err * err;
        s_r2 += hi[i] * hi[i];
    }
    M.ulp_mean = s_ulp / n;
    M.cr_frac = double(cr) / n;
    M.fro_rel = std::sqrt(s_d2 / s_r2);
    return M;
}

static void run_complex(sycl::queue& q, oz2::Ozaki2& oz, const std::string& dist, int k, int64_t m, int64_t n,
                        const std::vector<int>& svals, bool cpu) {
    using cd = std::complex<double>;
    std::vector<double> Ar, Ai, Br, Bi;
    gen(Ar, m, k, dist, 11 + k, true);
    gen(Ai, m, k, dist, 13 + k, true);
    gen(Br, k, n, dist, 77 + k, false);
    gen(Bi, k, n, dist, 79 + k, false);
    std::vector<cd> A(m * k), B(k * n), C(m * n);
    for (size_t i = 0; i < A.size(); ++i) A[i] = cd(Ar[i], Ai[i]);
    for (size_t i = 0; i < B.size(); ++i) B[i] = cd(Br[i], Bi[i]);
    cd* dA = sycl::malloc_device<cd>(A.size(), q);
    cd* dB = sycl::malloc_device<cd>(B.size(), q);
    cd* dC = sycl::malloc_device<cd>(m * n, q);
    double* dR = sycl::malloc_device<double>(5 * m * n, q);
    q.memcpy(dA, A.data(), A.size() * 16);
    q.memcpy(dB, B.data(), B.size() * 16).wait();
    const double* Ad = reinterpret_cast<const double*>(dA);
    const double* Bd = reinterpret_cast<const double*>(dB);
    q.parallel_for(sycl::range<2>(n, m), [=](sycl::id<2> id) {
        const int64_t j = id[0], i = id[1];
        double h[2] = {0, 0}, l[2] = {0, 0}, ab = 0;
        auto dd_add = [&](int c, double a, double b) {
            const double ph = a * b, pl = sycl::fma(a, b, -ph);
            const double s2 = h[c] + ph, bb = s2 - h[c];
            const double e = (h[c] - (s2 - bb)) + (ph - bb);
            const double t = e + l[c] + pl;
            h[c] = s2 + t;
            l[c] = t - (h[c] - s2);
        };
        for (int64_t p = 0; p < k; ++p) {
            const double ar = Ad[2 * (i + p * m)], ai = Ad[2 * (i + p * m) + 1];
            const double br = Bd[2 * (p + j * k)], bi = Bd[2 * (p + j * k) + 1];
            dd_add(0, ar, br); dd_add(0, -ai, bi);
            dd_add(1, ar, bi); dd_add(1, ai, br);
            ab += (sycl::fabs(ar) + sycl::fabs(ai)) * (sycl::fabs(br) + sycl::fabs(bi));
        }
        const int64_t o = i + j * m;
        dR[o] = h[0]; dR[m * n + o] = h[1]; dR[2 * m * n + o] = l[0]; dR[3 * m * n + o] = l[1]; dR[4 * m * n + o] = ab;
    }).wait();
    std::vector<double> R(5 * m * n);
    q.memcpy(R.data(), dR, R.size() * 8).wait();
    std::vector<double> hi(R.begin(), R.begin() + 2 * m * n), lo(R.begin() + 2 * m * n, R.begin() + 4 * m * n);
    std::vector<double> absAB(2 * m * n);
    for (int64_t i = 0; i < m * n; ++i) absAB[i] = absAB[m * n + i] = R[4 * m * n + i];
    std::vector<double> Cs(2 * m * n);
    auto row = [&](const char* method, int s, int tk) {
        for (int64_t i = 0; i < m * n; ++i) { Cs[i] = C[i].real(); Cs[m * n + i] = C[i].imag(); }
        Metrics M = measure(Cs, hi, lo, absAB);
        std::printf("%s,%d,%s,%d,%d,%.4g,%.4g,%.6f,%.4g,%.4g\n", ("z" + dist).c_str(), k, method, s, tk, M.ulp_max,
                    M.ulp_mean, M.cr_frac, M.ab_max, M.fro_rel);
        std::fflush(stdout);
    };
    mkl::blas::column_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::nontrans, m, n, k, cd(1), dA, m, dB, k,
                                  cd(0), dC, m).wait();
    q.memcpy(C.data(), dC, m * n * 16).wait();
    row("zgemm_gpu", 0, 53);
    if (cpu) {
        const cd one(1), zero(0);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, &one, A.data(), m, B.data(), k, &zero, C.data(), m);
        row("zgemm_cpu", 0, 53);
    }
    for (int s : svals) {
        int tk;
        try { tk = oz2::Ozaki2::precision_bits_complex(k, s); } catch (const std::exception&) { continue; }
        oz.zgemm(oz2::Trans::N, oz2::Trans::N, m, n, k, cd(1), dA, m, dB, k, cd(0), dC, m, s).wait();
        q.memcpy(C.data(), dC, m * n * 16).wait();
        row("ozaki2_3m", s, tk);
    }
    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q); sycl::free(dR, q);
}

int main(int argc, char** argv) {
    std::vector<int> ks = {256, 1024, 4096, 16384}, svals = parse_ints("4-24");
    std::vector<std::string> dists = {"urand", "phi0.5", "phi2", "phi4"};
    int64_t mn = 512;
    std::vector<int> Ls = {4, 5, 6, 7, 8, 9};
    bool cpu = false, cplx = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--k") ks = parse_ints(next());
        else if (a == "--s") svals = parse_ints(next());
        else if (a == "--dist") dists = split(next(), ',');
        else if (a == "--mn") mn = std::stoll(next());
        else if (a == "--cpu") cpu = true;
        else if (a == "--complex") cplx = true;
        else if (a == "--L") Ls = parse_ints(next());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    oz2::Ozaki2 oz(q);
    oz1::Ozaki1 oz1g(q);
    std::printf("dist,k,method,s,tk,ulp_max,ulp_mean,cr_frac,ab_max,fro_rel\n");

    const int64_t m = mn, n = mn;
    if (cplx) {
        for (const auto& dist : dists)
            for (int k : ks) run_complex(q, oz, dist, k, mn, mn, svals, cpu);
        return 0;
    }
    for (const auto& dist : dists)
        for (int k : ks) {
            std::vector<double> A, B;
            gen(A, m, k, dist, 11 + k, true);
            gen(B, k, n, dist, 77 + k, false);
            double* dA = sycl::malloc_device<double>(A.size(), q);
            double* dB = sycl::malloc_device<double>(B.size(), q);
            double* dC = sycl::malloc_device<double>(m * n, q);
            double* dH = sycl::malloc_device<double>(m * n, q);
            double* dL = sycl::malloc_device<double>(m * n, q);
            double* dAB = sycl::malloc_device<double>(m * n, q);
            q.memcpy(dA, A.data(), A.size() * 8);
            q.memcpy(dB, B.data(), B.size() * 8).wait();

            q.parallel_for(sycl::range<2>(n, m), [=](sycl::id<2> id) {
                const int64_t j = id[0], i = id[1];
                double sh = 0, sl = 0, ab = 0;
                for (int64_t p = 0; p < k; ++p) {
                    const double a = dA[i + p * m], b = dB[p + j * k];
                    const double ph = a * b, pl = sycl::fma(a, b, -ph);
                    const double s = sh + ph, bb = s - sh;
                    const double e = (sh - (s - bb)) + (ph - bb);
                    const double t = e + sl + pl;
                    sh = s + t;
                    sl = t - (sh - s);
                    ab += sycl::fabs(a) * sycl::fabs(b);
                }
                dH[i + j * m] = sh;
                dL[i + j * m] = sl;
                dAB[i + j * m] = ab;
            }).wait();
            std::vector<double> hi(m * n), lo(m * n), absAB(m * n), C(m * n);
            q.memcpy(hi.data(), dH, m * n * 8);
            q.memcpy(lo.data(), dL, m * n * 8);
            q.memcpy(absAB.data(), dAB, m * n * 8).wait();

            auto row = [&](const char* method, int s, int tk) {
                Metrics M = measure(C, hi, lo, absAB);
                std::printf("%s,%d,%s,%d,%d,%.4g,%.4g,%.6f,%.4g,%.4g\n", dist.c_str(), k, method, s, tk,
                            M.ulp_max, M.ulp_mean, M.cr_frac, M.ab_max, M.fro_rel);
                std::fflush(stdout);
            };

            mkl::blas::column_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::nontrans, m, n, k, 1.0,
                                          dA, m, dB, k, 0.0, dC, m).wait();
            q.memcpy(C.data(), dC, m * n * 8).wait();
            row("dgemm_gpu", 0, 53);
            if (cpu) {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A.data(), m, B.data(), k,
                            0.0, C.data(), m);
                row("dgemm_cpu", 0, 53);
            }
            for (int s : svals) {
                int tk;
                try { tk = oz2::Ozaki2::precision_bits(k, s); } catch (const std::exception&) { continue; }
                oz.gemm(oz2::Trans::N, oz2::Trans::N, m, n, k, 1.0, dA, m, dB, k, 0.0, dC, m, s).wait();
                q.memcpy(C.data(), dC, m * n * 8).wait();
                row("ozaki2", s, tk);
            }
            for (int L : Ls) {
                try { oz1g.gemm(m, n, k, dA, m, dB, k, dC, m, L).wait(); } catch (const std::exception&) { continue; }
                q.memcpy(C.data(), dC, m * n * 8).wait();
                row("ozaki1", oz1::Ozaki1::num_gemms(L), oz1::Ozaki1::precision_bits(L));
            }
            for (double* p : {dA, dB, dC, dH, dL, dAB}) sycl::free(p, q);
        }
    return 0;
}
