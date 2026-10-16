// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rostislavs Rostovskis, Aleksandrs Platonenko
//
// test_ozaki2.cpp - correctness tests for ozaki2_gpu.hpp against a double-double reference.

#include <cinttypes>
#include <complex>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ozaki2_gpu.hpp"

using oz2::Trans;

static int g_fail = 0, g_pass = 0;

struct Mat {
    int64_t rows = 0, cols = 0, ld = 0;
    std::vector<double> h;
    double& at(int64_t r, int64_t c) { return h[r + c * ld]; }
    double at(int64_t r, int64_t c) const { return h[r + c * ld]; }
};

enum class Dist { Uniform, Phi, Ints, MaxMag, SinCos };

static Mat make_mat(int64_t rows, int64_t cols, int64_t ld, Dist d, double phi, uint64_t seed) {
    Mat M;
    M.rows = rows; M.cols = cols; M.ld = ld;
    M.h.assign(static_cast<size_t>(ld) * cols, 0.0);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(-0.5, 0.5);
    std::normal_distribution<double> N(0.0, 1.0);
    std::uniform_int_distribution<int> I(-1000, 1000);
    for (int64_t c = 0; c < cols; ++c)
        for (int64_t r = 0; r < rows; ++r) {
            double v = 0;
            switch (d) {
                case Dist::Uniform: v = U(rng); break;
                case Dist::Phi: v = U(rng) * std::exp(phi * N(rng)); break;
                case Dist::Ints: v = I(rng); break;
                case Dist::MaxMag: v = 1.0 - std::ldexp(1.0, -53); break;
                case Dist::SinCos: v = std::sin(static_cast<double>(r + 3 * c)); break;
            }
            M.at(r, c) = v;
        }
    for (int64_t c = 0; c < cols; ++c)
        for (int64_t r = rows; r < ld; ++r) M.at(r, c) = 1e300;
    return M;
}

static double opv(const Mat& X, Trans t, int64_t r, int64_t c) { return t == Trans::N ? X.at(r, c) : X.at(c, r); }

static void dd_reference(sycl::queue& q, Trans ta, Trans tb, int64_t m, int64_t n, int64_t k,
                         const double* dA, int64_t lda, const double* dB, int64_t ldb,
                         std::vector<double>& hi, std::vector<double>& lo) {
    double* dhi = sycl::malloc_device<double>(m * n, q);
    double* dlo = sycl::malloc_device<double>(m * n, q);
    const bool tA = ta == Trans::T, tB = tb == Trans::T;
    q.parallel_for(sycl::range<2>(n, m), [=](sycl::id<2> id) {
        const int64_t j = id[0], i = id[1];
        double sh = 0, sl = 0;
        for (int64_t p = 0; p < k; ++p) {
            const double a = tA ? dA[p + i * lda] : dA[i + p * lda];
            const double b = tB ? dB[j + p * ldb] : dB[p + j * ldb];
            const double ph = a * b;
            const double pl = sycl::fma(a, b, -ph);
            const double s = sh + ph;
            const double bb = s - sh;
            const double e = (sh - (s - bb)) + (ph - bb);
            const double t = e + sl + pl;
            sh = s + t;
            sl = t - (sh - s);
        }
        dhi[i + j * m] = sh;
        dlo[i + j * m] = sl;
    }).wait();
    hi.resize(m * n); lo.resize(m * n);
    q.memcpy(hi.data(), dhi, m * n * sizeof(double));
    q.memcpy(lo.data(), dlo, m * n * sizeof(double)).wait();
    sycl::free(dhi, q); sycl::free(dlo, q);
}

static int frexp_exp(double v) { int e; std::frexp(v, &e); return e; }

struct CaseResult {
    double max_rel_bound = 0;
    double max_err_over_AB = 0;
    double max_rel_err = 0;
    bool bitexact = true;
};

static CaseResult run_case(sycl::queue& q, oz2::Ozaki2& oz, const char* name,
                           Trans ta, Trans tb, int64_t m, int64_t n, int64_t k, int s,
                           Dist dist, double phi, double alpha, double beta,
                           int64_t lda_extra = 0, int64_t ldc_extra = 0,
                           bool expect_exact = false, uint64_t seed = 1,
                           std::vector<double>* out_C = nullptr) {
    const int64_t Ar = ta == Trans::N ? m : k, Ac = ta == Trans::N ? k : m;
    const int64_t Br = tb == Trans::N ? k : n, Bc = tb == Trans::N ? n : k;
    Mat A = make_mat(Ar, Ac, Ar + lda_extra, dist, phi, seed);
    Mat B = make_mat(Br, Bc, Br + lda_extra, dist, phi, seed * 7 + 3);
    const int64_t ldc = m + ldc_extra;
    std::vector<double> C0(ldc * n);
    {
        std::mt19937_64 rng(seed + 99);
        std::uniform_real_distribution<double> U(-1, 1);
        for (auto& v : C0) v = U(rng);
    }

    double* dA = sycl::malloc_device<double>(A.h.size(), q);
    double* dB = sycl::malloc_device<double>(B.h.size(), q);
    double* dC = sycl::malloc_device<double>(C0.size(), q);
    q.memcpy(dA, A.h.data(), A.h.size() * 8);
    q.memcpy(dB, B.h.data(), B.h.size() * 8);
    q.memcpy(dC, C0.data(), C0.size() * 8).wait();

    oz.gemm(ta, tb, m, n, k, alpha, dA, A.ld, dB, B.ld, beta, dC, ldc, s).wait();
    std::vector<double> C(C0.size());
    q.memcpy(C.data(), dC, C.size() * 8).wait();

    std::vector<double> hi, lo;
    dd_reference(q, ta, tb, m, n, k, dA, A.ld, dB, B.ld, hi, lo);
    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);

    const int tk = oz2::Ozaki2::precision_bits(k, s);
    std::vector<int> D(m), E(n);
    std::vector<double> rowsumA(m, 0), colsumB(n, 0);
    for (int64_t i = 0; i < m; ++i) {
        double mx = 0;
        for (int64_t p = 0; p < k; ++p) {
            mx = std::max(mx, std::fabs(opv(A, ta, i, p)));
            rowsumA[i] += std::fabs(opv(A, ta, i, p));
        }
        D[i] = mx == 0 ? 0 : tk - frexp_exp(mx);
    }
    for (int64_t j = 0; j < n; ++j) {
        double mx = 0;
        for (int64_t p = 0; p < k; ++p) {
            mx = std::max(mx, std::fabs(opv(B, tb, p, j)));
            colsumB[j] += std::fabs(opv(B, tb, p, j));
        }
        E[j] = mx == 0 ? 0 : tk - frexp_exp(mx);
    }
    std::vector<double> absAB(m * n, 0);
#pragma omp parallel for
    for (int64_t j = 0; j < n; ++j)
        for (int64_t p = 0; p < k; ++p) {
            const double b = std::fabs(opv(B, tb, p, j));
            for (int64_t i = 0; i < m; ++i) absAB[i + j * m] += std::fabs(opv(A, ta, i, p)) * b;
        }

    CaseResult res;
    const double u = std::ldexp(1.0, -53);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i) {
            const double got = C[i + j * ldc];
            const double c0 = C0[i + j * ldc];
            const double ref_hi = hi[i + j * m], ref_lo = lo[i + j * m];
            const double tb_err = 0.5 * std::ldexp(colsumB[j], -D[i]) + 0.5 * std::ldexp(rowsumA[i], -E[j]) +
                                  0.25 * static_cast<double>(k) * std::ldexp(1.0, -D[i] - E[j]);
            double err, bound;
            if (alpha == 1.0 && beta == 0.0) {
                err = std::fabs((got - ref_hi) - ref_lo);
                bound = tb_err * (1 + 2 * u) + u * std::fabs(got) + 1e-300;
            } else {
                const double ref = alpha * (ref_hi + ref_lo) + beta * c0;
                err = std::fabs(got - ref);
                bound = std::fabs(alpha) * tb_err * (1 + 4 * u) +
                        4 * u * (std::fabs(alpha * ref_hi) + std::fabs(beta * c0)) + 1e-300;
            }
            res.max_rel_bound = std::max(res.max_rel_bound, err / bound);
            if (absAB[i + j * m] > 0) res.max_err_over_AB = std::max(res.max_err_over_AB, err / absAB[i + j * m]);
            if (std::fabs(ref_hi) > 0) res.max_rel_err = std::max(res.max_rel_err, err / std::fabs(ref_hi));
            if (got != ref_hi || ref_lo != 0) res.bitexact = false;
        }
    bool pad_ok = true;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = m; i < ldc; ++i)
            if (C[i + j * ldc] != C0[i + j * ldc]) pad_ok = false;

    const bool ok = res.max_rel_bound <= 1.0 && pad_ok && (!expect_exact || res.bitexact);
    std::printf("[%s] %-34s %c%c m=%-6" PRId64 " n=%-6" PRId64 " k=%-7" PRId64 " s=%-2d tk=%-2d err/bound=%.3f "
                "err/|A||B|=%.2e%s%s\n",
                ok ? "PASS" : "FAIL", name, ta == Trans::N ? 'N' : 'T', tb == Trans::N ? 'N' : 'T', m, n, k, s, tk,
                res.max_rel_bound, res.max_err_over_AB, expect_exact ? (res.bitexact ? " exact" : " NOT-EXACT") : "",
                pad_ok ? "" : " PAD-CORRUPTED");
    (ok ? g_pass : g_fail)++;
    if (out_C) *out_C = C;
    return res;
}

template <int S>
static void unit_split(sycl::queue& q, int tk) {
    using namespace oz2::detail;
    Plan p = make_plan(S, 16);
    const auto tab = make_split_tab<S>(p);
    const int64_t rows = 4096, cols = 3;
    std::mt19937_64 rng(1234 + S + tk);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> X(rows * cols);
    std::vector<int> ex(cols);
    for (int64_t r = 0; r < rows; ++r) {
        X[r] = U(rng);
        X[r + rows] = U(rng) * std::ldexp(1.0, -static_cast<int>(rng() % 120));
        X[r + 2 * rows] = (r % 7 == 0) ? 0.0 : (r % 7 == 1) ? -0.0 : U(rng) * std::ldexp(1.0, -1030 - static_cast<int>(rng() % 40));
    }
    for (int64_t c = 0; c < cols; ++c) {
        double mx = 0;
        for (int64_t r = 0; r < rows; ++r) mx = std::max(mx, std::fabs(X[r + c * rows]));
        ex[c] = tk - frexp_exp(mx);
    }
    double* dX = sycl::malloc_device<double>(X.size(), q);
    int* dE = sycl::malloc_device<int>(cols, q);
    int8_t* dP = sycl::malloc_device<int8_t>(rows * cols * S, q);
    q.memcpy(dX, X.data(), X.size() * 8);
    q.memcpy(dE, ex.data(), cols * 4).wait();
    if (tk > oz2::kNarrowScaleBits) launch_split<S, false, 5>(q, dX, rows, rows, cols, dE, dP, rows, rows * cols, tab, {}).wait();
    else launch_split<S, false, 3>(q, dX, rows, rows, cols, dE, dP, rows, rows * cols, tab, {}).wait();
    std::vector<int8_t> P(rows * cols * S);
    q.memcpy(P.data(), dP, P.size()).wait();
    sycl::free(dX, q); sycl::free(dE, q); sycl::free(dP, q);

    size_t bad = 0;
    for (int64_t c = 0; c < cols; ++c)
        for (int64_t r = 0; r < rows; ++r) {
            const double v = X[r + c * rows];
            int qe;
            const double fr = std::frexp(std::fabs(v), &qe);
            const __int128 mant = static_cast<__int128>(std::ldexp(fr, 53));
            const int sh = qe - 53 + ex[c];
            __int128 x;
            if (v == 0) x = 0;
            else if (sh >= 0) x = mant << sh;
            else if (sh < -60) x = 0;
            else {
                const __int128 qv = mant >> (-sh), rem = mant - (qv << (-sh)), half = static_cast<__int128>(1) << (-sh - 1);
                x = qv + ((rem > half || (rem == half && (qv & 1))) ? 1 : 0);
            }
            if (v < 0) x = -x;
            for (int t = 0; t < S; ++t) {
                const int m = static_cast<int>(p.m[t]), h = (m - 1) / 2;
                int rr = static_cast<int>(((x % m) + m) % m);
                rr = (rr + h) % m - h;
                if (static_cast<int8_t>(rr) != P[r + c * rows + t * rows * cols]) ++bad;
            }
        }
    const bool ok = bad == 0;
    std::printf("[%s] unit split: S=%-2d tk=%-3d (%s path), %lld residues, %zu mismatches\n", ok ? "PASS" : "FAIL", S, tk,
                tk > oz2::kNarrowScaleBits ? "wide" : "narrow", (long long)(rows * cols * S), bad);
    (ok ? g_pass : g_fail)++;
}

static double big_to_double_rne(const oz2::detail::Big& mag, bool neg, int scale) {
    int top = oz2::detail::Big::N * 32 - 1;
    auto bit = [&](int i) { return (mag.w[i / 32] >> (i % 32)) & 1u; };
    while (top >= 0 && !bit(top)) --top;
    if (top < 0) return 0.0;
    uint64_t m53 = 0;
    for (int i = top; i > top - 53; --i) m53 = (m53 << 1) | (i >= 0 ? bit(i) : 0);
    const int rb = top - 53;
    const unsigned round = rb >= 0 ? bit(rb) : 0;
    bool sticky = false;
    for (int i = rb - 1; i >= 0; --i) sticky |= bit(i);
    if (round && (sticky || (m53 & 1))) ++m53;
    const double v = std::ldexp(static_cast<double>(m53), top - 52 + scale);
    return neg ? -v : v;
}

template <int S>
static void unit_crt(sycl::queue& q) {
    using namespace oz2::detail;
    using oz2::detail::Big;
    Plan p = make_plan(S, 16);
    const auto tab = make_crt_tab<S>(p);
    const int64_t n = 8192;
    std::mt19937_64 rng(777 + S);
    std::vector<int32_t> Ct(n * S);
    std::vector<double> ref(n);
    std::vector<int> D(n), E(1, 3);
    Big Mq = p.M;
    Mq.divmod_small(4);
    int topbit = Big::N * 32 - 1;
    while (topbit > 0 && !((Mq.w[topbit / 32] >> (topbit % 32)) & 1u)) --topbit;
    for (int64_t i = 0; i < n; ++i) {
        Big x;
        const int bits = (i < 16) ? topbit : static_cast<int>(rng() % (topbit + 1));
        for (int l = 0; l < Big::N; ++l) x.w[l] = static_cast<uint32_t>(rng());
        for (int b = bits; b < Big::N * 32; ++b) x.w[b / 32] &= ~(1u << (b % 32));
        if (i < 8) x = Mq;
        if (x.cmp(Mq) > 0) x = Mq;
        if (i == 8) x = Big{};
        const bool neg = (rng() & 1) && i != 8;
        D[i] = static_cast<int>(rng() % 41) - 20;
        ref[i] = big_to_double_rne(x, neg, -D[i] - E[0]);
        for (int t = 0; t < S; ++t) {
            const int m = static_cast<int>(p.m[t]);
            int r = static_cast<int>(x.mod_small(m));
            if (neg) r = (m - r) % m;
            const int64_t mult = static_cast<int64_t>(rng() % 2000001) - 1000000;
            Ct[i + t * n] = static_cast<int32_t>(r + mult * m);
        }
    }
    int32_t* dC = sycl::malloc_device<int32_t>(Ct.size(), q);
    int* dD = sycl::malloc_device<int>(n, q);
    int* dE = sycl::malloc_device<int>(1, q);
    double* dO = sycl::malloc_device<double>(n, q);
    q.memcpy(dC, Ct.data(), Ct.size() * 4);
    q.memcpy(dD, D.data(), n * 4);
    q.memcpy(dE, E.data(), 4).wait();
    launch_crt<S, false>(q, dC, n, 1, tab, n, 1, dD, dE, 1.0, 0.0, dO, n, {}).wait();
    std::vector<double> out(n);
    q.memcpy(out.data(), dO, n * 8).wait();
    sycl::free(dC, q); sycl::free(dD, q); sycl::free(dE, q); sycl::free(dO, q);
    size_t bad = 0;
    for (int64_t i = 0; i < n; ++i)
        if (out[i] != ref[i]) ++bad;
    const bool ok = bad == 0;
    std::printf("[%s] unit CRT: S=%-2d log2M=%.1f NL=%d, %lld exact integers, %zu mismatches\n", ok ? "PASS" : "FAIL", S,
                p.log2M, p.NL, (long long)n, bad);
    (ok ? g_pass : g_fail)++;
}

static void run_zcase(sycl::queue& q, oz2::Ozaki2& oz, const char* name, Trans ta, Trans tb, int64_t m, int64_t n,
                      int64_t k, int s, Dist dist, double phi, std::complex<double> alpha, std::complex<double> beta,
                      int64_t ld_extra = 0, bool expect_exact = false, uint64_t seed = 3) {
    using cd = std::complex<double>;
    const int64_t Ar = ta == Trans::N ? m : k, Ac = ta == Trans::N ? k : m;
    const int64_t Br = tb == Trans::N ? k : n, Bc = tb == Trans::N ? n : k;
    const int64_t lda = Ar + ld_extra, ldb = Br + ld_extra, ldc = m + ld_extra;
    Mat Are = make_mat(Ar, Ac, lda, dist, phi, seed), Aim = make_mat(Ar, Ac, lda, dist, phi, seed + 1);
    Mat Bre = make_mat(Br, Bc, ldb, dist, phi, seed + 2), Bim = make_mat(Br, Bc, ldb, dist, phi, seed + 3);
    std::vector<cd> A(lda * Ac), B(ldb * Bc), C0(ldc * n);
    for (size_t i = 0; i < A.size(); ++i) A[i] = cd(Are.h[i], Aim.h[i]);
    for (size_t i = 0; i < B.size(); ++i) B[i] = cd(Bre.h[i], Bim.h[i]);
    std::mt19937_64 rng(seed + 9);
    std::uniform_real_distribution<double> U(-1, 1);
    for (auto& v : C0) v = cd(U(rng), U(rng));

    cd* dA = sycl::malloc_device<cd>(A.size(), q);
    cd* dB = sycl::malloc_device<cd>(B.size(), q);
    cd* dC = sycl::malloc_device<cd>(C0.size(), q);
    double* dR = sycl::malloc_device<double>(4 * m * n, q);
    q.memcpy(dA, A.data(), A.size() * 16);
    q.memcpy(dB, B.data(), B.size() * 16);
    q.memcpy(dC, C0.data(), C0.size() * 16).wait();
    oz.zgemm(ta, tb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc, s).wait();
    std::vector<cd> C(C0.size());
    q.memcpy(C.data(), dC, C.size() * 16).wait();

    const int tA = ta == Trans::N ? 0 : ta == Trans::T ? 1 : 2, tB = tb == Trans::N ? 0 : tb == Trans::T ? 1 : 2;
    const double* Ad = reinterpret_cast<const double*>(dA);
    const double* Bd = reinterpret_cast<const double*>(dB);
    q.parallel_for(sycl::range<2>(n, m), [=](sycl::id<2> id) {
        const int64_t j = id[0], i = id[1];
        double h[2] = {0, 0}, l[2] = {0, 0};
        auto dd_add = [&](int c, double a, double b) {
            const double ph = a * b, pl = sycl::fma(a, b, -ph);
            const double s2 = h[c] + ph, bb = s2 - h[c];
            const double e = (h[c] - (s2 - bb)) + (ph - bb);
            const double t = e + l[c] + pl;
            h[c] = s2 + t;
            l[c] = t - (h[c] - s2);
        };
        for (int64_t p = 0; p < k; ++p) {
            const int64_t ia = tA == 0 ? i + p * lda : p + i * lda;
            const int64_t ib = tB == 0 ? p + j * ldb : j + p * ldb;
            const double ar = Ad[2 * ia], ai = tA == 2 ? -Ad[2 * ia + 1] : Ad[2 * ia + 1];
            const double br = Bd[2 * ib], bi = tB == 2 ? -Bd[2 * ib + 1] : Bd[2 * ib + 1];
            dd_add(0, ar, br); dd_add(0, -ai, bi);
            dd_add(1, ar, bi); dd_add(1, ai, br);
        }
        const int64_t o = 4 * (i + j * m);
        dR[o] = h[0]; dR[o + 1] = l[0]; dR[o + 2] = h[1]; dR[o + 3] = l[1];
    }).wait();
    std::vector<double> R(4 * m * n);
    q.memcpy(R.data(), dR, R.size() * 8).wait();
    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q); sycl::free(dR, q);

    auto opA = [&](int64_t i, int64_t p) { cd v = ta == Trans::N ? A[i + p * lda] : A[p + i * lda]; return ta == Trans::C ? std::conj(v) : v; };
    auto opB = [&](int64_t p, int64_t j) { cd v = tb == Trans::N ? B[p + j * ldb] : B[j + p * ldb]; return tb == Trans::C ? std::conj(v) : v; };
    const int tk = oz2::Ozaki2::precision_bits_complex(k, s);
    std::vector<int> D(m), E(n);
    std::vector<double> sa(m, 0), sb(n, 0);
    for (int64_t i = 0; i < m; ++i) {
        double mx = 0;
        for (int64_t p = 0; p < k; ++p) {
            const cd v = opA(i, p);
            mx = std::max({mx, std::fabs(v.real()), std::fabs(v.imag())});
            sa[i] += std::fabs(v.real()) + std::fabs(v.imag());
        }
        D[i] = mx == 0 ? 0 : tk - frexp_exp(mx);
    }
    for (int64_t j = 0; j < n; ++j) {
        double mx = 0;
        for (int64_t p = 0; p < k; ++p) {
            const cd v = opB(p, j);
            mx = std::max({mx, std::fabs(v.real()), std::fabs(v.imag())});
            sb[j] += std::fabs(v.real()) + std::fabs(v.imag());
        }
        E[j] = mx == 0 ? 0 : tk - frexp_exp(mx);
    }
    const double u = std::ldexp(1.0, -53);
    const bool plain = alpha == cd(1, 0) && beta == cd(0, 0);
    double worst = 0;
    bool exact = true, pad_ok = true;
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            const double* r = &R[4 * (i + j * m)];
            const double tb_err = 0.5 * std::ldexp(sb[j], -D[i]) + 0.5 * std::ldexp(sa[i], -E[j]) +
                                  0.5 * static_cast<double>(k) * std::ldexp(1.0, -D[i] - E[j]);
            const cd got = C[i + j * ldc];
            for (int c = 0; c < 2; ++c) {
                const double g = c ? got.imag() : got.real();
                double err, bound;
                if (plain) {
                    err = std::fabs((g - r[2 * c]) - r[2 * c + 1]);
                    bound = tb_err * (1 + 4 * u) + u * std::fabs(g) + 1e-300;
                    if (g != r[2 * c] || r[2 * c + 1] != 0) exact = false;
                } else {
                    const cd ex = alpha * cd(r[0] + r[1], r[2] + r[3]) + beta * C0[i + j * ldc];
                    err = std::fabs(g - (c ? ex.imag() : ex.real()));
                    bound = std::abs(alpha) * tb_err * 2 * (1 + 8 * u) +
                            8 * u * (std::abs(alpha) * (std::fabs(r[0]) + std::fabs(r[2])) + std::abs(beta * C0[i + j * ldc])) + 1e-300;
                }
                worst = std::max(worst, err / bound);
            }
        }
        for (int64_t i = m; i < ldc; ++i)
            if (C[i + j * ldc] != C0[i + j * ldc]) pad_ok = false;
    }
    const auto tc = [](Trans t) { return t == Trans::N ? 'N' : t == Trans::T ? 'T' : 'C'; };
    const bool ok = worst <= 1.0 && pad_ok && (!expect_exact || exact);
    std::printf("[%s] ZGEMM %-28s %c%c m=%-5lld n=%-5lld k=%-6lld s=%-2d tk=%-2d err/bound=%.3f%s%s\n", ok ? "PASS" : "FAIL", name,
                tc(ta), tc(tb), (long long)m, (long long)n, (long long)k, s, tk, worst,
                expect_exact ? (exact ? " exact" : " NOT-EXACT") : "", pad_ok ? "" : " PAD-CORRUPTED");
    (ok ? g_pass : g_fail)++;
}

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    oz2::Ozaki2 oz(q);

    const Trans TT[2] = {Trans::N, Trans::T};
    for (Trans ta : TT)
        for (Trans tb : TT) {
            run_case(q, oz, "shape 1x1x1", ta, tb, 1, 1, 1, 8, Dist::Uniform, 0, 1, 0);
            run_case(q, oz, "odd shapes", ta, tb, 37, 53, 129, 12, Dist::Phi, 1.0, 1, 0, 3, 5);
            run_case(q, oz, "k=1", ta, tb, 130, 7, 1, 6, Dist::Phi, 2.0, 1, 0);
            run_case(q, oz, "alpha/beta", ta, tb, 77, 65, 300, 14, Dist::Uniform, 0, -1.5, 0.25, 0, 2);
        }

    for (int s = 2; s <= oz2::kMaxModuli; ++s) {
        try {
            run_case(q, oz, "accuracy vs s, phi=1", Trans::N, Trans::N, 256, 256, 1024, s, Dist::Phi, 1.0, 1, 0);
        } catch (const std::invalid_argument& e) {
            std::printf("[skip] s=%d: %s\n", s, e.what());
        }
    }
    for (double phi : {0.5, 2.0, 4.0})
        run_case(q, oz, ("dynamic range phi=" + std::to_string(phi).substr(0, 3)).c_str(), Trans::N, Trans::N,
                 200, 300, 2000, 16, Dist::Phi, phi, 1, 0);

    run_case(q, oz, "integers (exact)", Trans::N, Trans::N, 100, 120, 1000, 12, Dist::Ints, 0, 1, 0, 0, 0, true);
    run_case(q, oz, "integers T (exact)", Trans::T, Trans::T, 63, 31, 777, 14, Dist::Ints, 0, 1, 0, 1, 1, true);
    for (int s : {3, 4, 8, 12, 16, 20, 24})
        run_case(q, oz, "max magnitude (|C'|~M/4)", Trans::N, Trans::N, 64, 64, 4096, s, Dist::MaxMag, 0, 1, 0);

    {
        std::vector<double> C1, C2;
        run_case(q, oz, "tiling reference (1 tile)", Trans::N, Trans::T, 1000, 900, 512, 10, Dist::Phi, 1.0, 1, 0,
                 0, 0, false, 5, &C1);
        oz.set_workspace_limit(8u << 20);
        run_case(q, oz, "tiling (8 MiB workspace)", Trans::N, Trans::T, 1000, 900, 512, 10, Dist::Phi, 1.0, 1, 0,
                 0, 0, false, 5, &C2);
        const auto ti = oz.last_tiling();
        oz.set_workspace_limit(0);
        const bool same = C1 == C2;
        std::printf("[%s] tiling bitwise identical (tiles %" PRId64 "x%" PRId64 ", tile %" PRId64 "x%" PRId64 ")\n",
                    same ? "PASS" : "FAIL", ti.pm, ti.pn, ti.mb, ti.nb);
        (same ? g_pass : g_fail)++;
    }

    run_case(q, oz, "k chunks (k > 131071)", Trans::N, Trans::N, 16, 24, 300000, 18, Dist::Phi, 0.5, 1, 0);
    run_case(q, oz, "k chunks T", Trans::T, Trans::N, 8, 8, 140000, 18, Dist::Uniform, 0, 1, 0);

    {
        const Trans TC[3] = {Trans::N, Trans::T, Trans::C};
        for (Trans ta : TC)
            for (Trans tb : TC)
                run_zcase(q, oz, "shapes/ops", ta, tb, 37, 29, 101, 14, Dist::Phi, 1.0, {1, 0}, {0, 0}, 3);
        run_zcase(q, oz, "alpha/beta complex", Trans::N, Trans::C, 64, 48, 300, 16, Dist::Uniform, 0, {0.5, -1.25}, {0.25, 2.0}, 1);
        for (int s : {4, 8, 12, 16, 20, 24})
            run_zcase(q, oz, "accuracy vs s", Trans::N, Trans::N, 128, 128, 1024, s, Dist::Phi, 1.0, {1, 0}, {0, 0});
        run_zcase(q, oz, "integers (exact)", Trans::N, Trans::T, 60, 70, 900, 14, Dist::Ints, 0, {1, 0}, {0, 0}, 0, true);
        run_zcase(q, oz, "max magnitude", Trans::N, Trans::N, 32, 32, 2048, 12, Dist::MaxMag, 0, {1, 0}, {0, 0});
        run_zcase(q, oz, "max magnitude wide", Trans::C, Trans::N, 32, 32, 2048, 24, Dist::MaxMag, 0, {1, 0}, {0, 0});
        run_zcase(q, oz, "k chunks", Trans::N, Trans::N, 8, 12, 140000, 18, Dist::Uniform, 0, {1, 0}, {0, 0});
        oz.set_workspace_limit(4u << 20);
        run_zcase(q, oz, "tiling (4 MiB workspace)", Trans::T, Trans::C, 300, 260, 200, 10, Dist::Phi, 1.0, {1, 0}, {0, 0});
        oz.set_workspace_limit(0);
    }

    unit_split<8>(q, 24);
    unit_split<12>(q, 40);
    unit_split<16>(q, 55);
    unit_split<18>(q, 62);
    unit_split<20>(q, 70);
    unit_split<24>(q, 85);
    unit_split<24>(q, 100);
#define OZ2_UNIT_CRT(S) unit_crt<S>(q);
    OZ2_UNIT_CRT(2) OZ2_UNIT_CRT(3) OZ2_UNIT_CRT(4) OZ2_UNIT_CRT(5) OZ2_UNIT_CRT(6) OZ2_UNIT_CRT(7) OZ2_UNIT_CRT(8)
    OZ2_UNIT_CRT(9) OZ2_UNIT_CRT(10) OZ2_UNIT_CRT(11) OZ2_UNIT_CRT(12) OZ2_UNIT_CRT(13) OZ2_UNIT_CRT(14)
    OZ2_UNIT_CRT(15) OZ2_UNIT_CRT(16) OZ2_UNIT_CRT(17) OZ2_UNIT_CRT(18) OZ2_UNIT_CRT(19) OZ2_UNIT_CRT(20)
    OZ2_UNIT_CRT(21) OZ2_UNIT_CRT(22) OZ2_UNIT_CRT(23) OZ2_UNIT_CRT(24)
#undef OZ2_UNIT_CRT

    {
        const int64_t m = 32, n = 32, k = 64;
        std::vector<double> A(m * k, 0.5), B(k * n, 0.25), C(m * n, 0);
        A[3 + 10 * m] = INFINITY;
        B[5 + 7 * k] = NAN;
        double *dA = sycl::malloc_device<double>(A.size(), q), *dB = sycl::malloc_device<double>(B.size(), q),
               *dC = sycl::malloc_device<double>(C.size(), q);
        q.memcpy(dA, A.data(), A.size() * 8);
        q.memcpy(dB, B.data(), B.size() * 8).wait();
        oz.gemm(Trans::N, Trans::N, m, n, k, 1.0, dA, m, dB, k, 0.0, dC, m, 8).wait();
        q.memcpy(C.data(), dC, C.size() * 8).wait();
        bool ok = true;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i) {
                const double v = C[i + j * m];
                const bool expect_nan = i == 3 || j == 7;
                if (expect_nan ? !std::isnan(v) : v != 0.5 * 0.25 * k) ok = false;
            }
        std::printf("[%s] non-finite propagation\n", ok ? "PASS" : "FAIL");
        (ok ? g_pass : g_fail)++;
        sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
    }

    if (!quick) {
        run_case(q, oz, "large sincos", Trans::N, Trans::N, 2048, 2048, 4096, 16, Dist::SinCos, 0, 1, 0);
        run_case(q, oz, "large phi=1 s=8", Trans::N, Trans::N, 3000, 2500, 4000, 8, Dist::Phi, 1.0, 1, 0);
        run_case(q, oz, "large phi=1 s=20", Trans::T, Trans::N, 2500, 3000, 4000, 20, Dist::Phi, 1.0, 1, 0);
    }

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
