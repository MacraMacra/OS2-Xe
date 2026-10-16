// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rostislavs Rostovskis, Aleksandrs Platonenko
//
// ozaki1_gpu.hpp - Ozaki Scheme I (slicing) baseline on the same INT8 GEMM, used for comparison only.

#pragma once
#include "ozaki2_gpu.hpp"

namespace oz1 {
template <int L, bool PER_ROW, bool REV>
inline sycl::event launch_slices(sycl::queue& q, const double* X, int64_t ldx, int64_t rows, int64_t cols,
                                 const int* ex, int8_t* P, int64_t ldp, int64_t plane, sycl::event dep) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::range<2>(static_cast<size_t>(cols), static_cast<size_t>(ldp / 4)),
                       [=](sycl::id<2> id) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t c = id[0];
            const int64_t r0 = static_cast<int64_t>(id[1]) * 4;
            const int ec = PER_ROW ? 0 : ex[c];
            int64_t x[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int64_t r = r0 + e;
                const int ee = PER_ROW ? (r < rows ? ex[r] : 0) : ec;
                double v = 0.0;
                if (r < rows && ee != oz2::kNonFinite) v = X[r + c * ldx];
                uint64_t uh, ul;
                oz2::detail::scaled_offset_int<62>(v, ee, uh, ul);
                x[e] = static_cast<int64_t>(ul - (uint64_t{1} << 62));
            }
            uint32_t packed[L];
#pragma unroll
            for (int i = 0; i < L; ++i) packed[i] = 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
#pragma unroll
                for (int i = L - 1; i >= 0; --i) {
                    const int64_t d = ((x[e] + 64) & 127) - 64;
                    x[e] = (x[e] - d) >> 7;
                    packed[i] |= (static_cast<uint32_t>(d) & 0xFFu) << (8 * e);
                }
            }
#pragma unroll
            for (int i = 0; i < L; ++i)
                *reinterpret_cast<uint32_t*>(P + (REV ? L - 1 - i : i) * plane + r0 + c * ldp) = packed[i];
        });
    });
}

inline sycl::event launch_horner(sycl::queue& q, const int32_t* Ct, int64_t plane, int cnt, double* Th,
                                 double* Tl, bool first, bool last, int shift, const int* D, const int* E,
                                 double* C, int64_t m, int64_t n, int64_t ldc, sycl::event dep) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::range<2>(static_cast<size_t>(n), static_cast<size_t>(m)), [=](sycl::id<2> id) {
            const int64_t j = id[0], i = id[1];
            const int64_t idx = i + j * m;
            int64_t S = 0;
            for (int c = 0; c < cnt; ++c) S += Ct[c * plane + idx];
            double hi = first ? 0.0 : Th[idx] * 128.0, lo = first ? 0.0 : Tl[idx] * 128.0;
            const double sd = static_cast<double>(S);
            const double s = hi + sd, bb = s - hi;
            const double err = (hi - (s - bb)) + (sd - bb) + lo;
            hi = s + err;
            lo = err - (hi - s);
            if (last) {
                const int De = D[i], Ee = E[j];
                double v = sycl::ldexp(hi + lo, shift - De - Ee);
                if (De == oz2::kNonFinite || Ee == oz2::kNonFinite) v = std::numeric_limits<double>::quiet_NaN();
                C[i + j * ldc] = v;
            } else {
                Th[idx] = hi;
                Tl[idx] = lo;
            }
        });
    });
}

class Ozaki1 {
public:
    explicit Ozaki1(sycl::queue& q) : q_(q) {}
    ~Ozaki1() { release(); }

    static int precision_bits(int L) { return 7 * L - 2; }
    static int num_gemms(int L) { return L * (L + 1) / 2; }

    sycl::event gemm(int64_t m, int64_t n, int64_t k, const double* A, int64_t lda, const double* B, int64_t ldb,
                     double* C, int64_t ldc, int L) {
        if (L < 2 || L > 9) throw std::invalid_argument("oz1: L must be in [2, 9]");
        if (k * 4096 >= (int64_t{1} << 31)) throw std::invalid_argument("oz1: k too large (int32 GEMM)");
        switch (L) {
#define OZ1_CASE(LL) case LL: return run<LL>(m, n, k, A, lda, B, ldb, C, ldc);
            OZ1_CASE(2) OZ1_CASE(3) OZ1_CASE(4) OZ1_CASE(5) OZ1_CASE(6) OZ1_CASE(7) OZ1_CASE(8) OZ1_CASE(9)
#undef OZ1_CASE
        }
        return {};
    }
    void release() {
        if (ws_) { q_.wait(); sycl::free(ws_, q_); }
        ws_ = nullptr;
        ws_size_ = 0;
    }
    size_t workspace_bytes() const { return ws_size_; }

private:
    template <int L>
    sycl::event run(int64_t m, int64_t n, int64_t k, const double* A, int64_t lda, const double* B, int64_t ldb,
                    double* C, int64_t ldc) {
        using namespace oz2::detail;
        const int tk = precision_bits(L);
        const int64_t ldpA = round_up(m, 4), ldpB = round_up(k, 4);
        const int64_t planeA = ldpA * k, planeB = ldpB * n, planeC = m * n;
        size_t off = 0;
        auto carve = [&](size_t b) { size_t o = off; off = align_up(off + b, 256); return o; };
        const size_t oA = carve(planeA * L), oB = carve(planeB * L), oC = carve(planeC * L * 4),
                     oH = carve(planeC * 8), oL = carve(planeC * 8), oMa = carve(m * 8), oMb = carve(n * 8),
                     oD = carve(m * 4), oE = carve(n * 4);
        if (ws_size_ < off) {
            release();
            ws_ = sycl::malloc_device<std::byte>(off, q_);
            if (!ws_) throw std::runtime_error("oz1: workspace allocation failed");
            ws_size_ = off;
        }
        auto* Ap = reinterpret_cast<int8_t*>(ws_ + oA);
        auto* Bp = reinterpret_cast<int8_t*>(ws_ + oB);
        auto* Ct = reinterpret_cast<int32_t*>(ws_ + oC);
        auto* Th = reinterpret_cast<double*>(ws_ + oH);
        auto* Tl = reinterpret_cast<double*>(ws_ + oL);
        auto* maxA = reinterpret_cast<uint64_t*>(ws_ + oMa);
        auto* maxB = reinterpret_cast<uint64_t*>(ws_ + oMb);
        auto* D = reinterpret_cast<int*>(ws_ + oD);
        auto* E = reinterpret_cast<int*>(ws_ + oE);

        sycl::event ev = q_.memset(maxA, 0, m * 8);
        ev = launch_row_maxabs(q_, A, lda, m, k, maxA, ev);
        ev = launch_col_maxabs(q_, B, ldb, k, n, maxB, ev);
        ev = launch_exponents(q_, maxA, m, tk, D, ev);
        ev = launch_exponents(q_, maxB, n, tk, E, ev);
        ev = launch_slices<L, true, false>(q_, A, lda, m, k, D, Ap, ldpA, planeA, ev);
        ev = launch_slices<L, false, true>(q_, B, ldb, k, n, E, Bp, ldpB, planeB, ev);
        for (int d = 0; d < L; ++d) {
            ev = oneapi::mkl::blas::column_major::gemm_batch(
                q_, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, k, 1.0f,
                Ap, ldpA, planeA, Bp + (L - 1 - d) * planeB, ldpB, planeB, 0.0f, Ct, m, planeC, d + 1,
                oneapi::mkl::blas::compute_mode::unset, {ev});
            ev = launch_horner(q_, Ct, planeC, d + 1, Th, Tl, d == 0, d == L - 1,
                               7 * (L - 1), D, E, C, m, n, ldc, ev);
        }
        return ev;
    }

    sycl::queue& q_;
    std::byte* ws_ = nullptr;
    size_t ws_size_ = 0;
};

}
