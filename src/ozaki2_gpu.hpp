// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rostislavs Rostovskis, Aleksandrs Platonenko
//
// ozaki2_gpu.hpp - Ozaki Scheme II DGEMM/ZGEMM emulation on INT8 matrix engines of Intel GPUs (SYCL + oneMKL).

#pragma once
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <complex>
#include <vector>

#include "coprime_tables.hpp"

namespace oz2 {
inline constexpr int kMaxModuli = 24;
inline constexpr int kMaxLimbs = 6;
inline constexpr int kMaxScaleBits = 100;
inline constexpr int kNarrowScaleBits = 62;
inline constexpr int kNonFinite = std::numeric_limits<int>::min();
inline constexpr int64_t kMaxKChunk = 131071;

enum class Trans { N, T, C };

struct StageTimes {
    double scale_ms = 0, split_ms = 0, gemm_ms = 0, crt_ms = 0;
    double total_ms() const { return scale_ms + split_ms + gemm_ms + crt_ms; }
};

struct Tuning {
    bool batched_gemm = true;
    bool pad_lda = true;
    bool pow2_tiles = true;
};

struct TileInfo {
    int64_t mb = 0, nb = 0;
    int64_t pm = 0, pn = 0;
    int64_t k_chunks = 0;
    size_t workspace_bytes = 0;
};

namespace detail {
constexpr int limbs_for(int s) {
    const int nl = static_cast<int>((COPRIME_M[s] + 1.0) / 32.0) + 1;
    return nl < 2 ? 2 : nl;
}
static_assert(limbs_for(kMaxModuli) <= kMaxLimbs, "kMaxModuli needs more limbs");

template <int S>
struct SplitTab {
    uint32_t mu[S];
    uint16_t m[S];
    uint16_t c[S][4];
    uint16_t K3[S];
    uint16_t K5[S];
    uint16_t h[S];
};

template <int S>
struct CrtTab {
    static constexpr int L16 = 2 * limbs_for(S);
    uint32_t mu[S];
    float frac[S];
    uint16_t m[S];
    uint16_t K[S];
    uint16_t W16[S][L16];
    uint16_t M16[L16];
};

struct Big {
    static constexpr int N = 8;
    uint32_t w[N] = {};

    static Big from(uint64_t v) {
        Big b;
        b.w[0] = static_cast<uint32_t>(v);
        b.w[1] = static_cast<uint32_t>(v >> 32);
        return b;
    }
    void mul_small(uint32_t f) {
        uint64_t c = 0;
        for (int i = 0; i < N; ++i) {
            uint64_t t = static_cast<uint64_t>(w[i]) * f + c;
            w[i] = static_cast<uint32_t>(t);
            c = t >> 32;
        }
        if (c) throw std::overflow_error("oz2: Big overflow");
    }
    uint32_t divmod_small(uint32_t d) {
        uint64_t r = 0;
        for (int i = N - 1; i >= 0; --i) {
            uint64_t cur = (r << 32) | w[i];
            w[i] = static_cast<uint32_t>(cur / d);
            r = cur % d;
        }
        return static_cast<uint32_t>(r);
    }
    uint32_t mod_small(uint32_t d) const { Big t = *this; return t.divmod_small(d); }
    void shl(int bits) {
        Big r;
        const int limbs = bits / 32, b = bits % 32;
        for (int i = N - 1; i >= 0; --i) {
            const int src = i - limbs;
            if (src < 0) continue;
            uint64_t v = static_cast<uint64_t>(w[src]) << b;
            if (b && src > 0) v |= static_cast<uint64_t>(w[src - 1]) >> (32 - b);
            r.w[i] = static_cast<uint32_t>(v);
        }
        if (limbs < N && b && (w[N - 1 - limbs] >> (32 - b))) throw std::overflow_error("oz2: Big shl overflow");
        *this = r;
    }
    int cmp(const Big& o) const {
        for (int i = N - 1; i >= 0; --i)
            if (w[i] != o.w[i]) return w[i] < o.w[i] ? -1 : 1;
        return 0;
    }
    double log2() const {
        int top = N - 1;
        while (top > 0 && w[top] == 0) --top;
        double v = 0;
        for (int i = top; i >= std::max(0, top - 2); --i) v = v * 4294967296.0 + w[i];
        return std::log2(v) + 32.0 * std::max(0, top - 2);
    }
};

inline int64_t mod_inverse(int64_t a, int64_t m) {
    int64_t t = 0, nt = 1, r = m, nr = ((a % m) + m) % m;
    while (nr) {
        int64_t q = r / nr;
        std::tie(t, nt) = std::make_pair(nt, t - q * nt);
        std::tie(r, nr) = std::make_pair(nr, r - q * nr);
    }
    if (r != 1) throw std::runtime_error("oz2: moduli are not pairwise coprime");
    return (t % m + m) % m;
}

struct Plan {
    int s = 0;
    int NL = 0;
    int tk = 0;
    double log2M = 0;
    uint32_t m[kMaxModuli] = {}, y[kMaxModuli] = {};
    Big W[kMaxModuli];
    Big M;
};

inline Plan make_plan(int s, int64_t k) {
    if (s < 2 || s > kMaxModuli)
        throw std::invalid_argument("oz2: num_moduli must be in [2, " + std::to_string(kMaxModuli) + "]");
    if (k < 1) throw std::invalid_argument("oz2: k must be >= 1");

    Plan p;
    p.s = s;
    const uint16_t* mods = COPRIME_DATA + COPRIME_OFFSET[s];

    p.M = Big::from(1);
    for (int t = 0; t < s; ++t) {
        if (mods[t] < 2 || mods[t] > 256) throw std::runtime_error("oz2: modulus out of int8 range");
        p.M.mul_small(mods[t]);
    }
    p.log2M = p.M.log2();
    p.NL = std::max(2, static_cast<int>(std::floor(p.log2M / 32.0)) + 1);
    if (p.NL != limbs_for(s)) throw std::logic_error("oz2: limb count mismatch with COPRIME_M table");

    int tk = static_cast<int>(std::floor((p.log2M - 2.0 - std::log2(static_cast<double>(k))) / 2.0));
    tk = std::min(tk, kMaxScaleBits);
    for (;; --tk) {
        if (tk < 1)
            throw std::invalid_argument("oz2: num_moduli=" + std::to_string(s) +
                                        " is too small for k=" + std::to_string(k));
        Big lhs = Big::from(static_cast<uint64_t>(k));
        lhs.shl(2 * tk + 2);
        if (lhs.cmp(p.M) <= 0) break;
    }
    p.tk = tk;

    for (int t = 0; t < s; ++t) {
        const uint32_t m = mods[t];
        Big Mt = p.M;
        if (Mt.divmod_small(m) != 0) throw std::runtime_error("oz2: bad modulus table");
        const int64_t y = mod_inverse(Mt.mod_small(m), m);
        p.W[t] = Mt;
        p.W[t].mul_small(static_cast<uint32_t>(y));
        p.m[t] = m;
        p.y[t] = static_cast<uint32_t>(y);
    }
    return p;
}

template <int S>
SplitTab<S> make_split_tab(const Plan& p) {
    SplitTab<S> tb{};
    for (int t = 0; t < S; ++t) {
        const uint32_t m = p.m[t];
        const uint32_t h = (m - 1) / 2;
        tb.mu[t] = static_cast<uint32_t>((uint64_t{1} << 32) / m);
        tb.m[t] = static_cast<uint16_t>(m);
        uint32_t p2 = 1;
        for (int b = 1; b <= 104; ++b) {
            p2 = (p2 * 2) % m;
            if (b % 21 == 0 && b / 21 <= 4) tb.c[t][b / 21 - 1] = static_cast<uint16_t>(p2);
            if (b == 62) tb.K3[t] = static_cast<uint16_t>((m - p2 + h) % m);
            if (b == 104) tb.K5[t] = static_cast<uint16_t>((m - p2 + h) % m);
        }
        tb.h[t] = static_cast<uint16_t>(h);
    }
    return tb;
}

template <int S>
CrtTab<S> make_crt_tab(const Plan& p) {
    CrtTab<S> tb{};
    constexpr int L16 = CrtTab<S>::L16;
    for (int t = 0; t < S; ++t) {
        const uint32_t m = p.m[t];
        tb.mu[t] = static_cast<uint32_t>((uint64_t{1} << 32) / m);
        tb.frac[t] = static_cast<float>(static_cast<double>(p.y[t]) / m);
        tb.m[t] = static_cast<uint16_t>(m);
        tb.K[t] = static_cast<uint16_t>((m - (uint64_t{1} << 31) % m) % m);
        for (int l = 0; l < L16; ++l) tb.W16[t][l] = static_cast<uint16_t>(p.W[t].w[l / 2] >> (16 * (l % 2)));
    }
    for (int l = 0; l < L16; ++l) tb.M16[l] = static_cast<uint16_t>(p.M.w[l / 2] >> (16 * (l % 2)));
    return tb;
}

template <class F>
decltype(auto) with_moduli(int s, F&& f) {
    switch (s) {
#define OZ2_CASE(S) case S: return f(std::integral_constant<int, S>{});
        OZ2_CASE(2) OZ2_CASE(3) OZ2_CASE(4) OZ2_CASE(5) OZ2_CASE(6) OZ2_CASE(7) OZ2_CASE(8)
        OZ2_CASE(9) OZ2_CASE(10) OZ2_CASE(11) OZ2_CASE(12) OZ2_CASE(13) OZ2_CASE(14) OZ2_CASE(15)
        OZ2_CASE(16) OZ2_CASE(17) OZ2_CASE(18) OZ2_CASE(19) OZ2_CASE(20) OZ2_CASE(21) OZ2_CASE(22)
        OZ2_CASE(23) OZ2_CASE(24)
#undef OZ2_CASE
    }
    throw std::invalid_argument("oz2: unsupported num_moduli");
}

template <int OFF>
inline void scaled_offset_int(double v, int e, uint64_t& hi, uint64_t& lo) {
    const uint64_t bits = sycl::bit_cast<uint64_t>(v);
    const int bexp = static_cast<int>((bits >> 52) & 0x7FF);
    uint64_t mant = bits & 0xFFFFFFFFFFFFFull;
    if (bexp) mant |= 0x10000000000000ull;
    const int sh = (bexp ? bexp : 1) - 1075 + e;
    uint64_t mh = 0, ml;
    if (sh >= 0) {
        if constexpr (OFF < 64) {
            ml = mant << sh;
        } else {
            ml = sh < 64 ? mant << sh : 0;
            mh = sh == 0 ? 0 : (sh < 64 ? mant >> (64 - sh) : mant << (sh - 64));
        }
    } else if (sh < -54) {
        ml = 0;
    } else {
        const int rs = -sh;
        const uint64_t q = mant >> rs;
        const uint64_t rem = mant & ((uint64_t{1} << rs) - 1);
        const uint64_t half = uint64_t{1} << (rs - 1);
        ml = q + ((rem > half || (rem == half && (q & 1))) ? 1 : 0);
    }
    if constexpr (OFF < 64) {
        constexpr uint64_t off = uint64_t{1} << OFF;
        hi = 0;
        lo = (bits >> 63) ? off - ml : off + ml;
    } else {
        constexpr uint64_t off_hi = uint64_t{1} << (OFF - 64);
        if (bits >> 63) {
            lo = 0 - ml;
            hi = off_hi - mh - (ml != 0 ? 1 : 0);
        } else {
            lo = ml;
            hi = off_hi + mh;
        }
    }
}

template <int P>
inline uint32_t bits21(uint64_t hi, uint64_t lo) {
    constexpr uint64_t M = 0x1FFFFFu;
    if constexpr (P + 21 <= 64) return static_cast<uint32_t>((lo >> P) & M);
    else if constexpr (P >= 64) return static_cast<uint32_t>((hi >> (P - 64)) & M);
    else return static_cast<uint32_t>(((lo >> P) | (hi << (64 - P))) & M);
}

inline uint32_t mod_u32(uint32_t a, uint32_t m, uint32_t mu) {
    const uint32_t q = sycl::mul_hi(a, mu);
    const uint32_t r = a - q * m;
    return sycl::min(r, r - m);
}

inline uint32_t rep_i32(int32_t c, uint32_t m, uint32_t mu, uint32_t K) {
    return mod_u32(static_cast<uint32_t>(c) + 0x80000000u, m, mu) + K;
}

inline sycl::event launch_row_maxabs(sycl::queue& q, const double* X, int64_t ldx,
                                     int64_t rows, int64_t cols, uint64_t* out_bits,
                                     sycl::event dep) {
    constexpr int64_t WG = 256;
    const int64_t target_items = int64_t{1} << 21;
    int64_t chunks = std::max<int64_t>(1, std::min<int64_t>(cols, target_items / std::max<int64_t>(rows, 1)));
    const int64_t cw = (cols + chunks - 1) / chunks;
    chunks = (cols + cw - 1) / cw;
    const int64_t rows_pad = (rows + WG - 1) / WG * WG;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::nd_range<2>({static_cast<size_t>(chunks), static_cast<size_t>(rows_pad)},
                                         {1, static_cast<size_t>(WG)}),
                       [=](sycl::nd_item<2> it) {
            const int64_t r = it.get_global_id(1);
            if (r >= rows) return;
            const int64_t c0 = it.get_global_id(0) * cw;
            const int64_t c1 = sycl::min(cols, c0 + cw);
            uint64_t best = 0;
            for (int64_t c = c0; c < c1; ++c) {
                const uint64_t b = sycl::bit_cast<uint64_t>(X[r + c * ldx]) & 0x7FFFFFFFFFFFFFFFull;
                best = b > best ? b : best;
            }
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(out_bits[r]).fetch_max(best);
        });
    });
}

inline sycl::event launch_col_maxabs(sycl::queue& q, const double* X, int64_t ldx,
                                     int64_t rows, int64_t cols, uint64_t* out_bits,
                                     sycl::event dep) {
    constexpr size_t WG = 256;
    const int64_t groups = std::min<int64_t>(cols, int64_t{1} << 20);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(groups) * WG, WG), [=](sycl::nd_item<1> it) {
            for (int64_t c = it.get_group(0); c < cols; c += groups) {
                const double* col = X + c * ldx;
                uint64_t best = 0;
                for (int64_t r = it.get_local_id(0); r < rows; r += WG) {
                    const uint64_t b = sycl::bit_cast<uint64_t>(col[r]) & 0x7FFFFFFFFFFFFFFFull;
                    best = b > best ? b : best;
                }
                best = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<uint64_t>());
                if (it.get_local_id(0) == 0) out_bits[c] = best;
            }
        });
    });
}

inline sycl::event launch_exponents(sycl::queue& q, const uint64_t* bits, int64_t n, int tk,
                                    int* ex, sycl::event dep, bool pair = false) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
            const uint64_t b = pair ? sycl::max(bits[2 * i], bits[2 * i + 1]) : bits[i];
            int e = 0;
            if (b >= 0x7FF0000000000000ull) {
                e = kNonFinite;
            } else if (b != 0) {
                int e2;
                sycl::frexp(sycl::bit_cast<double>(b), sycl::private_ptr<int>(&e2));
                e = tk - e2;
            }
            ex[i] = e;
        });
    });
}

template <class V = double>
inline sycl::event launch_transpose(sycl::queue& q, const V* X, int64_t ldx, int64_t rows, int64_t cols,
                                    V* Y, int64_t ldy, sycl::event dep) {
    constexpr int T = 32, R = 8;
    const size_t gr = static_cast<size_t>((rows + T - 1) / T) * T;
    const size_t gc = static_cast<size_t>((cols + T - 1) / T) * R;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        sycl::local_accessor<V, 2> tile({T, T + 1}, h);
        h.parallel_for(sycl::nd_range<2>({gc, gr}, {R, T}), [=](sycl::nd_item<2> it) {
            const int ly = it.get_local_id(0), lx = it.get_local_id(1);
            const int64_t c0 = static_cast<int64_t>(it.get_group(0)) * T;
            const int64_t r0 = static_cast<int64_t>(it.get_group(1)) * T;
#pragma unroll
            for (int q4 = 0; q4 < T / R; ++q4) {
                const int64_t r = r0 + lx, c = c0 + ly + R * q4;
                tile[ly + R * q4][lx] = (r < rows && c < cols) ? X[r + c * ldx] : V{};
            }
            sycl::group_barrier(it.get_group());
#pragma unroll
            for (int q4 = 0; q4 < T / R; ++q4) {
                const int64_t c = c0 + lx, r = r0 + ly + R * q4;
                if (r < rows && c < cols) Y[c + r * ldy] = tile[lx][ly + R * q4];
            }
        });
    });
}

template <int S, bool PER_ROW, int NLIMB>
inline sycl::event launch_split(sycl::queue& q, const double* X, int64_t ldx,
                                int64_t rows, int64_t cols, const int* ex,
                                int8_t* P, int64_t ldp, int64_t plane_stride,
                                const SplitTab<S>& tab, sycl::event dep) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::range<2>(static_cast<size_t>(cols), static_cast<size_t>(ldp / 4)),
                       [=](sycl::id<2> id) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t c = id[0];
            const int64_t r0 = static_cast<int64_t>(id[1]) * 4;
            const double* col = X + c * ldx;
            const int ec = PER_ROW ? 0 : ex[c];

            uint32_t a[NLIMB][4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int64_t r = r0 + e;
                const int ee = PER_ROW ? (r < rows ? ex[r] : 0) : ec;
                double v = 0.0;
                if (r < rows && ee != kNonFinite) v = col[r];
                uint64_t uh, ul;
                scaled_offset_int<NLIMB == 3 ? 62 : 104>(v, ee, uh, ul);
                a[0][e] = bits21<0>(uh, ul);
                a[1][e] = bits21<21>(uh, ul);
                a[2][e] = bits21<42>(uh, ul);
                if constexpr (NLIMB == 5) {
                    a[3][e] = bits21<63>(uh, ul);
                    a[4][e] = bits21<84>(uh, ul);
                }
            }

            int8_t* out = P + r0 + c * ldp;
#pragma unroll
            for (int t = 0; t < S; ++t) {
                uint32_t packed = 0;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    uint32_t v = a[0][e] + (NLIMB == 3 ? tab.K3[t] : tab.K5[t]);
#pragma unroll
                    for (int j = 1; j < NLIMB; ++j) v += a[j][e] * uint32_t{tab.c[t][j - 1]};
                    const uint32_t r = mod_u32(v, tab.m[t], tab.mu[t]);
                    packed |= ((r - tab.h[t]) & 0xFFu) << (8 * e);
                }
                *reinterpret_cast<uint32_t*>(out + t * plane_stride) = packed;
            }
        });
    });
}

template <int S, bool PER_ROW, int NLIMB, bool CONJ>
inline sycl::event launch_split_cplx(sycl::queue& q, const double* X, int64_t ldx, int64_t rows, int64_t cols,
                                     const int* ex, int8_t* P, int64_t ldp, int64_t plane_stride,
                                     int64_t group_stride, const SplitTab<S>& tab, sycl::event dep) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::range<2>(static_cast<size_t>(cols), static_cast<size_t>(ldp / 4)),
                       [=](sycl::id<2> id) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t c = id[0];
            const int64_t r0 = static_cast<int64_t>(id[1]) * 4;
            const int ec = PER_ROW ? 0 : ex[c];
            uint32_t a[2][NLIMB][4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int64_t r = r0 + e;
                const int ee = PER_ROW ? (r < rows ? ex[r] : 0) : ec;
                double v[2] = {0.0, 0.0};
                if (r < rows && ee != kNonFinite) {
                    v[0] = X[2 * (r + c * ldx)];
                    v[1] = X[2 * (r + c * ldx) + 1];
                }
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    uint64_t uh, ul;
                    scaled_offset_int<NLIMB == 3 ? 62 : 104>(v[p], ee, uh, ul);
                    a[p][0][e] = bits21<0>(uh, ul);
                    a[p][1][e] = bits21<21>(uh, ul);
                    a[p][2][e] = bits21<42>(uh, ul);
                    if constexpr (NLIMB == 5) {
                        a[p][3][e] = bits21<63>(uh, ul);
                        a[p][4][e] = bits21<84>(uh, ul);
                    }
                }
            }
            int8_t* out = P + r0 + c * ldp;
#pragma unroll
            for (int t = 0; t < S; ++t) {
                const int32_t m = tab.m[t], hh = tab.h[t];
                uint32_t pk[3] = {0, 0, 0};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    int32_t sym[2];
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        uint32_t v = a[p][0][e] + (NLIMB == 3 ? tab.K3[t] : tab.K5[t]);
#pragma unroll
                        for (int j = 1; j < NLIMB; ++j) v += a[p][j][e] * uint32_t{tab.c[t][j - 1]};
                        sym[p] = static_cast<int32_t>(mod_u32(v, m, tab.mu[t])) - hh;
                    }
                    if constexpr (CONJ) sym[1] = -sym[1];
                    int32_t ss = sym[0] + sym[1];
                    ss = ss > m - 1 - hh ? ss - m : ss;
                    ss = ss < -hh ? ss + m : ss;
                    pk[0] |= (static_cast<uint32_t>(sym[0]) & 0xFFu) << (8 * e);
                    pk[1] |= (static_cast<uint32_t>(sym[1]) & 0xFFu) << (8 * e);
                    pk[2] |= (static_cast<uint32_t>(ss) & 0xFFu) << (8 * e);
                }
#pragma unroll
                for (int g = 0; g < 3; ++g)
                    *reinterpret_cast<uint32_t*>(out + g * group_stride + t * plane_stride) = pk[g];
            }
        });
    });
}

struct alignas(16) cplx64 {
    double re, im;
};

template <int S>
struct CrtAcc {
    static constexpr int NL = limbs_for(S);
    static constexpr int L16 = 2 * NL;
    float u = 0.0f;
    uint32_t acc[L16];

    CrtAcc() {
#pragma unroll
        for (int l = 0; l < L16; ++l) acc[l] = 0;
    }
    void add(const CrtTab<S>& tab, int t, uint32_t r) {
        u = sycl::fma(static_cast<float>(r), tab.frac[t], u);
        const uint16_t r16 = static_cast<uint16_t>(r);
#pragma unroll
        for (int l = 0; l < L16; ++l) acc[l] += static_cast<uint32_t>(r16) * static_cast<uint32_t>(tab.W16[t][l]);
    }
    double finish(const CrtTab<S>& tab, int shift) const {
        const int32_t qq = static_cast<int32_t>(sycl::rint(u));
        uint32_t w[NL];
        int32_t carry = 0;
#pragma unroll
        for (int l = 0; l < L16; ++l) {
            const int32_t v = static_cast<int32_t>(acc[l]) - qq * int32_t{tab.M16[l]} + carry;
            const uint32_t lo = static_cast<uint32_t>(v) & 0xFFFFu;
            if (l % 2 == 0) w[l / 2] = lo;
            else w[l / 2] |= lo << 16;
            carry = v >> 16;
        }
        const bool neg = (w[NL - 1] >> 31) != 0;
        if (neg) {
            uint32_t cy = 1;
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const uint32_t nw = ~w[l] + cy;
                cy = (cy && nw == 0) ? 1u : 0u;
                w[l] = nw;
            }
        }
        int sh32 = 0;
#pragma unroll
        for (int it = 0; it < NL - 1; ++it) {
            if (w[NL - 1] == 0) {
#pragma unroll
                for (int l = NL - 1; l > 0; --l) w[l] = w[l - 1];
                w[0] = 0;
                ++sh32;
            }
        }
        if (w[NL - 1] == 0) return 0.0;
        const int z = sycl::clz(w[NL - 1]);
        const uint64_t hi = (static_cast<uint64_t>(w[NL - 1]) << 32) | w[NL - 2];
        uint32_t nxt = 0;
        if constexpr (NL >= 3) nxt = w[NL - 3];
        uint64_t top = z ? (hi << z) | (nxt >> (32 - z)) : hi;
        bool sticky = (z ? (nxt << z) : nxt) != 0;
#pragma unroll
        for (int l = 0; l < NL - 3; ++l) sticky |= w[l] != 0;
        top |= sticky ? 1u : 0u;
        const double val = sycl::ldexp(static_cast<double>(top), 32 * NL - 64 - z - 32 * sh32 - shift);
        return neg ? -val : val;
    }
};

template <int S, bool MULTI_K>
inline uint32_t gemm_residue(const int32_t* Ct, int64_t first, int64_t chunk_stride, int k_chunks,
                             const CrtTab<S>& tab, int t) {
    const uint32_t m = tab.m[t];
    if constexpr (!MULTI_K) {
        return rep_i32(Ct[first], m, tab.mu[t], tab.K[t]);
    } else {
        uint32_t sum = 0;
        for (int c = 0; c < k_chunks; ++c) sum += rep_i32(Ct[first + c * chunk_stride], m, tab.mu[t], tab.K[t]);
        return mod_u32(sum, m, tab.mu[t]);
    }
}

template <int S, bool MULTI_K>
inline sycl::event launch_crt(sycl::queue& q, const int32_t* Ct, int64_t plane_stride,
                              int k_chunks, const CrtTab<S>& tab,
                              int64_t mb, int64_t nb, const int* D, const int* E,
                              double alpha, double beta, double* C, int64_t ldc,
                              sycl::event dep) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::range<2>(static_cast<size_t>(nb), static_cast<size_t>(mb)),
                       [=](sycl::id<2> id) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t j = id[0], i = id[1];
            const int64_t idx = i + j * mb;
            const int De = D[i], Ee = E[j];
            CrtAcc<S> acc;
#pragma unroll
            for (int t = 0; t < S; ++t)
                acc.add(tab, t, gemm_residue<S, MULTI_K>(Ct, t * plane_stride + idx, S * plane_stride, k_chunks, tab, t));
            double val = acc.finish(tab, De + Ee);
            if (De == kNonFinite || Ee == kNonFinite) val = std::numeric_limits<double>::quiet_NaN();
            double* cp = C + i + j * ldc;
            *cp = beta == 0.0 ? alpha * val : alpha * val + beta * *cp;
        });
    });
}

template <int S, bool MULTI_K>
inline sycl::event launch_crt_cplx(sycl::queue& q, const int32_t* Ct, int64_t plane_stride, int k_chunks,
                                   const CrtTab<S>& tab, int64_t mb, int64_t nb, const int* D, const int* E,
                                   double ar, double ai, double br, double bi, double* C, int64_t ldc,
                                   sycl::event dep) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::range<2>(static_cast<size_t>(nb), static_cast<size_t>(mb)),
                       [=](sycl::id<2> id) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t j = id[0], i = id[1];
            const int64_t idx = i + j * mb;
            const int De = D[i], Ee = E[j];
            const int64_t gs = S * plane_stride;
            CrtAcc<S> re, im;
#pragma unroll
            for (int t = 0; t < S; ++t) {
                const uint32_t m = tab.m[t], mu = tab.mu[t];
                const uint32_t p0 = gemm_residue<S, MULTI_K>(Ct, t * plane_stride + idx, 3 * gs, k_chunks, tab, t);
                const uint32_t p1 = gemm_residue<S, MULTI_K>(Ct, gs + t * plane_stride + idx, 3 * gs, k_chunks, tab, t);
                const uint32_t p2 = gemm_residue<S, MULTI_K>(Ct, 2 * gs + t * plane_stride + idx, 3 * gs, k_chunks, tab, t);
                re.add(tab, t, mod_u32(p0 + 2 * m - p1, m, mu));
                im.add(tab, t, mod_u32(p2 + 4 * m - p0 - p1, m, mu));
            }
            double vr = re.finish(tab, De + Ee), vi = im.finish(tab, De + Ee);
            if (De == kNonFinite || Ee == kNonFinite) vr = vi = std::numeric_limits<double>::quiet_NaN();
            double* cp = C + 2 * (i + j * ldc);
            double orr = ar * vr - ai * vi, oi = ar * vi + ai * vr;
            if (br != 0.0 || bi != 0.0) {
                const double cr = cp[0], ci = cp[1];
                orr += br * cr - bi * ci;
                oi += br * ci + bi * cr;
            }
            cp[0] = orr;
            cp[1] = oi;
        });
    });
}

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }
inline int64_t round_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

inline int64_t plane_ld_A(int64_t rows, bool pad = true) {
    const int64_t ld = round_up(rows, 4);
    return pad && ld % 12288 == 0 ? ld + 64 : ld;
}

inline constexpr int64_t kMaxRange = std::numeric_limits<int>::max();

}

class Ozaki2 {
public:
    explicit Ozaki2(sycl::queue& q) : q_(q) {
        profiling_ = q_.has_property<sycl::property::queue::enable_profiling>();
    }
    ~Ozaki2() {
        try { q_.wait(); } catch (...) {}
        release_workspace();
        if (co_) sycl::free(co_, q_);
    }
    Ozaki2(const Ozaki2&) = delete;
    Ozaki2& operator=(const Ozaki2&) = delete;

    static int precision_bits(int64_t k, int num_moduli) { return detail::make_plan(num_moduli, k).tk; }

    static int moduli_for_bits(int64_t k, int bits) {
        for (int s = 2; s <= kMaxModuli; ++s) {
            try {
                if (detail::make_plan(s, k).tk >= bits) return s;
            } catch (const std::invalid_argument&) {}
        }
        return kMaxModuli;
    }

    void set_workspace_limit(size_t bytes) { ws_limit_ = bytes; }
    void set_resident_hint(size_t bytes) { resident_hint_ = bytes; }
    void release_workspace() {
        if (!ws_) return;
        q_.wait();
        sycl::free(ws_, q_);
        ws_ = nullptr;
        ws_size_ = 0;
    }

    const TileInfo& last_tiling() const { return tiling_; }
    void set_tuning(const Tuning& t) { tune_ = t; }
    const Tuning& tuning() const { return tune_; }

    StageTimes stage_times() {
        StageTimes st;
        if (!profiling_ || stages_.empty()) return st;
        stages_.back().second.wait();
        uint64_t prev_end = 0;
        for (size_t i = 0; i < stages_.size(); ++i) {
            auto& ev = stages_[i].second;
            const uint64_t end = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            const uint64_t beg = i == 0 ? ev.get_profiling_info<sycl::info::event_profiling::command_start>()
                                        : prev_end;
            const double ms = end > beg ? (end - beg) * 1e-6 : 0.0;
            prev_end = end;
            switch (stages_[i].first) {
                case Stage::Begin: break;
                case Stage::Scale: st.scale_ms += ms; break;
                case Stage::Split: st.split_ms += ms; break;
                case Stage::Gemm: st.gemm_ms += ms; break;
                case Stage::Crt: st.crt_ms += ms; break;
            }
        }
        return st;
    }

    sycl::event gemm(Trans ta, Trans tb, int64_t m, int64_t n, int64_t k,
                     double alpha, const double* A, int64_t lda,
                     const double* B, int64_t ldb,
                     double beta, double* C, int64_t ldc,
                     int num_moduli, const std::vector<sycl::event>& deps = {}) {
        stages_.clear();
        if (m < 0 || n < 0 || k < 0) throw std::invalid_argument("oz2: negative dimension");
        if (lda < std::max<int64_t>(1, ta == Trans::N ? m : k) ||
            ldb < std::max<int64_t>(1, tb == Trans::N ? k : n) || ldc < std::max<int64_t>(1, m))
            throw std::invalid_argument("oz2: bad leading dimension");
        sycl::event last = q_.ext_oneapi_submit_barrier(deps);
        if (m == 0 || n == 0) return last;
        if (k == 0 || alpha == 0.0) return scale_c(m, n, beta, C, ldc, last);

        if (plan_.s != num_moduli || plan_k_ != k) {
            plan_ = detail::make_plan(num_moduli, k);
            plan_k_ = k;
        }
        return detail::with_moduli(num_moduli, [&](auto S_) {
            return run<decltype(S_)::value>(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, last);
        });
    }

    static int precision_bits_complex(int64_t k, int num_moduli) { return precision_bits(2 * k, num_moduli); }

    sycl::event zgemm(Trans ta, Trans tb, int64_t m, int64_t n, int64_t k,
                      std::complex<double> alpha, const std::complex<double>* A, int64_t lda,
                      const std::complex<double>* B, int64_t ldb,
                      std::complex<double> beta, std::complex<double>* C, int64_t ldc,
                      int num_moduli, const std::vector<sycl::event>& deps = {}) {
        stages_.clear();
        if (m < 0 || n < 0 || k < 0) throw std::invalid_argument("oz2: negative dimension");
        if (lda < std::max<int64_t>(1, ta == Trans::N ? m : k) ||
            ldb < std::max<int64_t>(1, tb == Trans::N ? k : n) || ldc < std::max<int64_t>(1, m))
            throw std::invalid_argument("oz2: bad leading dimension");
        sycl::event last = q_.ext_oneapi_submit_barrier(deps);
        if (m == 0 || n == 0) return last;
        double* Cd = reinterpret_cast<double*>(C);
        if (k == 0 || alpha == 0.0) return scale_cz(m, n, beta, Cd, ldc, last);
        if (plan_.s != num_moduli || plan_k_ != 2 * k) {
            plan_ = detail::make_plan(num_moduli, 2 * k);
            plan_k_ = 2 * k;
        }
        return detail::with_moduli(num_moduli, [&](auto S_) {
            return run_z<decltype(S_)::value>(ta, tb, m, n, k, alpha, reinterpret_cast<const double*>(A), lda,
                                              reinterpret_cast<const double*>(B), ldb, beta, Cd, ldc, last);
        });
    }

private:
    enum class Stage { Begin, Scale, Split, Gemm, Crt };

    template <int S>
    sycl::event run_z(Trans ta, Trans tb, int64_t m, int64_t n, int64_t k, std::complex<double> alpha,
                      const double* A, int64_t lda, const double* B, int64_t ldb, std::complex<double> beta,
                      double* C, int64_t ldc, sycl::event last) {
        using namespace detail;
        const SplitTab<S> stab = make_split_tab<S>(plan_);
        const CrtTab<S> ctab = make_crt_tab<S>(plan_);
        const int64_t kc_len = std::min<int64_t>(k, kMaxKChunk);
        const int64_t k_chunks = (k + kc_len - 1) / kc_len;
        choose_tiling(ta, tb, m, n, k, S, k_chunks, true);
        const int64_t mb = tiling_.mb, nb = tiling_.nb;

        const int64_t ldpA = ta == Trans::N ? plane_ld_A(mb, tune_.pad_lda) : plane_ld_A(k, tune_.pad_lda);
        const int64_t planeA = ldpA * (ta == Trans::N ? k : mb);
        const int64_t ldpB = round_up(k, 4);
        const int64_t planeB = ldpB * nb;
        const int64_t planeC = mb * nb;
        size_t off = 0;
        auto carve = [&](size_t bytes) { size_t o = off; off = align_up(off + bytes, 256); return o; };
        const size_t oA = carve(static_cast<size_t>(planeA) * 3 * S);
        const size_t oB = carve(static_cast<size_t>(planeB) * 3 * S);
        const size_t oC = carve(static_cast<size_t>(planeC) * 3 * S * k_chunks * sizeof(int32_t));
        const size_t oMa = carve(static_cast<size_t>(2 * m) * sizeof(uint64_t));
        const size_t oMb = carve(static_cast<size_t>(2 * n) * sizeof(uint64_t));
        const size_t oD = carve(static_cast<size_t>(m) * sizeof(int));
        const size_t oE = carve(static_cast<size_t>(n) * sizeof(int));
        const size_t oT = carve(tb != Trans::N ? static_cast<size_t>(k) * nb * 16 : 0);
        ensure_workspace(off, last);
        int8_t* Ap = reinterpret_cast<int8_t*>(ws_ + oA);
        int8_t* Bp = reinterpret_cast<int8_t*>(ws_ + oB);
        int32_t* Ct = reinterpret_cast<int32_t*>(ws_ + oC);
        uint64_t* maxA = reinterpret_cast<uint64_t*>(ws_ + oMa);
        uint64_t* maxB = reinterpret_cast<uint64_t*>(ws_ + oMb);
        int* D = reinterpret_cast<int*>(ws_ + oD);
        int* E = reinterpret_cast<int*>(ws_ + oE);
        cplx64* Btmp = reinterpret_cast<cplx64*>(ws_ + oT);

        record(Stage::Begin, last);
        if (ta == Trans::N) {
            last = q_.memset(maxA, 0, 2 * m * sizeof(uint64_t), last);
            last = launch_row_maxabs(q_, A, 2 * lda, 2 * m, k, maxA, last);
            last = launch_exponents(q_, maxA, m, plan_.tk, D, last, true);
        } else {
            last = launch_col_maxabs(q_, A, 2 * lda, 2 * k, m, maxA, last);
            last = launch_exponents(q_, maxA, m, plan_.tk, D, last);
        }
        if (tb == Trans::N) {
            last = launch_col_maxabs(q_, B, 2 * ldb, 2 * k, n, maxB, last);
            last = launch_exponents(q_, maxB, n, plan_.tk, E, last);
        } else {
            last = q_.memset(maxB, 0, 2 * n * sizeof(uint64_t), last);
            last = launch_row_maxabs(q_, B, 2 * ldb, 2 * n, k, maxB, last);
            last = launch_exponents(q_, maxB, n, plan_.tk, E, last, true);
        }
        record(Stage::Scale, last);

        const bool wide = plan_.tk > kNarrowScaleBits;
        auto split = [&](auto per_row, bool conj, const double* X, int64_t ldx, int64_t rows, int64_t cols,
                         const int* ex, int8_t* P, int64_t ldp, int64_t plane, sycl::event dep) {
            constexpr bool PR = decltype(per_row)::value;
            const int64_t gs = plane * S;
            if (wide) return conj ? launch_split_cplx<S, PR, 5, true>(q_, X, ldx, rows, cols, ex, P, ldp, plane, gs, stab, dep)
                                  : launch_split_cplx<S, PR, 5, false>(q_, X, ldx, rows, cols, ex, P, ldp, plane, gs, stab, dep);
            return conj ? launch_split_cplx<S, PR, 3, true>(q_, X, ldx, rows, cols, ex, P, ldp, plane, gs, stab, dep)
                        : launch_split_cplx<S, PR, 3, false>(q_, X, ldx, rows, cols, ex, P, ldp, plane, gs, stab, dep);
        };
        const std::true_type per_row;
        const std::false_type per_col;
        const auto mkl_ta = ta == Trans::N ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;

        for (int64_t j0 = 0; j0 < n; j0 += nb) {
            const int64_t nbj = std::min(nb, n - j0);
            if (tb == Trans::N) {
                last = split(per_col, false, B + 2 * j0 * ldb, ldb, k, nbj, E + j0, Bp, ldpB, planeB, last);
            } else {
                last = launch_transpose<cplx64>(q_, reinterpret_cast<const cplx64*>(B) + j0, ldb, nbj, k, Btmp, k, last);
                last = split(per_col, tb == Trans::C, reinterpret_cast<const double*>(Btmp), k, k, nbj, E + j0, Bp,
                             ldpB, planeB, last);
            }
            record(Stage::Split, last);
            for (int64_t i0 = 0; i0 < m; i0 += mb) {
                const int64_t mbi = std::min(mb, m - i0);
                if (tiling_.pm > 1 || j0 == 0) {
                    if (ta == Trans::N)
                        last = split(per_row, false, A + 2 * i0, lda, mbi, k, D + i0, Ap, ldpA, planeA, last);
                    else
                        last = split(per_col, ta == Trans::C, A + 2 * i0 * lda, lda, k, mbi, D + i0, Ap, ldpA, planeA, last);
                    record(Stage::Split, last);
                }
                for (int64_t c = 0; c < k_chunks; ++c) {
                    const int64_t p0 = c * kc_len;
                    const int64_t kl = std::min(kc_len, k - p0);
                    for (int g = 0; g < 3; ++g) {
                        const int8_t* a = Ap + g * S * planeA + (ta == Trans::N ? p0 * ldpA : p0);
                        const int8_t* b = Bp + g * S * planeB + p0;
                        last = oneapi::mkl::blas::column_major::gemm_batch(
                            q_, mkl_ta, oneapi::mkl::transpose::nontrans, mbi, nbj, kl, 1.0f, a, ldpA, planeA, b, ldpB,
                            planeB, 0.0f, Ct + (c * 3 + g) * S * planeC, mbi, planeC, S,
                            oneapi::mkl::blas::compute_mode::unset, {last});
                    }
                }
                record(Stage::Gemm, last);
                double* Cb = C + 2 * (i0 + j0 * ldc);
                if (k_chunks == 1)
                    last = launch_crt_cplx<S, false>(q_, Ct, planeC, 1, ctab, mbi, nbj, D + i0, E + j0, alpha.real(),
                                                     alpha.imag(), beta.real(), beta.imag(), Cb, ldc, last);
                else
                    last = launch_crt_cplx<S, true>(q_, Ct, planeC, static_cast<int>(k_chunks), ctab, mbi, nbj, D + i0,
                                                    E + j0, alpha.real(), alpha.imag(), beta.real(), beta.imag(), Cb,
                                                    ldc, last);
                record(Stage::Crt, last);
            }
        }
        return last;
    }

    sycl::event scale_cz(int64_t m, int64_t n, std::complex<double> beta, double* C, int64_t ldc, sycl::event dep) {
        const double br = beta.real(), bi = beta.imag();
        const int64_t cols = std::max<int64_t>(1, detail::kMaxRange / m);
        for (int64_t j0 = 0; j0 < n; j0 += cols) {
            const int64_t nc = std::min(cols, n - j0);
            double* Cj = C + 2 * j0 * ldc;
            dep = q_.submit([&](sycl::handler& h) {
                h.depends_on(dep);
                h.parallel_for(sycl::range<2>(static_cast<size_t>(nc), static_cast<size_t>(m)), [=](sycl::id<2> id) {
                    double* cp = Cj + 2 * (id[1] + static_cast<int64_t>(id[0]) * ldc);
                    if (br == 0.0 && bi == 0.0) {
                        cp[0] = 0.0;
                        cp[1] = 0.0;
                    } else {
                        const double r = cp[0], i = cp[1];
                        cp[0] = br * r - bi * i;
                        cp[1] = br * i + bi * r;
                    }
                });
            });
        }
        return dep;
    }

    void record(Stage st, const sycl::event& ev) {
        if (profiling_) stages_.emplace_back(st, ev);
    }

    template <int S>
    sycl::event run(Trans ta, Trans tb, int64_t m, int64_t n, int64_t k,
                    double alpha, const double* A, int64_t lda,
                    const double* B, int64_t ldb,
                    double beta, double* C, int64_t ldc, sycl::event last) {
        using namespace detail;
        const SplitTab<S> stab = make_split_tab<S>(plan_);
        const CrtTab<S> ctab = make_crt_tab<S>(plan_);

        const int64_t kc_len = std::min<int64_t>(k, kMaxKChunk);
        const int64_t k_chunks = (k + kc_len - 1) / kc_len;
        choose_tiling(ta, tb, m, n, k, S, k_chunks);
        const int64_t mb = tiling_.mb, nb = tiling_.nb;

        const int64_t ldpA = ta == Trans::N ? plane_ld_A(mb, tune_.pad_lda) : plane_ld_A(k, tune_.pad_lda);
        const int64_t planeA = ldpA * (ta == Trans::N ? k : mb);
        const int64_t ldpB = round_up(k, 4);
        const int64_t planeB = ldpB * nb;
        const int64_t planeC = mb * nb;
        size_t off = 0;
        auto carve = [&](size_t bytes) { size_t o = off; off = align_up(off + bytes, 256); return o; };
        const size_t oA = carve(static_cast<size_t>(planeA) * S);
        const size_t oB = carve(static_cast<size_t>(planeB) * S);
        const size_t oC = carve(static_cast<size_t>(planeC) * S * k_chunks * sizeof(int32_t));
        const size_t oMa = carve(static_cast<size_t>(m) * sizeof(uint64_t));
        const size_t oMb = carve(static_cast<size_t>(n) * sizeof(uint64_t));
        const size_t oD = carve(static_cast<size_t>(m) * sizeof(int));
        const size_t oE = carve(static_cast<size_t>(n) * sizeof(int));
        const size_t oT = carve(tb != Trans::N ? static_cast<size_t>(k) * nb * sizeof(double) : 0);
        ensure_workspace(off, last);
        double* Btmp = reinterpret_cast<double*>(ws_ + oT);
        int8_t* Ap = reinterpret_cast<int8_t*>(ws_ + oA);
        int8_t* Bp = reinterpret_cast<int8_t*>(ws_ + oB);
        int32_t* Ct = reinterpret_cast<int32_t*>(ws_ + oC);
        uint64_t* maxA = reinterpret_cast<uint64_t*>(ws_ + oMa);
        uint64_t* maxB = reinterpret_cast<uint64_t*>(ws_ + oMb);
        int* D = reinterpret_cast<int*>(ws_ + oD);
        int* E = reinterpret_cast<int*>(ws_ + oE);

        record(Stage::Begin, last);
        if (ta == Trans::N) {
            last = q_.memset(maxA, 0, m * sizeof(uint64_t), last);
            last = launch_row_maxabs(q_, A, lda, m, k, maxA, last);
        } else {
            last = launch_col_maxabs(q_, A, lda, k, m, maxA, last);
        }
        if (tb == Trans::N) {
            last = launch_col_maxabs(q_, B, ldb, k, n, maxB, last);
        } else {
            last = q_.memset(maxB, 0, n * sizeof(uint64_t), last);
            last = launch_row_maxabs(q_, B, ldb, n, k, maxB, last);
        }
        last = launch_exponents(q_, maxA, m, plan_.tk, D, last);
        last = launch_exponents(q_, maxB, n, plan_.tk, E, last);
        record(Stage::Scale, last);

        const bool wide = plan_.tk > kNarrowScaleBits;
        auto split = [&](auto per_row, const double* X, int64_t ldx, int64_t rows, int64_t cols, const int* ex,
                         int8_t* P, int64_t ldp, int64_t plane, sycl::event dep) {
            constexpr bool PR = decltype(per_row)::value;
            return wide ? launch_split<S, PR, 5>(q_, X, ldx, rows, cols, ex, P, ldp, plane, stab, dep)
                        : launch_split<S, PR, 3>(q_, X, ldx, rows, cols, ex, P, ldp, plane, stab, dep);
        };
        const std::true_type per_row;
        const std::false_type per_col;

        const auto mkl_ta = ta == Trans::N ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
        const auto mkl_tb = oneapi::mkl::transpose::nontrans;

        for (int64_t j0 = 0; j0 < n; j0 += nb) {
            const int64_t nbj = std::min(nb, n - j0);
            if (tb == Trans::N) {
                last = split(per_col, B + j0 * ldb, ldb, k, nbj, E + j0, Bp, ldpB, planeB, last);
            } else {
                last = launch_transpose(q_, B + j0, ldb, nbj, k, Btmp, k, last);
                last = split(per_col, Btmp, k, k, nbj, E + j0, Bp, ldpB, planeB, last);
            }
            record(Stage::Split, last);

            for (int64_t i0 = 0; i0 < m; i0 += mb) {
                const int64_t mbi = std::min(mb, m - i0);
                if (tiling_.pm > 1 || j0 == 0) {
                    if (ta == Trans::N)
                        last = split(per_row, A + i0, lda, mbi, k, D + i0, Ap, ldpA, planeA, last);
                    else
                        last = split(per_col, A + i0 * lda, lda, k, mbi, D + i0, Ap, ldpA, planeA, last);
                    record(Stage::Split, last);
                }
                for (int64_t c = 0; c < k_chunks; ++c) {
                    const int64_t p0 = c * kc_len;
                    const int64_t kl = std::min(kc_len, k - p0);
                    const int8_t* a = Ap + (ta == Trans::N ? p0 * ldpA : p0);
                    const int8_t* b = Bp + p0;
                    if (tune_.batched_gemm) {
                        last = oneapi::mkl::blas::column_major::gemm_batch(
                            q_, mkl_ta, mkl_tb, mbi, nbj, kl, 1.0f,
                            a, ldpA, planeA, b, ldpB, planeB, 0.0f,
                            Ct + c * S * planeC, mbi, planeC, S,
                            oneapi::mkl::blas::compute_mode::unset, {last});
                    } else {
                        for (int t = 0; t < S; ++t)
                            last = oneapi::mkl::blas::column_major::gemm_bias(
                                q_, mkl_ta, mkl_tb, oneapi::mkl::offset::fix, mbi, nbj, kl, 1.0f,
                                a + t * planeA, ldpA, int8_t{0}, b + t * planeB, ldpB, int8_t{0}, 0.0f,
                                Ct + (c * S + t) * planeC, mbi, zero_co(),
                                oneapi::mkl::blas::compute_mode::unset, {last});
                    }
                }
                record(Stage::Gemm, last);
                if (k_chunks == 1)
                    last = launch_crt<S, false>(q_, Ct, planeC, 1, ctab, mbi, nbj,
                                                D + i0, E + j0, alpha, beta, C + i0 + j0 * ldc, ldc, last);
                else
                    last = launch_crt<S, true>(q_, Ct, planeC, static_cast<int>(k_chunks), ctab, mbi, nbj,
                                               D + i0, E + j0, alpha, beta, C + i0 + j0 * ldc, ldc, last);
                record(Stage::Crt, last);
            }
        }
        return last;
    }

    sycl::event scale_c(int64_t m, int64_t n, double beta, double* C, int64_t ldc, sycl::event dep) {
        const int64_t cols = std::max<int64_t>(1, detail::kMaxRange / m);
        for (int64_t j0 = 0; j0 < n; j0 += cols) {
            const int64_t nc = std::min(cols, n - j0);
            double* Cj = C + j0 * ldc;
            dep = q_.submit([&](sycl::handler& h) {
                h.depends_on(dep);
                h.parallel_for(sycl::range<2>(static_cast<size_t>(nc), static_cast<size_t>(m)), [=](sycl::id<2> id) {
                    double* cp = Cj + id[1] + static_cast<int64_t>(id[0]) * ldc;
                    *cp = beta == 0.0 ? 0.0 : beta * *cp;
                });
            });
        }
        return dep;
    }

    size_t workspace_bytes(Trans ta, Trans tb, int64_t m, int64_t n, int64_t k, int s,
                           int64_t kch, int64_t mb, int64_t nb, bool cplx = false) const {
        using detail::round_up;
        const int64_t planeA = ta == Trans::N ? detail::plane_ld_A(mb, tune_.pad_lda) * k
                                              : detail::plane_ld_A(k, tune_.pad_lda) * mb;
        const int64_t planeB = round_up(k, 4) * nb;
        const size_t g = cplx ? 3 : 1;
        return g * s * static_cast<size_t>(planeA + planeB + 4 * kch * mb * nb) +
               (tb != Trans::N ? static_cast<size_t>(k) * nb * (cplx ? 16 : 8) : 0) +
               static_cast<size_t>(m + n) * (cplx ? 20 : 12) + 4 * 4096;
    }

    bool ranges_fit(Trans ta, Trans tb, int64_t k, int64_t mb, int64_t nb) const {
        using detail::kMaxRange;
        using detail::round_up;
        const int64_t splitA = ta == Trans::N ? k * (detail::plane_ld_A(mb, tune_.pad_lda) / 4)
                                              : mb * (detail::plane_ld_A(k, tune_.pad_lda) / 4);
        const int64_t splitB = nb * (round_up(k, 4) / 4);
        (void)tb;
        return mb * nb <= kMaxRange && splitA <= kMaxRange && splitB <= kMaxRange;
    }

    size_t budget() const {
        if (ws_limit_) return ws_limit_;
        const auto dev = q_.get_device();
        size_t free_b = 0;
        try {
            free_b = dev.get_info<sycl::ext::intel::info::device::free_memory>() + ws_size_;
        } catch (...) {
            free_b = dev.get_info<sycl::info::device::global_mem_size>();
        }
        const size_t total = dev.get_info<sycl::info::device::global_mem_size>();
        const size_t reserve = (size_t{512} << 20) + total / 32;
        const size_t max_alloc = dev.get_info<sycl::info::device::max_mem_alloc_size>();
        size_t lim = std::min(free_b > reserve ? free_b - reserve : 0, max_alloc);
        if (resident_hint_)
            lim = std::min(lim, total > reserve + resident_hint_ ? total - reserve - resident_hint_ : size_t{0});
        return lim;
    }

    static std::vector<int64_t> tile_candidates(int64_t x, bool pow2) {
        std::vector<int64_t> c{x};
        for (int64_t p = 2; p <= 256; ++p) {
            const int64_t b = detail::round_up((x + p - 1) / p, 64);
            if (b < x) c.push_back(b);
        }
        if (pow2)
            for (int64_t b = 1024; b < x; b *= 2) c.push_back(b);
        std::sort(c.begin(), c.end(), std::greater<>());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        return c;
    }

    void choose_tiling(Trans ta, Trans tb, int64_t m, int64_t n, int64_t k, int s, int64_t kch, bool cplx = false) {
        const size_t limit = budget();
        double best = std::numeric_limits<double>::infinity();
        TileInfo bestT;
        const double g = cplx ? 3.0 : 1.0;
        const double gemm_ops = 2.0 * m * n * k * s * g;
        const auto is_pow2 = [](int64_t v) { return (v & (v - 1)) == 0; };
        const auto nbs = tile_candidates(n, tune_.pow2_tiles);
        for (int64_t mb : tile_candidates(m, tune_.pow2_tiles)) {
            const int64_t pm = (m + mb - 1) / mb;
            for (int64_t nb : nbs) {
                const int64_t pn = (n + nb - 1) / nb;
                const size_t ws = workspace_bytes(ta, tb, m, n, k, s, kch, mb, nb, cplx);
                if (ws > limit || !ranges_fit(ta, tb, k, mb, nb)) continue;
                double eff = 1.0 / (1.0 + 256.0 / mb + 256.0 / nb);
                if (tune_.pow2_tiles && pm > 1 && !is_pow2(mb)) eff *= 0.88;
                const double splitA = g * (8.0 + s) * m * k * (pm > 1 ? pn : 1);
                const double splitB = g * (8.0 + s) * k * n;
                const double crt = g * (4.0 * s * kch + 16.0) * m * n;
                const double t = gemm_ops / (190e12 * eff) + (splitA + splitB + crt) / 400e9 +
                                 pm * pn * 4 * 20e-6;
                if (t < best) {
                    best = t;
                    bestT = TileInfo{mb, nb, pm, pn, kch, ws};
                }
                break;
            }
        }
        if (!std::isfinite(best))
            throw std::runtime_error("oz2: not enough device memory for workspace (limit " +
                                     std::to_string(limit >> 20) + " MiB)");
        tiling_ = bestT;
    }

    void ensure_workspace(size_t bytes, sycl::event& last) {
        if (ws_size_ >= bytes) return;
        last.wait();
        release_workspace();
        ws_ = sycl::malloc_device<std::byte>(bytes, q_);
        if (!ws_) throw std::runtime_error("oz2: workspace allocation failed");
        ws_size_ = bytes;
    }

    const int32_t* zero_co() {
        if (!co_) {
            co_ = sycl::malloc_device<int32_t>(1, q_);
            q_.memset(co_, 0, sizeof(int32_t)).wait();
        }
        return co_;
    }

    sycl::queue& q_;
    Tuning tune_{};
    int32_t* co_ = nullptr;
    bool profiling_ = false;
    detail::Plan plan_{};
    int64_t plan_k_ = -1;
    std::byte* ws_ = nullptr;
    size_t ws_size_ = 0;
    size_t ws_limit_ = 0;
    size_t resident_hint_ = 0;
    TileInfo tiling_{};
    std::vector<std::pair<Stage, sycl::event>> stages_;
};

}
