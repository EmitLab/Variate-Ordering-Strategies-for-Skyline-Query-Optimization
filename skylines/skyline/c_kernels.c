/* clock_gettime / CLOCK_MONOTONIC need POSIX visibility under -std=c11. */
#define _POSIX_C_SOURCE 199309L

#include "c_kernels.h"

#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define SKYLINES_X86 1
#else
#define SKYLINES_X86 0
#endif

/* Monotonic wall-clock nanoseconds. Shared by lex / monosort presort timing. */
static inline int64_t monosort_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

static inline DomResult make_result(int32_t dominating, int32_t fail_idx) {
    DomResult result;
    result.dominating = dominating;
    result.fail_idx = fail_idx;
    return result;
}

static inline int32_t skyline_cpu_has_sse2(void) {
#if SKYLINES_X86 && defined(__GNUC__)
    static int32_t cached = -1;
    if (cached < 0) {
        __builtin_cpu_init();
        cached = __builtin_cpu_supports("sse2") ? 1 : 0;
    }
    return cached;
#else
    return 0;
#endif
}

static inline int32_t skyline_cpu_has_avx2(void) {
#if SKYLINES_X86 && defined(__GNUC__)
    static int32_t cached = -1;
    if (cached < 0) {
        __builtin_cpu_init();
        cached = __builtin_cpu_supports("avx2") ? 1 : 0;
    }
    return cached;
#else
    return 0;
#endif
}

DomResult skyline_scalar_f32(const float *row1, const float *row2, int32_t n) {
    int32_t dominating = 0;

    for (int32_t i = 0; i < n; ++i) {
        if (row1[i] < row2[i]) {
            return make_result(0, i);
        }
        if (row1[i] > row2[i]) {
            dominating = 1;
        }
    }

    return make_result(dominating, -1);
}

#if SKYLINES_X86
__attribute__((target("sse2")))
static DomResult skyline_sse2_f32_impl(const float *row1, const float *row2, int32_t n) {
    const int32_t lanes = 4;
    const int32_t limit = n - (n % lanes);
    int32_t dominating = 0;

    for (int32_t i = 0; i < limit; i += lanes) {
        __m128 lhs = _mm_loadu_ps(row1 + i);
        __m128 rhs = _mm_loadu_ps(row2 + i);

        int32_t lt_mask = _mm_movemask_ps(_mm_cmplt_ps(lhs, rhs));
        if (lt_mask != 0) {
            return make_result(0, i + __builtin_ctz((unsigned)lt_mask));
        }

        if (_mm_movemask_ps(_mm_cmpgt_ps(lhs, rhs)) != 0) {
            dominating = 1;
        }
    }

    if (limit < n) {
        const int32_t tail = n - limit;
        __m128 lhs = _mm_setzero_ps();
        __m128 rhs = _mm_setzero_ps();
        if (tail == 1) {
            lhs = _mm_load_ss(row1 + limit);
            rhs = _mm_load_ss(row2 + limit);
        } else if (tail == 2) {
            lhs = _mm_loadl_pi(lhs, (const __m64 *)(row1 + limit));
            rhs = _mm_loadl_pi(rhs, (const __m64 *)(row2 + limit));
        } else { /* tail == 3 */
            __m128 lhs_lo = _mm_loadl_pi(_mm_setzero_ps(), (const __m64 *)(row1 + limit));
            __m128 rhs_lo = _mm_loadl_pi(_mm_setzero_ps(), (const __m64 *)(row2 + limit));
            __m128 lhs_hi = _mm_load_ss(row1 + limit + 2);
            __m128 rhs_hi = _mm_load_ss(row2 + limit + 2);
            lhs = _mm_movelh_ps(lhs_lo, lhs_hi);
            rhs = _mm_movelh_ps(rhs_lo, rhs_hi);
        }

        int32_t lt_mask = _mm_movemask_ps(_mm_cmplt_ps(lhs, rhs));
        if (lt_mask != 0) {
            return make_result(0, limit + __builtin_ctz((unsigned)lt_mask));
        }
        if (_mm_movemask_ps(_mm_cmpgt_ps(lhs, rhs)) != 0) {
            dominating = 1;
        }
    }

    return make_result(dominating, -1);
}

__attribute__((target("avx2")))
static DomResult skyline_avx2_f32_impl(const float *row1, const float *row2, int32_t n) {
    const int32_t lanes = 8;
    const int32_t limit = n - (n % lanes);
    int32_t dominating = 0;

    for (int32_t i = 0; i < limit; i += lanes) {
        __m256 lhs = _mm256_loadu_ps(row1 + i);
        __m256 rhs = _mm256_loadu_ps(row2 + i);

        int32_t lt_mask = _mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ));
        if (lt_mask != 0) {
            return make_result(0, i + __builtin_ctz((unsigned)lt_mask));
        }

        if (_mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ)) != 0) {
            dominating = 1;
        }
    }

    if (limit < n) {
        const int32_t tail = n - limit;
        const __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(tail), idx);
        __m256 lhs = _mm256_maskload_ps(row1 + limit, mask);
        __m256 rhs = _mm256_maskload_ps(row2 + limit, mask);

        int32_t lt_mask = _mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ));
        if (lt_mask != 0) {
            return make_result(0, limit + __builtin_ctz((unsigned)lt_mask));
        }
        if (_mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ)) != 0) {
            dominating = 1;
        }
    }

    return make_result(dominating, -1);
}
#endif

DomResult skyline_sse2_f32(const float *row1, const float *row2, int32_t n) {
#if SKYLINES_X86
    return skyline_sse2_f32_impl(row1, row2, n);
#else
    return skyline_scalar_f32(row1, row2, n);
#endif
}

DomResult skyline_avx2_f32(const float *row1, const float *row2, int32_t n) {
#if SKYLINES_X86
    return skyline_avx2_f32_impl(row1, row2, n);
#else
    return skyline_scalar_f32(row1, row2, n);
#endif
}

int32_t skyline_cpu_supports_sse2(void) {
    return skyline_cpu_has_sse2();
}

int32_t skyline_cpu_supports_avx2(void) {
    return skyline_cpu_has_avx2();
}

int32_t skyline_simd_lanes(void) {
    if (skyline_cpu_has_avx2()) {
        return 8;
    }
    if (skyline_cpu_has_sse2()) {
        return 4;
    }
    return 1;
}

DomResult skyline_simd_f32(const float *row1, const float *row2, int32_t n) {
#if SKYLINES_X86
    if (skyline_cpu_has_avx2()) {
        return skyline_avx2_f32_impl(row1, row2, n);
    }
    if (skyline_cpu_has_sse2()) {
        return skyline_sse2_f32_impl(row1, row2, n);
    }
#endif
    return skyline_scalar_f32(row1, row2, n);
}

/* ── float64 implementations ─────────────────────────────────────────────── */

DomResult skyline_scalar_f64(const double *row1, const double *row2, int32_t n) {
    int32_t dominating = 0;
    for (int32_t i = 0; i < n; ++i) {
        if (row1[i] < row2[i]) return make_result(0, i);
        if (row1[i] > row2[i]) dominating = 1;
    }
    return make_result(dominating, -1);
}

#if SKYLINES_X86
__attribute__((target("sse2")))
static DomResult skyline_sse2_f64_impl(const double *row1, const double *row2, int32_t n) {
    const int32_t lanes = 2;
    const int32_t limit = n - (n % lanes);
    int32_t dominating = 0;

    for (int32_t i = 0; i < limit; i += lanes) {
        __m128d lhs = _mm_loadu_pd(row1 + i);
        __m128d rhs = _mm_loadu_pd(row2 + i);

        int32_t lt_mask = _mm_movemask_pd(_mm_cmplt_pd(lhs, rhs));
        if (lt_mask != 0) {
            return make_result(0, i + __builtin_ctz((unsigned)lt_mask));
        }
        if (_mm_movemask_pd(_mm_cmpgt_pd(lhs, rhs)) != 0) {
            dominating = 1;
        }
    }
    if (limit < n) {
        /* lanes == 2, tail always == 1: load_sd zeros high lane */
        __m128d lhs = _mm_load_sd(row1 + limit);
        __m128d rhs = _mm_load_sd(row2 + limit);

        int32_t lt_mask = _mm_movemask_pd(_mm_cmplt_pd(lhs, rhs));
        if (lt_mask != 0) {
            return make_result(0, limit + __builtin_ctz((unsigned)lt_mask));
        }
        if (_mm_movemask_pd(_mm_cmpgt_pd(lhs, rhs)) != 0) {
            dominating = 1;
        }
    }
    return make_result(dominating, -1);
}

__attribute__((target("avx2")))
static DomResult skyline_avx2_f64_impl(const double *row1, const double *row2, int32_t n) {
    const int32_t lanes = 4;
    const int32_t limit = n - (n % lanes);
    int32_t dominating = 0;

    for (int32_t i = 0; i < limit; i += lanes) {
        __m256d lhs = _mm256_loadu_pd(row1 + i);
        __m256d rhs = _mm256_loadu_pd(row2 + i);

        int32_t lt_mask = _mm256_movemask_pd(_mm256_cmp_pd(lhs, rhs, _CMP_LT_OQ));
        if (lt_mask != 0) {
            return make_result(0, i + __builtin_ctz((unsigned)lt_mask));
        }
        if (_mm256_movemask_pd(_mm256_cmp_pd(lhs, rhs, _CMP_GT_OQ)) != 0) {
            dominating = 1;
        }
    }
    if (limit < n) {
        const int32_t tail = n - limit;
        const __m256i idx = _mm256_setr_epi64x(0, 1, 2, 3);
        const __m256i mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(tail), idx);
        __m256d lhs = _mm256_maskload_pd(row1 + limit, mask);
        __m256d rhs = _mm256_maskload_pd(row2 + limit, mask);

        int32_t lt_mask = _mm256_movemask_pd(_mm256_cmp_pd(lhs, rhs, _CMP_LT_OQ));
        if (lt_mask != 0) {
            return make_result(0, limit + __builtin_ctz((unsigned)lt_mask));
        }
        if (_mm256_movemask_pd(_mm256_cmp_pd(lhs, rhs, _CMP_GT_OQ)) != 0) {
            dominating = 1;
        }
    }
    return make_result(dominating, -1);
}
#endif

DomResult skyline_sse2_f64(const double *row1, const double *row2, int32_t n) {
#if SKYLINES_X86
    return skyline_sse2_f64_impl(row1, row2, n);
#else
    return skyline_scalar_f64(row1, row2, n);
#endif
}

DomResult skyline_avx2_f64(const double *row1, const double *row2, int32_t n) {
#if SKYLINES_X86
    return skyline_avx2_f64_impl(row1, row2, n);
#else
    return skyline_scalar_f64(row1, row2, n);
#endif
}

DomResult skyline_simd_f64(const double *row1, const double *row2, int32_t n) {
#if SKYLINES_X86
    if (skyline_cpu_has_avx2()) return skyline_avx2_f64_impl(row1, row2, n);
    if (skyline_cpu_has_sse2()) return skyline_sse2_f64_impl(row1, row2, n);
#endif
    return skyline_scalar_f64(row1, row2, n);
}

/* ── Numba-callable wrappers (int64 pointers, int32 packed return) ────────── */

static inline int32_t pack_dom_result(DomResult r) {
    if (r.fail_idx >= 0) return r.fail_idx;
    return r.dominating ? -2 : -1;
}

int32_t skyline_scalar_f32_nb(int64_t ptr1, int64_t ptr2, int32_t n) {
    return pack_dom_result(skyline_scalar_f32((const float *)ptr1, (const float *)ptr2, n));
}
int32_t skyline_sse2_f32_nb(int64_t ptr1, int64_t ptr2, int32_t n) {
    return pack_dom_result(skyline_sse2_f32((const float *)ptr1, (const float *)ptr2, n));
}
int32_t skyline_avx2_f32_nb(int64_t ptr1, int64_t ptr2, int32_t n) {
    return pack_dom_result(skyline_avx2_f32((const float *)ptr1, (const float *)ptr2, n));
}
int32_t skyline_simd_f32_nb(int64_t ptr1, int64_t ptr2, int32_t n) {
    return pack_dom_result(skyline_simd_f32((const float *)ptr1, (const float *)ptr2, n));
}

int32_t skyline_scalar_f64_nb(int64_t ptr1, int64_t ptr2, int32_t n) {
    return pack_dom_result(skyline_scalar_f64((const double *)ptr1, (const double *)ptr2, n));
}
int32_t skyline_sse2_f64_nb(int64_t ptr1, int64_t ptr2, int32_t n) {
    return pack_dom_result(skyline_sse2_f64((const double *)ptr1, (const double *)ptr2, n));
}
int32_t skyline_avx2_f64_nb(int64_t ptr1, int64_t ptr2, int32_t n) {
    return pack_dom_result(skyline_avx2_f64((const double *)ptr1, (const double *)ptr2, n));
}
int32_t skyline_simd_f64_nb(int64_t ptr1, int64_t ptr2, int32_t n) {
    return pack_dom_result(skyline_simd_f64((const double *)ptr1, (const double *)ptr2, n));
}

/* ── batched BNL outer loop ────────────────────────────────────────────────── */

static inline void apply_dom_metrics(
    DomResult r, int32_t n_dims, int32_t chunk_size,
    BnlMetrics *m, int64_t *out_vcheck_freq)
{
    int32_t sc, scanned, vidx;
    if (r.fail_idx >= 0) {
        sc = r.fail_idx + 1;
        scanned = r.fail_idx + 1;
        vidx = r.fail_idx;
    } else {
        sc = n_dims;
        scanned = n_dims;
        vidx = n_dims - 1;
    }
    m->sub_comparisons += sc;
    if (chunk_size > 1) {
        m->simd_chunks_compared += (scanned + chunk_size - 1) / chunk_size;
    }
    out_vcheck_freq[vidx] += 1;
}

#define BNL_DEFINE(NAME, T, KERNEL_FN, CHUNK_SIZE)                              \
static void NAME(                                                               \
    const T *rows, int64_t n_rows, int32_t n_dims, int32_t reverse_scan,        \
    int64_t *out_skyline, int64_t *out_vcheck_freq, BnlMetrics *m)              \
{                                                                               \
    int64_t skyline_size = 0;                                                   \
    for (int64_t i = 0; i < n_rows; ++i) {                                      \
        const T *row_i = rows + (size_t)i * (size_t)n_dims;                     \
        int dominated = 0;                                                      \
        int64_t k = reverse_scan ? skyline_size - 1 : 0;                        \
        while (reverse_scan ? (k >= 0) : (k < skyline_size)) {                  \
            int64_t j = out_skyline[k];                                         \
            const T *row_j = rows + (size_t)j * (size_t)n_dims;                 \
            m->comparisons += 1;                                                \
            DomResult r1 = KERNEL_FN(row_j, row_i, n_dims);                     \
            apply_dom_metrics(r1, n_dims, (CHUNK_SIZE), m, out_vcheck_freq);    \
            if (r1.dominating) { dominated = 1; break; }                        \
            m->comparisons += 1;                                                \
            DomResult r2 = KERNEL_FN(row_i, row_j, n_dims);                     \
            apply_dom_metrics(r2, n_dims, (CHUNK_SIZE), m, out_vcheck_freq);    \
            if (r2.dominating) {                                                \
                skyline_size -= 1;                                              \
                for (int64_t s = k; s < skyline_size; ++s)                      \
                    out_skyline[s] = out_skyline[s + 1];                        \
                if (reverse_scan) k -= 1;                                       \
            } else {                                                            \
                k += reverse_scan ? -1 : 1;                                     \
            }                                                                   \
        }                                                                       \
        if (!dominated) out_skyline[skyline_size++] = i;                        \
    }                                                                           \
    m->n_skyline = skyline_size;                                                \
}

BNL_DEFINE(bnl_run_scalar_f32, float,  skyline_scalar_f32, 1)
BNL_DEFINE(bnl_run_sse2_f32,   float,  skyline_sse2_f32,   4)
BNL_DEFINE(bnl_run_avx2_f32,   float,  skyline_avx2_f32,   8)
BNL_DEFINE(bnl_run_scalar_f64, double, skyline_scalar_f64, 1)
BNL_DEFINE(bnl_run_sse2_f64,   double, skyline_sse2_f64,   2)
BNL_DEFINE(bnl_run_avx2_f64,   double, skyline_avx2_f64,   4)

BnlMetrics skyline_bnl_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    int32_t reverse_scan,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0, 0, 0, 0};
    switch (kernel_mode) {
        case 3: bnl_run_scalar_f32((const float  *)data, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 4: bnl_run_avx2_f32  ((const float  *)data, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 5: bnl_run_sse2_f32  ((const float  *)data, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 6: bnl_run_scalar_f64((const double *)data, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 7: bnl_run_avx2_f64  ((const double *)data, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 8: bnl_run_sse2_f64  ((const double *)data, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        default: break;
    }
    return m;
}

/* ── batched SFS outer loop ────────────────────────────────────────────────── */

#define SFS_DEFINE(NAME, T, KERNEL_FN, CHUNK_SIZE)                              \
static void NAME(                                                               \
    const T *rows, int64_t n_rows, int32_t n_dims, int32_t reverse_scan,        \
    int64_t *out_skyline, int64_t *out_vcheck_freq, BnlMetrics *m)              \
{                                                                               \
    int64_t skyline_size = 0;                                                   \
    for (int64_t i = 0; i < n_rows; ++i) {                                      \
        const T *row_i = rows + (size_t)i * (size_t)n_dims;                     \
        int dominated = 0;                                                      \
        if (reverse_scan) {                                                     \
            for (int64_t k = skyline_size - 1; k >= 0; --k) {                   \
                int64_t j = out_skyline[k];                                     \
                const T *row_j = rows + (size_t)j * (size_t)n_dims;             \
                m->comparisons += 1;                                            \
                DomResult r = KERNEL_FN(row_j, row_i, n_dims);                  \
                apply_dom_metrics(r, n_dims, (CHUNK_SIZE), m, out_vcheck_freq); \
                if (r.dominating) { dominated = 1; break; }                     \
            }                                                                   \
        } else {                                                                \
            for (int64_t k = 0; k < skyline_size; ++k) {                        \
                int64_t j = out_skyline[k];                                     \
                const T *row_j = rows + (size_t)j * (size_t)n_dims;             \
                m->comparisons += 1;                                            \
                DomResult r = KERNEL_FN(row_j, row_i, n_dims);                  \
                apply_dom_metrics(r, n_dims, (CHUNK_SIZE), m, out_vcheck_freq); \
                if (r.dominating) { dominated = 1; break; }                     \
            }                                                                   \
        }                                                                       \
        if (!dominated) out_skyline[skyline_size++] = i;                        \
    }                                                                           \
    m->n_skyline = skyline_size;                                                \
}

SFS_DEFINE(sfs_run_scalar_f32, float,  skyline_scalar_f32, 1)
SFS_DEFINE(sfs_run_sse2_f32,   float,  skyline_sse2_f32,   4)
SFS_DEFINE(sfs_run_avx2_f32,   float,  skyline_avx2_f32,   8)
SFS_DEFINE(sfs_run_scalar_f64, double, skyline_scalar_f64, 1)
SFS_DEFINE(sfs_run_sse2_f64,   double, skyline_sse2_f64,   2)
SFS_DEFINE(sfs_run_avx2_f64,   double, skyline_avx2_f64,   4)

BnlMetrics skyline_sfs_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    int32_t reverse_scan,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0, 0, 0, 0};
    switch (kernel_mode) {
        case 3: sfs_run_scalar_f32((const float  *)data, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 4: sfs_run_avx2_f32  ((const float  *)data, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 5: sfs_run_sse2_f32  ((const float  *)data, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 6: sfs_run_scalar_f64((const double *)data, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 7: sfs_run_avx2_f64  ((const double *)data, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 8: sfs_run_sse2_f64  ((const double *)data, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        default: break;
    }
    return m;
}

/* ── batched DnC trivial-partition skyline ──────────────────────────────────── */
/* Reverse scan + removal, like BNL lexsort, but iterates a caller-supplied
 * subset of row indices (the partition) rather than the full data range. */

#define DNC_TRIVIAL_DEFINE(NAME, T, KERNEL_FN, CHUNK_SIZE)                      \
static void NAME(                                                               \
    const T *rows, const int64_t *index, int64_t n_index, int32_t n_dims,       \
    int64_t *out_skyline, int64_t *out_vcheck_freq, BnlMetrics *m)              \
{                                                                               \
    int64_t skyline_size = 0;                                                   \
    for (int64_t rel_i = 0; rel_i < n_index; ++rel_i) {                         \
        int64_t i = index[rel_i];                                               \
        const T *row_i = rows + (size_t)i * (size_t)n_dims;                     \
        int dominated = 0;                                                      \
        int64_t k = skyline_size - 1;                                           \
        while (k >= 0) {                                                        \
            int64_t j = out_skyline[k];                                         \
            const T *row_j = rows + (size_t)j * (size_t)n_dims;                 \
            m->comparisons += 1;                                                \
            DomResult r1 = KERNEL_FN(row_j, row_i, n_dims);                     \
            apply_dom_metrics(r1, n_dims, (CHUNK_SIZE), m, out_vcheck_freq);    \
            if (r1.dominating) { dominated = 1; break; }                        \
            m->comparisons += 1;                                                \
            DomResult r2 = KERNEL_FN(row_i, row_j, n_dims);                     \
            apply_dom_metrics(r2, n_dims, (CHUNK_SIZE), m, out_vcheck_freq);    \
            if (r2.dominating) {                                                \
                for (int64_t s = k; s < skyline_size - 1; ++s)                  \
                    out_skyline[s] = out_skyline[s + 1];                        \
                skyline_size -= 1;                                              \
            }                                                                   \
            k -= 1;                                                             \
        }                                                                       \
        if (!dominated) out_skyline[skyline_size++] = i;                        \
    }                                                                           \
    m->n_skyline = skyline_size;                                                \
}

DNC_TRIVIAL_DEFINE(dnc_trivial_run_scalar_f32, float,  skyline_scalar_f32, 1)
DNC_TRIVIAL_DEFINE(dnc_trivial_run_sse2_f32,   float,  skyline_sse2_f32,   4)
DNC_TRIVIAL_DEFINE(dnc_trivial_run_avx2_f32,   float,  skyline_avx2_f32,   8)
DNC_TRIVIAL_DEFINE(dnc_trivial_run_scalar_f64, double, skyline_scalar_f64, 1)
DNC_TRIVIAL_DEFINE(dnc_trivial_run_sse2_f64,   double, skyline_sse2_f64,   2)
DNC_TRIVIAL_DEFINE(dnc_trivial_run_avx2_f64,   double, skyline_avx2_f64,   4)

BnlMetrics skyline_dnc_trivial_run(
    const void *data,
    const int64_t *index,
    int64_t n_index,
    int32_t n_dims,
    int32_t kernel_mode,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0, 0, 0, 0};
    switch (kernel_mode) {
        case 3: dnc_trivial_run_scalar_f32((const float  *)data, index, n_index, n_dims, out_skyline, out_vcheck_freq, &m); break;
        case 4: dnc_trivial_run_avx2_f32  ((const float  *)data, index, n_index, n_dims, out_skyline, out_vcheck_freq, &m); break;
        case 5: dnc_trivial_run_sse2_f32  ((const float  *)data, index, n_index, n_dims, out_skyline, out_vcheck_freq, &m); break;
        case 6: dnc_trivial_run_scalar_f64((const double *)data, index, n_index, n_dims, out_skyline, out_vcheck_freq, &m); break;
        case 7: dnc_trivial_run_avx2_f64  ((const double *)data, index, n_index, n_dims, out_skyline, out_vcheck_freq, &m); break;
        case 8: dnc_trivial_run_sse2_f64  ((const double *)data, index, n_index, n_dims, out_skyline, out_vcheck_freq, &m); break;
        default: break;
    }
    return m;
}

/* ── batched SaLSa forward scan + pstop pruning ─────────────────────────────── */

#include <stdlib.h>

#define SALSA_FWD_DEFINE(NAME, T, KERNEL_FN, CHUNK_SIZE)                          \
static void NAME(                                                                 \
    const T *rows, int64_t n_rows, int32_t n_dims, const T *scores,               \
    int64_t *out_skyline, int64_t *out_vcheck_freq, BnlMetrics *m)                \
{                                                                                 \
    T *pstop = (T *)malloc(sizeof(T) * (size_t)n_dims);                           \
    if (!pstop) { m->n_skyline = 0; return; }                                     \
    int64_t skyline_size = 0;                                                     \
    int pstop_set = 0;                                                            \
    T pstop_min = (T)0;                                                           \
                                                                                  \
    for (int64_t i = 0; i < n_rows; ++i) {                                        \
        const T *row_i = rows + (size_t)i * (size_t)n_dims;                       \
        int dominated = 0;                                                        \
        for (int64_t k = 0; k < skyline_size; ++k) {                              \
            int64_t j = out_skyline[k];                                           \
            const T *row_j = rows + (size_t)j * (size_t)n_dims;                   \
            m->comparisons += 1;                                                  \
            DomResult r = KERNEL_FN(row_j, row_i, n_dims);                        \
            apply_dom_metrics(r, n_dims, (CHUNK_SIZE), m, out_vcheck_freq);       \
            if (r.dominating) { dominated = 1; break; }                           \
        }                                                                         \
        if (!dominated) {                                                         \
            out_skyline[skyline_size++] = i;                                      \
            T row_min = row_i[0];                                                 \
            for (int32_t d = 1; d < n_dims; ++d)                                  \
                if (row_i[d] < row_min) row_min = row_i[d];                       \
            if (!pstop_set || row_min > pstop_min) {                              \
                for (int32_t d = 0; d < n_dims; ++d) pstop[d] = row_i[d];         \
                pstop_set = 1;                                                    \
                pstop_min = row_min;                                              \
            }                                                                     \
        }                                                                         \
        if (pstop_set) {                                                          \
            T s0 = scores[(size_t)i * 2 + 0];                                     \
            T s1 = scores[(size_t)i * 2 + 1];                                     \
            int prune = 0;                                                        \
            if (s0 < pstop_min) prune = 1;                                        \
            else if (s0 > pstop_min) prune = 0;                                   \
            else if (s1 < pstop_min) prune = 1;                                   \
            else prune = 0;                                                       \
            if (prune) break;                                                     \
        }                                                                         \
    }                                                                             \
    m->n_skyline = skyline_size;                                                  \
    free(pstop);                                                                  \
}

SALSA_FWD_DEFINE(salsa_run_scalar_f32, float,  skyline_scalar_f32, 1)
SALSA_FWD_DEFINE(salsa_run_sse2_f32,   float,  skyline_sse2_f32,   4)
SALSA_FWD_DEFINE(salsa_run_avx2_f32,   float,  skyline_avx2_f32,   8)
SALSA_FWD_DEFINE(salsa_run_scalar_f64, double, skyline_scalar_f64, 1)
SALSA_FWD_DEFINE(salsa_run_sse2_f64,   double, skyline_sse2_f64,   2)
SALSA_FWD_DEFINE(salsa_run_avx2_f64,   double, skyline_avx2_f64,   4)

BnlMetrics skyline_salsa_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    const void *monotonic_scores,
    int32_t kernel_mode,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0, 0, 0, 0};
    switch (kernel_mode) {
        case 3: salsa_run_scalar_f32((const float  *)data, n_rows, n_dims, (const float  *)monotonic_scores, out_skyline, out_vcheck_freq, &m); break;
        case 4: salsa_run_avx2_f32  ((const float  *)data, n_rows, n_dims, (const float  *)monotonic_scores, out_skyline, out_vcheck_freq, &m); break;
        case 5: salsa_run_sse2_f32  ((const float  *)data, n_rows, n_dims, (const float  *)monotonic_scores, out_skyline, out_vcheck_freq, &m); break;
        case 6: salsa_run_scalar_f64((const double *)data, n_rows, n_dims, (const double *)monotonic_scores, out_skyline, out_vcheck_freq, &m); break;
        case 7: salsa_run_avx2_f64  ((const double *)data, n_rows, n_dims, (const double *)monotonic_scores, out_skyline, out_vcheck_freq, &m); break;
        case 8: salsa_run_sse2_f64  ((const double *)data, n_rows, n_dims, (const double *)monotonic_scores, out_skyline, out_vcheck_freq, &m); break;
        default: break;
    }
    return m;
}

/* ── batched SaLSa lexsort (reverse scan + reverse-lex pstop pruning) ───────── */

/* `rows` drives the pstop pruning + row materialization; `cmp_rows` carries
 * the column layout fed to the per-pair dominance kernel. All current callers
 * pass the same buffer for both; the separate `cmp_rows` parameter is kept so
 * future variants can decouple the two without changing the macro.
 *
 * `key_order` (NULL or n_dims-entry int32) is the lex sort's column priority,
 * highest-priority first. The pstop pruning walks columns in reverse lex
 * priority via this indirection: NULL means natural order [0..n-1]; any
 * permutation makes the early-termination sound for that NCF key order.
 * `pstop_last_nonone` is tracked in PRIORITY space (0 = highest-priority
 * column, n-1 = lowest) so the leading-zeros and tiebreak checks compose
 * with the indirection. */
#define SALSA_REV_DEFINE(NAME, T, KERNEL_FN, CHUNK_SIZE)                          \
static void NAME(                                                                 \
    const T *rows, const T *cmp_rows, int64_t n_rows, int32_t n_dims,             \
    const int32_t *key_order,                                                     \
    int64_t *out_skyline, int64_t *out_vcheck_freq, BnlMetrics *m)                \
{                                                                                 \
    T *pstop = (T *)malloc(sizeof(T) * (size_t)n_dims);                           \
    if (!pstop) { m->n_skyline = 0; return; }                                     \
    int64_t skyline_size = 0;                                                     \
    int pstop_set = 0;                                                            \
    int32_t pstop_last_nonone_pri = -1; /* priority position, not column index */ \
                                                                                  \
    for (int64_t i = 0; i < n_rows; ++i) {                                        \
        const T *row_i  = rows     + (size_t)i * (size_t)n_dims;                  \
        const T *crow_i = cmp_rows + (size_t)i * (size_t)n_dims;                  \
        int dominated = 0;                                                        \
        for (int64_t k = skyline_size - 1; k >= 0; --k) {                         \
            int64_t j = out_skyline[k];                                           \
            const T *crow_j = cmp_rows + (size_t)j * (size_t)n_dims;              \
            m->comparisons += 1;                                                  \
            DomResult r = KERNEL_FN(crow_j, crow_i, n_dims);                      \
            apply_dom_metrics(r, n_dims, (CHUNK_SIZE), m, out_vcheck_freq);       \
            if (r.dominating) { dominated = 1; break; }                           \
        }                                                                         \
        if (!dominated) {                                                         \
            out_skyline[skyline_size++] = i;                                      \
            int gt = 0;                                                           \
            if (!pstop_set) {                                                     \
                gt = 1;                                                           \
            } else {                                                              \
                for (int32_t p = n_dims - 1; p >= 0; --p) {                       \
                    int32_t d = key_order ? key_order[p] : p;                     \
                    if (row_i[d] > pstop[d]) { gt = 1; break; }                   \
                    if (row_i[d] < pstop[d]) { gt = 0; break; }                   \
                }                                                                 \
            }                                                                     \
            if (gt) {                                                             \
                for (int32_t d = 0; d < n_dims; ++d) pstop[d] = row_i[d];         \
                pstop_set = 1;                                                    \
                pstop_last_nonone_pri = -1;                                       \
                for (int32_t p = n_dims - 1; p >= 0; --p) {                       \
                    int32_t d = key_order ? key_order[p] : p;                     \
                    if (pstop[d] != (T)1.0) { pstop_last_nonone_pri = p; break; }  \
                }                                                                 \
            }                                                                     \
        }                                                                         \
        if (pstop_set) {                                                          \
            int prune = 0;                                                        \
            if (pstop_last_nonone_pri == -1) {                                    \
                for (int32_t d = 0; d < n_dims; ++d) {                            \
                    if (row_i[d] < pstop[d]) { prune = 1; break; }                \
                }                                                                 \
            } else {                                                              \
                int leading_zeros = 1;                                            \
                for (int32_t p = 0; p < pstop_last_nonone_pri; ++p) {             \
                    int32_t d = key_order ? key_order[p] : p;                     \
                    if (row_i[d] != (T)0.0) { leading_zeros = 0; break; }         \
                }                                                                 \
                int32_t d_last = key_order ? key_order[pstop_last_nonone_pri]     \
                                            : pstop_last_nonone_pri;              \
                if (leading_zeros && row_i[d_last] < pstop[d_last])               \
                    prune = 1;                                                    \
            }                                                                     \
            if (prune) break;                                                     \
        }                                                                         \
    }                                                                             \
    m->n_skyline = skyline_size;                                                  \
    free(pstop);                                                                  \
}

SALSA_REV_DEFINE(salsa_lexsort_run_scalar_f32, float,  skyline_scalar_f32, 1)
SALSA_REV_DEFINE(salsa_lexsort_run_sse2_f32,   float,  skyline_sse2_f32,   4)
SALSA_REV_DEFINE(salsa_lexsort_run_avx2_f32,   float,  skyline_avx2_f32,   8)
SALSA_REV_DEFINE(salsa_lexsort_run_scalar_f64, double, skyline_scalar_f64, 1)
SALSA_REV_DEFINE(salsa_lexsort_run_sse2_f64,   double, skyline_sse2_f64,   2)
SALSA_REV_DEFINE(salsa_lexsort_run_avx2_f64,   double, skyline_avx2_f64,   4)

BnlMetrics skyline_salsa_lexsort_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0, 0, 0, 0};
    /* Caller's lex sort is by natural column order; pstop walks reverse
     * natural order (key_order = NULL). */
    switch (kernel_mode) {
        case 3: salsa_lexsort_run_scalar_f32((const float  *)data, (const float  *)data, n_rows, n_dims, NULL, out_skyline, out_vcheck_freq, &m); break;
        case 4: salsa_lexsort_run_avx2_f32  ((const float  *)data, (const float  *)data, n_rows, n_dims, NULL, out_skyline, out_vcheck_freq, &m); break;
        case 5: salsa_lexsort_run_sse2_f32  ((const float  *)data, (const float  *)data, n_rows, n_dims, NULL, out_skyline, out_vcheck_freq, &m); break;
        case 6: salsa_lexsort_run_scalar_f64((const double *)data, (const double *)data, n_rows, n_dims, NULL, out_skyline, out_vcheck_freq, &m); break;
        case 7: salsa_lexsort_run_avx2_f64  ((const double *)data, (const double *)data, n_rows, n_dims, NULL, out_skyline, out_vcheck_freq, &m); break;
        case 8: salsa_lexsort_run_sse2_f64  ((const double *)data, (const double *)data, n_rows, n_dims, NULL, out_skyline, out_vcheck_freq, &m); break;
        default: break;
    }
    return m;
}

/* ── batched D&C full recursion + merge ──────────────────────────────────── */
/* Mirrors skylines/skyline/dnc/dnc.py {find_skyline_recursive, merge}.
 *
 * Each recursion / merge frame returns ownership of a malloc'd index buffer
 * to its caller, who is responsible for free(). This avoids hard-to-bound
 * upper limits on output size (the algorithm can briefly hold |L|+|R|-sized
 * intermediates inside merge before the outer concat). The buffers are
 * small (skyline-size scale), so the heap traffic is not a hotspot. */

#define DNC_RUN_DEFINE(T, KERNEL_FN, CHUNK_SIZE, SUFFIX)                                         \
                                                                                                 \
static T dnc_mean_##SUFFIX(const T *data, const int64_t *idx, int64_t n,                         \
                            int32_t col, int32_t n_dims) {                                       \
    T first = data[(size_t)idx[0] * (size_t)n_dims + (size_t)col];                               \
    int all_equal = 1;                                                                           \
    double sum = 0.0;                                                                            \
    for (int64_t i = 0; i < n; ++i) {                                                            \
        T v = data[(size_t)idx[i] * (size_t)n_dims + (size_t)col];                               \
        sum += (double)v;                                                                        \
        if (v != first) all_equal = 0;                                                           \
    }                                                                                            \
    if (all_equal) return first;                                                                 \
    return (T)(sum / (double)n);                                                                 \
}                                                                                                \
                                                                                                 \
static int64_t dnc_partition_##SUFFIX(const T *data, const int64_t *idx, int64_t n,              \
                                       int32_t col, int32_t n_dims, T pivot, int strict,         \
                                       int64_t *out_left, int64_t *out_right,                    \
                                       BnlMetrics *m) {                                          \
    int64_t nl = 0, nr = 0;                                                                      \
    m->comparisons += n;                                                                         \
    m->sub_comparisons += n;                                                                     \
    for (int64_t i = 0; i < n; ++i) {                                                            \
        T v = data[(size_t)idx[i] * (size_t)n_dims + (size_t)col];                               \
        int left = strict ? (v < pivot) : (v <= pivot);                                          \
        if (left) out_left[nl++] = idx[i];                                                       \
        else      out_right[nr++] = idx[i];                                                      \
    }                                                                                            \
    return nl;                                                                                   \
}                                                                                                \
                                                                                                 \
/* Trivial-leaf: BNL-style scan with removal over a subset of indices.                           \
 * reverse=0 -> forward (oldest->newest) iteration, mirrors dnc.py ref                           \
 *   `for j in list(skyline.items())` (dict insertion order).                                    \
 * reverse=1 -> reverse (newest->oldest) iteration, mirrors lexsort.py ref                       \
 *   `for j in reversed(list(skyline.items()))`.                                                 \
 * Early-exit on r1.dominating makes iteration order affect the comparison                       \
 * count, so the direction must match the EmitLab variant being mirrored. */                     \
static int64_t dnc_trivial_##SUFFIX(const T *data, const int64_t *idx, int64_t n_idx,            \
                                     int32_t n_dims, int reverse, int64_t *out,                  \
                                     BnlMetrics *m, int64_t *out_vcheck_freq) {                  \
    int64_t skyline_size = 0;                                                                    \
    for (int64_t rel_i = 0; rel_i < n_idx; ++rel_i) {                                            \
        int64_t i = idx[rel_i];                                                                  \
        const T *row_i = data + (size_t)i * (size_t)n_dims;                                      \
        int dominated = 0;                                                                       \
        if (reverse) {                                                                           \
            int64_t k = skyline_size - 1;                                                        \
            while (k >= 0) {                                                                     \
                int64_t j = out[k];                                                              \
                const T *row_j = data + (size_t)j * (size_t)n_dims;                              \
                m->comparisons += 1;                                                             \
                DomResult r1 = KERNEL_FN(row_j, row_i, n_dims);                                  \
                apply_dom_metrics(r1, n_dims, (CHUNK_SIZE), m, out_vcheck_freq);                 \
                if (r1.dominating) { dominated = 1; break; }                                     \
                m->comparisons += 1;                                                             \
                DomResult r2 = KERNEL_FN(row_i, row_j, n_dims);                                  \
                apply_dom_metrics(r2, n_dims, (CHUNK_SIZE), m, out_vcheck_freq);                 \
                if (r2.dominating) {                                                             \
                    for (int64_t s = k; s < skyline_size - 1; ++s)                               \
                        out[s] = out[s + 1];                                                     \
                    skyline_size -= 1;                                                           \
                }                                                                                \
                k -= 1;                                                                          \
            }                                                                                    \
        } else {                                                                                 \
            int64_t k = 0;                                                                       \
            while (k < skyline_size) {                                                           \
                int64_t j = out[k];                                                              \
                const T *row_j = data + (size_t)j * (size_t)n_dims;                              \
                m->comparisons += 1;                                                             \
                DomResult r1 = KERNEL_FN(row_j, row_i, n_dims);                                  \
                apply_dom_metrics(r1, n_dims, (CHUNK_SIZE), m, out_vcheck_freq);                 \
                if (r1.dominating) { dominated = 1; break; }                                     \
                m->comparisons += 1;                                                             \
                DomResult r2 = KERNEL_FN(row_i, row_j, n_dims);                                  \
                apply_dom_metrics(r2, n_dims, (CHUNK_SIZE), m, out_vcheck_freq);                 \
                if (r2.dominating) {                                                             \
                    for (int64_t s = k; s < skyline_size - 1; ++s)                               \
                        out[s] = out[s + 1];                                                     \
                    skyline_size -= 1;                                                           \
                    /* Don't advance k: surviving entry shifted into slot k. */                  \
                } else {                                                                         \
                    k += 1;                                                                      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
        if (!dominated) out[skyline_size++] = i;                                                 \
    }                                                                                            \
    return skyline_size;                                                                         \
}                                                                                                \
                                                                                                 \
/* Forward declaration (recursion <-> merge). */                                                 \
static int64_t dnc_merge_##SUFFIX(const T *data, int32_t n_dims,                                 \
                                   const int64_t *L, int64_t nL,                                 \
                                   const int64_t *R, int64_t nR,                                 \
                                   int32_t dimension,                                            \
                                   int64_t **out_buf, BnlMetrics *m,                             \
                                   int64_t *out_vcheck_freq);                                    \
                                                                                                 \
static int64_t dnc_recursive_##SUFFIX(const T *data, int32_t n_dims, int reverse,                \
                                       const int64_t *idx, int64_t n,                            \
                                       int32_t dimension,                                        \
                                       int64_t **out_buf, BnlMetrics *m,                         \
                                       int64_t *out_vcheck_freq) {                               \
    if (n == 1 || dimension == -1) {                                                             \
        int64_t *out = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));             \
        memcpy(out, idx, sizeof(int64_t) * (size_t)n);                                           \
        *out_buf = out;                                                                          \
        return n;                                                                                \
    }                                                                                            \
    if (n < 100) {                                                                               \
        int64_t *out = (int64_t *)malloc(sizeof(int64_t) * (size_t)n);                           \
        int64_t out_n = dnc_trivial_##SUFFIX(data, idx, n, n_dims, reverse, out, m,              \
                                              out_vcheck_freq);                                  \
        *out_buf = out;                                                                          \
        return out_n;                                                                            \
    }                                                                                            \
    T pivot = dnc_mean_##SUFFIX(data, idx, n, dimension, n_dims);                                \
    int64_t *L = (int64_t *)malloc(sizeof(int64_t) * (size_t)n);                                 \
    int64_t *R = (int64_t *)malloc(sizeof(int64_t) * (size_t)n);                                 \
    int64_t nL = dnc_partition_##SUFFIX(data, idx, n, dimension, n_dims, pivot,                  \
                                         /*strict*/1, L, R, m);                                  \
    int64_t nR = n - nL;                                                                         \
    int64_t result_n;                                                                            \
    if (nL == 0 || nR == 0) {                                                                    \
        result_n = dnc_recursive_##SUFFIX(data, n_dims, reverse, idx, n, dimension - 1,          \
                                           out_buf, m, out_vcheck_freq);                         \
    } else if (dimension == 0) {                                                                 \
        result_n = dnc_recursive_##SUFFIX(data, n_dims, reverse, R, nR, dimension,               \
                                           out_buf, m, out_vcheck_freq);                         \
    } else {                                                                                     \
        int64_t *LS = NULL, *RS = NULL;                                                          \
        int64_t nLS = dnc_recursive_##SUFFIX(data, n_dims, reverse, L, nL, dimension,            \
                                              &LS, m, out_vcheck_freq);                          \
        int64_t nRS = dnc_recursive_##SUFFIX(data, n_dims, reverse, R, nR, dimension,            \
                                              &RS, m, out_vcheck_freq);                          \
        int64_t *M = NULL;                                                                       \
        int64_t nM = dnc_merge_##SUFFIX(data, n_dims, LS, nLS, RS, nRS,                          \
                                         dimension, &M, m, out_vcheck_freq);                     \
        int64_t out_cap = nM + nRS;                                                              \
        int64_t *out = (int64_t *)malloc(sizeof(int64_t) * (size_t)(out_cap > 0 ? out_cap : 1)); \
        if (nM > 0) memcpy(out, M, sizeof(int64_t) * (size_t)nM);                                \
        if (nRS > 0) memcpy(out + nM, RS, sizeof(int64_t) * (size_t)nRS);                        \
        result_n = nM + nRS;                                                                     \
        *out_buf = out;                                                                          \
        free(M); free(RS); free(LS);                                                             \
    }                                                                                            \
    free(R); free(L);                                                                            \
    return result_n;                                                                             \
}                                                                                                \
                                                                                                 \
static int64_t dnc_merge_##SUFFIX(const T *data, int32_t n_dims,                                 \
                                   const int64_t *L, int64_t nL,                                 \
                                   const int64_t *R, int64_t nR,                                 \
                                   int32_t dimension,                                            \
                                   int64_t **out_buf, BnlMetrics *m,                             \
                                   int64_t *out_vcheck_freq) {                                   \
    if (nR == 0) {                                                                               \
        int64_t *out = (int64_t *)malloc(sizeof(int64_t) * (size_t)(nL > 0 ? nL : 1));           \
        if (nL > 0) memcpy(out, L, sizeof(int64_t) * (size_t)nL);                                \
        *out_buf = out;                                                                          \
        return nL;                                                                               \
    }                                                                                            \
    if (nL == 0) {                                                                               \
        *out_buf = (int64_t *)malloc(sizeof(int64_t));                                           \
        return 0;                                                                                \
    }                                                                                            \
    if (nR == 1) {                                                                               \
        const T *rrow = data + (size_t)R[0] * (size_t)n_dims;                                    \
        int64_t *out = (int64_t *)malloc(sizeof(int64_t) * (size_t)nL);                          \
        int64_t no = 0;                                                                          \
        for (int64_t i = 0; i < nL; ++i) {                                                       \
            const T *lrow = data + (size_t)L[i] * (size_t)n_dims;                                \
            m->comparisons += 1;                                                                 \
            DomResult r = KERNEL_FN(rrow, lrow, n_dims);                                         \
            apply_dom_metrics(r, n_dims, (CHUNK_SIZE), m, out_vcheck_freq);                      \
            if (!r.dominating) out[no++] = L[i];                                                 \
        }                                                                                        \
        *out_buf = out;                                                                          \
        return no;                                                                               \
    }                                                                                            \
    if (nL == 1) {                                                                               \
        const T *lrow = data + (size_t)L[0] * (size_t)n_dims;                                    \
        int dominated = 0;                                                                       \
        for (int64_t i = 0; i < nR; ++i) {                                                       \
            const T *rrow = data + (size_t)R[i] * (size_t)n_dims;                                \
            m->comparisons += 1;                                                                 \
            DomResult r = KERNEL_FN(rrow, lrow, n_dims);                                         \
            apply_dom_metrics(r, n_dims, (CHUNK_SIZE), m, out_vcheck_freq);                      \
            if (r.dominating) { dominated = 1; break; }                                          \
        }                                                                                        \
        int64_t *out = (int64_t *)malloc(sizeof(int64_t));                                       \
        if (!dominated) { out[0] = L[0]; *out_buf = out; return 1; }                             \
        *out_buf = out;                                                                          \
        return 0;                                                                                \
    }                                                                                            \
    if (dimension == 1) {                                                                        \
        T rmax = data[(size_t)R[0] * (size_t)n_dims + 0];                                        \
        for (int64_t i = 1; i < nR; ++i) {                                                       \
            T v = data[(size_t)R[i] * (size_t)n_dims + 0];                                       \
            if (v > rmax) rmax = v;                                                              \
        }                                                                                        \
        int64_t *out = (int64_t *)malloc(sizeof(int64_t) * (size_t)nL);                          \
        int64_t no = 0;                                                                          \
        for (int64_t i = 0; i < nL; ++i) {                                                       \
            T v = data[(size_t)L[i] * (size_t)n_dims + 0];                                       \
            m->comparisons += 1;                                                                 \
            m->sub_comparisons += 1;                                                             \
            if (v > rmax) out[no++] = L[i];                                                      \
        }                                                                                        \
        *out_buf = out;                                                                          \
        return no;                                                                               \
    }                                                                                            \
    /* General case. Partition L and R on dim-1 by mean(R[:, dim-1]). */                         \
    T pivot = dnc_mean_##SUFFIX(data, R, nR, dimension - 1, n_dims);                             \
    int64_t *LL = (int64_t *)malloc(sizeof(int64_t) * (size_t)nL);                               \
    int64_t *LR = (int64_t *)malloc(sizeof(int64_t) * (size_t)nL);                               \
    int64_t *RL = (int64_t *)malloc(sizeof(int64_t) * (size_t)nR);                               \
    int64_t *RR = (int64_t *)malloc(sizeof(int64_t) * (size_t)nR);                               \
    int64_t nLL = dnc_partition_##SUFFIX(data, L, nL, dimension - 1, n_dims, pivot,              \
                                          /*strict*/1, LL, LR, m);                               \
    int64_t nLR = nL - nLL;                                                                      \
    int64_t nRL = dnc_partition_##SUFFIX(data, R, nR, dimension - 1, n_dims, pivot,              \
                                          /*strict*/1, RL, RR, m);                               \
    int64_t nRR = nR - nRL;                                                                      \
    int64_t result_n;                                                                            \
    if (nLL == 0 && nRL == 0) {                                                                  \
        nLL = dnc_partition_##SUFFIX(data, L, nL, dimension - 1, n_dims, pivot,                  \
                                      /*strict*/0, LL, LR, m);                                   \
        nLR = nL - nLL;                                                                          \
        int64_t *S1 = NULL;                                                                      \
        int64_t nS1 = dnc_merge_##SUFFIX(data, n_dims, LL, nLL, RR, nRR,                         \
                                          dimension - 1, &S1, m, out_vcheck_freq);               \
        int64_t out_cap = nS1 + nLR;                                                             \
        int64_t *out = (int64_t *)malloc(sizeof(int64_t) * (size_t)(out_cap > 0 ? out_cap : 1)); \
        if (nS1 > 0) memcpy(out, S1, sizeof(int64_t) * (size_t)nS1);                             \
        if (nLR > 0) memcpy(out + nS1, LR, sizeof(int64_t) * (size_t)nLR);                       \
        result_n = nS1 + nLR;                                                                    \
        *out_buf = out;                                                                          \
        free(S1);                                                                                \
    } else {                                                                                     \
        int64_t *S1 = NULL, *S2 = NULL, *S3 = NULL;                                              \
        /* S1 = merge(LL, RL, dim): both <pivot on dim-1 (L and R sides).                        \
           S2 = merge(LR, RR, dim): both >=pivot on dim-1 (L and R sides).                       \
           S3 = merge(S1, RR, dim-1): cross-pivot pruning of S1 against RR. */                   \
        int64_t nS1 = dnc_merge_##SUFFIX(data, n_dims, LL, nLL, RL, nRL,                         \
                                          dimension, &S1, m, out_vcheck_freq);                   \
        int64_t nS2 = dnc_merge_##SUFFIX(data, n_dims, LR, nLR, RR, nRR,                         \
                                          dimension, &S2, m, out_vcheck_freq);                   \
        int64_t nS3 = dnc_merge_##SUFFIX(data, n_dims, S1, nS1, RR, nRR,                         \
                                          dimension - 1, &S3, m, out_vcheck_freq);               \
        int64_t out_cap = nS2 + nS3;                                                             \
        int64_t *out = (int64_t *)malloc(sizeof(int64_t) * (size_t)(out_cap > 0 ? out_cap : 1)); \
        if (nS2 > 0) memcpy(out, S2, sizeof(int64_t) * (size_t)nS2);                             \
        if (nS3 > 0) memcpy(out + nS2, S3, sizeof(int64_t) * (size_t)nS3);                       \
        result_n = nS2 + nS3;                                                                    \
        *out_buf = out;                                                                          \
        free(S3); free(S2); free(S1);                                                            \
    }                                                                                            \
    free(RR); free(RL); free(LR); free(LL);                                                      \
    return result_n;                                                                             \
}                                                                                                \
                                                                                                 \
static void dnc_run_##SUFFIX(const T *data, int64_t n_rows, int32_t n_dims,                      \
                              int reverse, int64_t *out_skyline,                                 \
                              int64_t *out_vcheck_freq, BnlMetrics *m) {                         \
    int64_t *idx = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n_rows > 0 ? n_rows : 1));       \
    for (int64_t i = 0; i < n_rows; ++i) idx[i] = i;                                             \
    int64_t *out = NULL;                                                                         \
    int64_t out_n = dnc_recursive_##SUFFIX(data, n_dims, reverse, idx, n_rows, n_dims - 1,       \
                                            &out, m, out_vcheck_freq);                           \
    if (out_n > 0) memcpy(out_skyline, out, sizeof(int64_t) * (size_t)out_n);                    \
    m->n_skyline = out_n;                                                                        \
    free(out); free(idx);                                                                        \
}

DNC_RUN_DEFINE(float,  skyline_scalar_f32, 1, scalar_f32)
DNC_RUN_DEFINE(float,  skyline_sse2_f32,   4, sse2_f32)
DNC_RUN_DEFINE(float,  skyline_avx2_f32,   8, avx2_f32)
DNC_RUN_DEFINE(double, skyline_scalar_f64, 1, scalar_f64)
DNC_RUN_DEFINE(double, skyline_sse2_f64,   2, sse2_f64)
DNC_RUN_DEFINE(double, skyline_avx2_f64,   4, avx2_f64)

BnlMetrics skyline_dnc_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0, 0, 0, 0};
    switch (kernel_mode) {
        case 3: dnc_run_scalar_f32((const float  *)data, n_rows, n_dims, 0, out_skyline, out_vcheck_freq, &m); break;
        case 4: dnc_run_avx2_f32  ((const float  *)data, n_rows, n_dims, 0, out_skyline, out_vcheck_freq, &m); break;
        case 5: dnc_run_sse2_f32  ((const float  *)data, n_rows, n_dims, 0, out_skyline, out_vcheck_freq, &m); break;
        case 6: dnc_run_scalar_f64((const double *)data, n_rows, n_dims, 0, out_skyline, out_vcheck_freq, &m); break;
        case 7: dnc_run_avx2_f64  ((const double *)data, n_rows, n_dims, 0, out_skyline, out_vcheck_freq, &m); break;
        case 8: dnc_run_sse2_f64  ((const double *)data, n_rows, n_dims, 0, out_skyline, out_vcheck_freq, &m); break;
        default: break;
    }
    return m;
}

/* ── stable lex permutation + per-algo lex_run wrappers ────────────────────────
 *
 * Stable bottom-up mergesort over int64 row indices, descending lex compare
 * across a caller-supplied key_order. Mirrors np.lexsort's stability guarantee
 * (equal-key ties retain input order) so the produced permutation matches what
 * the Python pre-sort path would emit byte-for-byte.
 *
 * Each `_lex_run` allocates the permutation, materializes a permuted-data
 * scratch buffer, dispatches the existing scan kernel, then maps surviving
 * indices back into the original (unsorted) row index space.
 */

#define LEX_PERM_DEFINE(T, SUFFIX)                                                   \
static int lex_compare_desc_##SUFFIX(const T *data, int64_t i, int64_t j,            \
                                      int32_t n_dims,                                \
                                      const int32_t *key_order, int32_t n_keys) {    \
    const T *ri = data + (size_t)i * (size_t)n_dims;                                 \
    const T *rj = data + (size_t)j * (size_t)n_dims;                                 \
    for (int32_t k = 0; k < n_keys; ++k) {                                           \
        int32_t d = key_order[k];                                                    \
        if (ri[d] > rj[d]) return -1;                                                \
        if (ri[d] < rj[d]) return 1;                                                 \
    }                                                                                \
    return 0;                                                                        \
}                                                                                    \
                                                                                     \
static void lex_merge_##SUFFIX(const T *data, int32_t n_dims,                        \
                                const int32_t *key_order, int32_t n_keys,            \
                                const int64_t *src, int64_t *dst,                    \
                                int64_t lo, int64_t mid, int64_t hi) {               \
    int64_t i = lo, j = mid, k = lo;                                                 \
    while (i < mid && j < hi) {                                                      \
        int c = lex_compare_desc_##SUFFIX(data, src[i], src[j], n_dims,              \
                                           key_order, n_keys);                       \
        if (c <= 0) dst[k++] = src[i++];                                             \
        else        dst[k++] = src[j++];                                             \
    }                                                                                \
    while (i < mid) dst[k++] = src[i++];                                             \
    while (j < hi)  dst[k++] = src[j++];                                             \
}                                                                                    \
                                                                                     \
static int lex_perm_##SUFFIX(const T *data, int64_t n_rows, int32_t n_dims,          \
                              const int32_t *key_order, int32_t n_keys,              \
                              int64_t *out_perm) {                                   \
    if (n_rows <= 0) return 0;                                                       \
    for (int64_t i = 0; i < n_rows; ++i) out_perm[i] = i;                            \
    if (n_rows == 1) return 0;                                                       \
    int64_t *buf = (int64_t *)malloc(sizeof(int64_t) * (size_t)n_rows);              \
    if (!buf) return -1;                                                             \
    int64_t *src = out_perm;                                                         \
    int64_t *dst = buf;                                                              \
    for (int64_t width = 1; width < n_rows; width *= 2) {                            \
        for (int64_t lo = 0; lo < n_rows; lo += width * 2) {                         \
            int64_t mid = lo + width;     if (mid > n_rows) mid = n_rows;            \
            int64_t hi  = lo + width * 2; if (hi  > n_rows) hi  = n_rows;            \
            lex_merge_##SUFFIX(data, n_dims, key_order, n_keys,                      \
                               src, dst, lo, mid, hi);                               \
        }                                                                            \
        int64_t *tmp = src; src = dst; dst = tmp;                                    \
    }                                                                                \
    if (src != out_perm)                                                             \
        memcpy(out_perm, src, sizeof(int64_t) * (size_t)n_rows);                     \
    free(buf);                                                                       \
    return 0;                                                                        \
}                                                                                    \
                                                                                     \
static T *lex_permute_data_##SUFFIX(const T *data, int64_t n_rows, int32_t n_dims,   \
                                     const int64_t *perm) {                          \
    if (n_rows <= 0) return NULL;                                                    \
    T *out = (T *)malloc(sizeof(T) * (size_t)n_rows * (size_t)n_dims);               \
    if (!out) return NULL;                                                           \
    for (int64_t i = 0; i < n_rows; ++i) {                                           \
        memcpy(out + (size_t)i * (size_t)n_dims,                                     \
               data + (size_t)perm[i] * (size_t)n_dims,                              \
               sizeof(T) * (size_t)n_dims);                                          \
    }                                                                                \
    return out;                                                                      \
}

LEX_PERM_DEFINE(float,  f32)
LEX_PERM_DEFINE(double, f64)

/* Per-column min-max normalize, mirroring the Python pre-sort line
 *   data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
 * All arithmetic in the native dtype to match numpy's float path exactly.
 * Constant columns yield IEEE nan/inf identical to numpy. Caller frees.
 * Defined before the *_lex_run / monosort entries that consume it. */
#define NORMALIZE_DEFINE(T, SUFFIX)                                              \
static T *normalize_##SUFFIX(const T *src, int64_t n_rows, int32_t n_dims) {     \
    T *out = (T *)malloc(sizeof(T) * (size_t)n_rows * (size_t)n_dims);           \
    T *mn  = (T *)malloc(sizeof(T) * (size_t)n_dims);                            \
    T *mx  = (T *)malloc(sizeof(T) * (size_t)n_dims);                            \
    if (!out || !mn || !mx) { free(out); free(mn); free(mx); return NULL; }      \
    for (int32_t k = 0; k < n_dims; ++k) { mn[k] = src[k]; mx[k] = src[k]; }     \
    for (int64_t i = 0; i < n_rows; ++i) {                                       \
        const T *r = src + (size_t)i * (size_t)n_dims;                           \
        for (int32_t k = 0; k < n_dims; ++k) {                                   \
            T v = r[k];                                                          \
            if (v < mn[k]) mn[k] = v;                                            \
            if (v > mx[k]) mx[k] = v;                                            \
        }                                                                        \
    }                                                                            \
    for (int64_t i = 0; i < n_rows; ++i) {                                       \
        const T *r = src + (size_t)i * (size_t)n_dims;                           \
        T *o = out + (size_t)i * (size_t)n_dims;                                 \
        for (int32_t k = 0; k < n_dims; ++k)                                     \
            o[k] = (r[k] - mn[k]) / (mx[k] - mn[k]);                             \
    }                                                                            \
    free(mn); free(mx);                                                          \
    return out;                                                                  \
}

NORMALIZE_DEFINE(float,  f32)
NORMALIZE_DEFINE(double, f64)

/* Build the natural-order key vector [0, 1, ..., n_dims-1]. */
static void fill_key_order_forward(int32_t *keys, int32_t n_dims) {
    for (int32_t d = 0; d < n_dims; ++d) keys[d] = d;
}

/* Build the reverse-order key vector [n_dims-1, n_dims-2, ..., 0]. Used by
 * dnc_lex_run because dnc/lexsort.py reverses preference columns before the
 * lexsort, so the C-side lex key must walk columns in reverse to recover the
 * original preference order. */
static void fill_key_order_reverse(int32_t *keys, int32_t n_dims) {
    for (int32_t d = 0; d < n_dims; ++d) keys[d] = n_dims - 1 - d;
}

/* Map surviving indices in `out_skyline` (currently in permuted space) back
 * to the original input row index via `perm`. */
static inline void map_perm_indices(int64_t *out_skyline, int64_t n,
                                     const int64_t *perm) {
    for (int64_t i = 0; i < n; ++i) out_skyline[i] = perm[out_skyline[i]];
}

/* Dispatch helper: run the appropriate dtype-specific lex permutation +
 * permuted-data buffer allocation. Returns the buffer (caller frees) or
 * NULL on failure. Fills `perm` in place. */
static void *lex_prepare(const void *data, int64_t n_rows, int32_t n_dims,
                          int32_t kernel_mode,
                          const int32_t *key_order, int32_t n_keys,
                          int64_t *perm) {
    if (kernel_mode >= 3 && kernel_mode <= 5) {
        if (lex_perm_f32((const float *)data, n_rows, n_dims,
                          key_order, n_keys, perm) != 0) return NULL;
        return lex_permute_data_f32((const float *)data, n_rows, n_dims, perm);
    }
    if (kernel_mode >= 6 && kernel_mode <= 8) {
        if (lex_perm_f64((const double *)data, n_rows, n_dims,
                          key_order, n_keys, perm) != 0) return NULL;
        return lex_permute_data_f64((const double *)data, n_rows, n_dims, perm);
    }
    return NULL;
}

/* Resolve the effective key_order for a lex_run entry. Returns 0 on success.
 * If `key_order_in` is non-NULL, the caller-supplied keys are used directly
 * (and *out_owned_keys is left NULL — caller does not own the buffer). If
 * `key_order_in` is NULL, a fresh int32 array is allocated and filled via
 * `fill_default_keys`; ownership transfers to the caller via *out_owned_keys.
 * On failure, returns -1; caller need not free anything in that path. */
typedef void (*key_filler_t)(int32_t *, int32_t);

static int resolve_key_order(const int32_t *key_order_in, int32_t n_keys_in,
                              int32_t n_dims, key_filler_t fill_default_keys,
                              const int32_t **out_keys, int32_t *out_n_keys,
                              int32_t **out_owned_keys) {
    *out_owned_keys = NULL;
    if (key_order_in != NULL) {
        *out_keys = key_order_in;
        *out_n_keys = n_keys_in > 0 ? n_keys_in : 0;
        return 0;
    }
    int32_t *keys = (int32_t *)malloc(sizeof(int32_t) * (size_t)n_dims);
    if (!keys) return -1;
    fill_default_keys(keys, n_dims);
    *out_owned_keys = keys;
    *out_keys = keys;
    *out_n_keys = n_dims;
    return 0;
}

BnlMetrics skyline_bnl_lex_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    const int32_t *key_order,
    int32_t n_keys,
    int32_t reverse_scan,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0};
    if (n_rows <= 0) return m;
    int64_t *perm = (int64_t *)malloc(sizeof(int64_t) * (size_t)n_rows);
    if (!perm) return m;
    const int32_t *keys; int32_t keys_n; int32_t *owned_keys;
    if (resolve_key_order(key_order, n_keys, n_dims, fill_key_order_forward,
                          &keys, &keys_n, &owned_keys) != 0) {
        free(perm); return m;
    }
    int64_t t_sort = monosort_now_ns();
    void *pdata = lex_prepare(data, n_rows, n_dims, kernel_mode, keys, keys_n, perm);
    if (!pdata) { free(perm); free(owned_keys); return m; }
    m.sort_ns = monosort_now_ns() - t_sort;

    switch (kernel_mode) {
        case 3: bnl_run_scalar_f32((const float  *)pdata, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 4: bnl_run_avx2_f32  ((const float  *)pdata, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 5: bnl_run_sse2_f32  ((const float  *)pdata, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 6: bnl_run_scalar_f64((const double *)pdata, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 7: bnl_run_avx2_f64  ((const double *)pdata, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 8: bnl_run_sse2_f64  ((const double *)pdata, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        default: break;
    }
    map_perm_indices(out_skyline, m.n_skyline, perm);
    free(pdata); free(perm); free(owned_keys);
    return m;
}

BnlMetrics skyline_sfs_lex_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    const int32_t *key_order,
    int32_t n_keys,
    int32_t reverse_scan,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0};
    if (n_rows <= 0) return m;
    int is_f32 = (kernel_mode >= 3 && kernel_mode <= 5);
    int is_f64 = (kernel_mode >= 6 && kernel_mode <= 8);
    if (!is_f32 && !is_f64) return m;
    int64_t *perm = (int64_t *)malloc(sizeof(int64_t) * (size_t)n_rows);
    if (!perm) return m;
    const int32_t *keys; int32_t keys_n; int32_t *owned_keys;
    if (resolve_key_order(key_order, n_keys, n_dims, fill_key_order_forward,
                          &keys, &keys_n, &owned_keys) != 0) {
        free(perm); return m;
    }

    int64_t t_norm = monosort_now_ns();
    void *ndata = is_f32
        ? (void *)normalize_f32((const float  *)data, n_rows, n_dims)
        : (void *)normalize_f64((const double *)data, n_rows, n_dims);
    if (!ndata) { free(perm); free(owned_keys); return m; }
    m.normalize_ns = monosort_now_ns() - t_norm;

    int64_t t_sort = monosort_now_ns();
    void *pdata = lex_prepare(ndata, n_rows, n_dims, kernel_mode, keys, keys_n, perm);
    free(ndata);
    if (!pdata) { free(perm); free(owned_keys); return m; }
    m.sort_ns = monosort_now_ns() - t_sort;

    /* The lex sort by `key_order` is the only permutation; the dominance scan
     * runs over the row-sorted buffer with columns in their original (strategy
     * VO) layout and uses the regular all-dimensions dominance kernel. */
    switch (kernel_mode) {
        case 3: sfs_run_scalar_f32((const float  *)pdata, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 4: sfs_run_avx2_f32  ((const float  *)pdata, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 5: sfs_run_sse2_f32  ((const float  *)pdata, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 6: sfs_run_scalar_f64((const double *)pdata, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 7: sfs_run_avx2_f64  ((const double *)pdata, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        case 8: sfs_run_sse2_f64  ((const double *)pdata, n_rows, n_dims, reverse_scan, out_skyline, out_vcheck_freq, &m); break;
        default: break;
    }
    map_perm_indices(out_skyline, m.n_skyline, perm);
    free(pdata); free(perm); free(owned_keys);
    return m;
}

BnlMetrics skyline_dnc_lex_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    const int32_t *key_order,
    int32_t n_keys,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0};
    if (n_rows <= 0) return m;
    int64_t *perm = (int64_t *)malloc(sizeof(int64_t) * (size_t)n_rows);
    if (!perm) return m;
    const int32_t *keys; int32_t keys_n; int32_t *owned_keys;
    if (resolve_key_order(key_order, n_keys, n_dims, fill_key_order_reverse,
                          &keys, &keys_n, &owned_keys) != 0) {
        free(perm); return m;
    }
    int64_t t_sort = monosort_now_ns();
    void *pdata = lex_prepare(data, n_rows, n_dims, kernel_mode, keys, keys_n, perm);
    if (!pdata) { free(perm); free(owned_keys); return m; }
    m.sort_ns = monosort_now_ns() - t_sort;

    switch (kernel_mode) {
        case 3: dnc_run_scalar_f32((const float  *)pdata, n_rows, n_dims, 1, out_skyline, out_vcheck_freq, &m); break;
        case 4: dnc_run_avx2_f32  ((const float  *)pdata, n_rows, n_dims, 1, out_skyline, out_vcheck_freq, &m); break;
        case 5: dnc_run_sse2_f32  ((const float  *)pdata, n_rows, n_dims, 1, out_skyline, out_vcheck_freq, &m); break;
        case 6: dnc_run_scalar_f64((const double *)pdata, n_rows, n_dims, 1, out_skyline, out_vcheck_freq, &m); break;
        case 7: dnc_run_avx2_f64  ((const double *)pdata, n_rows, n_dims, 1, out_skyline, out_vcheck_freq, &m); break;
        case 8: dnc_run_sse2_f64  ((const double *)pdata, n_rows, n_dims, 1, out_skyline, out_vcheck_freq, &m); break;
        default: break;
    }
    map_perm_indices(out_skyline, m.n_skyline, perm);
    free(pdata); free(perm); free(owned_keys);
    return m;
}

BnlMetrics skyline_salsa_lex_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    const int32_t *key_order,
    int32_t n_keys,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0};
    if (n_rows <= 0) return m;
    int is_f32 = (kernel_mode >= 3 && kernel_mode <= 5);
    int is_f64 = (kernel_mode >= 6 && kernel_mode <= 8);
    if (!is_f32 && !is_f64) return m;
    int64_t *perm = (int64_t *)malloc(sizeof(int64_t) * (size_t)n_rows);
    if (!perm) return m;
    const int32_t *keys; int32_t keys_n; int32_t *owned_keys;
    if (resolve_key_order(key_order, n_keys, n_dims, fill_key_order_forward,
                          &keys, &keys_n, &owned_keys) != 0) {
        free(perm); return m;
    }

    int64_t t_norm = monosort_now_ns();
    void *ndata = is_f32
        ? (void *)normalize_f32((const float  *)data, n_rows, n_dims)
        : (void *)normalize_f64((const double *)data, n_rows, n_dims);
    if (!ndata) { free(perm); free(owned_keys); return m; }
    m.normalize_ns = monosort_now_ns() - t_norm;

    int64_t t_sort = monosort_now_ns();
    void *pdata = lex_prepare(ndata, n_rows, n_dims, kernel_mode, keys, keys_n, perm);
    free(ndata);
    if (!pdata) { free(perm); free(owned_keys); return m; }
    m.sort_ns = monosort_now_ns() - t_sort;

    /* The lex sort by `key_order` is the only permutation; the dominance scan
     * runs the regular all-dimensions kernel over the row-sorted buffer in
     * its (strategy VO) column layout. The SaLSa pstop loop is passed the
     * same `keys` so its reverse-priority scan walks key_order[n-1..0] —
     * sound under arbitrary NCF orders without any data permutation. */
    const int32_t *pstop_keys = (keys_n == n_dims) ? keys : NULL;
    switch (kernel_mode) {
        case 3: salsa_lexsort_run_scalar_f32((const float  *)pdata, (const float  *)pdata, n_rows, n_dims, pstop_keys, out_skyline, out_vcheck_freq, &m); break;
        case 4: salsa_lexsort_run_avx2_f32  ((const float  *)pdata, (const float  *)pdata, n_rows, n_dims, pstop_keys, out_skyline, out_vcheck_freq, &m); break;
        case 5: salsa_lexsort_run_sse2_f32  ((const float  *)pdata, (const float  *)pdata, n_rows, n_dims, pstop_keys, out_skyline, out_vcheck_freq, &m); break;
        case 6: salsa_lexsort_run_scalar_f64((const double *)pdata, (const double *)pdata, n_rows, n_dims, pstop_keys, out_skyline, out_vcheck_freq, &m); break;
        case 7: salsa_lexsort_run_avx2_f64  ((const double *)pdata, (const double *)pdata, n_rows, n_dims, pstop_keys, out_skyline, out_vcheck_freq, &m); break;
        case 8: salsa_lexsort_run_sse2_f64  ((const double *)pdata, (const double *)pdata, n_rows, n_dims, pstop_keys, out_skyline, out_vcheck_freq, &m); break;
        default: break;
    }
    map_perm_indices(out_skyline, m.n_skyline, perm);
    free(pdata); free(perm); free(owned_keys);
    return m;
}

/* ── monotonic-score presort (SFS / SaLSa) ──────────────────────────────────
 *
 * Mirrors the Python pre-sort path that previously lived in sfs.py / salsa.py:
 *   SFS   : sort_order = np.argsort(-data.sum(axis=1))
 *   SaLSa : monotonic_scores = column_stack((max, sum));
 *           sort_order = np.lexsort(-monotonic_scores.T[::-1])
 *
 * Scores are accumulated in double regardless of data dtype, then a stable
 * bottom-up mergesort produces a descending permutation (ties keep input
 * order, matching np.lexsort stability; SFS output is canonicalized by the
 * caller so tie order is irrelevant there too). The whole score-build +
 * sort + data permutation region is timed into m.sort_ns. */

/* Descending compare over n_keys score columns; -1 if i precedes j. */
static int score_compare_desc(const double *scores, int n_keys,
                               int64_t i, int64_t j) {
    const double *si = scores + (size_t)i * (size_t)n_keys;
    const double *sj = scores + (size_t)j * (size_t)n_keys;
    for (int k = 0; k < n_keys; ++k) {
        if (si[k] > sj[k]) return -1;
        if (si[k] < sj[k]) return 1;
    }
    return 0;
}

static void score_merge(const double *scores, int n_keys,
                         const int64_t *src, int64_t *dst,
                         int64_t lo, int64_t mid, int64_t hi) {
    int64_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        int c = score_compare_desc(scores, n_keys, src[i], src[j]);
        if (c <= 0) dst[k++] = src[i++];
        else        dst[k++] = src[j++];
    }
    while (i < mid) dst[k++] = src[i++];
    while (j < hi)  dst[k++] = src[j++];
}

/* Stable descending mergesort of row indices keyed by `scores`. */
static int score_perm(const double *scores, int64_t n_rows, int n_keys,
                       int64_t *out_perm) {
    if (n_rows <= 0) return 0;
    for (int64_t i = 0; i < n_rows; ++i) out_perm[i] = i;
    if (n_rows == 1) return 0;
    int64_t *buf = (int64_t *)malloc(sizeof(int64_t) * (size_t)n_rows);
    if (!buf) return -1;
    int64_t *src = out_perm;
    int64_t *dst = buf;
    for (int64_t width = 1; width < n_rows; width *= 2) {
        for (int64_t lo = 0; lo < n_rows; lo += width * 2) {
            int64_t mid = lo + width;     if (mid > n_rows) mid = n_rows;
            int64_t hi  = lo + width * 2; if (hi  > n_rows) hi  = n_rows;
            score_merge(scores, n_keys, src, dst, lo, mid, hi);
        }
        int64_t *tmp = src; src = dst; dst = tmp;
    }
    if (src != out_perm)
        memcpy(out_perm, src, sizeof(int64_t) * (size_t)n_rows);
    free(buf);
    return 0;
}

#define MONO_SCORE_DEFINE(T, SUFFIX)                                             \
static void build_sum_score_##SUFFIX(const T *data, int64_t n_rows,              \
                                      int32_t n_dims, double *scores) {          \
    for (int64_t i = 0; i < n_rows; ++i) {                                       \
        const T *r = data + (size_t)i * (size_t)n_dims;                          \
        double s = 0.0;                                                          \
        for (int32_t k = 0; k < n_dims; ++k) s += (double)r[k];                  \
        scores[i] = s;                                                           \
    }                                                                            \
}                                                                                \
static void build_maxsum_score_##SUFFIX(const T *data, int64_t n_rows,           \
                                         int32_t n_dims, double *scores) {       \
    for (int64_t i = 0; i < n_rows; ++i) {                                       \
        const T *r = data + (size_t)i * (size_t)n_dims;                          \
        double s = 0.0, mx = (double)r[0];                                       \
        for (int32_t k = 0; k < n_dims; ++k) {                                   \
            double v = (double)r[k];                                             \
            s += v;                                                              \
            if (v > mx) mx = v;                                                  \
        }                                                                        \
        scores[2 * i]     = mx;  /* primary lex key */                           \
        scores[2 * i + 1] = s;   /* secondary lex key */                         \
    }                                                                            \
}

MONO_SCORE_DEFINE(float,  f32)
MONO_SCORE_DEFINE(double, f64)

BnlMetrics skyline_sfs_monosort_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    int32_t monotonic_fn,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0};
    if (n_rows <= 0) return m;
    int is_f32 = (kernel_mode >= 3 && kernel_mode <= 5);
    int is_f64 = (kernel_mode >= 6 && kernel_mode <= 8);
    if (!is_f32 && !is_f64) return m;

    /* monotonic_fn: 0 = descending row-sum (1 lex key),
     *               1 = descending (max, sum) lex (2 lex keys). */
    int n_keys = (monotonic_fn == 1) ? 2 : 1;
    double  *scores = (double  *)malloc(sizeof(double)  * (size_t)n_rows * (size_t)n_keys);
    int64_t *perm   = (int64_t *)malloc(sizeof(int64_t) * (size_t)n_rows);
    if (!scores || !perm) { free(scores); free(perm); return m; }

    int64_t t_norm = monosort_now_ns();
    void *ndata = is_f32
        ? (void *)normalize_f32((const float  *)data, n_rows, n_dims)
        : (void *)normalize_f64((const double *)data, n_rows, n_dims);
    if (!ndata) { free(scores); free(perm); return m; }
    m.normalize_ns = monosort_now_ns() - t_norm;

    int64_t t_sort = monosort_now_ns();
    if (monotonic_fn == 1) {
        if (is_f32) build_maxsum_score_f32((const float  *)ndata, n_rows, n_dims, scores);
        else        build_maxsum_score_f64((const double *)ndata, n_rows, n_dims, scores);
    } else {
        if (is_f32) build_sum_score_f32((const float  *)ndata, n_rows, n_dims, scores);
        else        build_sum_score_f64((const double *)ndata, n_rows, n_dims, scores);
    }

    if (score_perm(scores, n_rows, n_keys, perm) != 0) {
        free(ndata); free(scores); free(perm); return m;
    }
    void *pdata = is_f32
        ? (void *)lex_permute_data_f32((const float  *)ndata, n_rows, n_dims, perm)
        : (void *)lex_permute_data_f64((const double *)ndata, n_rows, n_dims, perm);
    free(ndata);
    if (!pdata) { free(scores); free(perm); return m; }
    m.sort_ns = monosort_now_ns() - t_sort;

    switch (kernel_mode) {
        case 3: sfs_run_scalar_f32((const float  *)pdata, n_rows, n_dims, 0, out_skyline, out_vcheck_freq, &m); break;
        case 4: sfs_run_avx2_f32  ((const float  *)pdata, n_rows, n_dims, 0, out_skyline, out_vcheck_freq, &m); break;
        case 5: sfs_run_sse2_f32  ((const float  *)pdata, n_rows, n_dims, 0, out_skyline, out_vcheck_freq, &m); break;
        case 6: sfs_run_scalar_f64((const double *)pdata, n_rows, n_dims, 0, out_skyline, out_vcheck_freq, &m); break;
        case 7: sfs_run_avx2_f64  ((const double *)pdata, n_rows, n_dims, 0, out_skyline, out_vcheck_freq, &m); break;
        case 8: sfs_run_sse2_f64  ((const double *)pdata, n_rows, n_dims, 0, out_skyline, out_vcheck_freq, &m); break;
        default: break;
    }
    map_perm_indices(out_skyline, m.n_skyline, perm);
    free(pdata); free(scores); free(perm);
    return m;
}

BnlMetrics skyline_salsa_monosort_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    int32_t monotonic_fn,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0};
    if (n_rows <= 0) return m;
    int is_f32 = (kernel_mode >= 3 && kernel_mode <= 5);
    int is_f64 = (kernel_mode >= 6 && kernel_mode <= 8);
    if (!is_f32 && !is_f64) return m;

    double  *scores = (double  *)malloc(sizeof(double)  * (size_t)n_rows * 2);
    int64_t *perm   = (int64_t *)malloc(sizeof(int64_t) * (size_t)n_rows);
    if (!scores || !perm) { free(scores); free(perm); return m; }

    int64_t t_norm = monosort_now_ns();
    void *ndata = is_f32
        ? (void *)normalize_f32((const float  *)data, n_rows, n_dims)
        : (void *)normalize_f64((const double *)data, n_rows, n_dims);
    if (!ndata) { free(scores); free(perm); return m; }
    m.normalize_ns = monosort_now_ns() - t_norm;

    int64_t t_sort = monosort_now_ns();
    if (is_f32) build_maxsum_score_f32((const float  *)ndata, n_rows, n_dims, scores);
    else        build_maxsum_score_f64((const double *)ndata, n_rows, n_dims, scores);

    /* monotonic_fn: 1 = descending (max, sum) lex (the [max, sum] built above);
     * 0 = descending row-sum — collapse the primary lex key onto the row sum so
     * both the presort and the pstop early-break key on sum alone. */
    if (monotonic_fn == 0) {
        for (int64_t i = 0; i < n_rows; ++i) scores[2 * i] = scores[2 * i + 1];
    }

    if (score_perm(scores, n_rows, 2, perm) != 0) {
        free(ndata); free(scores); free(perm); return m;
    }
    void *pdata = is_f32
        ? (void *)lex_permute_data_f32((const float  *)ndata, n_rows, n_dims, perm)
        : (void *)lex_permute_data_f64((const double *)ndata, n_rows, n_dims, perm);
    free(ndata);
    if (!pdata) { free(scores); free(perm); return m; }

    /* Rebuild the [max, sum] monotonic_scores in permuted order, in the data
     * dtype, for the SaLSa pstop early-break check. */
    void *pscores = NULL;
    if (is_f32) {
        float *ps = (float *)malloc(sizeof(float) * (size_t)n_rows * 2);
        if (ps) {
            for (int64_t i = 0; i < n_rows; ++i) {
                ps[2 * i]     = (float)scores[2 * perm[i]];
                ps[2 * i + 1] = (float)scores[2 * perm[i] + 1];
            }
        }
        pscores = ps;
    } else {
        double *ps = (double *)malloc(sizeof(double) * (size_t)n_rows * 2);
        if (ps) {
            for (int64_t i = 0; i < n_rows; ++i) {
                ps[2 * i]     = scores[2 * perm[i]];
                ps[2 * i + 1] = scores[2 * perm[i] + 1];
            }
        }
        pscores = ps;
    }
    if (!pscores) { free(pdata); free(scores); free(perm); return m; }
    m.sort_ns = monosort_now_ns() - t_sort;

    switch (kernel_mode) {
        case 3: salsa_run_scalar_f32((const float  *)pdata, n_rows, n_dims, (const float  *)pscores, out_skyline, out_vcheck_freq, &m); break;
        case 4: salsa_run_avx2_f32  ((const float  *)pdata, n_rows, n_dims, (const float  *)pscores, out_skyline, out_vcheck_freq, &m); break;
        case 5: salsa_run_sse2_f32  ((const float  *)pdata, n_rows, n_dims, (const float  *)pscores, out_skyline, out_vcheck_freq, &m); break;
        case 6: salsa_run_scalar_f64((const double *)pdata, n_rows, n_dims, (const double *)pscores, out_skyline, out_vcheck_freq, &m); break;
        case 7: salsa_run_avx2_f64  ((const double *)pdata, n_rows, n_dims, (const double *)pscores, out_skyline, out_vcheck_freq, &m); break;
        case 8: salsa_run_sse2_f64  ((const double *)pdata, n_rows, n_dims, (const double *)pscores, out_skyline, out_vcheck_freq, &m); break;
        default: break;
    }
    map_perm_indices(out_skyline, m.n_skyline, perm);
    free(pdata); free(pscores); free(scores); free(perm);
    return m;
}
