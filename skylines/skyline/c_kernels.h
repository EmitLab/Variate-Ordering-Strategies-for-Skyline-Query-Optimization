#ifndef SKYLINES_C_KERNELS_H
#define SKYLINES_C_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t dominating;
    int32_t fail_idx;
} DomResult;

/* float32 row-level kernels */
DomResult skyline_scalar_f32(const float *row1, const float *row2, int32_t n);
DomResult skyline_sse2_f32(const float *row1, const float *row2, int32_t n);
DomResult skyline_avx2_f32(const float *row1, const float *row2, int32_t n);
DomResult skyline_simd_f32(const float *row1, const float *row2, int32_t n);

/* float64 row-level kernels */
DomResult skyline_scalar_f64(const double *row1, const double *row2, int32_t n);
DomResult skyline_sse2_f64(const double *row1, const double *row2, int32_t n);
DomResult skyline_avx2_f64(const double *row1, const double *row2, int32_t n);
DomResult skyline_simd_f64(const double *row1, const double *row2, int32_t n);

int32_t skyline_cpu_supports_sse2(void);
int32_t skyline_cpu_supports_avx2(void);
int32_t skyline_simd_lanes(void);

/* Numba-callable variants: pointers passed as int64, result packed into int32.
 * Encoding: >= 0 = fail_idx (not dominating), -1 = not dominating (all checked), -2 = dominating */
int32_t skyline_scalar_f32_nb(int64_t ptr1, int64_t ptr2, int32_t n);
int32_t skyline_sse2_f32_nb(int64_t ptr1, int64_t ptr2, int32_t n);
int32_t skyline_avx2_f32_nb(int64_t ptr1, int64_t ptr2, int32_t n);
int32_t skyline_simd_f32_nb(int64_t ptr1, int64_t ptr2, int32_t n);

int32_t skyline_scalar_f64_nb(int64_t ptr1, int64_t ptr2, int32_t n);
int32_t skyline_sse2_f64_nb(int64_t ptr1, int64_t ptr2, int32_t n);
int32_t skyline_avx2_f64_nb(int64_t ptr1, int64_t ptr2, int32_t n);
int32_t skyline_simd_f64_nb(int64_t ptr1, int64_t ptr2, int32_t n);

/* ── batched BNL outer loop ──────────────────────────────────────────────────
 * Single-call entry that runs the full BNL skyline pass in C, eliminating the
 * per-pair Numba <-> ctypes boundary. Supports both forward (BNL) and reverse
 * (BNL with lexsort presort) scan orders. */
typedef struct {
    int64_t n_skyline;
    int64_t comparisons;
    int64_t sub_comparisons;
    int64_t simd_chunks_compared;
    /* Wall-clock nanoseconds spent in the C-internal per-column min-max
     * normalize. 0 for kernels that do not normalize. */
    int64_t normalize_ns;
    /* Wall-clock nanoseconds spent in the C-internal presort (score build +
     * stable sort + data permutation). For BBS this is the STR per-dim
     * MBR-center qsort time, plus the lex presort in the lexsort variant.
     * 0 for kernels that do not presort. */
    int64_t sort_ns;
    /* Wall-clock nanoseconds spent building the spatial index. For BBS this
     * is the STR R-tree node-construction time (bulk-load wall time minus
     * its qsort phase). 0 for kernels that build no index. */
    int64_t index_ns;
} BnlMetrics;

/*  data            : (n_rows, n_dims) row-major C-contiguous; dtype matches kernel_mode.
 *  kernel_mode     : 3 = scalar f32, 4 = avx2 f32, 5 = sse2 f32,
 *                    6 = scalar f64, 7 = avx2 f64, 8 = sse2 f64.
 *                    (Matches the kernel_mode constants in skylines/skyline/skyline.py:_MODE_CHUNK_SIZES.)
 *  reverse_scan    : 0 = forward scan (BNL),
 *                    1 = reverse scan (BNL + lexsort).
 *  out_skyline     : caller-allocated, len >= n_rows. Filled with surviving row indices.
 *  out_vcheck_freq : caller-allocated, len == n_dims. Caller zeros it. */
BnlMetrics skyline_bnl_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    int32_t reverse_scan,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

/* Batched SFS skyline (forward or reverse scan, no removal, single dom check
 * per pair). Same convention as skyline_bnl_run for kernel_mode/reverse_scan
 * and out buffers. */
BnlMetrics skyline_sfs_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    int32_t reverse_scan,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

/* Batched DnC trivial-partition skyline.
 *
 *   data            : (n_rows_total, n_dims) full data array.
 *   index           : int64[n_index] subset of row indices into data.
 *   n_dims          : columns to compare (== data.shape[1]).
 *   kernel_mode     : as above (3..8).
 *   out_skyline     : caller-allocated, len >= n_index. Filled with row
 *                     indices (in the data frame, not relative to partition).
 *   out_vcheck_freq : caller-allocated, len == n_dims. Caller zeros it. */
BnlMetrics skyline_dnc_trivial_run(
    const void *data,
    const int64_t *index,
    int64_t n_index,
    int32_t n_dims,
    int32_t kernel_mode,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

/* Batched SaLSa (forward scan + pstop pruning).
 *
 *   data              : (n_rows, n_dims), C-contiguous, dtype matches kernel_mode.
 *   monotonic_scores  : (n_rows, 2), C-contiguous, same dtype as data.
 *                       Per-row [max, sum] used for pstop early-break check.
 *   Other args identical to skyline_sfs_run. */
BnlMetrics skyline_salsa_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    const void *monotonic_scores,
    int32_t kernel_mode,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

/* Batched BBS (Branch-and-Bound Skyline).
 *
 * Builds an STR-packed R-tree internally and runs the heap-based BBS
 * traversal entirely in C. SIMD is irrelevant here — BBS is bound by
 * R-tree pruning and operates on MBR bounds (not raw row pairs); we
 * always run the scalar MBR dominance check.
 *
 * The STR bulk-load is split into its qsort phase (m.sort_ns) and the
 * R-tree node-construction phase (m.index_ns), both separate from the
 * heap-traversal run time.
 *
 *   data            : (n_rows, n_dims) row-major C-contiguous; dtype matches
 *                     kernel_mode (3..5 = f32, 6..8 = f64). SIMD modes fall
 *                     back to scalar arithmetic.
 *   n_rows / n_dims : full data extent.
 *   key_map_1       : int32[n_dims] — dim ordering used by STR bulk-load
 *                     (controls heap priority via leaf packing order).
 *   keys_2          : int32[n_dims] — dim iteration order in MBR dominance
 *                     check (controls vcheck-frequency accumulation order).
 *                     Value -1 in keys_2 terminates iteration early (length
 *                     can be < n_dims), but typically all n_dims entries
 *                     are populated.
 *   M               : R-tree fan-out (max entries per node). 0 = use default
 *                     `4096 / (16 * n_dims + 8)`.
 *   out_skyline     : caller-allocated, len >= n_rows. Filled with surviving
 *                     row indices.
 *   out_vcheck_freq : caller-allocated, len == n_dims. Caller zeros it. */
BnlMetrics skyline_bbs_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    const int32_t *key_map_1,
    const int32_t *keys_2,
    int32_t M,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

/* BBS lexsort variant: stable np.lexsort(-data.T[::-1]) presort of the leaf
 * entries before STR bulk-load (mirrors the historical bbs/lexsort.py
 * form_rtree). Same signature/semantics as skyline_bbs_run otherwise. */
BnlMetrics skyline_bbs_lex_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    const int32_t *key_map_1,
    const int32_t *keys_2,
    int32_t M,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

/* Batched D&C full recursion + merge.
 *
 *   data            : (n_rows, n_dims) row-major C-contiguous; dtype matches kernel_mode.
 *                     Caller must apply preference reversal / lexsort presort beforehand
 *                     (mirrors find_skyline preprocessing in dnc.py / lexsort.py).
 *   n_rows / n_dims : full data extent.
 *   kernel_mode     : as above (3..8).
 *   out_skyline     : caller-allocated, len >= n_rows. Filled with surviving row
 *                     indices (positions into the input data, 0..n_rows-1).
 *   out_vcheck_freq : caller-allocated, len == n_dims. Caller zeros it. */
BnlMetrics skyline_dnc_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

/* Batched SaLSa lexsort (reverse scan + reverse-lex pstop pruning).
 * No monotonic_scores — pstop check uses data rows directly. */
BnlMetrics skyline_salsa_lexsort_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

/* ── batched lexsort variants (internal stable lex permutation) ──────────────
 * Fold the np.lexsort + permute step into the C run. Each entry takes the
 * caller's already-shaped data (column slice / preference reversal / SFS-SaLSa
 * normalization applied) and computes a descending stable lex permutation
 * internally before running the existing scan kernel. Surviving indices in
 * out_skyline are mapped back to the original unsorted row index space.
 *
 *   data           : (n_rows, n_dims) row-major C-contiguous, dtype matches
 *                    kernel_mode (3..8).
 *   kernel_mode    : as above (3..8).
 *   key_order      : optional int32[n_keys] specifying the column order used
 *                    for the lex permutation. NULL = default (forward natural
 *                    order for bnl/sfs/salsa, reverse for dnc). Pass a custom
 *                    key_order to decouple the lex sort priority from the
 *                    column slicing used by the dominance scan.
 *   n_keys         : number of entries in key_order (ignored if key_order is
 *                    NULL). Typically equals n_dims.
 *   reverse_scan   : (skyline_bnl_lex_run / skyline_sfs_lex_run only) candidate
 *                    window scan direction — 0 = forward (oldest→newest skyline
 *                    buffer entry), 1 = reverse (newest→oldest). dnc/salsa lex
 *                    have no reverse-scan analog and omit this parameter.
 *   out_skyline    : caller-allocated, len >= n_rows. Filled with surviving
 *                    row indices into the input data (NOT the sorted order).
 *   out_vcheck_freq: caller-allocated, len == n_dims. Caller zeros it.
 *
 * Default lex key conventions (key_order == NULL):
 *   bnl/sfs/salsa : descending lex by columns (0, 1, ..., n_dims-1).
 *   dnc           : descending lex by columns (n_dims-1, ..., 0). Caller is
 *                   expected to have already reversed preference columns
 *                   (matching dnc/lexsort.py), so this column ordering
 *                   re-applies the original preference order as the lex key.
 */
BnlMetrics skyline_bnl_lex_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    const int32_t *key_order,
    int32_t n_keys,
    int32_t reverse_scan,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

BnlMetrics skyline_sfs_lex_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    const int32_t *key_order,
    int32_t n_keys,
    int32_t reverse_scan,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

BnlMetrics skyline_dnc_lex_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    const int32_t *key_order,
    int32_t n_keys,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

BnlMetrics skyline_salsa_lex_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    const int32_t *key_order,
    int32_t n_keys,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

/* ── batched monotonic-score presort variants ───────────────────────────────
 * Fold the SFS / SaLSa monotone presort + permute step into the C run,
 * replacing the Python np.argsort / np.lexsort path. `data` is the caller's
 * already-normalized C-contiguous array; the score build, stable descending
 * sort, and data permutation run internally and are timed into m.sort_ns.
 * Surviving indices in out_skyline map back to the original input rows.
 *
 *   monotonic_fn : selects the monotone presort function —
 *     0 = descending row sum     (mirrors np.argsort(-data.sum(1)))
 *     1 = descending (max, sum) lex, primary max then sum
 *         (mirrors np.lexsort(-[max,sum].T[::-1]))
 *
 * For salsa the permuted monotonic_scores are rebuilt internally for the
 * pstop early-break check; under monotonic_fn=0 both score columns carry the
 * row sum, so the pstop check keys on sum alone. */
BnlMetrics skyline_sfs_monosort_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    int32_t monotonic_fn,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

BnlMetrics skyline_salsa_monosort_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    int32_t monotonic_fn,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq);

#ifdef __cplusplus
}
#endif

#endif
