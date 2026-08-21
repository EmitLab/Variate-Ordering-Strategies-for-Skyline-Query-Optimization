/* Batched BBS (Branch-and-Bound Skyline) — full R-tree + heap traversal in C.
 *
 * Mirrors skylines/skyline/bbs/bbs.py and skylines/skyline/bbs/rtree: STR-packed R-tree built
 * once per call, BBS heap-based traversal with MBR-level dominance pruning,
 * counter accumulation matches the Python implementation.
 *
 * MBR coordinates are always double, so the dom-check SIMD width is independent
 * of the f32/f64 row dtype: SSE2 = 2 lanes, AVX2 = 4 lanes. CPU support is
 * detected at runtime (mbr_dom_select), with graceful downgrade. Scalar uses
 * a keys_2-ordered scan for byte-identical sub_comparisons/vcheck_freq
 * accounting; the SIMD paths materialize a contiguous keys_2 gather then run
 * lane-wide cmp+early-exit, and feed simd_chunks_compared via mbr_finish using
 * the same ceil(scanned/lanes) convention as apply_dom_metrics in c_kernels.c. */

/* clock_gettime / CLOCK_MONOTONIC need POSIX visibility under -std=c11. */
#define _POSIX_C_SOURCE 199309L

#include "c_kernels.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Monotonic wall-clock nanoseconds for the BBS STR bulk-load and lex
 * presort timing — see m->index_ns / m->sort_ns in BBS_RUN_DEFINE. */
static inline int64_t bbs_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define SKYLINES_X86 1
#include <immintrin.h>
#else
#define SKYLINES_X86 0
#endif

#if SKYLINES_X86 && defined(__GNUC__)
static int bbs_cpu_has_sse2(void) {
    static int c = -1;
    if (c < 0) { __builtin_cpu_init(); c = __builtin_cpu_supports("sse2") ? 1 : 0; }
    return c;
}
static int bbs_cpu_has_avx2(void) {
    static int c = -1;
    if (c < 0) { __builtin_cpu_init(); c = __builtin_cpu_supports("avx2") ? 1 : 0; }
    return c;
}
#else
static int bbs_cpu_has_sse2(void) { return 0; }
static int bbs_cpu_has_avx2(void) { return 0; }
#endif

/* ── data types ─────────────────────────────────────────────────────────── */

typedef struct BBSNode BBSNode;
typedef struct BBSEntry BBSEntry;

struct BBSEntry {
    double  *mbr_min;       /* len n_dims */
    double  *mbr_max;       /* len n_dims */
    BBSNode *child_node;    /* non-NULL for internal entries */
    int64_t  leaf_idx;      /* row index; valid iff child_node == NULL */
    BBSNode *node;          /* back-pointer to container node */
};

struct BBSNode {
    BBSEntry **entries;
    int32_t   n_entries;
    int32_t   cap;
    int32_t   is_leaf;      /* 1 = leaf, 0 = internal */
};

/* ── arena ──────────────────────────────────────────────────────────────── */

/* Tracks all heap allocations made during a single BBS run for bulk free
 * at the end. We keep two parallel lists: one for malloc'd pointers and
 * one for the size of each allocation (only used for debugging). */

typedef struct {
    void   **ptrs;
    int64_t  n;
    int64_t  cap;
} Arena;

static void arena_init(Arena *a) {
    a->ptrs = NULL;
    a->n = 0;
    a->cap = 0;
}

static void *arena_alloc(Arena *a, size_t bytes) {
    void *p = calloc(1, bytes);
    if (a->n == a->cap) {
        int64_t new_cap = a->cap == 0 ? 64 : a->cap * 2;
        a->ptrs = (void **)realloc(a->ptrs, sizeof(void *) * (size_t)new_cap);
        a->cap = new_cap;
    }
    a->ptrs[a->n++] = p;
    return p;
}

static void arena_free(Arena *a) {
    for (int64_t i = 0; i < a->n; ++i) free(a->ptrs[i]);
    free(a->ptrs);
}

/* ── MBR ops ────────────────────────────────────────────────────────────── */

/* STR bulk-load only needs centers + a final MBR-of-children for inner
 * nodes. No `area`, `combine` (point-pair), or insert/split helpers — those
 * belong to the regular R-tree insert path which BBS doesn't exercise. */

/* MBR-level dominance check matching `BBSSkyline.is_dominating(mbr1, mbr2)`:
 *   dominating iff for all i: mbr1.min[i] >= mbr2.max[i],
 *                with at least one strict mbr1.min[i] > mbr2.max[i].
 * Iterates dims in `keys_2` order; updates sub_comparisons/vcheck_freq
 * identically to the Python path. */
/* All three dom variants share this signature so the traversal can select
 * one via a function pointer. `sa`/`sb` are caller-owned scratch buffers of
 * length >= n_dims (used only by the SIMD variants for the keys_2 gather);
 * the scalar variant ignores them. Metric accounting (sub_comparisons,
 * vcheck_freq) is byte-identical across all three. */
typedef int (*mbr_dom_fn)(const double *, const double *, int32_t,
                          const int32_t *, BnlMetrics *, int64_t *,
                          double *, double *);

static int mbr_dominates(const double *mbr1_min, const double *mbr2_max,
                          int32_t n_dims, const int32_t *keys_2,
                          BnlMetrics *m, int64_t *out_vcheck_freq,
                          double *sa, double *sb) {
    (void)sa; (void)sb;
    int dominating = 0;
    for (int32_t pos = 0; pos < n_dims; ++pos) {
        int32_t i = keys_2[pos];
        m->sub_comparisons += 1;  /* single-count-per-dim convention */
        if (mbr1_min[i] < mbr2_max[i]) {
            out_vcheck_freq[i] += 1;
            return 0;
        }
        if (mbr1_min[i] > mbr2_max[i]) {
            dominating = 1;
        }
    }
    out_vcheck_freq[n_dims - 1] += 1;
    return dominating;
}

/* Gather mbr coords into keys_2 order so the scan is contiguous. n_dims is
 * small (typically 3..6); the gather cost is negligible vs the traversal. */
static inline void mbr_gather(const double *mbr1_min, const double *mbr2_max,
                              int32_t n_dims, const int32_t *keys_2,
                              double *sa, double *sb) {
    for (int32_t pos = 0; pos < n_dims; ++pos) {
        int32_t i = keys_2[pos];
        sa[pos] = mbr1_min[i];
        sb[pos] = mbr2_max[i];
    }
}

/* Shared tail of the SIMD variants: `firstpos` = lowest keys_2-order index
 * with sa < sb (or -1 if none), `any_gt` = sa > sb seen anywhere. Reproduces
 * the scalar early-exit accounting exactly. `lanes` is the SIMD width of the
 * caller (2 for SSE2, 4 for AVX2 — MBR coords are always double) and is used
 * to bump simd_chunks_compared by ceil(scanned / lanes), matching the
 * row-vs-row convention in apply_dom_metrics (c_kernels.c). */
static inline int mbr_finish(int32_t firstpos, int any_gt, int32_t n_dims,
                             const int32_t *keys_2, BnlMetrics *m,
                             int64_t *out_vcheck_freq, int32_t lanes) {
    int32_t scanned;
    int result;
    if (firstpos >= 0) {
        m->sub_comparisons += firstpos + 1;
        out_vcheck_freq[keys_2[firstpos]] += 1;
        scanned = firstpos + 1;
        result = 0;
    } else {
        m->sub_comparisons += n_dims;
        out_vcheck_freq[n_dims - 1] += 1;
        scanned = n_dims;
        result = any_gt;
    }
    m->simd_chunks_compared += (scanned + lanes - 1) / lanes;
    return result;
}

#if SKYLINES_X86
__attribute__((target("sse2")))
static int mbr_dominates_sse2(const double *mbr1_min, const double *mbr2_max,
                               int32_t n_dims, const int32_t *keys_2,
                               BnlMetrics *m, int64_t *out_vcheck_freq,
                               double *sa, double *sb) {
    mbr_gather(mbr1_min, mbr2_max, n_dims, keys_2, sa, sb);
    const int32_t lanes = 2;
    const int32_t limit = n_dims - (n_dims % lanes);
    int any_gt = 0;
    for (int32_t i = 0; i < limit; i += lanes) {
        __m128d a = _mm_loadu_pd(sa + i);
        __m128d b = _mm_loadu_pd(sb + i);
        int32_t lt = _mm_movemask_pd(_mm_cmplt_pd(a, b));
        if (lt != 0)
            return mbr_finish(i + __builtin_ctz((unsigned)lt), any_gt,
                              n_dims, keys_2, m, out_vcheck_freq, lanes);
        if (_mm_movemask_pd(_mm_cmpgt_pd(a, b)) != 0) any_gt = 1;
    }
    if (limit < n_dims) {  /* lanes == 2 -> tail is exactly 1 dim */
        __m128d a = _mm_load_sd(sa + limit);
        __m128d b = _mm_load_sd(sb + limit);
        int32_t lt = _mm_movemask_pd(_mm_cmplt_pd(a, b));
        if (lt != 0)
            return mbr_finish(limit + __builtin_ctz((unsigned)lt), any_gt,
                              n_dims, keys_2, m, out_vcheck_freq, lanes);
        if (_mm_movemask_pd(_mm_cmpgt_pd(a, b)) != 0) any_gt = 1;
    }
    return mbr_finish(-1, any_gt, n_dims, keys_2, m, out_vcheck_freq, lanes);
}

__attribute__((target("avx2")))
static int mbr_dominates_avx2(const double *mbr1_min, const double *mbr2_max,
                               int32_t n_dims, const int32_t *keys_2,
                               BnlMetrics *m, int64_t *out_vcheck_freq,
                               double *sa, double *sb) {
    mbr_gather(mbr1_min, mbr2_max, n_dims, keys_2, sa, sb);
    const int32_t lanes = 4;
    const int32_t limit = n_dims - (n_dims % lanes);
    int any_gt = 0;
    for (int32_t i = 0; i < limit; i += lanes) {
        __m256d a = _mm256_loadu_pd(sa + i);
        __m256d b = _mm256_loadu_pd(sb + i);
        int32_t lt = _mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
        if (lt != 0)
            return mbr_finish(i + __builtin_ctz((unsigned)lt), any_gt,
                              n_dims, keys_2, m, out_vcheck_freq, lanes);
        if (_mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_GT_OQ)) != 0) any_gt = 1;
    }
    if (limit < n_dims) {
        const int32_t tail = n_dims - limit;
        const __m256i idx = _mm256_setr_epi64x(0, 1, 2, 3);
        const __m256i msk = _mm256_cmpgt_epi64(_mm256_set1_epi64x(tail), idx);
        __m256d a = _mm256_maskload_pd(sa + limit, msk);
        __m256d b = _mm256_maskload_pd(sb + limit, msk);
        int32_t lt = _mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
        if (lt != 0)
            return mbr_finish(limit + __builtin_ctz((unsigned)lt), any_gt,
                              n_dims, keys_2, m, out_vcheck_freq, lanes);
        if (_mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_GT_OQ)) != 0) any_gt = 1;
    }
    return mbr_finish(-1, any_gt, n_dims, keys_2, m, out_vcheck_freq, lanes);
}
#endif /* SKYLINES_X86 */

/* dom_simd: 0 = scalar, 1 = SSE2, 2 = AVX2 (caller already CPU-gated). */
static mbr_dom_fn mbr_dom_select(int dom_simd) {
#if SKYLINES_X86
    if (dom_simd == 2) return mbr_dominates_avx2;
    if (dom_simd == 1) return mbr_dominates_sse2;
#else
    (void)dom_simd;
#endif
    return mbr_dominates;
}

/* ── node helpers ───────────────────────────────────────────────────────── */

static BBSNode *node_new(Arena *a, int32_t is_leaf) {
    BBSNode *n = (BBSNode *)arena_alloc(a, sizeof(BBSNode));
    n->is_leaf = is_leaf;
    n->n_entries = 0;
    n->cap = 0;
    n->entries = NULL;
    return n;
}

static void node_reserve(Arena *a, BBSNode *n, int32_t cap) {
    if (n->cap >= cap) return;
    BBSEntry **arr = (BBSEntry **)arena_alloc(a, sizeof(BBSEntry *) * (size_t)cap);
    if (n->n_entries > 0) {
        memcpy(arr, n->entries, sizeof(BBSEntry *) * (size_t)n->n_entries);
    }
    n->entries = arr;
    n->cap = cap;
}

static void node_compute_mbr(const BBSNode *n, double *out_min, double *out_max,
                              int32_t n_dims) {
    BBSEntry *e0 = n->entries[0];
    memcpy(out_min, e0->mbr_min, sizeof(double) * (size_t)n_dims);
    memcpy(out_max, e0->mbr_max, sizeof(double) * (size_t)n_dims);
    for (int32_t i = 1; i < n->n_entries; ++i) {
        BBSEntry *e = n->entries[i];
        for (int32_t d = 0; d < n_dims; ++d) {
            if (e->mbr_min[d] < out_min[d]) out_min[d] = e->mbr_min[d];
            if (e->mbr_max[d] > out_max[d]) out_max[d] = e->mbr_max[d];
        }
    }
}

/* ── STR bulk-load ──────────────────────────────────────────────────────── */

/* Sort context: stable, sortable by MBR-center on a single dim. We sort
 * pointer arrays in-place. Use a thread-local-style static dim/key_map
 * trick — qsort_r isn't portable, so we cheat with a tiny global. */

typedef struct {
    int32_t  dim;            /* logical dim index used for sort */
    int32_t  n_dims;
} SortCtx;

static SortCtx g_sort_ctx;   /* not thread-safe; BBS run is sequential */

static int cmp_entry_by_center(const void *a, const void *b) {
    const BBSEntry *ea = *(const BBSEntry *const *)a;
    const BBSEntry *eb = *(const BBSEntry *const *)b;
    int32_t d = g_sort_ctx.dim;
    double ca = 0.5 * (ea->mbr_min[d] + ea->mbr_max[d]);
    double cb = 0.5 * (eb->mbr_min[d] + eb->mbr_max[d]);
    if (ca < cb) return -1;
    if (ca > cb) return  1;
    return 0;
}

/* STR group partitioning. Mirrors `str_groups` in rtree.py. Splits `items`
 * into contiguous slabs that are recursively sorted along subsequent dims,
 * yielding a flat list of groups each of size ≤ M. */
static void str_partition(BBSEntry **items, int64_t n,
                           int32_t M, int32_t n_dims, const int32_t *key_map_1,
                           int32_t dim,
                           BBSEntry ****out_groups, int64_t *out_n_groups,
                           int64_t **out_group_sizes,
                           int64_t *out_total_offset, int64_t *out_groups_cap) {
    /* Trivial case: small enough — emit one group. */
    if (n <= M) {
        if (*out_n_groups == *out_groups_cap) {
            int64_t new_cap = (*out_groups_cap == 0) ? 16 : (*out_groups_cap) * 2;
            *out_groups       = (BBSEntry ***)realloc(*out_groups,       sizeof(BBSEntry **) * (size_t)new_cap);
            *out_group_sizes  = (int64_t    *)realloc(*out_group_sizes,  sizeof(int64_t)     * (size_t)new_cap);
            *out_groups_cap = new_cap;
        }
        (*out_groups)[*out_n_groups]      = items;
        (*out_group_sizes)[*out_n_groups] = n;
        (*out_n_groups)++;
        *out_total_offset += n;
        return;
    }

    int32_t k = n_dims - dim;

    /* Last dim: sort and chunk. */
    if (k == 1) {
        g_sort_ctx.dim = key_map_1[dim];
        g_sort_ctx.n_dims = n_dims;
        qsort(items, (size_t)n, sizeof(BBSEntry *), cmp_entry_by_center);
        for (int64_t i = 0; i < n; i += M) {
            int64_t take = (n - i < M) ? (n - i) : M;
            if (*out_n_groups == *out_groups_cap) {
                int64_t new_cap = (*out_groups_cap == 0) ? 16 : (*out_groups_cap) * 2;
                *out_groups       = (BBSEntry ***)realloc(*out_groups,       sizeof(BBSEntry **) * (size_t)new_cap);
                *out_group_sizes  = (int64_t    *)realloc(*out_group_sizes,  sizeof(int64_t)     * (size_t)new_cap);
                *out_groups_cap = new_cap;
            }
            (*out_groups)[*out_n_groups]      = items + i;
            (*out_group_sizes)[*out_n_groups] = take;
            (*out_n_groups)++;
            *out_total_offset += take;
        }
        return;
    }

    /* General case: sort by current dim and recurse into slabs. */
    g_sort_ctx.dim = key_map_1[dim];
    g_sort_ctx.n_dims = n_dims;
    qsort(items, (size_t)n, sizeof(BBSEntry *), cmp_entry_by_center);

    /* P = ceil(n / M); slab_size = max(M, M * ceil(P^((k-1)/k))). */
    int64_t P = (n + M - 1) / M;
    double exponent = (double)(k - 1) / (double)k;
    int64_t slab_size = (int64_t)((double)M * ceil(pow((double)P, exponent)));
    if (slab_size < M) slab_size = M;

    for (int64_t i = 0; i < n; i += slab_size) {
        int64_t take = (n - i < slab_size) ? (n - i) : slab_size;
        str_partition(items + i, take, M, n_dims, key_map_1, dim + 1,
                      out_groups, out_n_groups, out_group_sizes,
                      out_total_offset, out_groups_cap);
    }
}

/* Generic version that operates on a polymorphic item array — used at upper
 * levels where items are nodes (we treat them via their MBR). To avoid a
 * second qsort comparator, we materialize node-level entries (each pointing
 * to the child node's MBR) and reuse the entry-based STR. */

static BBSNode *bulk_load_str(Arena *a, BBSEntry **leaves, int64_t n_rows,
                               int32_t n_dims, int32_t M, const int32_t *key_map_1,
                               double *mbr_pool, int64_t *mbr_pool_cursor) {
    /* Step 1: build leaf nodes by STR-grouping the leaf entries. */
    BBSEntry ***groups = NULL;
    int64_t   *group_sizes = NULL;
    int64_t    n_groups = 0;
    int64_t    total = 0;
    int64_t    groups_cap = 0;

    str_partition(leaves, n_rows, M, n_dims, key_map_1, 0,
                  &groups, &n_groups, &group_sizes, &total, &groups_cap);

    BBSNode **level = (BBSNode **)arena_alloc(a, sizeof(BBSNode *) * (size_t)n_groups);
    int64_t  level_n = n_groups;
    for (int64_t i = 0; i < n_groups; ++i) {
        BBSNode *leaf = node_new(a, /*is_leaf=*/1);
        node_reserve(a, leaf, (int32_t)group_sizes[i]);
        for (int64_t j = 0; j < group_sizes[i]; ++j) {
            BBSEntry *e = groups[i][j];
            e->node = leaf;
            leaf->entries[leaf->n_entries++] = e;
        }
        level[i] = leaf;
    }
    free(groups); free(group_sizes);

    /* Single-leaf-node case: that leaf is the root. */
    if (level_n == 1) return level[0];

    /* Step 2+: build upper levels until 1 node remains. */
    while (level_n > 1) {
        /* Materialize entries pointing into each child node's MBR. */
        BBSEntry **node_entries = (BBSEntry **)arena_alloc(
            a, sizeof(BBSEntry *) * (size_t)level_n);
        for (int64_t i = 0; i < level_n; ++i) {
            BBSEntry *e = (BBSEntry *)arena_alloc(a, sizeof(BBSEntry));
            e->mbr_min = mbr_pool + (*mbr_pool_cursor); *mbr_pool_cursor += n_dims;
            e->mbr_max = mbr_pool + (*mbr_pool_cursor); *mbr_pool_cursor += n_dims;
            node_compute_mbr(level[i], e->mbr_min, e->mbr_max, n_dims);
            e->child_node = level[i];
            e->leaf_idx = -1;
            e->node = NULL;     /* set below when packed into a parent */
            node_entries[i] = e;
        }

        /* STR-pack the new entry list into parent nodes. */
        BBSEntry ***pgroups = NULL;
        int64_t   *pgroup_sizes = NULL;
        int64_t    pn_groups = 0;
        int64_t    ptotal = 0;
        int64_t    pgroups_cap = 0;

        str_partition(node_entries, level_n, M, n_dims, key_map_1, 0,
                      &pgroups, &pn_groups, &pgroup_sizes,
                      &ptotal, &pgroups_cap);

        BBSNode **next_level = (BBSNode **)arena_alloc(
            a, sizeof(BBSNode *) * (size_t)pn_groups);
        for (int64_t i = 0; i < pn_groups; ++i) {
            BBSNode *parent = node_new(a, /*is_leaf=*/0);
            node_reserve(a, parent, (int32_t)pgroup_sizes[i]);
            for (int64_t j = 0; j < pgroup_sizes[i]; ++j) {
                BBSEntry *e = pgroups[i][j];
                e->node = parent;
                parent->entries[parent->n_entries++] = e;
            }
            next_level[i] = parent;
        }
        free(pgroups); free(pgroup_sizes);

        level = next_level;
        level_n = pn_groups;
    }

    return level[0];
}

/* ── max-heap on sum(max_dim) ────────────────────────────────────────────── */

/* Heap stores BBSEntry pointers. Pop order: largest sum(max_dim) first.
 * Mirrors /home/cc/Causal-Skylines bbs/rtree/entry.py:
 *   priority = sum(mbr.max_dim); __lt__: self.priority > other.priority
 * i.e. under Python's min-heap (heapq) the max-sum entry pops first.
 * Sum is over raw column order; keys_2 is irrelevant to pop priority. */

typedef struct {
    BBSEntry **buf;
    int64_t    n;
    int64_t    cap;
    int32_t    n_dims;
} Heap;

static int heap_lt(const Heap *h, const BBSEntry *a, const BBSEntry *b) {
    /* "a < b in heap order" iff a should pop first iff
     * sum(a.max_dim) > sum(b.max_dim)  (max-heap on scalar sum). */
    double sa = 0.0, sb = 0.0;
    for (int32_t d = 0; d < h->n_dims; ++d) {
        sa += a->mbr_max[d];
        sb += b->mbr_max[d];
    }
    return sa > sb;
}

static void heap_init(Heap *h, int32_t n_dims) {
    h->buf = NULL; h->n = 0; h->cap = 0; h->n_dims = n_dims;
}

static void heap_reserve(Heap *h, int64_t cap) {
    if (h->cap >= cap) return;
    int64_t new_cap = h->cap == 0 ? 64 : h->cap;
    while (new_cap < cap) new_cap *= 2;
    h->buf = (BBSEntry **)realloc(h->buf, sizeof(BBSEntry *) * (size_t)new_cap);
    h->cap = new_cap;
}

static void heap_swap(Heap *h, int64_t i, int64_t j) {
    BBSEntry *tmp = h->buf[i]; h->buf[i] = h->buf[j]; h->buf[j] = tmp;
}

static void heap_sift_up(Heap *h, int64_t i) {
    while (i > 0) {
        int64_t parent = (i - 1) >> 1;
        if (heap_lt(h, h->buf[i], h->buf[parent])) {
            heap_swap(h, i, parent);
            i = parent;
        } else break;
    }
}

static void heap_sift_down(Heap *h, int64_t i) {
    for (;;) {
        int64_t l = 2 * i + 1, r = 2 * i + 2, smallest = i;
        if (l < h->n && heap_lt(h, h->buf[l], h->buf[smallest])) smallest = l;
        if (r < h->n && heap_lt(h, h->buf[r], h->buf[smallest])) smallest = r;
        if (smallest == i) break;
        heap_swap(h, i, smallest);
        i = smallest;
    }
}

static void heap_push(Heap *h, BBSEntry *e) {
    heap_reserve(h, h->n + 1);
    h->buf[h->n++] = e;
    heap_sift_up(h, h->n - 1);
}

static BBSEntry *heap_pop(Heap *h) {
    BBSEntry *top = h->buf[0];
    h->n -= 1;
    if (h->n > 0) {
        h->buf[0] = h->buf[h->n];
        heap_sift_down(h, 0);
    }
    return top;
}

static void heap_free(Heap *h) {
    free(h->buf); h->buf = NULL; h->n = 0; h->cap = 0;
}

/* ── BBS run ────────────────────────────────────────────────────────────── */

static int32_t default_M(int32_t n_dims) {
    /* Mirror `M = 4096 // (16 * n_dims + 8)` from bbs.py. */
    int32_t M = 4096 / (16 * n_dims + 8);
    return M < 2 ? 2 : M;
}

/* ── stable lexsort presort (BBS lexsort variant) ───────────────────────────
 *
 * Mirrors `np.lexsort(-data.T[::-1])` from the historical bbs/lexsort.py
 * form_rtree: descending across dims [0..n_dims-1], equal-key ties retain
 * input (row-index) order. Leaf MBRs are points (min == max), so compare on
 * mbr_min. Bottom-up mergesort = stable, matching np.lexsort byte-for-byte.
 * Sorts the leaf pointer array in place before STR bulk-load. */
static int bbs_lex_cmp_desc(const BBSEntry *a, const BBSEntry *b, int32_t n_dims) {
    for (int32_t d = 0; d < n_dims; ++d) {
        if (a->mbr_min[d] > b->mbr_min[d]) return -1;
        if (a->mbr_min[d] < b->mbr_min[d]) return 1;
    }
    return 0;
}

static void bbs_lex_merge(BBSEntry **src, BBSEntry **dst,
                          int64_t lo, int64_t mid, int64_t hi, int32_t n_dims) {
    int64_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        if (bbs_lex_cmp_desc(src[i], src[j], n_dims) <= 0) dst[k++] = src[i++];
        else                                               dst[k++] = src[j++];
    }
    while (i < mid) dst[k++] = src[i++];
    while (j < hi)  dst[k++] = src[j++];
}

static void bbs_lex_sort_leaves(BBSEntry **leaves, int64_t n_rows, int32_t n_dims) {
    if (n_rows <= 1) return;
    BBSEntry **buf = (BBSEntry **)malloc(sizeof(BBSEntry *) * (size_t)n_rows);
    if (!buf) return;
    BBSEntry **src = leaves, **dst = buf;
    for (int64_t width = 1; width < n_rows; width *= 2) {
        for (int64_t lo = 0; lo < n_rows; lo += width * 2) {
            int64_t mid = lo + width;     if (mid > n_rows) mid = n_rows;
            int64_t hi  = lo + width * 2; if (hi  > n_rows) hi  = n_rows;
            bbs_lex_merge(src, dst, lo, mid, hi, n_dims);
        }
        BBSEntry **t = src; src = dst; dst = t;
    }
    if (src != leaves) memcpy(leaves, src, sizeof(BBSEntry *) * (size_t)n_rows);
    free(buf);
}

#define BBS_RUN_DEFINE(T, SUFFIX)                                                       \
static void bbs_run_##SUFFIX(                                                           \
    const T *data, int64_t n_rows, int32_t n_dims,                                      \
    const int32_t *key_map_1, const int32_t *keys_2, int32_t M_in, int lex_sort,        \
    int reverse_scan,                                                                   \
    int dom_simd, int64_t *out_skyline, int64_t *out_vcheck_freq, BnlMetrics *m)        \
{                                                                                       \
    Arena arena; arena_init(&arena);                                                    \
    mbr_dom_fn dom = mbr_dom_select(dom_simd);                                          \
    double *dom_sa = (double *)malloc(sizeof(double) * (size_t)(n_dims > 0 ? n_dims : 1)); \
    double *dom_sb = (double *)malloc(sizeof(double) * (size_t)(n_dims > 0 ? n_dims : 1)); \
                                                                                        \
    int32_t M = M_in > 0 ? M_in : default_M(n_dims);                                    \
                                                                                        \
    /* Pool for MBR coords: n_rows leaf MBRs (point: min == max copies of data row)     \
     * + at most ~2*n_rows internal MBRs across all upper levels (geometric series).    \
     * 4*n_rows total upper bound; round up generously. */                              \
    int64_t mbr_capacity = (4 * n_rows + 16) * (int64_t)n_dims;                         \
    double *mbr_pool = (double *)calloc((size_t)mbr_capacity, sizeof(double));          \
    int64_t mbr_cursor = 0;                                                             \
                                                                                        \
    /* Build leaf entries — one per row, MBR == point. */                               \
    BBSEntry **leaves = (BBSEntry **)arena_alloc(&arena, sizeof(BBSEntry *) * (size_t)n_rows); \
    for (int64_t i = 0; i < n_rows; ++i) {                                              \
        BBSEntry *e = (BBSEntry *)arena_alloc(&arena, sizeof(BBSEntry));                \
        e->mbr_min = mbr_pool + mbr_cursor; mbr_cursor += n_dims;                       \
        e->mbr_max = mbr_pool + mbr_cursor; mbr_cursor += n_dims;                       \
        const T *row = data + (size_t)i * (size_t)n_dims;                               \
        for (int32_t d = 0; d < n_dims; ++d) {                                          \
            e->mbr_min[d] = (double)row[d];                                             \
            e->mbr_max[d] = (double)row[d];                                             \
        }                                                                               \
        e->child_node = NULL;                                                           \
        e->leaf_idx = i;                                                                \
        e->node = NULL;                                                                 \
        leaves[i] = e;                                                                  \
    }                                                                                   \
                                                                                        \
    /* lexsort variant: stable descending lex presort before STR bulk-load. */          \
    if (lex_sort) {                                                                     \
        int64_t _t_sort = bbs_now_ns();                                                 \
        bbs_lex_sort_leaves(leaves, n_rows, n_dims);                                     \
        m->sort_ns += bbs_now_ns() - _t_sort;                                           \
    }                                                                                   \
                                                                                        \
    /* Build R-tree by STR. STR = Sort-Tile-Recursive: the per-dim                      \
     * MBR-center qsort is an intrinsic sub-step of index construction (no              \
     * sort, no STR), so the whole bulk-load wall time is index_ns. Only the            \
     * lexsort variant's separate dataset presort above counts as sort_ns.              \
     * Recorded apart from the BBS heap traversal so the caller can subtract            \
     * it from the BBS run time. */                                                     \
    int64_t _t_str = bbs_now_ns();                                                      \
    BBSNode *root = bulk_load_str(&arena, leaves, n_rows, n_dims, M, key_map_1,         \
                                   mbr_pool, &mbr_cursor);                              \
    m->index_ns += bbs_now_ns() - _t_str;                                               \
                                                                                        \
    /* BBS traversal. */                                                                \
    Heap H; heap_init(&H, n_dims);                                                      \
    /* Push root's entries (matches Python `heap.push(rtree.root)`). */                 \
    for (int32_t i = 0; i < root->n_entries; ++i) heap_push(&H, root->entries[i]);      \
                                                                                        \
    /* Skyline survivors as entry pointers. */                                          \
    BBSEntry **sky = (BBSEntry **)malloc(sizeof(BBSEntry *) * (size_t)(n_rows > 0 ? n_rows : 1)); \
    int64_t n_sky = 0;                                                                  \
                                                                                        \
    while (H.n > 0) {                                                                   \
        BBSEntry *entry = heap_pop(&H);                                                 \
        int dominated = 0;                                                              \
        /* reverse_scan=1 mirrors Causal lexsort.py `for item in reversed(skyline)`; \
         * reverse_scan=0 mirrors Causal bbs.py `for item in skyline` (forward). */     \
        int64_t _s0 = reverse_scan ? n_sky - 1 : 0;                                     \
        int64_t _sd = reverse_scan ? -1 : 1;                                            \
        for (int64_t s = _s0, _sc = 0; _sc < n_sky; s += _sd, ++_sc) {                  \
            BBSEntry *item = sky[s];                                                    \
            m->comparisons += 1;                                                        \
            if (dom(item->mbr_min, entry->mbr_max,                                      \
                    n_dims, keys_2, m, out_vcheck_freq, dom_sa, dom_sb)) {              \
                dominated = 1;                                                          \
                break;                                                                  \
            }                                                                           \
        }                                                                               \
        if (!dominated) {                                                               \
            if (entry->node->is_leaf) {                                                 \
                sky[n_sky++] = entry;                                                   \
            } else {                                                                    \
                BBSNode *child = entry->child_node;                                     \
                for (int32_t i = 0; i < child->n_entries; ++i)                          \
                    heap_push(&H, child->entries[i]);                                   \
            }                                                                           \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    /* Materialize skyline indices. */                                                  \
    for (int64_t s = 0; s < n_sky; ++s) out_skyline[s] = sky[s]->leaf_idx;              \
    m->n_skyline = n_sky;                                                               \
    /* simd_chunks_compared was accumulated inside the SSE2/AVX2 dom kernels via        \
     * mbr_finish (ceil(scanned/lanes) per call). Scalar dom_simd path never touches    \
     * it, matching the BNL/SFS/SaLSa scalar convention. */                             \
                                                                                        \
    free(sky);                                                                          \
    free(dom_sa);                                                                       \
    free(dom_sb);                                                                       \
    heap_free(&H);                                                                      \
    free(mbr_pool);                                                                     \
    arena_free(&arena);                                                                 \
}

BBS_RUN_DEFINE(float,  f32)
BBS_RUN_DEFINE(double, f64)

/* kernel_mode SIMD family -> dom-check variant, CPU-gated with graceful
 * downgrade (AVX2 -> SSE2 -> scalar). MBR coords are always double, so the
 * dom-check width is independent of the f32/f64 data dtype.
 *   3,6 = scalar   4,7 = AVX2   5,8 = SSE2  (matches _MODE_CHUNK_SIZES). */
static int bbs_dom_simd_for_mode(int32_t kernel_mode) {
    if (kernel_mode == 4 || kernel_mode == 7)
        return bbs_cpu_has_avx2() ? 2 : (bbs_cpu_has_sse2() ? 1 : 0);
    if (kernel_mode == 5 || kernel_mode == 8)
        return bbs_cpu_has_sse2() ? 1 : 0;
    return 0;
}

BnlMetrics skyline_bbs_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    const int32_t *key_map_1,
    const int32_t *keys_2,
    int32_t M,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0};
    int ds = bbs_dom_simd_for_mode(kernel_mode);
    /* kernel_mode 3..5 = f32 data; 6..8 = f64 data. */
    if (kernel_mode >= 3 && kernel_mode <= 5) {
        bbs_run_f32((const float *)data, n_rows, n_dims, key_map_1, keys_2, M, 0,
                    /*reverse_scan=*/0, ds, out_skyline, out_vcheck_freq, &m);
    } else {
        bbs_run_f64((const double *)data, n_rows, n_dims, key_map_1, keys_2, M, 0,
                    /*reverse_scan=*/0, ds, out_skyline, out_vcheck_freq, &m);
    }
    return m;
}

BnlMetrics skyline_bbs_lex_run(
    const void *data,
    int64_t n_rows,
    int32_t n_dims,
    int32_t kernel_mode,
    const int32_t *key_map_1,
    const int32_t *keys_2,
    int32_t M,
    int64_t *out_skyline,
    int64_t *out_vcheck_freq)
{
    BnlMetrics m = {0, 0, 0, 0};
    int ds = bbs_dom_simd_for_mode(kernel_mode);
    if (kernel_mode >= 3 && kernel_mode <= 5) {
        bbs_run_f32((const float *)data, n_rows, n_dims, key_map_1, keys_2, M, 1,
                    /*reverse_scan=*/1, ds, out_skyline, out_vcheck_freq, &m);
    } else {
        bbs_run_f64((const double *)data, n_rows, n_dims, key_map_1, keys_2, M, 1,
                    /*reverse_scan=*/1, ds, out_skyline, out_vcheck_freq, &m);
    }
    return m;
}
