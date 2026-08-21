"""SaLSa backend dispatcher (C-only)."""
import numpy as np

from skylines.skyline.c_kernels_loader import BnlMetrics, _load_library


def _coerce(data, mode):
    expected_f32 = mode in (3, 4, 5)
    target_dtype = np.float32 if expected_f32 else np.float64
    if data.dtype != target_dtype or not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data, dtype=target_dtype)
    return data


def salsa_positions(data, monotonic_scores, mode):
    lib = _load_library()
    n_rows, n_dims = data.shape
    data = _coerce(data, mode)

    target_dtype = data.dtype
    if monotonic_scores.dtype != target_dtype or not monotonic_scores.flags["C_CONTIGUOUS"]:
        monotonic_scores = np.ascontiguousarray(monotonic_scores, dtype=target_dtype)

    out_skyline = np.empty(n_rows, dtype=np.int64)
    out_vcheck = np.zeros(n_dims, dtype=np.int64)

    metrics: BnlMetrics = lib.skyline_salsa_run(
        data.ctypes.data,
        np.int64(n_rows),
        np.int32(n_dims),
        monotonic_scores.ctypes.data,
        np.int32(mode),
        out_skyline.ctypes.data,
        out_vcheck.ctypes.data,
    )

    return (
        out_skyline[: metrics.n_skyline],
        metrics.comparisons,
        metrics.sub_comparisons,
        metrics.simd_chunks_compared,
        out_vcheck,
    )


def salsa_lexsort_positions(data, mode):
    lib = _load_library()
    n_rows, n_dims = data.shape
    data = _coerce(data, mode)

    out_skyline = np.empty(n_rows, dtype=np.int64)
    out_vcheck = np.zeros(n_dims, dtype=np.int64)

    metrics: BnlMetrics = lib.skyline_salsa_lexsort_run(
        data.ctypes.data,
        np.int64(n_rows),
        np.int32(n_dims),
        np.int32(mode),
        out_skyline.ctypes.data,
        out_vcheck.ctypes.data,
    )

    return (
        out_skyline[: metrics.n_skyline],
        metrics.comparisons,
        metrics.sub_comparisons,
        metrics.simd_chunks_compared,
        out_vcheck,
    )


def salsa_lex_positions(data, mode, key_order=None):
    """SaLSa with internal lex permutation fused inside `skyline_salsa_lex_run`.
    `data` is the normalized unsorted array.

    The lex sort by `key_order` is the only permutation: rows are reordered so
    the columns listed in `key_order` are monotone in priority sequence, but
    the column layout is unchanged. The dominance scan uses the regular all-
    dimensions kernel; the SaLSa pstop loop walks columns in reverse
    `key_order` priority via an index indirection (no data permutation), which
    keeps the early-termination sound under arbitrary NCF orders.

    `key_order` (optional int32 sequence) overrides the default forward lex key
    order, letting callers decouple lex sort priority from the data column
    layout used by the dominance scan."""
    lib = _load_library()
    n_rows, n_dims = data.shape
    data = _coerce(data, mode)

    out_skyline = np.empty(n_rows, dtype=np.int64)
    out_vcheck = np.zeros(n_dims, dtype=np.int64)

    if key_order is None:
        key_ptr = 0
        n_keys = 0
    else:
        key_arr = np.ascontiguousarray(key_order, dtype=np.int32)
        key_ptr = key_arr.ctypes.data
        n_keys = key_arr.shape[0]

    metrics: BnlMetrics = lib.skyline_salsa_lex_run(
        data.ctypes.data,
        np.int64(n_rows),
        np.int32(n_dims),
        np.int32(mode),
        key_ptr,
        np.int32(n_keys),
        out_skyline.ctypes.data,
        out_vcheck.ctypes.data,
    )

    return (
        out_skyline[: metrics.n_skyline],
        metrics.comparisons,
        metrics.sub_comparisons,
        metrics.simd_chunks_compared,
        out_vcheck,
        metrics.normalize_ns,
        metrics.sort_ns,
    )


def salsa_monosort_positions(data, mode, monotonic_fn=1):
    """SaLSa with the monotone presort fused inside `skyline_salsa_monosort_run`.
    `data` is the normalized unsorted array; the monotonic_scores are rebuilt
    internally in C for the pstop check. `monotonic_fn` selects the presort
    function: 0 = descending row-sum, 1 = descending (max, sum) lex. Also
    returns the C-measured presort time in nanoseconds (`sort_ns`)."""
    lib = _load_library()
    n_rows, n_dims = data.shape
    data = _coerce(data, mode)

    out_skyline = np.empty(n_rows, dtype=np.int64)
    out_vcheck = np.zeros(n_dims, dtype=np.int64)

    metrics: BnlMetrics = lib.skyline_salsa_monosort_run(
        data.ctypes.data,
        np.int64(n_rows),
        np.int32(n_dims),
        np.int32(mode),
        np.int32(monotonic_fn),
        out_skyline.ctypes.data,
        out_vcheck.ctypes.data,
    )

    return (
        out_skyline[: metrics.n_skyline],
        metrics.comparisons,
        metrics.sub_comparisons,
        metrics.simd_chunks_compared,
        out_vcheck,
        metrics.normalize_ns,
        metrics.sort_ns,
    )
