"""DnC backend dispatcher (C-only).

Thin ctypes wrappers around the batched-C D&C entries:
  * `skyline_dnc_run`         — full recursion + merge.
  * `skyline_dnc_lex_run`     — fused lex-permutation + recursion.
  * `skyline_dnc_trivial_run` — direct trivial-leaf entry (kernel-level parity).

Caller is responsible for any preprocessing (preference column reversal,
optional lexsort presort) before invoking these dispatchers.
"""
import numpy as np

from skylines.skyline.c_kernels_loader import BnlMetrics, _load_library


def _coerce(data, mode):
    expected_f32 = mode in (3, 4, 5)
    target_dtype = np.float32 if expected_f32 else np.float64
    if data.dtype != target_dtype or not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data, dtype=target_dtype)
    return data


# ── Full D&C (recursion + merge) ────────────────────────────────────────────


def dnc_positions(data, mode):
    """Run full D&C skyline on `data` (caller-preprocessed)."""
    lib = _load_library()
    n_rows, n_dims = data.shape
    data = _coerce(data, mode)

    out_skyline = np.empty(n_rows, dtype=np.int64)
    out_vcheck = np.zeros(n_dims, dtype=np.int64)

    metrics: BnlMetrics = lib.skyline_dnc_run(
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


def dnc_lex_positions(data, mode, key_order=None):
    """D&C with internal lex permutation fused inside `skyline_dnc_lex_run`.
    `data` is the reverse-pref column slice (unsorted).

    `key_order` (optional int32 sequence) overrides the default reverse lex key
    order. Indices in `key_order` reference columns of `data` (the reversed-pref
    space), so callers must apply the reversal when mapping preference names to
    column positions."""
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

    metrics: BnlMetrics = lib.skyline_dnc_lex_run(
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


# ── Trivial-leaf direct entry (kept for kernel-level parity tests) ──────────


def dnc_trivial_positions(data, index, mode):
    lib = _load_library()
    _, n_dims = data.shape
    data = _coerce(data, mode)

    if index.dtype != np.int64 or not index.flags["C_CONTIGUOUS"]:
        index = np.ascontiguousarray(index, dtype=np.int64)

    n_index = index.shape[0]
    out_skyline = np.empty(n_index, dtype=np.int64)
    out_vcheck = np.zeros(n_dims, dtype=np.int64)

    metrics: BnlMetrics = lib.skyline_dnc_trivial_run(
        data.ctypes.data,
        index.ctypes.data,
        np.int64(n_index),
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
