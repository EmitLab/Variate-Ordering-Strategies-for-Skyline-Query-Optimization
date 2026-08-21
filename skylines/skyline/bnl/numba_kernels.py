"""BNL backend dispatcher (C-only).

Thin ctypes wrapper around the batched-C BNL entries:
  * `skyline_bnl_run`     — outer BNL loop with optional reverse-scan flag.
  * `skyline_bnl_lex_run` — fused lex-permutation + scan.

Filename retained for backwards compatibility with algo imports.
"""
import numpy as np

from skylines.skyline.c_kernels_loader import BnlMetrics, _load_library


def _coerce(data, mode):
    expected_f32 = mode in (3, 4, 5)
    target_dtype = np.float32 if expected_f32 else np.float64
    if data.dtype != target_dtype or not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data, dtype=target_dtype)
    return data


def _bnl_c(data, mode, reverse_scan):
    lib = _load_library()
    n_rows, n_dims = data.shape
    data = _coerce(data, mode)

    out_skyline = np.empty(n_rows, dtype=np.int64)
    out_vcheck = np.zeros(n_dims, dtype=np.int64)

    metrics: BnlMetrics = lib.skyline_bnl_run(
        data.ctypes.data,
        np.int64(n_rows),
        np.int32(n_dims),
        np.int32(mode),
        np.int32(reverse_scan),
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


def bnl_positions(data, mode):
    return _bnl_c(data, mode, reverse_scan=0)


def bnl_lexsort_positions(data, mode):
    return _bnl_c(data, mode, reverse_scan=1)


def bnl_lex_positions(data, mode, key_order=None, reverse_scan=1):
    """Run BNL with internal lex permutation fused inside `skyline_bnl_lex_run`.
    `data` is the raw unsorted column-sliced array; surviving indices come back
    in that array's original row index space.

    `key_order` (optional int32 sequence) overrides the default forward lex key
    order, letting callers decouple lex sort priority from the data column
    layout used by the dominance scan.

    `reverse_scan` selects the candidate-window scan direction: 1 = reverse
    (newest→oldest skyline buffer entry; default, preserves prior behavior),
    0 = forward (oldest→newest)."""
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

    metrics: BnlMetrics = lib.skyline_bnl_lex_run(
        data.ctypes.data,
        np.int64(n_rows),
        np.int32(n_dims),
        np.int32(mode),
        key_ptr,
        np.int32(n_keys),
        np.int32(reverse_scan),
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
