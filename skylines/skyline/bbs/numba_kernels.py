"""BBS backend dispatcher (C-only).

Thin ctypes wrapper around `skyline_bbs_run`.

Caller passes already-shaped data plus the two int32 key arrays:
  key_map_1 : STR bulk-load dim ordering (controls heap priority via leaf packing)
  keys_2    : MBR-dominance iteration order (controls vcheck_freq accumulation)

`M` is the R-tree fan-out; pass 0 to use the default `4096 / (16*n_dims + 8)`.
"""
import numpy as np

from skylines.skyline.c_kernels_loader import BnlMetrics, _load_library


def _coerce(data, mode):
    expected_f32 = mode in (3, 4, 5)
    target_dtype = np.float32 if expected_f32 else np.float64
    if data.dtype != target_dtype or not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data, dtype=target_dtype)
    return data


def bbs_positions(data, mode, key_map_1, keys_2, M=0, lex_sort=False):
    """Run full BBS skyline on `data` (caller-preprocessed).

    lex_sort=True dispatches the lexsort variant: a stable
    np.lexsort(-data.T[::-1]) presort of the leaf entries inside the C
    kernel before STR bulk-load.

    Timing metrics from the C kernel: `index_ns` is the full STR bulk-load
    wall time — STR is Sort-Tile-Recursive, so its per-dim MBR-center qsort
    is an intrinsic part of index construction and is counted here.
    `sort_ns` is the lex presort time when lex_sort=True, else 0.
    `normalize_ns` is 0 (BBS does not normalize).
    """
    lib = _load_library()
    n_rows, n_dims = data.shape
    data = _coerce(data, mode)

    if isinstance(key_map_1, dict):
        km1 = np.array([int(key_map_1[i]) for i in range(n_dims)], dtype=np.int32)
    else:
        km1 = np.ascontiguousarray(key_map_1, dtype=np.int32)
    k2 = np.ascontiguousarray(keys_2, dtype=np.int32)

    out_skyline = np.empty(n_rows, dtype=np.int64)
    out_vcheck  = np.zeros(n_dims, dtype=np.int64)

    bbs_run = lib.skyline_bbs_lex_run if lex_sort else lib.skyline_bbs_run
    metrics: BnlMetrics = bbs_run(
        data.ctypes.data,
        np.int64(n_rows),
        np.int32(n_dims),
        np.int32(mode),
        km1.ctypes.data,
        k2.ctypes.data,
        np.int32(M),
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
        metrics.index_ns,
    )
