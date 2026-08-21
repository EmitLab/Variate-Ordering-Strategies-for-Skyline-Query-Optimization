from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from threading import Lock

import numpy as np


_SOURCE_DIR = Path(__file__).resolve().parent
_LIB_PATH = _SOURCE_DIR / "libcskyline.so"
_BUILD_INPUTS = (
    _SOURCE_DIR / "c_kernels.c",
    _SOURCE_DIR / "c_kernels.h",
    _SOURCE_DIR / "Makefile",
)

_LOAD_LOCK = Lock()
_LIB = None


class DomResult(ctypes.Structure):
    _fields_ = [
        ("dominating", ctypes.c_int32),
        ("fail_idx", ctypes.c_int32),
    ]


class BnlMetrics(ctypes.Structure):
    _fields_ = [
        ("n_skyline", ctypes.c_int64),
        ("comparisons", ctypes.c_int64),
        ("sub_comparisons", ctypes.c_int64),
        ("simd_chunks_compared", ctypes.c_int64),
        ("normalize_ns", ctypes.c_int64),
        ("sort_ns", ctypes.c_int64),
        ("index_ns", ctypes.c_int64),
    ]


SIMD_MODES = {"auto", "sse", "avx"}


def _needs_rebuild() -> bool:
    if not _LIB_PATH.exists():
        return True

    lib_mtime = _LIB_PATH.stat().st_mtime
    return any(path.stat().st_mtime > lib_mtime for path in _BUILD_INPUTS)


def _build_library() -> None:
    result = subprocess.run(
        ["make", "libcskyline.so"],
        cwd=_SOURCE_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to build C dominance kernels:\n"
            f"{result.stdout}{result.stderr}"
        )


def _load_library():
    global _LIB

    if _LIB is not None:
        return _LIB

    with _LOAD_LOCK:
        if _LIB is None:
            if _needs_rebuild():
                _build_library()

            lib = ctypes.CDLL(str(_LIB_PATH))
            lib.skyline_cpu_supports_sse2.argtypes = []
            lib.skyline_cpu_supports_sse2.restype = ctypes.c_int32
            lib.skyline_cpu_supports_avx2.argtypes = []
            lib.skyline_cpu_supports_avx2.restype = ctypes.c_int32

            _ptr_args = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32]
            for name in (
                "skyline_scalar_f32_nb", "skyline_sse2_f32_nb",
                "skyline_avx2_f32_nb", "skyline_simd_f32_nb",
                "skyline_scalar_f64_nb", "skyline_sse2_f64_nb",
                "skyline_avx2_f64_nb", "skyline_simd_f64_nb",
            ):
                fn = getattr(lib, name)
                fn.argtypes = _ptr_args
                fn.restype = ctypes.c_int32

            lib.skyline_bnl_run.restype = BnlMetrics
            lib.skyline_bnl_run.argtypes = [
                ctypes.c_void_p,   # data
                ctypes.c_int64,    # n_rows
                ctypes.c_int32,    # n_dims
                ctypes.c_int32,    # kernel_mode (3..8)
                ctypes.c_int32,    # reverse_scan
                ctypes.c_void_p,   # out_skyline
                ctypes.c_void_p,   # out_vcheck_freq
            ]

            lib.skyline_sfs_run.restype = BnlMetrics
            lib.skyline_sfs_run.argtypes = [
                ctypes.c_void_p,   # data
                ctypes.c_int64,    # n_rows
                ctypes.c_int32,    # n_dims
                ctypes.c_int32,    # kernel_mode
                ctypes.c_int32,    # reverse_scan
                ctypes.c_void_p,   # out_skyline
                ctypes.c_void_p,   # out_vcheck_freq
            ]

            lib.skyline_dnc_trivial_run.restype = BnlMetrics
            lib.skyline_dnc_trivial_run.argtypes = [
                ctypes.c_void_p,   # data
                ctypes.c_void_p,   # index (int64*)
                ctypes.c_int64,    # n_index
                ctypes.c_int32,    # n_dims
                ctypes.c_int32,    # kernel_mode
                ctypes.c_void_p,   # out_skyline
                ctypes.c_void_p,   # out_vcheck_freq
            ]

            lib.skyline_dnc_run.restype = BnlMetrics
            lib.skyline_dnc_run.argtypes = [
                ctypes.c_void_p,   # data
                ctypes.c_int64,    # n_rows
                ctypes.c_int32,    # n_dims
                ctypes.c_int32,    # kernel_mode
                ctypes.c_void_p,   # out_skyline
                ctypes.c_void_p,   # out_vcheck_freq
            ]

            lib.skyline_bbs_run.restype = BnlMetrics
            lib.skyline_bbs_run.argtypes = [
                ctypes.c_void_p,   # data
                ctypes.c_int64,    # n_rows
                ctypes.c_int32,    # n_dims
                ctypes.c_int32,    # kernel_mode
                ctypes.c_void_p,   # key_map_1 (int32*)
                ctypes.c_void_p,   # keys_2 (int32*)
                ctypes.c_int32,    # M (0 = default)
                ctypes.c_void_p,   # out_skyline
                ctypes.c_void_p,   # out_vcheck_freq
            ]

            lib.skyline_bbs_lex_run.restype = BnlMetrics
            lib.skyline_bbs_lex_run.argtypes = [
                ctypes.c_void_p,   # data
                ctypes.c_int64,    # n_rows
                ctypes.c_int32,    # n_dims
                ctypes.c_int32,    # kernel_mode
                ctypes.c_void_p,   # key_map_1 (int32*)
                ctypes.c_void_p,   # keys_2 (int32*)
                ctypes.c_int32,    # M (0 = default)
                ctypes.c_void_p,   # out_skyline
                ctypes.c_void_p,   # out_vcheck_freq
            ]

            lib.skyline_salsa_run.restype = BnlMetrics
            lib.skyline_salsa_run.argtypes = [
                ctypes.c_void_p,   # data
                ctypes.c_int64,    # n_rows
                ctypes.c_int32,    # n_dims
                ctypes.c_void_p,   # monotonic_scores
                ctypes.c_int32,    # kernel_mode
                ctypes.c_void_p,   # out_skyline
                ctypes.c_void_p,   # out_vcheck_freq
            ]

            lib.skyline_salsa_lexsort_run.restype = BnlMetrics
            lib.skyline_salsa_lexsort_run.argtypes = [
                ctypes.c_void_p,   # data
                ctypes.c_int64,    # n_rows
                ctypes.c_int32,    # n_dims
                ctypes.c_int32,    # kernel_mode
                ctypes.c_void_p,   # out_skyline
                ctypes.c_void_p,   # out_vcheck_freq
            ]

            _lex_run_args = [
                ctypes.c_void_p,   # data (unsorted)
                ctypes.c_int64,    # n_rows
                ctypes.c_int32,    # n_dims
                ctypes.c_int32,    # kernel_mode
                ctypes.c_void_p,   # key_order (int32*, NULL = default)
                ctypes.c_int32,    # n_keys (ignored if key_order is NULL)
                ctypes.c_void_p,   # out_skyline
                ctypes.c_void_p,   # out_vcheck_freq
            ]
            for name in (
                "skyline_dnc_lex_run",
                "skyline_salsa_lex_run",
            ):
                fn = getattr(lib, name)
                fn.restype = BnlMetrics
                fn.argtypes = _lex_run_args

            # SFS/SaLSa monosort_run fold the monotone presort into C; they
            # take no key_order/n_keys (unlike *_lex_run) but do take a
            # monotonic_fn selector (0 = sum, 1 = max_sum). Signature:
            # c_kernels.h (skyline_sfs/salsa_monosort_run).
            _monosort_run_args = [
                ctypes.c_void_p,   # data
                ctypes.c_int64,    # n_rows
                ctypes.c_int32,    # n_dims
                ctypes.c_int32,    # kernel_mode
                ctypes.c_int32,    # monotonic_fn (0 = sum, 1 = max_sum)
                ctypes.c_void_p,   # out_skyline
                ctypes.c_void_p,   # out_vcheck_freq
            ]
            for name in ("skyline_sfs_monosort_run", "skyline_salsa_monosort_run"):
                fn = getattr(lib, name)
                fn.restype = BnlMetrics
                fn.argtypes = _monosort_run_args

            # BNL/SFS lex_run additionally take a reverse_scan flag (candidate
            # window scan direction) after n_keys; dnc/salsa/monosort do not.
            _lex_run_args_rev = [
                ctypes.c_void_p,   # data (unsorted)
                ctypes.c_int64,    # n_rows
                ctypes.c_int32,    # n_dims
                ctypes.c_int32,    # kernel_mode
                ctypes.c_void_p,   # key_order (int32*, NULL = default)
                ctypes.c_int32,    # n_keys (ignored if key_order is NULL)
                ctypes.c_int32,    # reverse_scan (0 = forward, 1 = reverse)
                ctypes.c_void_p,   # out_skyline
                ctypes.c_void_p,   # out_vcheck_freq
            ]
            for name in ("skyline_bnl_lex_run", "skyline_sfs_lex_run"):
                fn = getattr(lib, name)
                fn.restype = BnlMetrics
                fn.argtypes = _lex_run_args_rev

            _LIB = lib

    return _LIB


def c_backend_available() -> bool:
    try:
        _load_library()
    except Exception:
        return False
    return True


def _normalize_simd_mode(mode: str) -> str:
    value = mode.lower()
    if value not in SIMD_MODES:
        raise ValueError(f"invalid c SIMD mode {mode!r}; expected one of {sorted(SIMD_MODES)}")
    return value


def c_supported_simd_modes() -> set[str]:
    lib = _load_library()
    modes = {"auto"}
    if lib.skyline_cpu_supports_sse2():
        modes.add("sse")
    if lib.skyline_cpu_supports_avx2():
        modes.add("avx")
    return modes
