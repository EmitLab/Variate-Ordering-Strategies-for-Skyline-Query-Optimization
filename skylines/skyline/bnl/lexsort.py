from typing import override

import numpy as np

import skylines
from skylines.dataset import *
from skylines.skyline import Skyline
from skylines.skyline.bnl.numba_kernels import bnl_lex_positions


class BNLSkylineLexsort(Skyline):

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    @override
    def find_skyline(self, index: np.ndarray, *args) -> np.ndarray:
        data = np.ascontiguousarray(self.data[np.ix_(index, list(self.preferences.values()))])
        # Lex permutation runs inside the C kernel; sort_ns is the C-measured
        # presort wall time (BNL does not normalize → normalize_ns 0).
        reverse_scan = 1 if getattr(skylines, "lex_reverse_scan", True) else 0
        positions, comp, sub, simd_chunks, vcheck_freq, normalize_ns, sort_ns = bnl_lex_positions(
            data, self._bnl_numba_mode(), reverse_scan=reverse_scan
        )
        self.normalize_time = normalize_ns / 1e9
        self.sort_time = sort_ns / 1e9
        self._apply_numba_metrics(comp, sub, simd_chunks, vcheck_freq)
        return index[positions]
