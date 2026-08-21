from typing import override

import numpy as np

from skylines.dataset import Dataset
from skylines.skyline import Skyline
from skylines.skyline.dnc.numba_kernels import dnc_lex_positions


class DnCSkylineLexsort(Skyline):

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    @override
    def find_skyline(self, index: np.ndarray, *args) -> np.ndarray:
        data = self.data[np.ix_(index, list(self.preferences.values())[::-1])]
        data = np.ascontiguousarray(data)
        # Lex permutation runs inside the C kernel; sort_ns is the C-measured
        # presort wall time (D&C does not normalize → normalize_ns 0).
        skyline, comp, sub, simd_chunks, vcheck_freq, normalize_ns, sort_ns = dnc_lex_positions(
            data, self._bnl_numba_mode()
        )
        self.normalize_time = normalize_ns / 1e9
        self.sort_time = sort_ns / 1e9
        self._apply_numba_metrics(comp, sub, simd_chunks, vcheck_freq)
        return index[skyline]
