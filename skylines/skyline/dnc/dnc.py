from typing import override

import numpy as np

from skylines.dataset import Dataset
from skylines.skyline import Skyline
from skylines.skyline.dnc.numba_kernels import dnc_positions


class DnCSkyline(Skyline):

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    @override
    def find_skyline(self, index: np.ndarray, *args) -> np.ndarray:
        data = self.data[np.ix_(index, list(self.preferences.values())[::-1])]
        skyline, comparisons, sub_comparisons, simd_chunks, vcheck_freq = dnc_positions(
            data, self._bnl_numba_mode()
        )
        self._apply_numba_metrics(comparisons, sub_comparisons, simd_chunks, vcheck_freq)
        return index[skyline]
