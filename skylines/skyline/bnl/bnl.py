from typing import override

import numpy as np

from skylines.dataset import *
from skylines.skyline import Skyline
from skylines.skyline.bnl.numba_kernels import bnl_positions


class BNLSkyline(Skyline):

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    @override
    def find_skyline(self, index: np.ndarray, *args) -> np.ndarray:
        data = np.ascontiguousarray(self.data[np.ix_(index, list(self.preferences.values()))])
        skyline_positions, comparisons, sub_comparisons, simd_chunks, vcheck_freq = bnl_positions(
            data,
            self._bnl_numba_mode(),
        )
        self._apply_numba_metrics(comparisons, sub_comparisons, simd_chunks, vcheck_freq)
        return index[skyline_positions]
