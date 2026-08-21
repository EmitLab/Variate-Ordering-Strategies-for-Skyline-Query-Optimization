from typing import override

import numpy as np

from skylines.dataset import *
from skylines.skyline import Skyline
from skylines.skyline.sfs.numba_kernels import sfs_monosort_positions


# Monotone presort function -> C kernel `monotonic_fn` selector.
#   'sum'     -> descending row-sum
#   'max_sum' -> descending (max, sum) lexicographic
_MONOTONIC_CODES = {'sum': 0, 'max_sum': 1}


class SFSSkyline(Skyline):

    def __init__(self, dataset: Dataset, monotonic_fn: str = 'sum'):
        super().__init__(dataset)
        if monotonic_fn not in _MONOTONIC_CODES:
            raise ValueError(
                f"invalid monotonic_fn={monotonic_fn!r}; expected 'sum' | 'max_sum'")
        self.monotonic_fn = monotonic_fn

    @override
    def find_skyline(self, index: np.ndarray, *args) -> np.ndarray:
        data = np.ascontiguousarray(self.data[np.ix_(index, list(self.preferences.values()))])

        # Per-column min-max normalize + monotone presort + permute run inside
        # the C kernel; `monotonic_fn` selects the presort key ('sum' descending
        # row-sum, or 'max_sum' descending (max, sum) lex). normalize_ns /
        # sort_ns are the C-measured per-phase wall times.
        skyline_positions, comparisons, sub_comparisons, simd_chunks, vcheck_freq, normalize_ns, sort_ns = sfs_monosort_positions(
            data,
            self._bnl_numba_mode(),
            _MONOTONIC_CODES[self.monotonic_fn],
        )
        self.normalize_time = normalize_ns / 1e9
        self.sort_time = sort_ns / 1e9
        self._apply_numba_metrics(comparisons, sub_comparisons, simd_chunks, vcheck_freq)
        return index[skyline_positions]
