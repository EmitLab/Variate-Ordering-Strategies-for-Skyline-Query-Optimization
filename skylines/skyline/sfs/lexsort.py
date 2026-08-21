from typing import override

import numpy as np

import skylines
from skylines.dataset import *
from skylines.skyline import Skyline
from skylines.skyline.sfs.numba_kernels import sfs_lex_positions


class SFSSkylineLexsort(Skyline):

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    @override
    def find_skyline(self, index: np.ndarray, *args) -> np.ndarray:
        data = np.ascontiguousarray(self.data[np.ix_(index, list(self.preferences.values()))])

        # Per-column min-max normalize + lex permutation now run inside the C
        # kernel; normalize_ns / sort_ns are the C-measured per-phase times.
        # `key_order` maps dataset.lex_preference (the NCF lex/sort order) onto
        # the v-check column positions, so the row-sort priority decouples from
        # the v-check column layout. None ⇒ matched-VO.
        key_order = self._lex_key_order(list(self.preferences.keys()))
        reverse_scan = 1 if getattr(skylines, "lex_reverse_scan", True) else 0
        positions, comp, sub, simd_chunks, vcheck_freq, normalize_ns, sort_ns = sfs_lex_positions(
            data, self._bnl_numba_mode(), key_order=key_order, reverse_scan=reverse_scan
        )
        self.normalize_time = normalize_ns / 1e9
        self.sort_time = sort_ns / 1e9
        self._apply_numba_metrics(comp, sub, simd_chunks, vcheck_freq)
        return index[positions]
