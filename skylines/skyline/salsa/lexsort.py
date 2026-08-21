from typing import override

import numpy as np

from skylines.dataset import *
from skylines.skyline import Skyline
from skylines.skyline.salsa.numba_kernels import salsa_lex_positions


class SaLSaSkylineLexsort(Skyline):

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    @override
    def find_skyline(self, index: np.ndarray, *args) -> np.ndarray:
        data = np.ascontiguousarray(self.data[np.ix_(index, list(self.preferences.values()))])

        # Per-column min-max normalize + lex permutation now run inside the C
        # kernel; normalize_ns / sort_ns are the C-measured per-phase times.
        # `key_order` maps dataset.lex_preference (the NCF lex/sort order) onto
        # the v-check column positions, so the row-sort priority decouples from
        # the v-check column layout. The dominance scan walks the v-check
        # column order; the SaLSa pstop loop walks reverse `key_order` priority
        # via an index indirection so its early-termination stays sound under
        # arbitrary NCF orders. None ⇒ matched-VO.
        key_order = self._lex_key_order(list(self.preferences.keys()))
        positions, comp, sub, simd_chunks, vcheck_freq, normalize_ns, sort_ns = salsa_lex_positions(
            data, self._bnl_numba_mode(), key_order=key_order
        )
        self.normalize_time = normalize_ns / 1e9
        self.sort_time = sort_ns / 1e9
        self._apply_numba_metrics(comp, sub, simd_chunks, vcheck_freq)
        return index[positions]
