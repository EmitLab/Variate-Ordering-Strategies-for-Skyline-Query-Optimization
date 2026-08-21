from typing import override

import numpy as np

from skylines.dataset import Dataset
from skylines.skyline import Skyline
from skylines.skyline.bbs.numba_kernels import bbs_positions


class BBSSkyline(Skyline):

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

        # lexsort variant flag (set by BBSSkylineLexsort): stable
        # np.lexsort(-data.T[::-1]) leaf presort inside the C kernel.
        self._lex_sort = False

        n_dims = len(self.preferences)

        # Match /home/cc/Causal-Skylines/skylines/skyline/bbs: vos column
        # order, no key remapping. key_map_1/keys_2 kept as identity so the
        # C kernel's STR sort + MBR-dominance scan run in raw dim order
        # (= Causal's `range(n_dims)` / un-remapped STR). The C public ABI
        # is unchanged; identity maps are the no-op configuration.
        self._data_cols = list(self.preferences.values())
        self.key_map_1 = {i: i for i in range(n_dims)}
        self.keys_2 = list(range(n_dims))

    @override
    def find_skyline(self, index: np.ndarray, *args) -> np.ndarray:
        data = self.data[np.ix_(index, self._data_cols)]
        skyline, comparisons, sub_comparisons, simd_chunks, vcheck_freq, normalize_ns, sort_ns, index_ns = bbs_positions(
            data, self._bnl_numba_mode(), self.key_map_1, self.keys_2, M=0,
            lex_sort=self._lex_sort,
        )
        self.normalize_time = normalize_ns / 1e9
        # The STR bulk-load wall time is split into two phases, both recorded
        # separately from the BBS traversal run time:
        #   sort_time  = per-dim MBR-center qsort (+ lex presort in the
        #                lexsort variant).
        #   index_time = R-tree node-construction. Non-zero only for BBS —
        #                no other algorithm builds a spatial index.
        self.sort_time = sort_ns / 1e9
        self.index_time = index_ns / 1e9
        self._apply_numba_metrics(comparisons, sub_comparisons, simd_chunks, vcheck_freq)
        return index[skyline]
