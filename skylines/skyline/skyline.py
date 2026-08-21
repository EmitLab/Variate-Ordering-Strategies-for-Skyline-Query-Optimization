import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from skylines.common import utils
from skylines.common.index import Index
from skylines.common.utils import time_function
from skylines.dataset import Dataset
from skylines.dataset import Dominance
from skylines import simd, dtype


# Per-mode SIMD chunk width. Mirrors the kernel_mode encoding in
# skylines/skyline/c_kernels.h (modes 3..8).
_MODE_CHUNK_SIZES = {
    3: 1,  # C scalar f32
    4: 8,  # C AVX2 f32
    5: 4,  # C SSE2 f32
    6: 1,  # C scalar f64
    7: 4,  # C AVX2 f64
    8: 2,  # C SSE2 f64
}


class Skyline(ABC):

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.data, self.variates, self.preferences = dataset.to_numpy()

        self.comparisons = 0
        self.sub_comparisons = 0
        self.simd_chunks_compared = 0
        self._chunk_size: int | None = None

        self.cluster_time = 0
        self.compute_time = []
        self.merge_time = 0
        self.time_handled = False

        self.t_compute = 0.0
        self.sort_time = 0.0
        self.normalize_time = 0.0
        # Spatial-index build time. Non-zero only for BBS (STR R-tree
        # construction); 0.0 for every algorithm that builds no index.
        self.index_time = 0.0

        self.vcheck_freq = [0] * len(self.preferences)

    @property
    def total_time(self):
        return self.cluster_time + sum(self.compute_time) + self.merge_time

    @property
    def parallel_time(self):
        return self.cluster_time + max(self.compute_time) + self.merge_time

    @property
    def t_python(self):
        return max(self.t_compute - self.total_time, 0.0)

    # ------------------------------------------------------------------
    # Phase 1 derived metrics — software-derived from the core counters.
    # All safe on zero denominators; see docs/ISSUES/1/plan.md.
    # ------------------------------------------------------------------

    @property
    def chunk_size(self) -> int:
        if self._chunk_size is not None:
            return self._chunk_size
        return _MODE_CHUNK_SIZES[self._bnl_numba_mode()]

    @property
    def lane_utilization(self) -> float:
        denom = self.simd_chunks_compared * self.chunk_size
        return self.sub_comparisons / denom if denom > 0 else 0.0

    @property
    def early_exit_rate(self) -> float:
        n_dims = len(self.preferences)
        denom = self.comparisons * n_dims
        return 1.0 - (self.sub_comparisons / denom) if denom > 0 else 0.0

    @property
    def scalar_tail_fraction(self) -> float:
        if self.sub_comparisons <= 0:
            return 0.0
        tail = self.sub_comparisons - self.simd_chunks_compared * self.chunk_size
        return max(tail / self.sub_comparisons, 0.0)

    @property
    def simd_amortization(self) -> float:
        return self.simd_chunks_compared / self.comparisons if self.comparisons > 0 else 0.0

    @property
    def dom_check_throughput(self) -> float:
        return self.comparisons / self.total_time if self.total_time > 0 else 0.0

    @property
    def sub_compare_throughput(self) -> float:
        return self.sub_comparisons / self.total_time if self.total_time > 0 else 0.0

    @property
    def simd_chunk_throughput(self) -> float:
        return self.simd_chunks_compared / self.total_time if self.total_time > 0 else 0.0

    def _bnl_numba_mode(self) -> int:
        """Resolve the C kernel_mode (3..8) for the configured simd width + dtype."""
        dtype_offset = 3 if self.data.dtype == np.float64 else 0
        if simd == 'scalar':
            return 3 + dtype_offset
        if simd == 'sse':
            return 5 + dtype_offset
        if simd == 'avx':
            return 4 + dtype_offset
        raise ValueError(f"invalid simd={simd!r}; expected 'scalar' | 'sse' | 'avx'")

    def _lex_key_order(self, data_column_names: list[str]) -> list[int] | None:
        """Map dataset.lex_preference to column positions in the data layout the
        kernel sees, so the lex permutation can use a different priority than
        the v-check dominance scan. `data_column_names` lists preference names
        in the order they appear as columns in the kernel's input array (the
        v-check VO order). Returns None when dataset.lex_preference is unset,
        leaving the kernel on its default forward key order (matched-VO)."""
        lex_pref = getattr(self.dataset, 'lex_preference', None)
        if lex_pref is None:
            return None
        name_to_pos = {name: i for i, name in enumerate(data_column_names)}
        return [name_to_pos[name] for name in lex_pref]

    def _apply_numba_metrics(
        self,
        comparisons: int,
        sub_comparisons: int,
        simd_chunks_compared: int,
        vcheck_freq: np.ndarray,
    ) -> None:
        """Stamp metric counters from a single batched-C call (BNL/SFS/SaLSa)."""
        self.comparisons = int(comparisons)
        self.sub_comparisons = int(sub_comparisons)
        self.simd_chunks_compared = int(simd_chunks_compared)
        self._chunk_size = _MODE_CHUNK_SIZES[self._bnl_numba_mode()]
        self.vcheck_freq = vcheck_freq.tolist()

    def _accumulate_numba_metrics(
        self,
        comparisons: int,
        sub_comparisons: int,
        simd_chunks_compared: int,
        vcheck_freq: np.ndarray,
    ) -> None:
        """Additive variant for algorithms that call the C entry multiple times (DnC)."""
        self.comparisons += int(comparisons)
        self.sub_comparisons += int(sub_comparisons)
        self.simd_chunks_compared += int(simd_chunks_compared)
        self._chunk_size = _MODE_CHUNK_SIZES[self._bnl_numba_mode()]
        for i, v in enumerate(vcheck_freq):
            self.vcheck_freq[i] += int(v)

    def is_dominating(self, row1, row2) -> bool:
        """Pure-Python row-level dominance check. Used by BBSSkyline override
        (for MBR-level checks) and as a generic fallback."""
        self.comparisons += 1
        dominating = False
        for i in range(len(self.preferences)):
            self.sub_comparisons += 1
            if row1[i] < row2[i]:
                self.vcheck_freq[i] += 1
                return False
            if row1[i] > row2[i]:
                dominating = True
        self.vcheck_freq[-1] += 1
        return dominating

    def compute(self) -> pd.DataFrame:
        _t0 = time.perf_counter()

        # get the index
        index: np.ndarray = np.arange(len(self.data))

        # preprocess
        args = self.preprocess(index)

        # find the skyline for the entire data
        skyline, elapsed = time_function(self.find_skyline, index, *args)
        if not self.time_handled:
            self.compute_time.append(elapsed)

        # convert indices back to data
        skyline = self.dataset.data.iloc[skyline]
        skyline = skyline.sort_index().sort_values(by=self.dataset.preference)

        self.t_compute = time.perf_counter() - _t0
        return skyline

    def preprocess(self, index: np.ndarray) -> list:
        return []

    @abstractmethod
    def find_skyline(self, index: np.ndarray, *args) -> np.ndarray:
        pass
