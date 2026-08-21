"""
SFS / SaLSa correctness-demonstration dataset.

Low-cardinality integer data (values 0..2) plus a single global dominating
corner. After min-max normalization this yields the exact 0.0/1.0 boundary
structure the SaLSa-lexsort `pstop` prune needs, so SaLSa fires pstop hard
while SFS does not.

Purpose: in the benchmark Summary, SFS and SaLSa produce the *same skyline*
(Experiment.execute() asserts all algorithms' skylines are identical — that
equality IS the correctness check) but markedly different dominance/v-check
counts. SaLSa ≪ SFS proves SaLSa's pstop is implemented and working, not a
silent clone of SFS (which is what happens on continuous data where pstop
is dormant).
"""

import numpy as np
import pandas as pd

from skylines.dataset import Dominance
from skylines.dataset.synthetic.vos_synthetic import VariateOrderingSyntheticDataset

_PREF = ['A', 'B', 'C', 'D', 'E']


class SFSSaLSaDemo(VariateOrderingSyntheticDataset):
    """5-variate low-cardinality integer dataset with a dominating corner.

    All-MAX dominance. SaLSa-lexsort pstop fires (boundary structure) →
    SaLSa dominance checks ≪ SFS while the skyline is identical, exercising
    and validating both kernels against each other in the pipeline.
    """

    # DnC C kernel segfaults on heavily tied integer data (see Flight_4var);
    # experiment.py substitutes zero placeholders for D&C. BNL/SFS/SaLSa/BBS
    # still run and their skylines are cross-checked for equality.
    skip_algos = ('D&C',)

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = 10000,
                 seed: int = 42,
                 preference: list[str] = None,
                 cov_matrix=np.eye(5),
                 mean_vector=np.zeros(5),
                 effect=None,
                 cov_matrix_type='demo',
                 cov_matrix_spread='demo'):

        if preference is None:
            preference = list(_PREF)

        if dominance is None:
            dominance = {p: Dominance.MAX for p in preference}

        super().__init__(
            dominance=dominance,
            size=size,
            seed=seed,
            preference=preference,
            cov_matrix=cov_matrix,
            mean_vector=mean_vector,
            effect=effect,
            cov_matrix_type=cov_matrix_type,
            cov_matrix_spread=cov_matrix_spread,
        )

    def _generate(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = self.size
        # Low-cardinality integers -> many exact ties; after min-max the
        # column max maps to exactly 1.0 and min to exactly 0.0.
        data = rng.integers(0, 3, size=(n, 5)).astype(np.int64)
        # Global dominating corner: max in every column -> normalized all-1.0
        # row -> SaLSa pstop becomes all-ones and prunes the remaining scan.
        data[0] = 2
        df = pd.DataFrame(data, columns=_PREF)
        return df[self.preference]
