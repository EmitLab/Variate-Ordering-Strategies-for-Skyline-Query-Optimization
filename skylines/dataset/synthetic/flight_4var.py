"""
Flight 4-variate dataset.

Mirrors the `make_data()` generator from the lex_vs_nolex study notebook:
travel_time_hours / layovers / price_usd / distance with fixed correlations
and integer-clipped outputs. Used to exercise XCF lex-sort key behavior
(skylines.lex_sort_key="XCF") under realistic ties.
"""

import numpy as np
import pandas as pd

from skylines.dataset import Dominance
from skylines.dataset.synthetic.vos_synthetic import VariateOrderingSyntheticDataset


# correlation matrix order: time, layovers, price, distance
_CORR = np.array([
    [ 1.00,  0.75, -0.60, -0.50],
    [ 0.75,  1.00, -0.50, -0.40],
    [-0.60, -0.50,  1.00,  0.30],
    [-0.50, -0.40,  0.30,  1.00],
])
_STD = np.array([2.0, 1.0, 300.0, 800.0])
_MEAN = np.array([6.0, 2.0, 500.0, 2000.0])
_COV = np.outer(_STD, _STD) * _CORR

# clip floors per make_data() — keep outputs physically meaningful
_FLOORS = {
    'travel_time_hours': 1,
    'layovers': 0,
    'price_usd': 50,
    'distance': 100,
}


class Flight_4var(VariateOrderingSyntheticDataset):
    """4-variate flight dataset: travel_time, layovers, price, distance.

    All preferences are MIN by default (cheaper / faster / fewer stops /
    shorter trip — all four lower-is-better). VO heuristics (min_corr,
    max_corr, …) choose NCF vs XCF column ordering at experiment time.
    """

    # DnC C kernel segfaults on integer-tied data; experiment.py honors
    # this attribute and substitutes zero placeholders for D&C metrics.
    skip_algos = ('D&C',)

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = 10000,
                 seed: int = 42,
                 preference: list[str] = None,
                 cov_matrix=_COV,
                 mean_vector=_MEAN,
                 effect=None,
                 cov_matrix_type='mixed',
                 cov_matrix_spread='wide'):

        if preference is None:
            preference = ['travel_time_hours', 'layovers', 'price_usd', 'distance']

        if dominance is None:
            dominance = {p: Dominance.MIN for p in preference}

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
        # Reproduce make_data(): sample from multivariate normal in
        # (time, layovers, price, distance) order, then clip+round to
        # integer floors so realistic ties surface in lex-sort.
        np.random.seed(self.seed)

        cov = (self.cov_matrix + self.cov_matrix.T) / 2
        eigvals, eigvecs = np.linalg.eigh(cov)
        cov_psd = eigvecs @ np.diag(np.clip(eigvals, 1e-5, None)) @ eigvecs.T

        data = np.random.multivariate_normal(self.mean_vector, cov_psd, size=self.size)

        # Column order in `data` is fixed by _MEAN: time, layovers, price, distance.
        raw = {
            'travel_time_hours': np.clip(np.round(data[:, 0]), _FLOORS['travel_time_hours'], None).astype(int),
            'layovers':          np.clip(np.round(data[:, 1]), _FLOORS['layovers'],          None).astype(int),
            'price_usd':         np.clip(np.round(data[:, 2]), _FLOORS['price_usd'],         None).astype(int),
            'distance':          np.clip(np.round(data[:, 3]), _FLOORS['distance'],          None).astype(int),
        }

        df = pd.DataFrame(raw)
        # Honor the preference column order requested by the caller.
        df = df[self.preference]
        return df
