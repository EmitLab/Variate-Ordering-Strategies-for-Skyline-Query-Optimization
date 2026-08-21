import numpy as np
from skylines.dataset.synthetic.vos_synthetic import VariateOrderingSyntheticDataset


class TwentyFourVar(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24'],
                    cov_matrix=np.zeros((24, 24)),
                    mean_vector=np.zeros(24),
                    effect = None,
                    cov_matrix_type = '',
                    cov_matrix_spread = ''):
              super().__init__(
                     dominance=dominance,
                     size=size,
                     seed=seed,
                     preference=preference,
                     cov_matrix=cov_matrix,
                     mean_vector=mean_vector,
                     effect=effect,
                     cov_matrix_type=cov_matrix_type,
                     cov_matrix_spread=cov_matrix_spread
              )