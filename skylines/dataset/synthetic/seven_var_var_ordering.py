import numpy as np

from skylines.dataset.synthetic.vos_synthetic import VariateOrderingSyntheticDataset

class PQRSTUV(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S', 'T', 'U', 'V'],
                    cov_matrix=np.zeros((7, 7)),
                    mean_vector=np.zeros(7),
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