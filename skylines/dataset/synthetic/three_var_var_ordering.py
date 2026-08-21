"""
       2 variates test datasets
"""

import numpy as np

from skylines.dataset.synthetic.vos_synthetic import VariateOrderingSyntheticDataset

class PQR(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R'],
                    cov_matrix=np.zeros((3, 3)),
                    mean_vector=np.zeros(3),
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
class PQR_MixedCorr_1(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R'],
                    cov_matrix=np.array([[1.0, -0.9, -0.4], [-0.9, 1.0, 0.6], [-0.4, 0.6, 1.0]]),
                    mean_vector=np.zeros(3),
                    effect = None,
                    cov_matrix_type = 'mixed',
                    cov_matrix_spread = 'wide'):
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

class PQR_MixedCorr_2(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R'],
                    cov_matrix=np.array([[1.0, -0.1, -0.8], [-0.1, 1.0, 0.9], [-0.8, 0.9, 1.0]]),
                    mean_vector=np.zeros(3),
                    effect = None,
                    cov_matrix_type = 'mixed',
                    cov_matrix_spread = 'wide'):
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
              
class PQR_MixedCorr_3(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R'],
                    cov_matrix=np.array([[1.0, 0.85, -0.24], [0.85, 1.0, -0.9], [-0.24, -0.8, 1.0]]),
                    mean_vector=np.zeros(3),
                    effect = None,
                    cov_matrix_type = 'mixed',
                    cov_matrix_spread = 'wide'):
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
              
class PQR_MixedCorr_4(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R'],
                    cov_matrix=np.array([[1.0, 0.12, 0.6], [0.12, 1.0, -0.8], [0.6, -0.8, 1.0]]),
                    mean_vector=np.zeros(3),
                    effect = None,
                    cov_matrix_type = 'mixed',
                    cov_matrix_spread = 'wide'):
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
              
class PQR_MixedCorr_5(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R'],
                    cov_matrix=np.array([[1.0, 0.5, -0.8], [0.5, 1.0, 0.1], [-0.8, 0.1, 1.0]]),
                    mean_vector=np.zeros(3),
                    effect = None,
                    cov_matrix_type = 'mixed',
                    cov_matrix_spread = 'wide'):
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