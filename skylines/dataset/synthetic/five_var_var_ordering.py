import numpy as np

from skylines.dataset.synthetic.vos_synthetic import VariateOrderingSyntheticDataset

class PQRST(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S', 'T'],
                    cov_matrix=np.zeros((5, 5)),
                    mean_vector=np.zeros(5),
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

class PQRST_MixedCorr_1(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S', 'T'],
                    cov_matrix=np.array([
                            [1.0, -0.7, 0.5, -0.3, 0.1],
                            [-0.7, 1.0, -0.4, 0.95, -0.2],
                            [0.5, -0.4, 1.0, 0.8, -0.5],
                            [-0.3, 0.95, 0.8, 1.0, 0.1],
                            [0.1, -0.2, -0.5, 0.1, 1.0]
                     ]),
                    mean_vector=np.zeros(5),
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


class PQRST_MixedCorr_2(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S', 'T'],
                    cov_matrix=np.array([
                            [ 1.0,  0.9, -0.4,  0.6,  0.8],
                            [ 0.9,  1.0, -0.6,  0.2, -0.2],
                            [-0.4, -0.6,  1.0, -0.7, 0.6],
                            [ 0.6,  0.2, -0.7,  1.0, -0.4],
                            [ 0.8, -0.2, 0.6, -0.4,  1.0]
                     ]),
                    mean_vector=np.zeros(5),
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


class PQRST_MixedCorr_3(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S', 'T'],
                    cov_matrix=np.array([
                            [1.0, 0.1, -0.9, -0.6, 0.6],
                            [0.1, 1.0, 0.6, -0.4, 0.05],
                            [-0.9, 0.6, 1.0, 0.1, -0.2],
                            [-0.6, -0.4, 0.1, 1.0, 0.7],
                            [0.6, 0.05, -0.2, 0.7, 1.0]
                     ]),
                    mean_vector=np.zeros(5),
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


class PQRST_MixedCorr_4(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S', 'T'],
                    cov_matrix=np.array([
                            [ 1.0, -1.0,  0.8, -0.6, -0.2],
                            [-1.0,  1.0,  0.2, -0.3,  0.9],
                            [ 0.8,  0.2,  1.0,  0.4, -0.4],
                            [-0.6, -0.3,  0.4,  1.0, 0.8],
                            [-0.2,  0.9, -0.4, 0.8,  1.0]
                     ]),
                    mean_vector=np.zeros(5),
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

 
class PQRST_MixedCorr_5(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S', 'T'],
                    cov_matrix=np.array([
                            [ 1.0, 0.5, -0.9, -0.2, -0.2],
                            [0.5,  1.0,  0.6, -0.2,  0.9],
                            [-0.9,  0.6,  1.0,  0.4, -0.3],
                            [-0.2, -0.2,  0.4,  1.0, -0.2],
                            [-0.2,  0.9, -0.3, -0.2,  1.0]
                     ]),
                    mean_vector=np.zeros(5),
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
