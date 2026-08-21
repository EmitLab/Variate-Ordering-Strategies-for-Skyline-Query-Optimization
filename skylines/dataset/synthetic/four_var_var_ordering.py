"""
       4 variates test datasets
"""

import numpy as np

from skylines.dataset.synthetic.vos_synthetic import VariateOrderingSyntheticDataset

class PQRS(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.zeros((4, 4)),
                    mean_vector=np.zeros(4),
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
              
class PQRS_MixedCorr_1(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, -0.9, -0.6, 0.2], [-0.9, 1.0, 0.4, 0.1], 
                                         [-0.6, 0.4, 1.0, 0.6], [0.2, 0.1, 0.6, 1.0]]),
                    mean_vector=np.zeros(4),
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

class PQRS_MixedCorr_2(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, 0.2, 0.6, 0.4], [0.2, 1.0, -0.85, -0.6],
                                         [0.6, -0.85, 1.0, 0.2], [0.4, -0.6, 0.2, 1.0]]),
                    mean_vector=np.zeros(4),
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
              
class PQRS_MixedCorr_3(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, 0.9, 0.5, 0.1], [0.9, 1.0, -0.2, -0.4],
                                         [0.5, -0.2, 1.0, -0.9], [0.1, -0.4, -0.9, 1.0]]),
                    mean_vector=np.zeros(4),
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
              
class PQRS_MixedCorr_4(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, -0.2, -0.6, -0.8], [-0.2, 1.0, 0.4, 0.8],
                                         [-0.6, 0.4, 1.0, -0.4], [-0.8, 0.8, -0.4, 1.0]]),
                    mean_vector=np.zeros(4),
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
              
class PQRS_MixedCorr_5(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, 0.2, -0.8, 0.7], [0.2, 1.0, -0.5, -0.8],
                                         [-0.8, -0.5, 1.0, -0.3], [0.7, -0.8, -0.3, 1.0]]),
                    mean_vector=np.zeros(4),
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
              
class PQRS_PositiveCorr_1(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, 0.9, 0.7, 0.1], [0.9, 1.0, 0.05, 0.1],
                                         [0.7, 0.05, 1.0, 0.2], [0.1, 0.1, 0.2, 1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'positive',
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

class PQRS_PositiveCorr_2(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, 0.7, 0.4, 0.9], [0.7, 1.0, 0.1, 0.05],
                                         [0.4, 0.1, 1.0, 0.02], [0.9, 0.05, 0.02, 1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'positive',
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
              
class PQRS_PositiveCorr_3(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, 0.01, 0.2, 0.04], [0.01, 1.0, 0.6, 0.9],
                                         [0.2, 0.6, 1.0, 0.1], [0.04, 0.9, 0.1, 1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'positive',
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
              
class PQRS_PositiveCorr_4(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, 1.0, 0.8, 0.6], [1.0, 1.0, 0.01, 0.2],
                                         [0.8, 0.01, 1.0, 0.1], [0.6, 0.2, 0.1, 1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'positive',
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
              
class PQRS_PositiveCorr_5(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, 0.95, 0.5, 0.7], [0.95, 1.0, 0.4, 0.1],
                                         [0.5, 0.4, 1.0, 0.15], [0.7, 0.1, 0.15, 1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'positive',
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


class PQRS_NegativeCorr_1(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[ 1.0, -0.6, -0.5, -0.1],
                                         [-0.6,  1.0, -0.3, -0.1],
                                         [-0.5, -0.3,  1.0, -0.2],
                                         [-0.1, -0.1, -0.2,  1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'negative',
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


class PQRS_NegativeCorr_2(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[ 1.0, -0.2, -0.8, -0.3],
                                         [-0.2,  1.0, -0.1, -0.5],
                                         [-0.8, -0.1,  1.0, -0.7],
                                         [-0.3, -0.5, -0.7,  1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'negative',
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


class PQRS_NegativeCorr_3(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[ 1.0, -0.3, -0.6, -0.9],
                                         [-0.3,  1.0, -0.8, -0.2],
                                         [-0.6, -0.8,  1.0, -0.4],
                                         [-0.9, -0.2, -0.4,  1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'negative',
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


class PQRS_NegativeCorr_4(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[ 1.0, -0.7, -0.4, -0.1],
                                         [-0.7,  1.0, -0.2, -0.5],
                                         [-0.4, -0.2,  1.0, -0.9],
                                         [-0.1, -0.5, -0.9,  1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'negative',
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


class PQRS_NegativeCorr_5(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[ 1.0, -0.2, -0.8, -0.3],
                                         [-0.2,  1.0, -0.4, -0.7],
                                         [-0.8, -0.4,  1.0, -0.5],
                                         [-0.3, -0.7, -0.5,  1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'negative',
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


class PQRS_MixedCorr_6(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, -0.3, -0.35, -0.4], [-0.3, 1.0, -0.4, 0.2],
                                         [-0.35, -0.4, 1.0, -0.45], [-0.4, 0.2, -0.45, 1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'mixed',
                    cov_matrix_spread = 'narrow'):
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

class PQRS_MixedCorr_7(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, 0.1, -0.1, -0.15], [0.1, 1.0, 0.15, -0.2],
                                         [-0.1, 0.15, 1.0, 0.2], [-0.15, -0.2, 0.2, 1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'mixed',
                    cov_matrix_spread = 'narrow'):
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
              
class PQRS_MixedCorr_8(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, -0.1, -0.2, 0.35], [-0.1, 1.0, 0.24, 0.3],
                                        [-0.2, 0.24, 1.0, -0.4], [0.35, 0.3, -0.4, 1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'mixed',
                    cov_matrix_spread = 'narrow'):
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
              
class PQRS_MixedCorr_9(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, -0.8, -0.9, 0.4], [-0.8, 1.0, 0.65, -0.7],
                                         [-0.9, 0.65, 1.0, -0.4], [0.4, -0.7, -0.4, 1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'mixed',
                    cov_matrix_spread = 'narrow'):
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
              
class PQRS_MixedCorr_10(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S'],
                    cov_matrix=np.array([[1.0, -0.2, -0.3, 0.1], [-0.2, 1.0, 0.01, -0.1],
                                         [-0.3, 0.01, 1.0, 0.1], [0.1, -0.1, 0.1, 1.0]]),
                    mean_vector=np.zeros(4),
                    effect = None,
                    cov_matrix_type = 'mixed',
                    cov_matrix_spread = 'narrow'):
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