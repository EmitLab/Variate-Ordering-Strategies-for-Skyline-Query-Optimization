"""
       2 variates test datasets
"""

import numpy as np
import pandas as pd

from skylines.dataset.synthetic.vos_synthetic import VariateOrderingSyntheticDataset
from skylines.dataset.dataset import Dataset


class PQ_PositiveCorr_1(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, 1.0], [1.0, 1.0]]),
                    mean_vector=np.zeros(2),
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

class PQ_PositiveCorr_2(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, 0.9], [0.9, 1.0]]),
                    mean_vector=np.zeros(2),
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
              
class PQ_PositiveCorr_3(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, 0.8], [0.8, 1.0]]),
                    mean_vector=np.zeros(2),
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
              
class PQ_PositiveCorr_4(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, 0.75], [0.75, 1.0]]),
                    mean_vector=np.zeros(2),
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
              
class PQ_PositiveCorr_5(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, 0.7], [0.7, 1.0]]),
                    mean_vector=np.zeros(2),
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
              
class PQ_PositiveCorr_6(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, 0], [0, 1.0]]),
                    mean_vector=np.zeros(2),
                    effect = None,
                    cov_matrix_type = 'positive',
                    cov_matrix_spread = 'close'):
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

class PQ_PositiveCorr_7(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, 0.1], [0.1, 1.0]]),
                    mean_vector=np.zeros(2),
                    effect = None,
                    cov_matrix_type = 'positive',
                    cov_matrix_spread = 'close'):
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
              
class PQ_PositiveCorr_8(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, 0.2], [0.2, 1.0]]),
                    mean_vector=np.zeros(2),
                    effect = None,
                    cov_matrix_type = 'positive',
                    cov_matrix_spread = 'close'):
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
              
class PQ_PositiveCorr_9(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, 0.3], [0.3, 1.0]]),
                    mean_vector=np.zeros(2),
                    effect = None,
                    cov_matrix_type = 'positive',
                    cov_matrix_spread = 'close'):
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
              
class PQ_PositiveCorr_10(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, 0.4], [0.4, 1.0]]),
                    mean_vector=np.zeros(2),
                    effect = None,
                    cov_matrix_type = 'positive',
                    cov_matrix_spread = 'close'):
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
              
class PQ_NegativeCorr_1(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, -1.0], [-1.0, 1.0]]),
                    mean_vector=np.zeros(2),
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

class PQ_NegativeCorr_2(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, -0.9], [-0.9, 1.0]]),
                    mean_vector=np.zeros(2),
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
              
class PQ_NegativeCorr_3(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, -0.8], [-0.8, 1.0]]),
                    mean_vector=np.zeros(2),
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
              
class PQ_NegativeCorr_4(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, -0.75], [-0.75, 1.0]]),
                    mean_vector=np.zeros(2),
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
              
class PQ_NegativeCorr_5(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, -0.7], [-0.7, 1.0]]),
                    mean_vector=np.zeros(2),
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
              
class PQ_NegativeCorr_6(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, 0], [0, 1.0]]),
                    mean_vector=np.zeros(2),
                    effect = None,
                    cov_matrix_type = 'negative',
                    cov_matrix_spread = 'close'):
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

class PQ_NegativeCorr_7(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, -0.1], [-0.1, 1.0]]),
                    mean_vector=np.zeros(2),
                    effect = None,
                    cov_matrix_type = 'negative',
                    cov_matrix_spread = 'close'):
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
              
class PQ_NegativeCorr_8(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, -0.2], [-0.2, 1.0]]),
                    mean_vector=np.zeros(2),
                    effect = None,
                    cov_matrix_type = 'negative',
                    cov_matrix_spread = 'close'):
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
              
class PQ_NegativeCorr_9(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, -0.3], [-0.3, 1.0]]),
                    mean_vector=np.zeros(2),
                    effect = None,
                    cov_matrix_type = 'negative',
                    cov_matrix_spread = 'close'):
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
              
class PQ_NegativeCorr_10(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q'],
                    cov_matrix=np.array([[1.0, -0.4], [-0.4, 1.0]]),
                    mean_vector=np.zeros(2),
                    effect = None,
                    cov_matrix_type = 'negative',
                    cov_matrix_spread = 'close'):
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
              
class PQ_WorstCase_AntiCorr(Dataset):
      """BNL worst case: P ~ Gauss(0,1), Q = -P, pref={P,Q} both MAX.

      Perfect negative correlation => no row dominates any other =>
      every row is in the skyline (skyline size == n). BNL window never
      prunes, so dominance checks ~ n(n-1)/2. Presort stays O(n log n).
      Deterministic correctness oracle: |skyline| must equal n.
      """

      def __init__(self, dominance = None,
                   size: int = 10000, seed: int = 42,
                   preference: list[str] = None):
          super().__init__(
              control=[],
              preference=preference or ['P', 'Q'],
              effect={'P': {}, 'Q': {}},
              dominance=dominance,          # None => all MAX (Dataset default)
              size=size,
              seed=seed,
          )
          rng = np.random.default_rng(self.seed)
          p = rng.normal(loc=0.0, scale=1.0, size=self.size)
          self._df = pd.DataFrame({'P': p, 'Q': -p})

      @property
      def data(self) -> pd.DataFrame:
          return self._df
   
class PQ_BestCase_AntiCorr(Dataset):
      """BNL worst case: P ~ Gauss(0,1), Q = -P, pref={P,Q} both MAX.

      Perfect negative correlation => no row dominates any other =>
      every row is in the skyline (skyline size == n). BNL window never
      prunes, so dominance checks ~ n(n-1)/2. Presort stays O(n log n).
      Deterministic correctness oracle: |skyline| must equal n.
      """

      def __init__(self, dominance = None,
                   size: int = 10000, seed: int = 42,
                   preference: list[str] = None):
          super().__init__(
              control=[],
              preference=preference or ['P', 'Q'],
              effect={'P': {}, 'Q': {}},
              dominance=dominance,          # None => all MAX (Dataset default)
              size=size,
              seed=seed,
          )
          rng = np.random.default_rng(self.seed)
          p = rng.normal(loc=0.0, scale=1.0, size=self.size)
          self._df = pd.DataFrame({'P': p, 'Q': p})

      @property
      def data(self) -> pd.DataFrame:
          return self._df