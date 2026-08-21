import numpy as np
import pandas as pd

from skylines.dataset.dataset import Dataset
from skylines.dataset import Dominance


class VariateOrderingSyntheticDataset(Dataset):
       """
       A base class for synthetic datasets that require specific ordering of variates.
       This class is intended to be extended by other synthetic dataset classes.
       """
       
       def __init__(self,
                    dominance: dict[str, Dominance] = None,
                    size: int = 10000,
                    seed: int = 42,
                    preference: list[str] = None,
                    cov_matrix = None,
                    mean_vector = None,
                    effect: dict[str, dict[str, float]] = None,
                    cov_matrix_type: str = 'mixed',
                    cov_matrix_spread: str | float = 'wide',
                    base_corr: float = 0.9):

              super().__init__(preference=preference,
                               control=[],
                                   effect={
                                          pref: {} for pref in preference
                                          },
                                   dominance=dominance,
                                   size=size,
                                   seed=seed)
              
              self.cov_matrix = cov_matrix
              self.mean_vector = mean_vector
              self.cov_matrix_type = cov_matrix_type
              self.cov_matrix_spread = cov_matrix_spread
              self.base_corr = base_corr
              
              if cov_matrix is None or len(cov_matrix) == 0:
                     self._data = self._generate_data_without_corr_matrix()
              else:
                     self._data = self._generate()
                     
       
       @property
       def data(self) -> pd.DataFrame:
              return self._data
       
       def _generate_data_without_corr_matrix(self) -> pd.DataFrame:
              np.random.seed(self.seed)
              n_samples = self.size
              n_vars = len(self.preference)
              base_corr = self.base_corr
              noise_scale = self.cov_matrix_spread if isinstance(self.cov_matrix_spread, float) else 0.1

              # Step 1: Create base correlation matrix
              corr_matrix = np.full((n_vars, n_vars), base_corr)
              np.fill_diagonal(corr_matrix, 1.0)

              # Step 2: Add small random noise to off-diagonal entries
              noise = np.random.uniform(-noise_scale, noise_scale, size=(n_vars, n_vars))
              noise = (noise + noise.T) / 2  # make symmetric
              np.fill_diagonal(noise, 0)
              corr_matrix += noise

              # Step 3: Clip to [0, 0.99] to ensure all positive correlations
              if self.cov_matrix_type == 'positive':
                     corr_matrix = np.clip(corr_matrix, 0.01, 0.99)

              # Step 4: Ensure positive-definiteness
              def nearest_positive_definite(A):
                     """Find the nearest positive-definite matrix to input."""
                     def is_positive_definite(x):
                            return np.all(np.linalg.eigvals(x) > 0)
                     B = (A + A.T) / 2
                     _, s, V = np.linalg.svd(B)
                     H = np.dot(V.T * s, V)
                     A2 = (B + H) / 2
                     A3 = (A2 + A2.T) / 2
                     if is_positive_definite(A3):
                            return A3
                     spacing = np.spacing(np.linalg.norm(A))
                     I = np.eye(A.shape[0])
                     k = 1
                     while not is_positive_definite(A3):
                            mineig = np.min(np.real(np.linalg.eigvals(A3)))
                            A3 += I * (-mineig * k**2 + spacing)
                            k += 1
                     return A3

              corr_matrix = nearest_positive_definite(corr_matrix)

              # Step 5: Sample from multivariate normal
              mean = np.zeros(n_vars)
              data = np.random.multivariate_normal(mean, corr_matrix, size=n_samples)

              pref_copy = self.preference.copy()
              pref_copy.sort()
              df = pd.DataFrame(data, columns=pref_copy)

              # order the data according to the preference
              df = df[self.preference]

              # store the data
            #   file_path = timestamp.get_file('data', f'{self.name}.csv')
            #   df.to_csv(file_path)

              return df
       
       def _generate(self) -> pd.DataFrame:
              # specify seed for reproducibility
              np.random.seed(seed=self.seed)

              cov = self.cov_matrix

              # Make covariance matrix symmetric
              cov = (cov + cov.T) / 2

              # Compute eigenvalues and eigenvectors
              eigvals, eigvecs = np.linalg.eigh(cov)

              # Replace negative eigenvalues with small positive numbers
              eigvals_clipped = np.clip(eigvals, 1e-5, None)

              # Reconstruct the PSD covariance matrix
              cov_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

              # Mean of each variate
              mean = self.mean_vector

              # Generate multivariate data
              data = np.random.multivariate_normal(mean, cov_psd, size=self.size)
              
              # Put in DataFrame for inspection
              pref_copy = self.preference.copy()
              pref_copy.sort()
              df = pd.DataFrame(data, columns=pref_copy)

              # order the data according to the preference
              df = df[self.preference]

              # store the data
            #   file_path = timestamp.get_file('data', f'{self.name}.csv')
            #   df.to_csv(file_path)

              return df