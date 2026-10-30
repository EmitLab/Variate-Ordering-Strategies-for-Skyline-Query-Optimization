# VOS: Variate Ordering Strategies for Skyline Query Optimization

This repository contains the implementation of VoS: Variate Ordering Strategies for Skyline Query Optimization.

## File Structure

### `/skylines/__init__.py`

Contains all configurations for the experiment setup.

### `/skylines/dataset/`

Contains all datasets used for experiments.

- `synthetic/`: Contains data sets for experiments with synthetic data; including the covariance matrix, preference attribute sets, dominance criteria, and scripts to generate the synthetic data.
- `real/`: Contains data sets for experiments with real-world data; including the preference attributes sets, dominance criteria, and scripts to generate the (augmented) real-world data.
- `real/data/`: Contains raw unaugmented data files for real-world datasets in tabular CSV format, where rows represent tuples and columns represent variables.

### `/skylines/strategies/`

Contains implementations of strategies for finding best variate ordering.

- `algorithms.py`: Contains implementation of the `min-abs-correlation-first`,  `min-correlation-first`, `inc-min-correlation-first`, `inc-max-correlation-last`, `min-pairwise-correlation`, and `max-correlation-first` variate ordering strategies.

### `/skylines/skyline/`

Contains implementations naive and causal versions of all skyline algorithms using in our experimentation.

- `bnl/`: Implementation of naive and causal versions of Block Nested Loop (BNL) algorithm.

- `sfs/`: Implementation of naive and causal versions of Sort-Filter-Skyline (SFS) algorithm.

- `salsa/`: Implementation of naive and causal versions of Sort and Limit Skyline algorithm (SaLSa).

- `dnc/`: Implementation of naive and causal versions of Divide and Conquer (D&C) skyline algorithm.

- `bbs/`: Implementation of naive and causal versions of Branch and Bound Skyline (BBS) algorithm.

### `/skylines/experiment/core/`

Contains the core code for execution of the experiments.

- `benchmark.py`: Creates dataset instances based on the configuration and executes the skyline experiments.
- `summarize.py`: Summarize the executed skyline experiments for further analysis.

### `/skylines/common/`

Contains definition of various common boilerplate code.

`/constants`: Defines constants used for the experiments including definition of all experiment sets.

`/kmeans`: Implementation of K-Means algorithm.

`/utils`: Defines various common utility methods.

## Datasets

As mentioned before, datasets are defined inside `/skylines/dataset/`.

### Synthetic Datasets

A typical synthetic dataset definition looks like the following:

    class PQRST_MixedCorr_1(VariateOrderingSyntheticDataset):
       def __init__(self, dominance = None,
                    size = 10000,
                    seed = 42,
                    preference = ['P', 'Q', 'R', 'S', 'T'],
                    cov_matrix=np.array([
                            [ 1.0, -0.7,  0.5, -0.3,  0.1],
                            [-0.7,  1.0, -0.4, 0.95, -0.2],
                            [ 0.5, -0.4,  1.0,  0.8, -0.5],
                            [-0.3, 0.95,  0.8,  1.0,  0.1],
                            [ 0.1, -0.2, -0.5,  0.1,  1.0]
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

The `cov_matrix` is given as input to `np.random.multivariate_normal` function.

### Real-World Datasets

A typical real-world dataset definition looks like the following:

    class Abalone_3var_mixed(RealDataset):

        def __init__(self,
                    dominance: dict[str, Dominance] = None,
                    size: int = None,
                    seed: int = 42):
            super().__init__(file_name='abalone',
                            preference=['Length', 'Whole Weight', 'Sex'],
                            dominance=dominance,
                            size=size,
                            seed=seed,
                            provided=False)

Here, a covariance matrix is not required since we have the actual data available.

## Requirements

The experiments were run using Python 3.12 with [uv](https://docs.astral.sh/uv/) for environment and dependency management. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. The process to set up the environment is defined below.

### Installing uv

Install uv by following the instructions at
[https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/), or with the following command:

    curl -LsSf https://astral.sh/uv/install.sh | sh

### Creating environment

Create the virtual environment and install the exact locked dependencies using the following command:

    uv sync

This creates a `.venv/` directory in the project root, downloading Python 3.12 if it is not already available.

To also install the development dependencies (`pytest`), use the following command:

    uv sync --group dev

### Running inside the environment

Prefix commands with `uv run` to execute them inside the environment:

    uv run python -m skylines.experiment.core.benchmark

Alternatively, activate the environment directly:

    source .venv/bin/activate

## How to Run

### Configure the experiment setup

The experiments are configured through the `skylines/__init__.py` file.

    """ Number of tuples """
    n_samples = 200_000

    """ Number of runs for each experiment (to ensure stability) """
    n_runs = 5

    """ Lexicographic sort based on variate order """
    lex_sort = True

    """ Experiment type to run (enable only one line below) """
    # experiment_type = 'Synthetic'
    # experiment_type = 'Real'

### Run skyline computation experiments

These experiments execute skyline queries on all variate orders for all the skyline algorithms listed above, and reports number of dominance checks, number of per-variate dominance checks and execution time.

To run the experiment use the following command:

    uv run python -m skylines.experiment.core.benchmark

To summarize the experiment results use the following command:

    uv run python -m skylines.experiment.core.summarize

This summarizes all the experiment results, reporting the best variate orders for all the proposed variate ordering strategies in our work. The dominance checks, per-variate dominance checks and execution time results are also reported for every variate ordering strategy.
