"""
Download and prepare the Breast Cancer Wisconsin (Diagnostic) dataset.

Source: scikit-learn built-in (originally UCI ML Repository #17)
        30 continuous features, 569 records.

Usage:
    python prepare_breast_cancer.py

Output: breast_cancer.csv in this directory (569 rows × 30 columns).

Column naming convention (snake_case):
    mean_radius, mean_texture, ..., mean_fractal_dimension   (10 mean features)
    se_radius, se_texture, ..., se_fractal_dimension          (10 SE features)
    worst_radius, worst_texture, ..., worst_fractal_dimension (10 worst features)
"""

from pathlib import Path
import sys

OUT = Path(__file__).parent / 'breast_cancer.csv'

BASE = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
        'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

COLUMNS = (
    [f'mean_{b}' for b in BASE] +
    [f'se_{b}' for b in BASE] +
    [f'worst_{b}' for b in BASE]
)


def main():
    if OUT.exists():
        print(f'{OUT} already exists. Delete it to re-download.')
        return

    try:
        from sklearn.datasets import load_breast_cancer
        import pandas as pd
    except ImportError:
        print('scikit-learn not installed. Run: pip install scikit-learn')
        sys.exit(1)

    print('Loading Breast Cancer Wisconsin from scikit-learn…')
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=COLUMNS)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    print(f'Dataset shape: {df.shape}')
    print('Columns:', list(df.columns))
    df.to_csv(OUT, index=False)
    print(f'Saved → {OUT}')


if __name__ == '__main__':
    main()
