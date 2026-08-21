"""
Download and prepare the Elevators (Aircraft Control) dataset.

Source: OpenML dataset #216 (originally from the Delve benchmark suite)
        18 continuous input features + 1 regression target, 16,599 records.
        https://www.openml.org/d/216

Usage:
    pip install scikit-learn
    python prepare_elevators.py

Output: elevators.csv in this directory (16,599 rows × 19 columns: elev_f00..elev_f17 + elev_y).
"""

from pathlib import Path
import sys

OUT = Path(__file__).parent / 'elevators.csv'


def main():
    if OUT.exists():
        print(f'{OUT} already exists. Delete it to re-download.')
        return

    try:
        from sklearn.datasets import fetch_openml
        import pandas as pd
    except ImportError:
        print('scikit-learn not installed. Run: pip install scikit-learn')
        sys.exit(1)

    print('Fetching Elevators dataset from OpenML (ID 216)…')
    try:
        data = fetch_openml(data_id=216, as_frame=True, parser='auto')
    except Exception:
        data = fetch_openml(name='elevators', as_frame=True, parser='auto')

    X = data.data.copy()
    y = data.target.copy()

    X.columns = [f'elev_f{i:02d}' for i in range(X.shape[1])]
    if hasattr(y, 'name'):
        y = y.rename('elev_y')
    else:
        y = pd.Series(y, name='elev_y')

    df = pd.concat([X, y], axis=1)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    print(f'Dataset shape: {df.shape}')
    print('Columns:', list(df.columns))
    df.to_csv(OUT, index=False)
    print(f'Saved → {OUT}')


if __name__ == '__main__':
    main()
