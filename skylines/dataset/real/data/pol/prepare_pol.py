"""
Download and prepare the Pol (Robot Arm Control) dataset.

Source: OpenML dataset #60 (originally from the Delve benchmark suite)
        26 continuous input features + 1 regression target, 15,000 records.
        https://www.openml.org/d/60

Usage:
    pip install scikit-learn
    python prepare_pol.py

Output: pol.csv in this directory (15,000 rows × 27 columns: pol_f00..pol_f25 + pol_y).
"""

from pathlib import Path
import sys

OUT = Path(__file__).parent / 'pol.csv'

N_FEATURES = 26


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

    print('Fetching Pol dataset from OpenML (ID 60)…')
    try:
        data = fetch_openml(data_id=60, as_frame=True, parser='auto')
    except Exception:
        data = fetch_openml(name='pol', as_frame=True, parser='auto')

    X = data.data.copy()
    y = data.target.copy()

    if X.shape[1] != N_FEATURES:
        print(f'Warning: expected {N_FEATURES} features, got {X.shape[1]}. OpenML schema may have changed.')
        sys.exit(1)

    # Rename features to pol_f00 .. pol_f{N-1} for stability
    X.columns = [f'pol_f{i:02d}' for i in range(N_FEATURES)]
    if hasattr(y, 'name'):
        y = y.rename('pol_y')
    else:
        y = pd.Series(y, name='pol_y')

    df = pd.concat([X, y], axis=1)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    print(f'Dataset shape: {df.shape}')
    print('Columns:', list(df.columns))
    df.to_csv(OUT, index=False)
    print(f'Saved → {OUT}')


if __name__ == '__main__':
    main()
