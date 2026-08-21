"""
Download and prepare the CASP (Physicochemical Properties of Protein Tertiary Structure) dataset.

Source: UCI ML Repository #265
        9 physicochemical features + RMSD target, 45,730 records.
        https://archive.ics.uci.edu/dataset/265

Usage:
    python prepare_casp.py

Output: casp.csv in this directory (45,730 rows × 10 columns: RMSD + F1-F9).
"""

from pathlib import Path
import sys

OUT = Path(__file__).parent / 'casp.csv'

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv'


def main():
    if OUT.exists():
        print(f'{OUT} already exists. Delete it to re-download.')
        return

    try:
        import requests
        import pandas as pd
        from io import StringIO
    except ImportError as e:
        print(f'Missing dependency: {e}. Run: pip install requests pandas')
        sys.exit(1)

    print(f'Downloading CASP from {URL} …')
    r = requests.get(URL, timeout=120)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    # CSV header is already: RMSD, F1, F2, ..., F9
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    print(f'Dataset shape: {df.shape}')
    print('Columns:', list(df.columns))
    df.to_csv(OUT, index=False)
    print(f'Saved → {OUT}')


if __name__ == '__main__':
    main()
