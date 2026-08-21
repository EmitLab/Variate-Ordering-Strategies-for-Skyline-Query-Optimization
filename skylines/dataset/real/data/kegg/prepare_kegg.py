"""
Download and prepare the KEGG Metabolic Reaction Network (Undirected) dataset.

Source: UCI ML Repository #221
        27 continuous network topology features, ~63,608 records.
        https://archive.ics.uci.edu/dataset/221

Usage:
    python prepare_kegg.py

Output: kegg.csv in this directory (rows × 27 columns: kegg_f00..kegg_f26).

Raw file format: no header, comma-separated.
    Column 0: gene/pathway ID string (dropped)
    Columns 1-27: 27 numerical network topology features
    Column 28: extra integer column (dropped — not a standard feature)
"""

from pathlib import Path
import sys

OUT = Path(__file__).parent / 'kegg.csv'

ZIP_URL = 'https://archive.ics.uci.edu/static/public/221/kegg+metabolic+reaction+network+undirected.zip'
DATA_FILE = 'Reaction Network (Undirected).data'


def main():
    if OUT.exists():
        print(f'{OUT} already exists. Delete it to re-download.')
        return

    try:
        import requests
        import pandas as pd
        import zipfile
        import io
    except ImportError as e:
        print(f'Missing dependency: {e}. Run: pip install requests pandas')
        sys.exit(1)

    print(f'Downloading KEGG zip from UCI …')
    r = requests.get(ZIP_URL, timeout=120)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(DATA_FILE).decode('utf-8', errors='replace')

    df = pd.read_csv(io.StringIO(raw), header=None)
    print(f'Raw shape: {df.shape}')

    # Column 0 is the gene/pathway string ID — drop it
    df = df.drop(columns=[0])

    # Keep the first 27 numeric columns after the ID
    numeric_cols = df.select_dtypes(include='number').columns[:27]
    df = df[numeric_cols].copy()
    df.columns = [f'kegg_f{i:02d}' for i in range(df.shape[1])]

    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    print(f'Dataset shape: {df.shape}')
    print('Columns:', list(df.columns))
    df.to_csv(OUT, index=False)
    print(f'Saved → {OUT}')


if __name__ == '__main__':
    main()
