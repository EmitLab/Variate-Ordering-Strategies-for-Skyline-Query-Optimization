"""
Download and prepare the Parkinson's Telemonitoring dataset.

Source: UCI ML Repository #189
        22 columns (voice biomarkers + UPDRS scores), 5,875 records.
        https://archive.ics.uci.edu/dataset/189

Usage:
    pip install ucimlrepo
    python prepare_parkinsons.py

Output: parkinsons.csv in this directory with clean snake_case column names.

Columns saved:
    age, sex, test_time, motor_updrs, total_updrs,
    jitter_pct, jitter_abs, jitter_rap, jitter_ppq5, jitter_ddp,
    shimmer, shimmer_db, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda,
    nhr, hnr, rpde, dfa, ppe
    (subject# dropped as it is an identifier)
"""

from pathlib import Path
import sys

OUT = Path(__file__).parent / 'parkinsons.csv'

RENAME = {
    'subject#': None,       # drop — identifier
    'age': 'age',
    'sex': 'sex',
    'test_time': 'test_time',
    'motor_UPDRS': 'motor_updrs',
    'total_UPDRS': 'total_updrs',
    'Jitter(%)': 'jitter_pct',
    'Jitter(Abs)': 'jitter_abs',
    'Jitter:RAP': 'jitter_rap',
    'Jitter:PPQ5': 'jitter_ppq5',
    'Jitter:DDP': 'jitter_ddp',
    'Shimmer': 'shimmer',
    'Shimmer(dB)': 'shimmer_db',
    'Shimmer:APQ3': 'shimmer_apq3',
    'Shimmer:APQ5': 'shimmer_apq5',
    'Shimmer:APQ11': 'shimmer_apq11',
    'Shimmer:DDA': 'shimmer_dda',
    'NHR': 'nhr',
    'HNR': 'hnr',
    'RPDE': 'rpde',
    'DFA': 'dfa',
    'PPE': 'ppe',
}


def main():
    if OUT.exists():
        print(f'{OUT} already exists. Delete it to re-download.')
        return

    try:
        from ucimlrepo import fetch_ucirepo
        import pandas as pd
    except ImportError:
        print('ucimlrepo not installed. Run: pip install ucimlrepo')
        sys.exit(1)

    print("Fetching Parkinson's Telemonitoring from UCI repository (ID 189)…")
    dataset = fetch_ucirepo(id=189)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    df = pd.concat([X, y], axis=1)

    # Normalise column names
    col_map = {}
    for col in df.columns:
        for src, dst in RENAME.items():
            if col.strip() == src:
                col_map[col] = dst
                break
        else:
            # fallback: lowercase + replace non-alphanum with _
            import re
            col_map[col] = re.sub(r'[^a-z0-9]+', '_', col.strip().lower()).strip('_')

    df = df.rename(columns=col_map)
    # Drop identifier columns (mapped to None)
    drop_cols = [new for new in df.columns if new is None]
    df = df[[c for c in df.columns if c is not None and c not in drop_cols]]

    # Drop subject identifier if still present
    for cname in ['subject', 'subject_', 'subject#']:
        if cname in df.columns:
            df = df.drop(columns=[cname])

    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    print(f'Dataset shape: {df.shape}')
    print('Columns:', list(df.columns))
    df.to_csv(OUT, index=False)
    print(f'Saved → {OUT}')


if __name__ == '__main__':
    main()
