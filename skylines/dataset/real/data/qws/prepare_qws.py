"""
Download and prepare the QWS (Quality of Web Services) v2 dataset.

Source: Zheng et al. "Investigating QoS of Real-World Web Services" (2014)
        9 QoS numerical attributes, 2,507 web service records.

The QWS dataset is not available for automated download.
Manual steps:
  1. Visit https://qwsdata.github.io and request / download QWS2.csv
     (or contact the authors: zbzheng@cse.cuhk.edu.hk)
  2. Place the file in this directory as qws_raw.csv
  3. Run: python prepare_qws.py --local

Output: qws.csv in this directory (2,507 rows × 9 QoS columns).

Expected QoS column names (case-insensitive match):
    ResponseTime, Availability, Throughput, Successability,
    Reliability, Compliance, BestPractices, Latency, Documentation
"""

from pathlib import Path
import sys

OUT = Path(__file__).parent / 'qws.csv'

QOS_COLS = ['ResponseTime', 'Availability', 'Throughput', 'Successability',
            'Reliability', 'Compliance', 'BestPractices', 'Latency', 'Documentation']


def _find_qos_cols(df):
    """Return DataFrame with only the 9 QoS numeric columns."""
    import pandas as pd
    rename = {}
    for col in df.columns:
        for qos in QOS_COLS:
            if qos.lower() in col.lower().replace(' ', '').replace('_', ''):
                rename[col] = qos
                break
    if rename:
        df = df.rename(columns=rename)
    present = [c for c in QOS_COLS if c in df.columns]
    if len(present) == 9:
        return df[present]

    num = df.select_dtypes(include='number')
    if num.shape[1] >= 9:
        sub = num.iloc[:, :9].copy()
        sub.columns = QOS_COLS[:sub.shape[1]]
        return sub

    raise ValueError(f'Could not find 9 QoS columns. Got: {list(df.columns)}')


def from_local():
    import pandas as pd
    for name in ['qws_raw.csv', 'QWS2.csv', 'qws2.csv', 'QWS.csv', 'qws.csv']:
        p = Path(__file__).parent / name
        if p.exists() and p != OUT:
            print(f'Loading from {p} …')
            df = pd.read_csv(p, encoding='utf-8', encoding_errors='replace')
            df = _find_qos_cols(df)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            print(f'Loaded: {len(df):,} rows × {df.shape[1]} cols')
            print('Columns:', list(df.columns))
            return df
    print('No local QWS file found.')
    print('Place the raw file (e.g. QWS2.csv) in this directory and re-run with --local')
    sys.exit(1)


if __name__ == '__main__':
    if OUT.exists():
        print(f'{OUT} already exists. Delete it to re-download.')
        sys.exit(0)

    if '--local' not in sys.argv:
        print('QWS dataset requires manual download.')
        print('  1. Visit https://qwsdata.github.io and download QWS2.csv')
        print('  2. Place it here as qws_raw.csv')
        print('  3. Run: python prepare_qws.py --local')
        sys.exit(1)

    df = from_local()
    df.to_csv(OUT, index=False)
    print(f'Saved → {OUT}')
