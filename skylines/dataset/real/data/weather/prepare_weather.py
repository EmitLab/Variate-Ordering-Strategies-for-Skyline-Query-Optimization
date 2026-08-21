"""
Download and prepare the SkyBench WEATHER dataset (CRU TS climate data).

Source: SkyBench — https://github.com/sean-chester/SkyBench
        Filename: elv_weather-U-15-566268.csv (~75 MB)
        15 meteorological attributes (normalised [0,1]), 566,268 records.

Usage:
    python prepare_weather.py

Output: weather.csv in this directory (566,268 rows × 15 columns).

Column assignment (CRU TS-inspired names, positional):
    tmp, tmn, tmx, dtr, pre, cld, vap, wet, frs, pet, rhm, ssh, wnd,
    stn_pressure, grnd_frost
"""

from pathlib import Path
import sys

OUT = Path(__file__).parent / 'weather.csv'

COLUMN_NAMES = ['tmp', 'tmn', 'tmx', 'dtr', 'pre', 'cld', 'vap', 'wet',
                'frs', 'pet', 'rhm', 'ssh', 'wnd', 'stn_pressure', 'grnd_frost']

URL = ('https://raw.githubusercontent.com/sean-chester/SkyBench/master/'
       'workloads/elv_weather-U-15-566268.csv')


def download() -> 'pd.DataFrame':
    import requests
    import pandas as pd
    from io import StringIO

    print(f'Downloading weather data (~75 MB) from SkyBench …')
    r = requests.get(URL, timeout=300, stream=True)
    r.raise_for_status()

    chunks = []
    downloaded = 0
    for chunk in r.iter_content(chunk_size=1 << 20):
        chunks.append(chunk)
        downloaded += len(chunk)
        print(f'\r  {downloaded / 1e6:.1f} MB', end='', flush=True)
    print()

    content = b''.join(chunks).decode('utf-8', errors='replace')
    first_line = content.split('\n')[0]
    sep = '\t' if '\t' in first_line else ','

    try:
        float(first_line.split(sep)[0])
        df = pd.read_csv(StringIO(content), sep=sep, header=None)
    except ValueError:
        df = pd.read_csv(StringIO(content), sep=sep)

    # Drop trailing all-NaN columns (SkyBench files have trailing commas)
    df = df.dropna(axis=1, how='all')

    if df.shape[1] == 15:
        df.columns = COLUMN_NAMES
    else:
        df.columns = [f'attr_{i:02d}' for i in range(df.shape[1])]

    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    print(f'Downloaded: {len(df):,} rows × {df.shape[1]} cols')
    return df


def from_local() -> 'pd.DataFrame':
    import pandas as pd

    candidates = [
        Path(__file__).parent / 'weather_raw.csv',
        Path(__file__).parent / 'elv_weather-U-15-566268.csv',
        Path(__file__).parent / 'weather-U-15-566268.csv',
    ]
    for p in candidates:
        if p.exists():
            print(f'Loading from {p} …')
            with open(p) as f:
                first = f.readline()
            sep = '\t' if '\t' in first else ','
            try:
                float(first.split(sep)[0])
                df = pd.read_csv(p, sep=sep, header=None)
            except ValueError:
                df = pd.read_csv(p, sep=sep)
            if df.shape[1] == 15:
                df.columns = COLUMN_NAMES
            else:
                df.columns = [f'attr_{i:02d}' for i in range(df.shape[1])]
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            return df

    print('No local file found. Provide elv_weather-U-15-566268.csv in this directory.')
    sys.exit(1)


if __name__ == '__main__':
    if OUT.exists():
        print(f'{OUT} already exists. Delete it to re-download.')
        sys.exit(0)

    try:
        import requests, pandas
    except ImportError as e:
        print(f'Missing dependency: {e}. Run: pip install requests pandas')
        sys.exit(1)

    df = from_local() if '--local' in sys.argv else download()
    df.to_csv(OUT, index=False)
    print(f'Saved → {OUT}')
