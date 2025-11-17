#!/usr/bin/env python3
import sys
from pathlib import Path

PARQUET_PATH = Path('train-00000-of-00001.parquet')
CSV_PATH = Path('train-00000-of-00001.csv')

def main():
    if not PARQUET_PATH.exists():
        print(f"Parquet file not found: {PARQUET_PATH}", file=sys.stderr)
        sys.exit(1)
    try:
        import pyarrow.parquet as pq
        import pyarrow.csv as pcsv
    except ImportError:
        print("pyarrow not installed. Install with: pip install pyarrow", file=sys.stderr)
        sys.exit(2)

    table = pq.read_table(PARQUET_PATH)
    pcsv.write_csv(table, CSV_PATH)
    print(f"Wrote CSV: {CSV_PATH} with {table.num_rows} rows and {table.num_columns} columns")
    print("Columns:", table.schema.names)

if __name__ == '__main__':
    main()
