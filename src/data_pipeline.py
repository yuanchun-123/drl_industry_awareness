"""
data_pipeline.py (robust)

Improved loader that accepts varied column names (Date or date, Close/Adj_Close/close, etc.)
and produces:
 - panel_long.parquet : original long panel (cleaned)
 - returns_wide.parquet : pivoted wide returns (index=date, columns=ticker)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def _find_col(ci, candidates):
    for c in candidates:
        if c in ci:
            return c
    # try lower-case matching
    ci_lower = {c.lower(): c for c in ci}
    for cand in candidates:
        if cand.lower() in ci_lower:
            return ci_lower[cand.lower()]
    return None

def load_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # normalize column names: strip and keep original case mapping
    df.columns = [c.strip() for c in df.columns]
    # find date column
    date_col = _find_col(df.columns, ['date','Date','trade_date'])
    if date_col is None:
        raise ValueError("No date column found. Expected 'date' or 'Date'.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: 'date'})
    # find ticker column
    ticker_col = _find_col(df.columns, ['ticker','Ticker','symbol','Symbol'])
    if ticker_col is None:
        raise ValueError("No ticker column found. Expected 'ticker' or 'Ticker' or 'symbol'.")
    df = df.rename(columns={ticker_col: 'ticker'})
    return df

def prepare_panel(df: pd.DataFrame, price_col_candidates=['adj_close','Adj Close','adjclose','Close','close','price']):
    # Ensure date and ticker exist
    assert 'date' in df.columns and 'ticker' in df.columns, "Dataset must contain 'date' and 'ticker' columns"
    # attempt to find price column
    price_col = None
    for c in price_col_candidates:
        if c in df.columns:
            price_col = c
            break
    if price_col is None and 'return' not in df.columns:
        raise ValueError("No price or return column found. Provide a price column or 'return'.")
    df = df.sort_values(['ticker','date'])
    if 'return' not in df.columns:
        # compute returns via pct_change per ticker
        df['return'] = df.groupby('ticker')[price_col].pct_change()
    df = df.dropna(subset=['return'])
    # pivot returns
    returns_wide = df.pivot(index='date', columns='ticker', values='return').sort_index()
    return df, returns_wide

def save_processed(df: pd.DataFrame, returns_wide: pd.DataFrame, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(out_dir) / "panel_long.parquet")
    returns_wide.to_parquet(Path(out_dir) / "returns_wide.parquet")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="sp500_daily_features.xlsx")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()
    df = load_excel(args.data_path)
    panel, returns = prepare_panel(df)
    save_processed(panel, returns, args.out_dir)
    print("Processed saved to", args.out_dir)