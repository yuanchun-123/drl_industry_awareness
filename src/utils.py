"""
utils.py - helper functions
"""
import pandas as pd

def build_env_from_files(returns_wide_path: str, panel_long_path: str, tickers: list = None, start_idx: int = 0):
    from src.envs.market_env import MarketEnv
    returns_wide = pd.read_parquet(returns_wide_path)
    panel_long = pd.read_parquet(panel_long_path)
    if tickers is None:
        tickers = list(returns_wide.columns)
    env = MarketEnv(returns_wide, panel_long, tickers=tickers, start_idx=start_idx)
    return env