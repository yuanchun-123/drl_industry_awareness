"""
market_env.py

A simplified OpenAI Gym environment for portfolio allocation.
See README for notes.
"""
import gym
from gym import spaces
import numpy as np
import pandas as pd

class MarketEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, returns_wide: pd.DataFrame, features_long: pd.DataFrame,
                 tickers: list = None, initial_weights=None,
                 transaction_cost=0.001, reward_scaling=1.0, start_idx=None):
        super().__init__()
        self.returns_wide = returns_wide.sort_index()
        self.features_long = features_long.copy()
        self.tickers = tickers if tickers is not None else list(self.returns_wide.columns)
        numeric_cols = self.features_long.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['return']]
        if len(numeric_cols) == 0:
            numeric_cols = ['return']
        self.feature_cols = numeric_cols
        self.date_list = sorted(list(set(self.returns_wide.index)))
        self.n_assets = len(self.tickers)
        self.n_features = len(self.feature_cols)
        self.feature_matrix = {}
        for d in self.date_list:
            df_d = self.features_long[self.features_long['date'] == d]
            df_d = df_d.set_index('ticker').reindex(self.tickers)
            arr = df_d[self.feature_cols].to_numpy(dtype=float)
            if arr.shape[0] != self.n_assets:
                arr = np.nan_to_num(arr, nan=0.0)
            self.feature_matrix[d] = arr
        self.start_idx = start_idx if start_idx is not None else 0
        self.t = self.start_idx
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.n_assets,), dtype=np.float32)
        obs_dim = self.n_assets * self.n_features + self.n_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.prev_weights = np.zeros(self.n_assets, dtype=np.float32)
        self.done = False

    def _get_obs(self):
        date = self.date_list[self.t]
        feat = self.feature_matrix.get(date, np.zeros((self.n_assets,self.n_features),dtype=float))
        return np.concatenate([feat.flatten(), self.prev_weights]).astype(np.float32)

    def reset(self):
        self.t = self.start_idx
        self.prev_weights = np.zeros(self.n_assets, dtype=np.float32)
        self.done = False
        return self._get_obs()

    def step(self, action):
        exp = np.exp(action - np.max(action))
        weights = exp / (np.sum(exp) + 1e-8)
        if self.t + 1 >= len(self.date_list):
            self.done = True
            return self._get_obs(), 0.0, True, {}
        next_date = self.date_list[self.t + 1]
        returns_next = self.returns_wide.loc[next_date, self.tickers].to_numpy(dtype=float)
        portfolio_return = np.dot(weights, returns_next)
        turnover = np.sum(np.abs(weights - self.prev_weights))
        cost = self.transaction_cost * turnover
        reward = (portfolio_return - cost) * self.reward_scaling
        self.prev_weights = weights
        self.t += 1
        obs = self._get_obs()
        done = self.done or (self.t >= len(self.date_list)-1)
        info = {
            "portfolio_return": float(portfolio_return),
            "turnover": float(turnover),
            "cost": float(cost),
            "date": str(next_date)
        }
        return obs, float(reward), done, info

    def render(self, mode='human'):
        print(f"Date: {self.date_list[self.t]} | Prev weights sum: {self.prev_weights.sum():.3f}")