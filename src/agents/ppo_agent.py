"""
ppo_agent.py

Train a PPO agent with Stable-Baselines3. This example uses ActorCriticPolicy
and shows where a valuation network would be plugged in.
"""
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def train(env, save_path: str, total_timesteps: int = 100000):
    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec_env, verbose=1)
    checkpoint_cb = CheckpointCallback(save_freq=5000, save_path=save_path, name_prefix="ppo_model")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)
    model.save(os.path.join(save_path, "ppo_final"))
    return model

def evaluate_policy(model, env, n_steps=252):
    obs = env.reset()
    done = False
    total = 0.0
    returns = []
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total += reward
        returns.append(reward)
        if done:
            break
    return returns, total