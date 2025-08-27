# DRL Industry-Aware Portfolio 

**This bundle contains**
- `src/data_pipeline.py` : load & preprocess the uploaded Excel to a wide panel suitable for the Gym env.
- `src/envs/market_env.py` : OpenAI-Gym style environment for portfolio allocation.
- `src/models/valuation.py` : PyTorch ValuationNet (value function approximator).
- `src/agents/ppo_agent.py` : Wrapper to train a PPO agent using Stable-Baselines3 with a custom valuation network.
- `run_experiment.py` : simple orchestrator script to train & evaluate.
- `requirements.txt` : suggested Python packages.

See the README content in the generated repo for run instructions.
