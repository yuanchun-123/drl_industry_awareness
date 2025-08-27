"""
run_experiment.py - orchestrator
"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="sp500_daily_features.xlsx")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--total_timesteps", type=int, default=20000)
    parser.add_argument("--mode", type=str, choices=["train","eval"], default="train")
    args = parser.parse_args()

    from src.data_pipeline import load_excel, prepare_panel, save_processed
    from src.utils import build_env_from_files
    from src.agents.ppo_agent import train, evaluate_policy

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_excel(args.data_path)
    panel_long, returns_wide = prepare_panel(df)
    processed_dir = out / "processed"
    processed_dir.mkdir(exist_ok=True)
    save_processed(panel_long, returns_wide, processed_dir)

    tickers = list(returns_wide.columns)[:20]  # use small subset for quick runs
    env = build_env_from_files(str(processed_dir / "returns_wide.parquet"),
                               str(processed_dir / "panel_long.parquet"),
                               tickers=tickers, start_idx=0)

    if args.mode == "train":
        models_dir = out / "models"
        models_dir.mkdir(exist_ok=True)
        model = train(env, save_path=str(models_dir), total_timesteps=args.total_timesteps)
        print("Training completed. Models saved to", models_dir)
    else:
        from stable_baselines3 import PPO
        model = PPO.load(str(models_dir / "ppo_final.zip"))
        returns, total = evaluate_policy(model, env)
        print("Eval total reward:", total)

if __name__ == "__main__":
    main()