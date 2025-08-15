import os
import argparse
import random

import numpy as np
import torch as th
import wandb


from stable_baselines3 import SPMA
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.logger import configure

if __name__ in "__main__":

    def str2bool(v):
        """Convert string to boolean"""
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", type=str2bool, default=False, required=True)
    parser.add_argument("--exp_name", type=str, default='test', required=True)

    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)

    algo_parsers = parser.add_subparsers(dest="algo", required=True)

    spma_parser = algo_parsers.add_parser("SPMA")
    spma_parser.add_argument("--n_steps", type=int, default=2048)
    spma_parser.add_argument("--batch_size", type=int, default=2048)
    spma_parser.add_argument("--use_armijo_actor", type=str2bool, default=True)
    spma_parser.add_argument("--use_armijo_critic", type=str2bool, default=True)
    spma_parser.add_argument("--total_timesteps", type=int, default=4096000)
    spma_parser.add_argument("--n_epochs", type=int, default=5)
    spma_parser.add_argument("--eta", type=float)

    args = parser.parse_args()


    config = vars(args)
    
    print(config)

    # exclude arguments not apart of SPMA parameters
    use_wandb = config.pop("use_wandb")
    exp_name = config.pop("exp_name")
    env_id = config.pop('env_id')
    config.pop('algo')
    timesteps = config.pop('total_timesteps')

    log_dir_parts = [
        "logs",
        exp_name,
        f"env_{env_id}",
        f"eta_{args.eta}",
        f"seed_{args.seed}",
    ]

    log_dir = os.path.join(*log_dir_parts)
    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)

    if use_wandb:
        wandb.init(
            project=exp_name,
            name=f"SPMA_eta{args.eta}_seed{args.seed}",
            config=config,
            sync_tensorboard=True,
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.backends.cudnn.deterministic = True

    model = SPMA("MlpPolicy", env_id, timesteps, **config)
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    model.learn(total_timesteps=model.timesteps)

