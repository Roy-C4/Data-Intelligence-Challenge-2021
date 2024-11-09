"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.

In this example training script, we use command line arguments. Feel free to
change this to however you want it to work.
"""
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import time

import torch
from tqdm import trange

try:
    from world import Environment

    # Add your agents here
    from agents.q_learning_agent import QLearningAgent
    from agents.dqn_and_ddqn_agent import DQLAgent
    from agents.duel_and_ddqn_agent import DuelQLAgent
    from agents.dqn_per_agent import PERDQLAgent
    from agents.ddqn_duel_and_per_agent import PERDuelDQLAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.extend(root_path)

    from world import Environment

    # Add your agents here
    from agents.q_learning_agent import QLearningAgent
    from agents.dqn_and_ddqn_agent import DQLAgent
    from agents.duel_and_ddqn_agent import DuelQLAgent
    from agents.dqn_per_agent import PERDQLAgent
    from agents.ddqn_duel_and_per_agent import PERDuelDQLAgent

def plot(y):
    x = np.arange(len(y))
    plt.plot(x, y)
    # Adding labels and title
    plt.xlabel('The times of reset the envrionment')
    plt.ylabel('The iters used in before reset ')
    # Displaying the chart
    plt.show()

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--out", type=Path, default=Path("results/"),
                   help="Where to save training results.")
    p.add_argument("--save_agent_model", action="store_true",
                   help="Enables saving of the agent model.")
    p.add_argument("--cat", action="store_true",
                   help="Enables a moving object on the grid (cat)..")

    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, out: Path, random_seed: int, save_agent_model: bool, cat: bool):
    """Main loop of the program."""

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    for grid in grid_paths:
        # Set up the environment and reset it to its initial state
        start_pos = [(1, 1)]
        use_random_obstacle = cat
        env = Environment(grid, no_gui, n_agents=1, agent_start_pos=start_pos,
                          reward_fn=Environment.simple_reward_function,
                          sigma=sigma, target_fps=fps, random_seed=random_seed,
                          random_obstacle=use_random_obstacle)
        obs, info = env.get_observation()
        channels_used = 4

        # Set up the agents from scratch for every grid
        # Add your agents here
        agents = [
            # QLearningAgent(agent_number=0),
            DQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=False
            ),
            DQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=True
            ),
            # DuelQLAgent(
            #     agent_number=0,
            #     input_dim=(channels_used, len(obs), len(obs[0])),
            #     ddqn=True
            # ),
            # PERDQLAgent(
            #     agent_number=0,
            #     input_dim=(channels_used, len(obs), len(obs[0])),
            #     ddqn=True
            # ),
            # PERDuelDQLAgent(
            #     agent_number=0,
            #     input_dim=(channels_used, len(obs), len(obs[0])),
            #     ddqn=True
            # ),
        ]

        model_paths = ['models/BaseModel-small-2023-06-23__15-09-08.pth',
                       'models/BaseModelWithDouble-small-2023-06-23__15-15-48.pth']
        for grid in grid_paths:
            # Iterate through each agent for `iters` iterations
            for agent, model in zip(agents, model_paths):
                agent.epsilon = 0
                agent.training = False

                # Load the saved model and put it in evaluation mode.
                agent.dqn.load_state_dict(torch.load(model, map_location=device))
                agent.dqn.eval()
                Environment.evaluate_agent(grid, [agent], 1000, out,
                                           sigma, agent_start_pos=start_pos,
                                           no_gui=False, eval=True, cat=cat)



if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.out,
         args.random_seed, args.save_agent_model, args.cat)