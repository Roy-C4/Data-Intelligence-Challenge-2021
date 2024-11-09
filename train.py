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

        # Choose which agents you want to run on for training
        agents = [
            QLearningAgent(agent_number=0),
            # DQLAgent(
            #     agent_number=0,
            #     input_dim=(channels_used, len(obs), len(obs[0])),
            #     ddqn=False
            # ),
            # DQLAgent(
            #     agent_number=0,
            #     input_dim=(channels_used, len(obs), len(obs[0])),
            #     ddqn=True
            # ),
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

        # Iterate through each agent for `iters` iterations
        for agent in agents:
            start_time = time.time()
            for _ in trange(iters):
                old_state = agent.get_state_from_info(
                    obs, info)  # Convert observation to state
                action = agent.take_action(obs, info)

                # Perform the action
                obs, reward, terminated, info, actual_action = env.step([action], agent)
                new_state = agent.get_state_from_info(
                    obs, info)  # Convert next observation to state

                # Process the reward
                reward = agent.process_reward(
                    obs,
                    reward,
                    info,
                    new_state,
                    actual_action,
                    old_state
                )

                # Decay the epsilon of the agent
                agent.decay_epsilon(iters)

                # If agent is a normal q learning model, the way of updating Q
                # Values is different, so we should check for that
                if isinstance(agent, QLearningAgent):
                    agent.update_q_values(
                        old_state,
                        actual_action,
                        reward,
                        new_state
                    )
                else:
                    agent.remember(old_state, actual_action, reward, new_state, terminated)
                    agent.update_q_values()

                    if _ % (iters/40) == 0:
                        agent.synchronize_target_network()

                # If the agent is terminated, we reset the env and agent
                # parameters.
                if terminated:
                    obs, info, world_stats = env.reset()
                    agent.reset_parameters()

            # Compute training time
            training_time = time.time() - start_time

            # Reset parameters and disable training mode
            agent.reset_parameters()
            agent.training = False

            # If --save_agent_model flag is given to command, save the dqn
            # model of the agent
            if not isinstance(agent, QLearningAgent) and save_agent_model:
                # Put agent in evaluation mode
                agent.dqn.eval()
                file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
                grid_name = str(grid).replace('grid_configs/', '').replace('.grd', '')
                torch.save(agent.dqn.state_dict(), f'models/{str(agent)}-{grid_name}-{file_name}.pth')

            obs, info, world_stats = env.reset()
            print(world_stats)
            print(agent)
            Environment.evaluate_agent(grid, [agent], 1000, out, training_time,
                                       sigma, agent_start_pos=start_pos, cat=cat)



if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.out,
         args.random_seed, args.save_agent_model, args.cat)
