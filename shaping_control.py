#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import numpy as np
import pickle
import torch.nn as nn

from network_env import NetworkEnv
from lib.vdn import DQNAgent
from lib.preprocessors import NetworkStatePreprocessor
from lib.memory_replay import NetworkReplayMemory
from lib.utils import DQNetwork


def create_model(model_name='deep q_network'):
    """Create the Q-network model.

    Parameters
    ----------
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    the Q-model.
    """
    # TODO: Hard code the state and action size for now, update in the future.
    if model_name == "linear q_network":
        q_model = nn.Sequential(
            nn.Linear(3, 2))
    else:
        q_model = DQNetwork()
    print(model_name, " is created")
    return q_model


def get_output_folder(parent_dir):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, 'run{}'.format(experiment_id))
    return parent_dir


def main():
    parser = argparse.ArgumentParser(description='Run VDN on multi-flow traffic shaping network')
    parser.add_argument('flow_profile', type=str, help='Path to the flow profile file')
    parser.add_argument('flow_path', type=str, help='Path to the flow path file')
    parser.add_argument('--model', default='deep', type=str, choices=["linear", "deep"], help='Q network structure')
    parser.add_argument('--method', default='dqn', type=str, choices=["dqn", "double"], help='policy update method')
    parser.add_argument('--interval', default=1, type=int, help="time interval for the agents to take action")
    parser.add_argument('--memory-size', default=10000, type=int, help="Replay memory size")
    parser.add_argument(
        '--num-burn-in', default=5000, type=int, help="Number of states to collect before training VDN")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--batch-size', default=32, type=int, help="Batch size")
    parser.add_argument(
        '--target-update-freq', default=1000, type=int, help="The frequency with which the target network is updated")
    parser.add_argument(
        '--train-freq', default=4, type=int, help="The frequency with which the main DQN is trained")
    parser.add_argument(
        '--num-iterations', default=50000, type=int, help="Number of samples/updates to perform in training")
    parser.add_argument(
        '--num-episodes', default=100, type=int, help="Number of episodes to perform in evaluation")
    parser.add_argument(
        '--check-freq', default=500, type=int, help="The frequency with which a checkpoint is added")
    parser.add_argument(
        '-o', '--output', default='output', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    # Load the input data.
    with open(args.flow_profile, 'rb') as f:
        flow_profile_data = pickle.load(f)
    flow_path_data = np.load(args.flow_path)
    # Create the output directory.
    args.output = get_output_folder(args.output)
    os.makedirs(args.output, exist_ok=True)
    # Set the random generator seed.
    np.random.seed(args.seed)
    # Establish the network environment.
    env = NetworkEnv(flow_profile_data["flow_profile"], flow_path_data, flow_profile_data["reprofiling_delay"], args.interval)
    # Create the q network.
    model_name = "linear q_network" if args.model == "linear" else "deep q_network"
    q_model = create_model(model_name)
    preprocessor = NetworkStatePreprocessor()
    replay_memory = NetworkReplayMemory(args.memory_size)
    agent = DQNAgent(q_model, args.method, preprocessor, replay_memory, args.gamma, args.target_update_freq,
                     args.num_burn_in, args.train_freq, args.batch_size, env, args.check_freq, args.num_episodes,
                     args.output)
    agent.fit(args.num_iterations)
    np.savez(os.path.join(args.output, "reward_history.npz"), reward_mean_history=agent.reward_mean_history,
             reward_std_history=agent.reward_std_history,
             training_iteration=np.arange(args.check_freq, args.num_iterations + 1, args.check_freq))


if __name__ == '__main__':
    main()
