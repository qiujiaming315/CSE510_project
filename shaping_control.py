#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch
import torch.nn as nn

from network_env import NetworkEnv
from lib.vdn import DQNAgent
from lib.preprocessors import NetworkStatePreprocessor
from lib.memory_replay import NetworkReplayMemory


class DQNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fully1 = nn.Linear(3, 16)
        self.fully2 = nn.Linear(16, 16)
        self.fully3 = nn.Linear(16, 16)
        self.fully4 = nn.Linear(16, 2)

    def forward(self, output):
        output = nn.functional.relu(self.fully1(output))
        output = nn.functional.relu(self.fully2(output))
        output = nn.functional.relu(self.fully3(output))
        return self.fully4(output)


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
    parser.add_argument('input', type=str, help='Path to the input file')
    parser.add_argument('--model', default='deep', type=str, choices=["linear", "deep"], help='Q network structure')
    parser.add_argument('--method', default='dqn', type=str, choices=["dqn", "double"], help='policy update method')
    parser.add_argument('--interval', default=3, type=int, help="time interval for the agents to take action")
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
    with open(args.input, 'rb') as f:
        input_data = pickle.load(f)
    # Create the output directory.
    args.output = get_output_folder(args.output)
    os.makedirs(args.output, exist_ok=True)
    # Set the random generator seed.
    np.random.seed(args.seed)
    # Establish the network environment.
    env = NetworkEnv(input_data["flow_profile"], input_data["arrival_pattern"], input_data["reprofiling_delay"],
                     args.interval)
    # Create the q network.
    model_name = "linear q_network" if args.model == "linear" else "deep q_network"
    q_model = create_model(model_name)
    preprocessor = NetworkStatePreprocessor()
    replay_memory = NetworkReplayMemory(args.memory_size)
    agent = DQNAgent(q_model, args.method, preprocessor, replay_memory, args.gamma, args.target_update_freq,
                     args.num_burn_in, args.train_freq, args.batch_size, env, args.check_freq, args.output)
    agent.fit(args.num_iterations)
    # # Load an exiting model alternatively.
    # agent.q_model.load_state_dict(
    #     torch.load("./atari-v0/local_deep_dqn_backup/checkpoint_500000.pth", map_location=torch.device('cpu')))
    # reward_avg, reward_std = agent.evaluate(args.num_episodes)
    # print(f"Average reward: {reward_avg}.")
    # print(f"Reward std:P{reward_std}")

    # states = env.reset()
    # done = False
    # while not done:
    #     # actions = np.ones((env.num_flow,))
    #     actions = np.zeros((env.num_flow,))
    #     # actions = np.random.randint(2, size=env.num_flow)
    #     next_states, reward, terminated, truncated = env.step(actions)
    #     done = terminated
    #     # done = terminated or truncated

    # TODO: save the output data.
    # arrival_time = np.array(env.arrival_time)
    # departure_time = np.array(env.departure_time)
    # end_to_end_delay = departure_time - arrival_time
    # end_to_end_delay /= env.latency_target[:, np.newaxis]
    # end_to_end_delay *= 100
    #
    # plt.rc("font", family="DejaVu Sans")
    # plt.rcParams['figure.figsize'] = (15, 10)
    # ax0 = plt.subplot()
    # ax0.tick_params(axis='x', labelsize=30)
    # ax0.tick_params(axis='y', labelsize=30)
    # ax0.spines['top'].set_color('#606060')
    # ax0.spines['bottom'].set_color('#606060')
    # ax0.spines['left'].set_color('#606060')
    # ax0.spines['right'].set_color('#606060')
    # ax0.grid(True, color='#bfbfbf', linewidth=1)
    # ax0.set_ylabel("End-to-end Delay", labelpad=10, color='#333333', size=40)
    # ax0.set_xlabel("Packet Arrival Time", labelpad=15, color='#333333', size=40)
    # ax0.yaxis.set_major_formatter(mtick.PercentFormatter())
    #
    # colors = ["#66b2ff", "#ffb266", "#66ff66"]
    # labels = ["flow 1", "flow 2", "flow 3"]
    # for xdata, ydata, color, label in zip(arrival_time, end_to_end_delay, colors, labels):
    #     ax0.plot(xdata, ydata, 'o-', color=color, label=label, linewidth=3, markersize=12)
    # ax0.hlines(100, np.amin(arrival_time), np.amax(arrival_time), colors="red", linewidth=5,
    #            label='hard delay bound')
    # plt.legend(fontsize=35)
    # plt.tight_layout()
    # # plt.savefig("./results/toy_data/shaping_on.png", bbox_inches='tight')
    # plt.savefig("./results/toy_data/shaping_off.png", bbox_inches='tight')
    # # plt.savefig("./results/toy_data/random_action.png", bbox_inches='tight')


if __name__ == '__main__':
    main()
