#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import numpy as np
import gym
from gym import wrappers
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch
import torch.nn as nn

from network_env import NetworkEnv
import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.preprocessors import HistoryPreprocessor, AtariPreprocessor, PreprocessorSequence
from deeprl_hw2.memory_replay import AtariReplayMemory


class DQNetwork(nn.Module):

    def __init__(self, window, input_size, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(window, 16, kernel_size=8, stride=4)
        input_size = [(x - 8) // 4 + 1 for x in input_size]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        input_size = [(x - 4) // 2 + 1 for x in input_size]
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        input_size = [(x - 3) + 1 for x in input_size]
        self.fully4 = nn.Linear(input_size[0] * input_size[1] * 32, 256)
        self.fully5 = nn.Linear(256, num_actions)

    def forward(self, output):
        output = nn.functional.relu(self.conv1(output))
        output = nn.functional.relu(self.conv2(output))
        output = nn.functional.relu(self.conv3(output))
        output = nn.functional.relu(self.fully4(output.view(output.size(0), -1)))
        return self.fully5(output)


def create_model(window, input_shape, num_actions,
                 model_name='deep q_network'):
    """Create the Q-network model.

    You can use any DL library you like, including Tensorflow, Keras or PyTorch.

    If you use Tensorflow or Keras, we highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understand your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------

      The Q-model.
    """
    if model_name == "linear q_network":
        q_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * input_shape[0] * input_shape[1], num_actions))
    else:
        q_model = DQNetwork(window, input_shape, num_actions)
    print(model_name, " is created")
    return q_model


def get_output_folder(parent_dir, env_name):
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
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument('--model', default='deep', type=str, choices=["linear", "deep"], help='Q network structure')
    parser.add_argument('--method', default='dqn', type=str, choices=["dqn", "double"], help='policy update method')
    parser.add_argument('--input_shape', default=[84, 84], nargs=2, type=int, help="Size of each frame")
    parser.add_argument('--window', default=4, type=int, help="Window size (number of frames in each state)")
    parser.add_argument('--memory-size', default=1000000, type=int, help="Replay memory size")
    parser.add_argument(
        '--num-burn-in', default=50000, type=int, help="Number of frames to collect before training DQN")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--batch-size', default=32, type=int, help="Batch size")
    parser.add_argument(
        '--target-update-freq', default=10000, type=int, help="The frequency with which the target network is updated")
    parser.add_argument(
        '--train-freq', default=4, type=int, help="The frequency with which the main DQN is trained")
    parser.add_argument(
        '--num-iterations', default=50000000, type=int, help="Number of samples/updates to perform in training")
    parser.add_argument(
        '--num-episodes', default=100, type=int, help="Number of episodes to perform in evaluation")
    parser.add_argument(
        '--check-freq', default=1000, type=int, help="The frequency with which a checkpoint is added")
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    with open("./input/data/data3.pickle", 'rb') as f:
        input_data = pickle.load(f)
    np.random.seed(0)
    env = NetworkEnv(input_data["flow_profile"], input_data["arrival_pattern"], input_data["reprofiling_delay"], 3)
    state = env.reset()
    done = False
    while not done:
        # action = np.ones((env.num_flow,))
        action = np.zeros((env.num_flow,))
        # action = np.random.randint(2, size=env.num_flow)
        next_state, reward, terminated, truncated = env.step(action)
        done = terminated
        # done = terminated or truncated
    arrival_time = np.array(env.arrival_time)
    departure_time = np.array(env.departure_time)
    end_to_end_delay = departure_time - arrival_time
    end_to_end_delay /= env.latency_target[:, np.newaxis]
    end_to_end_delay *= 100

    plt.rc("font", family="DejaVu Sans")
    plt.rcParams['figure.figsize'] = (15, 10)
    ax0 = plt.subplot()
    ax0.tick_params(axis='x', labelsize=30)
    ax0.tick_params(axis='y', labelsize=30)
    ax0.spines['top'].set_color('#606060')
    ax0.spines['bottom'].set_color('#606060')
    ax0.spines['left'].set_color('#606060')
    ax0.spines['right'].set_color('#606060')
    ax0.grid(True, color='#bfbfbf', linewidth=1)
    ax0.set_ylabel("End-to-end Delay", labelpad=10, color='#333333', size=40)
    ax0.set_xlabel("Packet Arrival Time", labelpad=15, color='#333333', size=40)
    ax0.yaxis.set_major_formatter(mtick.PercentFormatter())

    colors = ["#66b2ff", "#ffb266", "#66ff66"]
    labels = ["flow 1", "flow 2", "flow 3"]
    for xdata, ydata, color, label in zip(arrival_time, end_to_end_delay, colors, labels):
        ax0.plot(xdata, ydata, 'o-', color=color, label=label, linewidth=3, markersize=12)
    ax0.hlines(100, np.amin(arrival_time), np.amax(arrival_time), colors="red", linewidth=5,
               label='hard delay bound')
    plt.legend(fontsize=35)
    plt.tight_layout()
    # plt.savefig("./results/toy_data/shaping_on.png", bbox_inches='tight')
    plt.savefig("./results/toy_data/shaping_off.png", bbox_inches='tight')
    # plt.savefig("./results/toy_data/random_action.png", bbox_inches='tight')

    # args.input_shape = tuple(args.input_shape)
    # args.output = get_output_folder(args.output, args.env)
    # os.makedirs(args.output, exist_ok=True)
    #
    # # here is where you should start up a session,
    # # create your DQN agent, create your model, etc.
    # # then you can run your training method.
    #
    # # Set the random generator seed.
    # np.random.seed(args.seed)
    # # Establish the gym Atari environment.
    # env = gym.make(args.env)
    # # Create the q network.
    # model_name = "linear q_network" if args.model == "linear" else "deep q_network"
    # q_model = create_model(args.window, args.input_shape, env.action_space.n, model_name)
    # history_processor = HistoryPreprocessor(args.window, args.input_shape)
    # atari_processor = AtariPreprocessor(args.input_shape)
    # preprocessor = PreprocessorSequence(history_processor, atari_processor)
    # replay_memory = AtariReplayMemory(args.memory_size, args.window, args.input_shape)
    # agent = DQNAgent(q_model, args.method, preprocessor, replay_memory, args.gamma, args.target_update_freq,
    #                  args.num_burn_in, args.train_freq, args.batch_size, env, args.check_freq, args.output)
    # # agent.fit(args.num_iterations)
    # # Load an exiting model alternatively.
    # agent.q_model.load_state_dict(
    #     torch.load("./atari-v0/local_deep_dqn_backup/checkpoint_500000.pth", map_location=torch.device('cpu')))
    # # reward_avg, reward_std = agent.evaluate(args.num_episodes)
    # # print(f"Average reward: {reward_avg}.")
    # # print(f"Reward std:P{reward_std}")
    #
    # # Generate a video capture.
    # # Setup a wrapper to be able to record a video of the game
    # record_video = True
    # should_record = lambda i: record_video
    # agent.stage = "evaluation"
    # agent.q_model.eval()
    # env = wrappers.Monitor(env, args.output, video_callable=should_record, force=True)
    # frame = env.reset()
    # preprocessor.reset()
    # done = False
    # while not done:
    #     frame = preprocessor.game_processor.process_state_for_memory(frame)
    #     action = agent.select_action(frame)
    #     next_frame, _, done, _ = env.step(action)
    #     frame = next_frame
    # record_video = False
    # env.close()


if __name__ == '__main__':
    main()
