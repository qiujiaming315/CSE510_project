import numpy as np
import os
import pickle
import torch
import torch.nn as nn

from network_env import NetworkEnv
from lib.utils import DQNetwork
from viz.packet_delay_demo import plot_delay


def get_latency_stats(env, control_policy, num_episodes=100, plot_output=None, plot_name=""):
    delay_aggregate = []
    for _ in range(num_episodes):
        states = env.reset()
        done = False
        while not done:
            states = np.array(states, dtype=np.float32)
            actions = control_policy(states)
            next_states, _, terminated, truncated = env.step(actions)
            states = next_states
            done = terminated
        arrival_time = env.arrival_time
        departure_time = env.departure_time
        latency_target = env.latency_target
        end_to_end_delay = [[(d[-1] - a) / target * 100 for a, d in zip(arrival, departure)] for
                            arrival, departure, target in zip(arrival_time, departure_time, latency_target)]
        if plot_output is not None:
            plot_delay(arrival_time, end_to_end_delay, plot_output, plot_name, segment=(0, 100))
            # Only plot the result once.
            plot_output = None
        for flow_delay in end_to_end_delay:
            delay_aggregate.extend(flow_delay)
    delay_aggregate = np.array(delay_aggregate)
    avg_delay = (np.mean(delay_aggregate), np.std(delay_aggregate))
    violation = np.sum(delay_aggregate > 100) / len(delay_aggregate)
    print(f"{plot_name}: {violation * 100: .3f}% delay violation")
    return avg_delay, violation


if __name__ == '__main__':
    flow_profile = "./input/data/flow_profile/flow3/data1.pickle"
    flow_path = "./input/data/flow_path/flow3/data4.npy"
    vdn_path = "./output/flow3/linear_dqn/checkpoint_last.pth"
    # vdn_path = "./output/deep_dqn/checkpoint_last.pth"
    output_path = "./figures/test/cyclic_0.1/"
    os.makedirs(output_path, exist_ok=True)
    with open(flow_profile, 'rb') as f:
        flow_profile_data = pickle.load(f)
    flow_path_data = np.load(flow_path)
    # Load the vdn model.
    vdn_model = nn.Sequential(nn.Linear(3, 2))
    # vdn_model = DQNetwork()
    vdn_model.load_state_dict(torch.load(vdn_path, map_location=torch.device('cpu')))
    np.random.seed(0)
    # Establish the network environment.
    env = NetworkEnv(flow_profile_data["flow_profile"], flow_path_data, flow_profile_data["reprofiling_delay"], 1,
                     sleep_prob=0.1)
    num_flow = env.num_flow
    on_policy = lambda states: np.ones((num_flow,))
    off_policy = lambda states: np.zeros((num_flow,))
    random_policy = lambda states: np.random.randint(2, size=num_flow)
    # vdn_policy = lambda states: np.argmax(
    #     vdn_model(torch.from_numpy(np.array(states, dtype=np.float32)).to("cpu")).detach().numpy(), axis=1)
    on_stats = get_latency_stats(env, on_policy, num_episodes=1, plot_output=output_path, plot_name="shaper_on")
    off_stats = get_latency_stats(env, off_policy, num_episodes=1, plot_output=output_path, plot_name="shaper_off")
    rand_stats = get_latency_stats(env, random_policy, num_episodes=1, plot_output=output_path, plot_name="random")
    # vdn_stats = get_latency_stats(env, vdn_policy, plot_output=output_path, plot_name="vdn")
    print()
