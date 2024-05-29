import copy
import numpy as np


class NetworkEnv:
    """A network environment for RL sampling similar to a openai.gym environment."""

    def __init__(self, flow_profile, reprofiling_delay, interval, terminate_time=1000, sleep_prob=0.1, high_reward=1,
                 low_reward=0.1, penalty=-10, tor=0.003):
        # TODO: add network profile later.
        flow_profile = np.array(flow_profile)
        self.flow_profile = flow_profile
        self.num_flow = len(flow_profile)
        self.reprofiling_delay = reprofiling_delay
        self.interval = interval
        self.terminate_time = terminate_time
        self.sleep_prob = sleep_prob
        self.high_reward = high_reward
        self.low_reward = low_reward
        self.penalty = penalty
        self.arrival_pattern = self.generate_arrival_pattern()
        self.arrival_time = copy.deepcopy(self.arrival_pattern)
        bandwidth = np.sum([f[1] / d for f, d in zip(flow_profile, reprofiling_delay)])
        self.latency_target = (flow_profile[:, 2] + 1 / bandwidth) * (1 + tor)
        # Configure the network components.
        self.token_buckets = [TokenBucket(f[0], f[1], interval) for f in flow_profile]
        self.reprofilers = [[TokenBucket(f[0], f[1] - f[0] * d, interval), TokenBucket(f[1] / d, 0, interval)] for
                            f, d in zip(flow_profile, reprofiling_delay)]
        self.scheduler = FIFOScheduler(bandwidth, interval)
        # Set the internal variables.
        self.time = 0
        self.packet_count = [0] * len(self.arrival_pattern)
        self.departure_time = [[] for _ in range(len(self.arrival_pattern))]
        return

    def reward_function(self, end_to_end_delay, flow_idx):
        # Compute the reward given an end-to-end delay
        reward = 0
        if end_to_end_delay <= self.latency_target[flow_idx]:
            reward = self.low_reward + (1 - end_to_end_delay / self.latency_target[flow_idx]) * (
                    self.high_reward - self.low_reward)
        return reward

    def step(self, action):
        states = [[] for _ in range(self.num_flow)]
        self.time += self.interval
        scheduler_arrival = []
        # Forward the packets into the network and compute the time they reach the destination.
        for flow_idx, (flow_arrival, a) in enumerate(zip(self.arrival_pattern, action)):
            new_arrival = []
            while len(flow_arrival) > 0 and flow_arrival[0] < self.time:
                new_arrival.append(flow_arrival.pop(0))
            _, token, departure = self.token_buckets[flow_idx].forward(new_arrival, 1)
            states[flow_idx].append(token)
            backlog1, _, departure = self.reprofilers[flow_idx][0].forward(departure, a)
            backlog2, _, departure = self.reprofilers[flow_idx][1].forward(departure, a)
            states[flow_idx].append(backlog1 + backlog2)
            scheduler_arrival.append(departure)
        backlog, departure = self.scheduler.forward(scheduler_arrival)
        # Append the scheduler backlog to the state of each flow.
        for state in states:
            state.append(backlog)
        # Compute the end-to-end latency experienced by each packet.
        # Compute the reward and determine whether the episode terminates.
        terminate, exceed_target = True, False
        reward = 0
        for flow_idx, dpt in enumerate(departure):
            self.departure_time[flow_idx].extend(dpt)
            flow_reward = 0
            for d in dpt:
                end_to_end = d - self.arrival_time[flow_idx][self.packet_count[flow_idx]]
                self.packet_count[flow_idx] += 1
                flow_reward += self.reward_function(end_to_end, flow_idx)
                if end_to_end > self.latency_target[flow_idx]:
                    exceed_target = True
            if self.packet_count[flow_idx] < len(self.arrival_time[flow_idx]):
                terminate = False
            flow_reward = 0 if len(dpt) == 0 else flow_reward / len(dpt)
            reward += flow_reward
        if exceed_target:
            reward = self.penalty
        return states, reward, terminate, exceed_target

    def generate_arrival_pattern(self):
        arrival_pattern = []
        for flow_idx in range(self.num_flow):
            flow_arrival_pattern = []
            flow_token = self.flow_profile[flow_idx, 1]
            time = 0
            burst_size = int(flow_token)
            residual_token = flow_token - burst_size
            while time <= self.terminate_time:
                flow_arrival_pattern.extend([time] * burst_size)
                sleep = np.random.rand() <= self.sleep_prob
                if sleep and flow_token > 1:
                    sleep_duration = np.random.rand() * (flow_token - 1) + 1
                    backlog = sleep_duration + residual_token
                else:
                    sleep_duration = 1 - residual_token
                    backlog = 1
                burst_size = int(backlog)
                residual_token = backlog - burst_size
                time += sleep_duration * (1 / self.flow_profile[flow_idx, 0])
            arrival_pattern.append(flow_arrival_pattern)
        return arrival_pattern

    def reset(self):
        self.arrival_pattern = self.generate_arrival_pattern()
        self.arrival_time = copy.deepcopy(self.arrival_pattern)
        for token_bucket, reprofiler in zip(self.token_buckets, self.reprofilers):
            token_bucket.reset()
            reprofiler[0].reset()
            reprofiler[1].reset()
        self.scheduler.reset()
        self.time = 0
        self.packet_count = [0] * len(self.arrival_pattern)
        self.departure_time = [[] for _ in range(len(self.arrival_pattern))]
        states = [[] for _ in range(self.num_flow)]
        for state, f in zip(states, self.flow_profile):
            state.append(f[1])
            state.append(0)
            state.append(0)
        return states


class TokenBucket:

    def __init__(self, rate, burst, interval):
        self.rate = rate
        self.burst = burst
        self.interval = interval
        self.backlog = []
        self.token = burst
        self.time = 0
        self.depart = 0
        return

    def forward(self, arrival, action):
        old_backlog = self.backlog.copy()
        old_time = self.time
        self.backlog.extend(arrival)
        self.time += self.interval
        departure = []
        if action == 0:
            # The reprofiler is turned off.
            departure = [old_time] * len(old_backlog)
            departure.extend(arrival)
            self.backlog = []
            self.token = min(self.token + self.rate * (old_time - self.depart), self.burst)
            self.depart = self.time
            return 0, self.token, departure
        while len(self.backlog) > 0:
            # Examine the next packet.
            next_arrival = self.backlog.pop(0)
            token = self.token
            if next_arrival > self.depart:
                token = min(self.token + self.rate * (next_arrival - self.depart), self.burst)
            delay = 0
            if token < 1:
                delay = (1 - token) / self.rate
                token = 1
            next_depart = max(next_arrival, self.depart) + delay
            # Check whether the packet depart at this next time step.
            if next_depart >= self.time:
                self.backlog.insert(0, next_arrival)
                break
            else:
                self.token = token - 1
                self.depart = next_depart
                departure.append(next_depart)
        return len(self.backlog), self.token, departure

    def reset(self):
        self.backlog = []
        self.token = self.burst
        self.time = 0
        self.depart = 0
        return


class FIFOScheduler:

    def __init__(self, bandwidth, interval):
        self.bandwidth = bandwidth
        self.interval = interval
        self.backlog = []
        self.backlog_flow = []
        self.time = 0
        self.depart = 0
        return

    def forward(self, arrival):
        # Multiplex the flows.
        mpx_flow, mpx_arrival = [], []
        for flow_idx, flow_arrival in enumerate(arrival):
            mpx_flow.extend([flow_idx] * len(flow_arrival))
            mpx_arrival.extend(flow_arrival)
        sort_idx = np.argsort(mpx_arrival)
        for idx in sort_idx:
            self.backlog.append(mpx_arrival[idx])
            self.backlog_flow.append(mpx_flow[idx])
        self.time += self.interval
        departure = [[] for _ in range(len(arrival))]
        while len(self.backlog) > 0:
            # Examine the next packet.
            next_arrival = self.backlog.pop(0)
            next_flow = self.backlog_flow.pop(0)
            next_depart = max(next_arrival, self.depart) + 1 / self.bandwidth
            # Check whether the packet depart at this next time step.
            if next_depart >= self.time:
                self.backlog.insert(0, next_arrival)
                self.backlog_flow.insert(0, next_flow)
                break
            else:
                self.depart = next_depart
                departure[next_flow].append(next_depart)
        return len(self.backlog), departure

    def reset(self):
        self.backlog = []
        self.backlog_flow = []
        self.time = 0
        self.depart = 0
        return
