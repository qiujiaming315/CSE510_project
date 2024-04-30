import copy
import numpy as np


class NetworkEnv:
    """A network environment for RL sampling similar to a openai.gym environment."""

    def __init__(self, flow_profile, arrival_pattern, reprofiling_delay, interval, high_reward=10, low_reward=1,
                 penalty=-100, tor=0.003):
        # TODO: add network profile later.
        flow_profile = np.array(flow_profile)
        self.flow_profile = flow_profile
        self.arrival_pattern = arrival_pattern
        self.reprofiling_delay = reprofiling_delay
        self.interval = interval
        self.high_reward = high_reward
        self.low_reward = low_reward
        self.penalty = penalty
        self.num_flow = len(flow_profile)
        self.arrival_time = copy.deepcopy(arrival_pattern)
        bandwidth = np.sum([f[1] / d for f, d in zip(flow_profile, reprofiling_delay)])
        self.latency_target = (flow_profile[:, 2] + self.num_flow / bandwidth) * (1 + tor)
        # Configure the network components.
        self.token_buckets = [TokenBucket(f[0], f[1] + 1, interval) for f in flow_profile]
        self.reprofilers = [[TokenBucket(f[0], f[1] + 1 - f[0] * d, interval), TokenBucket(f[1] / d, 1, interval)] for
                            f, d in zip(flow_profile, reprofiling_delay)]
        self.scheduler = FIFOScheduler(bandwidth, interval)
        # Set the internal variables.
        self.time = 0
        self.packet_count = [0] * len(arrival_pattern)
        self.departure_time = [[] for _ in range(len(arrival_pattern))]
        return

    def reward_function(self, end_to_end_delay, flow_idx):
        # Compute the reward given an end-to-end delay
        reward = 0
        if end_to_end_delay <= self.latency_target[flow_idx]:
            reward = self.low_reward + (1 - end_to_end_delay / self.latency_target[flow_idx]) * (
                    self.high_reward - self.low_reward)
        return reward

    def step(self, action):
        state = []
        self.time += self.interval
        scheduler_arrival = []
        # Forward the packets into the network and compute the time they reach the destination.
        for flow_idx, (flow_arrival, a) in enumerate(zip(self.arrival_pattern, action)):
            new_arrival = []
            while len(flow_arrival) > 0 and flow_arrival[0] < self.time:
                new_arrival.append(flow_arrival.pop(0))
            _, token, departure = self.token_buckets[flow_idx].forward(new_arrival, 1)
            state.append(token)
            backlog1, _, departure = self.reprofilers[flow_idx][0].forward(departure, a)
            backlog2, _, departure = self.reprofilers[flow_idx][1].forward(departure, a)
            state.append(backlog1 + backlog2)
            scheduler_arrival.append(departure)
        backlog, departure = self.scheduler.forward(scheduler_arrival)
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
        return state, reward, terminate, exceed_target

    def reset(self):
        for token_bucket, reprofiler in zip(self.token_buckets, self.reprofilers):
            token_bucket.reset()
            reprofiler[0].reset()
            reprofiler[1].reset()
        self.time = 0
        self.packet_count = [0] * len(self.arrival_pattern)
        self.departure_time = [[] for _ in range(len(self.arrival_pattern))]
        state = []
        for f in self.flow_profile:
            state.append(f[1] + 1)
            state.append(0)
        state.append(0)
        return state


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
        self.residual_delay = -1
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
            delay = self.residual_delay if self.residual_delay != -1 else 1 / self.bandwidth
            next_depart = max(next_arrival, self.depart) + delay
            # Check whether the packet depart at this next time step.
            if next_depart >= self.time:
                self.backlog.insert(0, next_arrival)
                self.backlog_flow.insert(0, next_flow)
                self.residual_delay = next_depart - self.time
                break
            else:
                self.depart = next_depart
                departure[next_flow].append(next_depart)
                self.residual_delay = -1
        return len(self.backlog), departure

    def reset(self):
        self.backlog = []
        self.backlog_flow = []
        self.time = 0
        self.depart = 0
        self.residual_delay = -1
        return
