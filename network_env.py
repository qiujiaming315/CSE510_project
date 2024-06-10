import copy
import heapq
import numpy as np
from dataclasses import dataclass
from enum import Enum

from network_component import NetworkComponent, TokenBucket, MultiSlopeShaper, FIFOScheduler


class NetworkEnv:
    """A network environment for RL sampling similar to a openai.gym environment."""

    def __init__(self, flow_profile, flow_path, reprofiling_delay, interval, terminate_time=1000, sleep_prob=0.1,
                 high_reward=1, low_reward=0.1, penalty=-10, tor=0.003):
        flow_profile = np.array(flow_profile)
        flow_path = np.array(flow_path)
        reprofiling_delay = np.array(reprofiling_delay)
        self.flow_profile = flow_profile
        self.num_flow = len(flow_profile)
        self.num_link = flow_path.shape[1]
        assert self.num_flow == len(flow_path), "Inconsistent number of flows in flow profile and flow path."
        self.reprofiling_delay = reprofiling_delay
        self.interval = interval
        self.terminate_time = terminate_time
        self.sleep_prob = sleep_prob
        self.high_reward = high_reward
        self.low_reward = low_reward
        self.penalty = penalty
        self.arrival_pattern = self.generate_arrival_pattern()
        self.arrival_time = copy.deepcopy(self.arrival_pattern)
        # Compute the expected latency bound (with small tolerance for numerical instability).
        packetization_delay = 2 * np.sum(flow_path > 0, axis=1) * reprofiling_delay / flow_profile[:, 1]
        self.latency_target = (flow_profile[:, 2] + packetization_delay) * (1 + tor)
        # Configure the network components.
        reprofiling_rate = flow_profile[:, 1] / reprofiling_delay
        reprofiling_burst = flow_profile[:, 1] - flow_profile[:, 0] * reprofiling_delay
        self.token_buckets = [TokenBucket(f[0], f[1], flow_idx, False) for flow_idx, f in enumerate(flow_profile)]
        self.reprofilers = [[MultiSlopeShaper(flow_idx) for flow_idx in range(self.num_flow)] for _ in
                            range(self.num_link)]
        self.schedulers = []
        for link_idx in range(self.num_link):
            link_flow_mask = flow_path[:, link_idx] > 0
            link_bandwidth = np.sum(reprofiling_rate[link_flow_mask])
            self.schedulers.append(FIFOScheduler(link_bandwidth, self.num_flow))
            for flow_idx in np.arange(self.num_flow)[link_flow_mask]:
                self.reprofilers[link_idx][flow_idx] = MultiSlopeShaper(flow_idx,
                                                                        TokenBucket(flow_profile[flow_idx, 0],
                                                                                    reprofiling_burst[flow_idx]),
                                                                        TokenBucket(reprofiling_rate[flow_idx], 0))
        # Connect the network components.
        self.flow_path = []
        for flow_idx, path in enumerate(flow_path):
            flow_links = np.where(path)[0]
            assert len(flow_links) > 0, "Every flow should traverse at least one hop."
            flow_links = flow_links[np.argsort(path[flow_links])]
            self.flow_path.append(flow_links)
            self.token_buckets[flow_idx].next = self.reprofilers[flow_links[0]][flow_idx]
            # Append an empty component as the terminal.
            self.schedulers[flow_links[-1]].terminal[flow_idx] = True
            self.schedulers[flow_links[-1]].next[flow_idx] = NetworkComponent()
            for link_idx in flow_links:
                self.reprofilers[link_idx][flow_idx].next = self.schedulers[link_idx]
            for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                self.schedulers[cur_link].next[flow_idx] = self.reprofilers[next_link][flow_idx]
        # Set the internal variables.
        self.time = 0
        self.packet_count = [0] * self.num_flow
        self.departure_time = [[[] for _ in range(len(self.arrival_time[flow_idx]))] for flow_idx in
                               range(self.num_flow)]
        self.event_pool = []
        # Add packet arrival events to the event pool.
        for flow_idx, flow_arrival in enumerate(self.arrival_pattern):
            for arrival in flow_arrival:
                event = Event(arrival, EventType.ARRIVAL, flow_idx, self.token_buckets[flow_idx])
                heapq.heappush(self.event_pool, event)
        # Add a summary event at each time interval to collect a snapshot of the network.
        for time_step in np.arange(interval, terminate_time + interval, interval):
            event = Event(time_step, EventType.SUMMARY)
            heapq.heappush(self.event_pool, event)
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
        end_to_end = [[] for _ in range(self.num_flow)]
        # Enforce the reprofiling control actions
        for flow_idx, a in enumerate(action):
            flow_links = self.flow_path[flow_idx]
            for link_idx in flow_links:
                self.reprofilers[link_idx][flow_idx].activate(a, self.time)
                # Add a packet forward event if the shaper is turned off.
                if not a:
                    for tb in self.reprofilers[link_idx][flow_idx].token_buckets:
                        event = Event(self.time, EventType.FORWARD, component=tb)
                        heapq.heappush(self.event_pool, event)
        self.time += self.interval
        while len(self.event_pool) > 0:
            event = heapq.heappop(self.event_pool)
            if event.event_type == EventType.ARRIVAL:
                # Start a busy period by creating a forward event if the component is idle upon arrival.
                if event.component.arrive(event.time, event.flow_idx):
                    forward_event = Event(event.time, EventType.FORWARD, component=event.component)
                    heapq.heappush(self.event_pool, forward_event)
            elif event.event_type == EventType.FORWARD:
                next_depart, idle, (flow_idx, packet_number, next_component) = event.component.forward(event.time)
                # Submit the next forward event if the component is currently busy.
                if not idle:
                    forward_event = Event(next_depart, EventType.FORWARD, component=event.component)
                    heapq.heappush(self.event_pool, forward_event)
                # Create a packet arrival event for the next component.
                departed = next_component is not None
                is_internal = isinstance(event.component, TokenBucket) and event.component.internal_component
                is_terminal = isinstance(event.component, FIFOScheduler) and event.component.terminal[flow_idx]
                if departed:
                    if isinstance(next_component, MultiSlopeShaper) and not is_internal:
                        # Create an arrival event for every token bucket from the multi-slope shaper.
                        for tb in next_component.token_buckets:
                            arrival_event = Event(event.time, EventType.ARRIVAL, flow_idx, tb)
                            heapq.heappush(self.event_pool, arrival_event)
                    elif not is_terminal:
                        arrival_event = Event(event.time, EventType.ARRIVAL, flow_idx, next_component)
                        heapq.heappush(self.event_pool, arrival_event)
                    # Record the packet departure time.
                    if not is_internal:
                        self.departure_time[flow_idx][packet_number - 1].append(event.time)
                    # Update the packet count and compute end-to-end latency.
                    if is_terminal:
                        end_to_end[flow_idx].append(event.time - self.arrival_time[flow_idx][packet_number - 1])
                        self.packet_count[flow_idx] += 1
            elif event.event_type == EventType.SUMMARY:
                # Record the network status.
                for state, tb in zip(states, self.token_buckets):
                    token_num, _ = tb.peek(event.time)
                    state.append(token_num)
                reprofiler_backlog = [[0] * self.num_link for _ in range(self.num_flow)]
                scheduler_backlog = [[0] * self.num_link for _ in range(self.num_flow)]
                for flow_idx, flow_links in enumerate(self.flow_path):
                    for link_idx in flow_links:
                        rb = self.reprofilers[link_idx][flow_idx].peek(event.time)
                        reprofiler_backlog[flow_idx][link_idx] = rb
                        sb = self.schedulers[link_idx].peek(event.time)
                        scheduler_backlog[flow_idx][link_idx] = sb
                for state, rb, sb in zip(states, reprofiler_backlog, scheduler_backlog):
                    state.extend(rb)
                    state.extend(sb)
                break
        # Compute the reward based on the end-to-end latency and determine whether the episode terminates.
        terminate, exceed_target = True, False
        reward = 0
        for flow_idx, end in enumerate(end_to_end):
            flow_reward = 0
            for e in end:
                flow_reward += self.reward_function(e, flow_idx)
                if e > self.latency_target[flow_idx]:
                    exceed_target = True
            if self.packet_count[flow_idx] < len(self.arrival_time[flow_idx]):
                terminate = False
            flow_reward = 0 if len(end) == 0 else flow_reward / len(end)
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
        for token_bucket in self.token_buckets:
            token_bucket.reset()
        for link_reprofiler, scheduler in zip(self.reprofilers, self.schedulers):
            for reprofiler in link_reprofiler:
                reprofiler.reset()
            scheduler.reset()
        self.time = 0
        self.packet_count = [0] * len(self.arrival_pattern)
        self.departure_time = [[[] for _ in range(len(self.arrival_time[flow_idx]))] for flow_idx in
                               range(self.num_flow)]
        self.event_pool = []
        # Add packet arrival events to the event pool.
        for flow_idx, flow_arrival in enumerate(self.arrival_pattern):
            for arrival in flow_arrival:
                event = Event(arrival, EventType.ARRIVAL, flow_idx, self.token_buckets[flow_idx])
                heapq.heappush(self.event_pool, event)
        # Add a summary event at each time interval to collect a snapshot of the network.
        for time_step in np.arange(self.interval, self.terminate_time + self.interval, self.interval):
            event = Event(time_step, EventType.SUMMARY)
            heapq.heappush(self.event_pool, event)
        # Set the initial state.
        states = [[0] * (2 * self.num_link + 1) for _ in range(self.num_flow)]
        for state, f in zip(states, self.flow_profile):
            state[0] = f[1]
        return states


class EventType(Enum):
    SUMMARY = 1
    FORWARD = 2
    ARRIVAL = 3


@dataclass
class Event:
    time: float
    event_type: EventType
    flow_idx: int = 0
    component: NetworkComponent = NetworkComponent()

    def __lt__(self, other):
        return (self.time, self.event_type.value) < (other.time, other.event_type.value)
