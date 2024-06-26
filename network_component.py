import numpy as np


class NetworkComponent:

    def __init__(self):
        self.next = None
        self.idle = True
        return

    def arrive(self, time, component_idx):
        """Method to add an arriving packet to backlog."""
        return

    def forward(self, time):
        """Method to release a packet from the backlog."""
        return

    def peek(self, time):
        """Method to check the status of the network component."""
        return

    def reset(self):
        self.idle = True
        return


class TokenBucket(NetworkComponent):

    def __init__(self, rate, burst, component_idx=0, internal_component=True):
        self.rate = rate
        self.burst = burst
        self.component_idx = component_idx
        self.internal_component = internal_component
        self.active = True
        self.backlog = []
        self.token = burst
        self.depart = 0
        self.packet_count = 0
        super().__init__()
        return

    def arrive(self, time, component_idx):
        self.backlog.append(time)
        return self.idle

    def forward(self, time):
        if len(self.backlog) == 0:
            # Redundant forward event. Ignore.
            return time, self.idle, (0, 0, None)
        component_idx, next_component = 0, None
        if self.idle:
            # Initiate a busy period.
            if self.active:
                token, _ = self.peek(time)
                self.token = token
                self.depart = time
            self.idle = False
        else:
            # Release the forwarded packet.
            self.backlog.pop(0)
            self.packet_count += 1
            if self.active:
                self.token += self.rate * (time - self.depart) - 1
                self.depart = time
            component_idx, next_component = self.component_idx, self.next
            if len(self.backlog) == 0:
                # Terminate a busy period.
                self.idle = True
                return time, self.idle, (component_idx, self.packet_count, next_component)
        # Examine the next packet.
        next_arrival = self.backlog[0]
        next_depart = time
        if self.active:
            delay = 0
            if self.token < 1:
                delay = (1 - self.token) / self.rate
            next_depart = max(next_arrival, self.depart) + delay
        return next_depart, self.idle, (component_idx, self.packet_count, next_component)

    def peek(self, time):
        # Update the token bucket state.
        token = min(self.token + self.rate * (time - self.depart), self.burst) if self.idle else 0
        return token, len(self.backlog)

    def activate(self, action, time):
        if action != self.active:
            if self.active:
                token, _ = self.peek(time)
                self.token = token
            self.depart = time
        self.active = action
        return

    def reset(self):
        self.active = True
        self.backlog = []
        self.token = self.burst
        self.depart = 0
        self.packet_count = 0
        super().reset()
        return


class MultiSlopeShaper(NetworkComponent):

    def __init__(self, flow_idx, *args):
        self.flow_idx = flow_idx
        # Set each token bucket from the input list.
        for tb_idx, tb in enumerate(args):
            assert isinstance(tb, TokenBucket), "Every argument passed into MultiSlopeShaper " \
                                                "must be a TokenBucket instance."
            tb.component_idx = tb_idx
            tb.next = self
        self.token_buckets = args
        self.eligible_packets = [[] for _ in range(len(args))]
        self.packet_count = 0
        super().__init__()
        return

    def arrive(self, time, component_idx):
        # A packet is eligible if released by all the token bucket shapers.
        self.eligible_packets[component_idx].append(time)
        return all(len(ep) > 0 for ep in self.eligible_packets)

    def forward(self, time):
        # Release an eligible packet.
        for ep in self.eligible_packets:
            ep.pop(0)
        self.packet_count += 1
        return time, True, (self.flow_idx, self.packet_count, self.next)

    def peek(self, time):
        # Return the maximum number of backlogged packets across all the token buckets.
        max_backlog = 0
        for tb in self.token_buckets:
            _, backlog = tb.peek(time)
            if backlog > max_backlog:
                max_backlog = backlog
        return max_backlog

    def activate(self, action, time):
        # Turn on or turn off all the token bucket shapers.
        for tb in self.token_buckets:
            tb.activate(action, time)
        return

    def reset(self):
        self.eligible_packets = [[] for _ in range(len(self.token_buckets))]
        self.packet_count = 0
        for tb in self.token_buckets:
            tb.reset()
        super().reset()
        return


class FIFOScheduler(NetworkComponent):

    def __init__(self, bandwidth, num_flow):
        self.bandwidth = bandwidth
        self.num_flow = num_flow
        self.backlog = []
        self.backlog_flow = []
        self.depart = 0
        self.packet_count = [0] * num_flow
        self.terminal = [False] * num_flow
        super().__init__()
        self.next = [None] * num_flow
        return

    def arrive(self, time, component_idx):
        # Add the packet and its flow index to the backlog.
        self.backlog.append(time)
        self.backlog_flow.append(component_idx)
        return self.idle

    def forward(self, time):
        if len(self.backlog) == 0:
            # Redundant forward event. Ignore.
            return time, self.idle, (0, 0, None)
        # Update the last packet departure time.
        self.depart = time
        flow_idx, next_component = 0, None
        if self.idle:
            # Initiate a busy period.
            self.idle = False
        else:
            # Release the forwarded packet.
            self.backlog.pop(0)
            flow_idx = self.backlog_flow.pop(0)
            self.packet_count[flow_idx] += 1
            next_component = self.next[flow_idx]
            if len(self.backlog) == 0:
                # Terminate a busy period.
                self.idle = True
                return time, self.idle, (flow_idx, self.packet_count[flow_idx], next_component)
        # Examine the next packet.
        next_arrival = self.backlog[0]
        next_depart = max(next_arrival, self.depart) + 1 / self.bandwidth
        return next_depart, self.idle, (flow_idx, self.packet_count[flow_idx], next_component)

    def peek(self, time):
        # Return the number of backlogged packets.
        return len(self.backlog)

    def reset(self):
        self.backlog = []
        self.backlog_flow = []
        self.depart = 0
        self.packet_count = [0] * self.num_flow
        super().reset()
        return
