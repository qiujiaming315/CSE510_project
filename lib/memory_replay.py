import numpy as np

from lib.core import Sample, ReplayMemory


class NetworkReplayMemory(ReplayMemory):
    """An implementation of memory replay using a ring buffer."""

    def __init__(self, max_size):
        super().__init__(max_size)
        self.index = 0
        self.full = False
        self.final_state = dict()

    def __getitem__(self, index):
        state, next_state, is_terminal = self.get_state(index)
        action = self.action_buffer[index]
        reward = self.reward_buffer[index]
        return Sample(state, action, reward, next_state, is_terminal)

    def set_index(self, index):
        return index % self.max_size

    def get_state(self, index):
        state = self.state_buffer[self.set_index(index)]
        # Check if the next state is the final state.
        next_state = self.final_state.get(index, None)
        is_terminal = next_state is not None
        if not is_terminal:
            next_state = self.state_buffer[self.set_index(index + 1)]
        return state, next_state, is_terminal

    def append(self, state, action, reward):
        # Store the state, action and reward.
        self.state_buffer[self.index] = state
        self.action_buffer[self.index] = action
        self.reward_buffer[self.index] = reward
        # Overwrite previously stored final state.
        if self.index in self.final_state.keys():
            del self.final_state[self.index]
        # Update index of the ring buffer.
        self.index = self.set_index(self.index + 1)
        if self.index == 0:
            self.full = True
        return

    def end_episode(self, final_state):
        self.final_state[self.set_index(self.index - 1)] = final_state
        return

    def sample(self, batch_size, indexes=None):
        if indexes is None:
            # If the replay buffer is not full, sample only from the filled part.
            sample_range = self.max_size if self.full else self.index
            # If the current index is not a final state, avoid sampling the current index.
            if self.set_index(self.index - 1) not in self.final_state.keys():
                indexes = np.random.randint(sample_range - 1, size=batch_size)
                indexes = np.where(indexes >= self.set_index(self.index - 1), indexes + 1, indexes)
            else:
                indexes = np.random.randint(sample_range, size=batch_size)
        assert len(indexes) == batch_size
        # Randomly sample from the replay buffer.
        samples = [self.__getitem__(index) for index in indexes]
        return samples

    def clear(self):
        # Clear the replay buffer and restore internal state.
        self.state_buffer = [None] * self.max_size
        self.action_buffer = [None] * self.max_size
        self.reward_buffer = [None] * self.max_size
        self.index = 0
        self.final_state = dict()
