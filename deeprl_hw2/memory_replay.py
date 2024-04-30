import numpy as np

from deeprl_hw2.core import Sample, ReplayMemory


class AtariReplayMemory(ReplayMemory):
    """An implementation of memory replay using a ring buffer."""

    def __init__(self, max_size, window_length, input_shape):
        super().__init__(max_size, window_length)
        self.input_shape = input_shape
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
        # Get the frames for a fixed window size.
        state = []
        for i in range(self.window_length):
            frame = self.state_buffer[self.set_index(index - i)]
            # Check if the sample spans uninitialized region.
            if frame is None:
                frame = np.zeros(self.input_shape, dtype=np.uint8)
            state.append(frame)
        state.reverse()
        state = np.array(state)
        # If the next state is the final state, retrieve the next frame from the final states.
        next_frame = self.final_state.get(index, None)
        is_terminal = next_frame is not None
        if next_frame is None:
            next_frame = self.state_buffer[self.set_index(index + 1)]
        # Set frame to zero if any of the previous state is the final state.
        for i in range(1, self.window_length):
            if self.set_index(index - i) in self.final_state.keys():
                state[0:self.window_length - i] = 0
                break
        # Stack the next frame onto the current state.
        next_state = np.concatenate((state[1:], next_frame[np.newaxis, :, :]), axis=0)
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

    def end_episode(self, final_state, is_terminal):
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
