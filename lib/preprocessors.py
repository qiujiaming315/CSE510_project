"""Suggested Preprocessors."""
import torch
import numpy as np

from lib.core import Preprocessor


class NetworkStatePreprocessor(Preprocessor):
    """Processor for the network state."""

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        state, action, reward, next_state = [], [], [], []
        for sample in samples:
            state.append(self.process_state_for_network(sample.state))
            action.append(sample.action)
            reward.append(self.process_reward(sample.reward))
            next_state.append(self.process_state_for_network(sample.next_state))
        state = torch.from_numpy(np.array(state, dtype=np.float32))
        action = torch.from_numpy(np.array(action, dtype=int))
        reward = torch.from_numpy(np.array(reward, dtype=np.float32))
        next_state = torch.from_numpy(np.array(next_state, dtype=np.float32))
        return state, action, reward, next_state
