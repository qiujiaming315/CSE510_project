"""Common functions you may find useful in your implementation."""

import torch
import os
import torch.nn as nn


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


def get_hard_target_model_updates(target, source):
    """Return list of target model update ops.

    These are hard target updates. The source weights are copied
    directly to the target network.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    # hard_updates = []
    # for t, s in zip(target, source):
    #     hard_updates.append((t, s))
    # return torch.nn.ModuleList(hard_updates)
    target.load_state_dict(source.state_dict())


def save_checkpoint(model, path, name="last"):
    torch.save(model.state_dict(), os.path.join(path, f"checkpoint_{name}.pth"))
