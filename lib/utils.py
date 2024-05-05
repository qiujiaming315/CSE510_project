"""Common functions you may find useful in your implementation."""

import torch
import numpy as np
import os
from matplotlib import pyplot as plt


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


def plot_loss(loss_data, save_path, cycle=100):
    # Plot the loss values, average every cycle loss values.
    loss_data = np.array(loss_data)
    loss_data = np.mean(loss_data.reshape(-1, cycle), axis=1)
    x_data = (np.arange(len(loss_data)) + 1) * cycle
    plt.plot(x_data, loss_data)
    plt.ylabel('loss value')
    plt.savefig(os.path.join(save_path, "loss.png"))
    # plt.show()
    plt.clf()


def save_checkpoint(model, path):
    torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
