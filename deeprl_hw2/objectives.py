"""Loss functions."""

import torch
# import tensorflow as tf
# import semver


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor, torch.Tensor
      Target value.
    y_pred: np.array, tf.Tensor, torch.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor, torch.Tensor
      The huber loss.
    """
    residue = torch.abs(y_pred - y_true)
    check = residue <= max_grad

    squared_loss = 0.5 * residue ** 2
    linear_loss = max_grad * (residue - 0.5 * max_grad)

    return torch.where(check, squared_loss, linear_loss)


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor, torch.Tensor
      Target value.
    y_pred: np.array, tf.Tensor, torch.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor, torch.Tensor
      The mean huber loss.
    """
    loss = huber_loss(y_true, y_pred, max_grad)

    return torch.mean(loss)
