#!/usr/bin/env python3
# coding: utf-8
"""
Created in April 2022
@author: Simon Delarue <sdelarue@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

from typing import Union

import numpy as np

from sknetwork.gnn.base_activation import BaseLoss
from sknetwork.gnn.activation import Sigmoid, Softmax


class CrossEntropy(BaseLoss, Softmax):
    """Cross entropy loss with softmax activation.

    For a single sample with value :math:`x` and true label :math:`y`, the cross-entropy loss
    is:

     :math:`-\\sum_i 1_{\\{y=i\\}}  \\log (p_i)`

     with

     :math:`p_i = e^{x_i} / \\sum_j e^{x_j}`.

    For :math:`n` samples, return the average loss.
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.name = 'Cross entropy'

    @staticmethod
    def loss(signal: np.ndarray, labels: np.ndarray) -> float:
        """Get loss value.

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples, n_channels)
            Input signal (before activation).
            The number of channels must be at least 2.
        labels : np.ndarray, shape (n_samples)
            True labels.

        Returns
        -------
        value : float
            Loss value.
        """
        n = len(labels)
        probs = Softmax.output(signal)

        # for numerical stability
        eps = 1e-10
        probs = np.clip(probs, eps, 1 - eps)

        value = -np.log(probs[np.arange(n), labels]).sum()

        return value / n

    @staticmethod
    def loss_gradient(signal: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Get the gradient of the loss function (including activation).

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples, n_channels)
            Input signal (before activation).
        labels : np.ndarray, shape (n_samples)
            True labels.
        Returns
        -------
        gradient: float
            Gradient of the loss function.
        """
        probs = Softmax.output(signal)
        one_hot_encoding = np.zeros_like(probs)
        one_hot_encoding[np.arange(len(labels)), labels] = 1
        gradient = probs - one_hot_encoding

        return gradient


class BinaryCrossEntropy(BaseLoss, Sigmoid):
    """Binary cross entropy loss with sigmoid activation.

    For a single sample with true label :math:`y` and predicted probability :math:`p`, the binary cross-entropy loss
    is:

     :math:`-y  \\log (p) - (1-y)  \\log (1 - p).`

    For :math:`n` samples, return the average loss.
    """
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()
        self.name = 'Binary cross entropy'

    @staticmethod
    def loss(signal: np.ndarray, labels: np.ndarray) -> float:
        """Get loss value.

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples, n_channels)
            Input signal (before activation).
            The number of channels must be at least 2.
        labels : np.ndarray, shape (n_samples)
            True labels.

        Returns
        -------
        value : float
            Loss value.
        """
        probs = Sigmoid.output(signal)
        n = len(labels)

        # for numerical stability
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)

        if probs.shape[1] == 1:
            # binary labels
            value = -np.log(probs[labels > 0]).sum()
            value -= np.log((1 - probs)[labels == 0]).sum()
        else:
            # general case
            value = -np.log(1 - probs)
            value[np.arange(n), labels] = -np.log(probs[np.arange(n), labels])
            value = value.sum()

        return value / n

    @staticmethod
    def loss_gradient(signal: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Get the gradient of the loss function (including activation).

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples, n_channels)
            Input signal (before activation).
        labels : np.ndarray, shape (n_samples)
            True labels.
        Returns
        -------
        gradient: float
            Gradient of the loss function.
        """
        probs = Sigmoid.output(signal)
        gradient = (probs.T - labels).T

        return gradient


def get_loss(loss: Union[BaseLoss, str] = 'CrossEntropyLoss') -> BaseLoss:
    """Instantiate loss function according to parameters.

    Parameters
    ----------
    loss : str or loss function.
        Which loss function to use. Can be ``'CrossEntropy'`` or ``'BinaryCrossEntropy'`` or custom loss.

    Returns
    -------
    Loss function object.
    """
    if issubclass(type(loss), BaseLoss):
        return loss
    elif type(loss) == str:
        loss = loss.lower().replace(' ', '')
        if loss in ['crossentropy', 'ce']:
            return CrossEntropy()
        elif loss in ['binarycrossentropy', 'bce']:
            return BinaryCrossEntropy()
        else:
            raise ValueError("Loss must be either \"CrossEntropy\" or \"BinaryCrossEntropy\".")
    else:
        raise TypeError("Loss must be either an \"BaseLoss\" object or a string.")
