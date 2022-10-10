#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 22 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""

from typing import Callable

import numpy as np


def cross_entropy_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    """Computes cross entropy loss.

    For a single sample with binary label :math:`y` and probability estimate :math:`p`, the cross-entropy loss
    is :math:`-(y \\log (p) + (1 - y) \\log (1 - p))`.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    probs : np.ndarray
        Predicted probabilities.
    eps: float (default = 1e-15)
        Clipping parameter so that probabilities are in :math:`(\\eps, 1 - \\eps)`.

    Returns
    -------
    loss: float
        Loss value.

    References
    ----------
    Bishop, C. M. (2006).
    `Pattern recognition and machine learning.
    <https://www.academia.edu/download/30428242/bg0137.pdf>`_
    """
    n = len(y_true)

    # Clipping
    probs = np.clip(probs, eps, 1 - eps)

    if probs.ndim == 1:
        loglikelihood = y_true.dot(np.log(probs.T)) + (1 - y_true).dot(np.log((1 - probs)).T)
    else:
        nb_classes = probs.shape[1]
        y_true_ohe = np.eye(nb_classes)[y_true]
        probs /= probs.sum(axis=1)[:, np.newaxis]
        loglikelihood = np.sum(y_true_ohe * np.log(probs))

    loss = -loglikelihood / n

    return loss


def cross_entropy_prime(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """ Derivative of log loss function.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    probs : np.ndarray
        Predicted probabilities.
    eps: float (default = 1e-15)
        Clipping parameter so that probabilities are in :math:`(\\eps, 1 - \\eps)`.
    """
    # Clipping
    probs = np.clip(probs, eps, 1 - eps)

    if probs.shape[0] == 1:
        prime_loss = -(y_true / probs) + (1 - y_true) / (1 - probs)
    else:
        prime_loss = probs - y_true

    return prime_loss


def get_loss_function(loss_name: str = 'CrossEntropyLoss') -> Callable[..., float]:
    """Returns loss function according to `loss_name`.

    Parameters
    ----------
    loss_name : str
        Which loss function to use. Can be ``'CrossEntropyLoss'``.

    Returns
    -------
    Callable[..., float]
        Loss function

    Raises
    ------
    ValueError
        Error raised if loss does not exist.
    """
    loss_name = loss_name.lower()
    if loss_name == 'crossentropyloss':
        return cross_entropy_loss
    else:
        raise ValueError("Loss must be either \"CrossEntropyLoss\"")


def get_prime_loss_function(loss_name: str = 'CrossEntropyLoss') -> Callable[..., np.ndarray]:
    """Returns loss function derivative according to `loss_name`.

    Parameters
    ----------
    loss_name : str
        Which loss function derivative to use. Can be ``'CrossEntropyLoss'``.

    Returns
    -------
    Callable[..., np.ndarray]
        Loss function derivative.

    Raises
    ------
    ValueError
        Error raised if loss does not exist.
    """
    loss_name = loss_name.lower()
    if loss_name == 'crossentropyloss':
        return cross_entropy_prime
    else:
        raise ValueError("Loss must be either \"CrossEntropyLoss\"")
