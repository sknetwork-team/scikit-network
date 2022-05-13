#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 22 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""

import numpy as np


def cross_entropy_loss(y_true: np.ndarray, logits: np.ndarray, eps: float = 1e-15) -> float:
    """Computes cross entropy loss, i.e log-loss.

    For a single sample with label :math:`y \in \{0, 1\}`, and a probability estimate :math:`p`, the log-loss
    is as follows;
    :math:`L_{\log}(y, p) = -(y \log (p) + (1 - y) \log (1 - p))`

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    logits : np.ndarray
        Predicted probabilites.
    eps: float (default=1e-15)
        Binary cross entropy is undefined for p=0 or p=1, thus predicated probabilities are clipped wrt to `eps`.

    Returns
    -------
    loss: float
    """
    n = len(y_true)

    # Clipping
    logits = np.clip(logits, eps, 1 - eps)

    if logits.ndim == 1:
        logprobs = y_true.dot(np.log(logits.T)) + (1 - y_true).dot(np.log((1 - logits)).T)
    else:
        nb_classes = logits.shape[1]
        y_true_ohe = np.eye(nb_classes)[y_true]
        logits /= logits.sum(axis=1)[:, np.newaxis]
        logprobs = np.sum(y_true_ohe * np.log(logits))

    return -logprobs / n


def cross_entropy_prime(y_true: np.ndarray, logits: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """ Derivative of log loss function.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    logits : np.ndarray
        Predicted probabilites.
    eps: float (default=1e-15)
        Binary cross entropy is undefined for p=0 or p=1, thus predicated probabilities are clipped wrt to `eps`.
    """
    # Clipping
    logits = np.clip(logits, eps, 1 - eps)

    if logits.shape[0] == 1:
        prime_loss = -(y_true / logits) + (1 - y_true) / (1 - logits)
    else:
        prime_loss = logits - y_true

    return prime_loss


def get_loss_function(loss_name: str) -> object:
    """Returns loss function according to `loss_name`.

    Parameters
    ----------
    loss_name : str
        Loss function name

    Returns
    -------
    object
        Loss function

    Raises
    ------
    ValueError
        Error raised if loss does not exists.
    """
    if loss_name == 'CrossEntropyLoss':
        return cross_entropy_loss
    else:
        raise ValueError("Loss must be either \"CrossEntropyLoss\"")


def get_prime_loss_function(loss_name: str) -> object:
    """Returns loss function derivative according to `loss_name`.

    Parameters
    ----------
    loss_name : str
        Loss function name.

    Returns
    -------
    object
        Loss function derivative.

    Raises
    ------
    ValueError
        Error raised if loss does not exists.
    """
    if loss_name == 'CrossEntropyLoss':
        return cross_entropy_prime
    else:
        raise ValueError("Loss must be either \"CrossEntropyLoss\"")