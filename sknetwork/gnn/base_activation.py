#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
import numpy as np


class BaseActivation:
    """Base class for activation functions.
    Parameters
    ----------
    name : str
        Name of the activation function.
    """
    def __init__(self, name: str = 'custom'):
        self.name = name

    @staticmethod
    def output(signal: np.ndarray) -> np.ndarray:
        """Output of the activation function.

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples, n_channels)
            Input signal.

        Returns
        -------
        output : np.ndarray, shape (n_samples, n_channels)
            Output signal.
        """
        output = signal
        return output

    @staticmethod
    def gradient(signal: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Gradient of the activation function.

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples, n_channels)
            Input signal.
        direction : np.ndarray, shape (n_samples, n_channels)
            Direction where the gradient is taken.

        Returns
        -------
        gradient : np.ndarray, shape (n_samples, n_channels)
            Gradient.
        """
        gradient = direction
        return gradient


class BaseLoss(BaseActivation):
    """Base class for loss functions."""
    @staticmethod
    def loss(signal: np.ndarray, labels: np.ndarray) -> float:
        """Get the loss value.

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples, n_channels)
            Input signal (before activation).
        labels : np.ndarray, shape (n_samples)
            True labels.
        """
        return 0

    @staticmethod
    def loss_gradient(signal: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Gradient of the loss function.

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples, n_channels)
            Input signal.
        labels : np.ndarray, shape (n_samples,)
            True labels.

        Returns
        -------
        gradient : np.ndarray, shape (n_samples, n_channels)
            Gradient.
        """
        gradient = np.ones_like(signal)
        return gradient
