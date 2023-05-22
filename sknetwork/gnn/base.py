#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2022
@author: Simon Delarue <sdelarue@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
from abc import ABC

from typing import Union

import numpy as np
from collections import defaultdict
from scipy import sparse

from sknetwork.gnn.loss import BaseLoss, get_loss
from sknetwork.gnn.optimizer import BaseOptimizer, get_optimizer
from sknetwork.base import Algorithm
from sknetwork.log import Log


class BaseGNN(ABC, Algorithm, Log):
    """Base class for GNNs.

    Parameters
    ----------
    loss : str or custom loss (default = ``'Cross entropy'``)
        Loss function.
    optimizer : str or custom optimizer (default = ``'Adam'``)
        Optimizer used for training.

        * ``'Adam'``, a stochastic gradient-based optimizer.
        * ``'GD'``, gradient descent.
    learning_rate : float
        Learning rate.
    verbose : bool
        Verbose mode

    Attributes
    ----------
    layers: list
        List of layers.
    labels_: np.ndarray
        Predicted labels.
    history_: dict
        Training history per epoch: {'embedding', 'loss', 'train_accuracy', 'test_accuracy'}.
    """
    def __init__(self, loss: Union[BaseLoss, str] = 'CrossEntropy', optimizer: Union[BaseOptimizer, str] = 'Adam',
                 learning_rate: float = 0.01, verbose: bool = False):
        Log.__init__(self, verbose)
        self.optimizer = get_optimizer(optimizer, learning_rate)
        self.loss = get_loss(loss)
        self.layers = []
        self.derivative_weight = []
        self.derivative_bias = []
        self.train_mask = None
        self.test_mask = None
        self.val_mask = None
        self.embedding_ = None
        self.output_ = None
        self.labels_ = None
        self.history_ = defaultdict(list)

    def fit(self, *args, **kwargs):
        """Fit Algorithm to the data."""
        raise NotImplementedError

    def predict(self):
        """Return the predicted labels."""
        return self.labels_

    def fit_predict(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels of the nodes.
        """
        self.fit(*args, **kwargs)
        return self.predict()

    def predict_proba(self):
        """Return the probability distribution over labels."""
        probs = self.output_
        if probs is not None:
            if probs.shape[1] == 1:
                probs = np.vstack(1 - probs, probs)
        return probs

    def fit_predict_proba(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the distribution over labels. Same parameters as the ``fit`` method.

        Returns
        -------
        probs : np.ndarray
            Probability distribution over labels.
        """
        self.fit(*args, **kwargs)
        return self.predict_proba()

    def transform(self):
        """Return the embedding of nodes."""
        return self.embedding_

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the embedding of the nodes. Same parameters as the ``fit`` method.

        Returns
        -------
        embedding : np.ndarray
            Embedding of the nodes.
        """
        self.fit(*args, **kwargs)
        return self.transform()

    def backward(self, features: sparse.csr_matrix, labels: np.ndarray, mask: np.ndarray):
        """Compute backpropagation.

        Parameters
        ----------
        features : sparse.csr_matrix
            Features, array of shape (n_nodes, n_features).
        labels : np.ndarray
            Labels, array of shape (n_nodes,).
        mask: np.ndarray
            Boolean mask, array of shape (n_nodes,).
        """
        derivative_weight = []
        derivative_bias = []

        # discard missing labels
        mask = mask & (labels >= 0)
        labels = labels[mask]

        # backpropagation
        n_layers = len(self.layers)
        layers_reverse: list = list(reversed(self.layers))
        signal = layers_reverse[0].embedding
        signal = signal[mask]
        gradient = layers_reverse[0].activation.loss_gradient(signal, labels)

        for i in range(n_layers):
            if i < n_layers - 1:
                signal = layers_reverse[i + 1].output
            else:
                signal = features
            signal = signal[mask]

            derivative_weight.append(signal.T.dot(gradient))
            derivative_bias.append(np.mean(gradient, axis=0, keepdims=True))

            if i < n_layers - 1:
                signal = layers_reverse[i + 1].embedding
                signal = signal[mask]
                direction = layers_reverse[i].weight.dot(gradient.T).T
                gradient = layers_reverse[i + 1].activation.gradient(signal, direction)

        self.derivative_weight = list(reversed(derivative_weight))
        self.derivative_bias = list(reversed(derivative_bias))

    def _check_fitted(self):
        if self.output_ is None:
            raise ValueError("This embedding instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments before using this method.")
        else:
            return self

    def __repr__(self) -> str:
        """String representation of the `GNN`, layers by layers.

        Returns
        -------
        str
            String representation of object.
        """
        string = f'{self.__class__.__name__}(\n'
        for layer in self.layers:
            string += f'  {layer}\n'
        string += ')'
        return string
