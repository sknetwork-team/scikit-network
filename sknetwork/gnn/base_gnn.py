#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""

import numpy as np
from collections import defaultdict

from sknetwork.gnn.layers import GCNConv
from sknetwork.gnn.activation import *
from sknetwork.gnn.loss import *
from sknetwork.gnn.optimizer import *
from sknetwork.utils.base import Algorithm
from sknetwork.utils.verbose import VerboseMixin


class BaseGNNClassifier(Algorithm, VerboseMixin):
    """Base class for GNN classifiers.

    Parameters
    ----------
    opt: str (default='Adam')
        Optimizer name:
        - 'SGD', stochastic gradient descent.
        - 'Adam', refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba.
        - 'none', gradient descent.

    Attributes
    ----------
    dW, db: np.ndarray, np.ndarray
        Derivative of trainable weight matrix and bias vector.
    layers: list
        List of layers object (e.g `GCNConv`).
    nb_layers: int
        Number of layers in model.
    activations: list
        List of layer activation function.
    labels_: np.ndarray
        Predicted node labels.
    history_: dict
        Training history per epoch: {'embedding', 'loss', 'train_accuracy', 'test_accuracy'}.
    """

    def __init__(self, opt: str = 'Adam', verbose: bool = False, **kwargs):
        VerboseMixin.__init__(self, verbose)
        self.opt = optimizer_factory(self, opt, **kwargs)
        self.layers = []
        self.train_mask = None
        self.test_mask = None
        self.nb_layers = 0
        self.activations = []
        self.dW, self.db = [], []
        self.labels_ = None
        self.history_ = defaultdict(list)

    def fit(self, *args, **kwargs):
        """Fit Algorithm to the data."""
        raise NotImplementedError

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        self.fit(*args, **kwargs)
        return self.labels_

    def _get_layers(self) -> list:
        """Get layer objects in model.

        Returns
        -------
        list of layer objects.
        """
        available_types = [GCNConv]

        return [l for l in list(self.__dict__.values()) if any([isinstance(l, t) for t in available_types])]

    @staticmethod
    def _get_activations(layers: list) -> list:
        """Get activation functions for each layer in model.

        Parameters
        ----------
        layers: list
            list of layer objects in model.

        Returns
        -------
        list of activation functions as strings.
        """
        return [l.activation for l in layers]

    def backward(self, feat: np.ndarray, y_true: np.ndarray, loss: str):
        """ Compute backpropagation.

        Parameters
        ----------
        feat : np.ndarray
            Input feature of shape (n, d) with n the number of nodes in the graph and d the size of the embedding.
        y_true : np.ndarray
            Label vectors of lenght n, with n the number of nodes in `adjacency`
        loss : str
            Loss function name.
        """
        n = len(y_true)
        prime_loss_function = get_prime_loss_function(loss)

        # Get information from model
        self.layers = self._get_layers()
        self.nb_layers = len(self.layers)
        activations = self._get_activations(self.layers)
        activation_primes = [DERIVATIVES[act] for act in activations]

        # Initialize parameters derivatives
        self.dW = [0] * self.nb_layers
        self.db = [0] * self.nb_layers

        # Backpropagation
        A = self.layers[self.nb_layers - 1].emb
        if self.layers[-1].activation == 'softmax':
            nb_classes = len(np.unique(y_true))
            y_true_ohe = np.eye(nb_classes)[y_true]
            dZ = (A - y_true_ohe).T
        else:
            dA = prime_loss_function(y_true, A.T)
            dZ = dA * activation_primes[-1](A.T)
        A_prev = self.layers[self.nb_layers - 2].emb

        dW = dZ.dot(A_prev).T
        db = np.sum(dZ, axis=1, keepdims=True) / n
        dA_prev = self.layers[self.nb_layers - 1].W.dot(dZ)

        self.dW[self.nb_layers - 1] = dW
        self.db[self.nb_layers - 1] = db.T

        for l in range(self.nb_layers - 1, 0, -1):
            dZ = dA_prev * activation_primes[l - 1](A_prev.T)
            if l == 1:
                A_prev = feat
            dW = A_prev.T.dot(dZ.T)  # SpM on left hand side
            db = np.sum(dZ, axis=1, keepdims=True) / n
            if l > 1:
                dA_prev = self.layers[l - 1].W.dot(dZ)

            self.dW[l - 1] = dW
            self.db[l - 1] = db.T

    def __repr__(self) -> str:
        """ String representation of object

        Returns
        -------
        str
            String representation of object
        """

        lines = ''

        layers = self._get_layers()
        lines += f'{self.__class__.__name__}(\n'

        for l in layers:
            lines += f'  {l}\n'
        lines += ')'

        return lines
