#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from collections import defaultdict

import numpy as np

from sknetwork.gnn.activation import get_prime_activation_function
from sknetwork.gnn.layers import GCNConv
from sknetwork.gnn.loss import get_prime_loss_function
from sknetwork.gnn.optimizer import get_optimizer
from sknetwork.utils.verbose import VerboseMixin


class BaseGNNClassifier(VerboseMixin):
    """Base class for GNN classifiers.

    Parameters
    ----------
    opt: str (default = ``'Adam'``)
        Optimizer name:
        * ``'Adam'``, refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba.
        * ``'None'``, gradient descent.

    Attributes
    ----------
    prime_weight, prime_bias: np.ndarray, np.ndarray
        Derivatives of trainable weight matrix and bias vector.
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
        self.opt = get_optimizer(self, opt, **kwargs)
        self.layers = []
        self.train_mask = None
        self.test_mask = None
        self.nb_layers = 0
        self.activations = []
        self.prime_weight, self.prime_bias = [], []
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
            Labels of the nodes.
        """
        self.fit(*args, **kwargs)
        return self.labels_

    def _get_layers(self) -> list:
        """Get layer objects in model.

        Returns
        -------
        List of layer objects.
        """
        available_types = [GCNConv]

        return [layer for layer in list(self.__dict__.values()) if any([isinstance(layer, t) for t in available_types])]

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
        return [layer.activation for layer in layers]

    def backward(self, feat: np.ndarray, y_true: np.ndarray, loss: str):
        """Compute backpropagation.

        Parameters
        ----------
        feat : np.ndarray
            Input feature of shape :math:`(n, d)` with :math:`n` the number of nodes in the graph and :math:`d`
            the size of the embedding.
        y_true : np.ndarray
            Label vectors of length :math:`n`, with :math:`n` the number of nodes in `adjacency`.
        loss : str
            Loss function name.
        """
        n = len(y_true)
        prime_loss_function = get_prime_loss_function(loss)

        # Get information from model
        self.layers = self._get_layers()
        self.nb_layers = len(self.layers)
        activations = self._get_activations(self.layers)
        activation_primes = [get_prime_activation_function(act) for act in activations]

        # Initialize parameters derivatives
        self.prime_weight = [0] * self.nb_layers
        self.prime_bias = [0] * self.nb_layers

        # Backpropagation
        output = self.layers[self.nb_layers - 1].emb
        if self.layers[-1].activation == 'softmax':
            nb_classes = len(np.unique(y_true))
            y_true_ohe = np.eye(nb_classes)[y_true]
            prime_update = (output - y_true_ohe).T
        else:
            prime_output = prime_loss_function(y_true, output.T)
            prime_update = prime_output * activation_primes[-1](output.T)
        output_prev = self.layers[self.nb_layers - 2].emb

        prime_weight = prime_update.dot(output_prev).T
        prime_bias = np.sum(prime_update, axis=1, keepdims=True) / n
        prime_output_prev = self.layers[self.nb_layers - 1].weight.dot(prime_update)

        self.prime_weight[self.nb_layers - 1] = prime_weight
        self.prime_bias[self.nb_layers - 1] = prime_bias.T

        for layer_idx in range(self.nb_layers - 1, 0, -1):
            prime_update = prime_output_prev * activation_primes[layer_idx - 1](output_prev.T)
            if layer_idx == 1:
                output_prev = feat
            prime_weight = output_prev.T.dot(prime_update.T)  # SpM on left-hand side
            prime_bias = np.sum(prime_update, axis=1, keepdims=True) / n
            if layer_idx > 1:
                prime_output_prev = self.layers[layer_idx - 1].weight.dot(prime_update)

            self.prime_weight[layer_idx - 1] = prime_weight
            self.prime_bias[layer_idx - 1] = prime_bias.T

    def __repr__(self) -> str:
        """String representation of the `GNN`, layer by layer.

        Returns
        -------
        str
            String representation of object.
        """

        lines = ''

        layers = self._get_layers()
        lines += f'{self.__class__.__name__}(\n'

        for layer in layers:
            lines += f'  {layer}\n'
        lines += ')'

        return lines
