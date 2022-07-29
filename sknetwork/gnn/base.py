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
    optimizer: str (default = ``'Adam'``)

        * ``'Adam'``, a stochastic gradient-based optimizer.
        * ``'None'``, gradient descent.

    Attributes
    ----------
    prime_weight, prime_bias: np.ndarray, np.ndarray
        Derivatives of trainable weight matrix and bias vector.
    layers: list
        List of layer object (e.g `GCNConv`).
    nb_layers: int
        Number of layers in model.
    activations: list
        List of activation functions.
    labels_: np.ndarray
        Predicted node labels.
    history_: dict
        Training history per epoch: {'embedding', 'loss', 'train_accuracy', 'test_accuracy'}.
    """

    def __init__(self, optimizer: str = 'Adam', verbose: bool = False, **kwargs):
        VerboseMixin.__init__(self, verbose)
        self.opt = get_optimizer(self, optimizer, **kwargs)
        self.layers = []
        self.train_mask = None
        self.test_mask = None
        self.val_mask = None
        self.nb_layers = 0
        self.activations = []
        self.prime_weight, self.prime_bias = [], []
        self.labels_ = None
        self.embedding_ = None
        self.history_ = defaultdict(list)

    @staticmethod
    def _init_layer(layer: str = 'GCNConv', *args) -> object:
        """Instantiate layer according to parameters.

        Parameters
        ----------
        layer : str
            Which layer to use. Can be ``'GCNConv'``.

        Returns
        -------
        Layer object.
        """
        layer = layer.lower()
        if layer == 'gcnconv':
            return GCNConv(*args)
        else:
            raise ValueError("Layer must be \"GCNConv\".")

    def fit(self, *args, **kwargs):
        """Fit Algorithm to the data."""
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Predict labels."""
        raise NotImplementedError

    def fit_predict(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels of the nodes.
        """
        self.fit(*args, **kwargs)
        return self.predict(*args[:2])

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the embedding of the nodes. Same parameters as the ``fit`` method.

        Returns
        -------
        Embedding : np.ndarray
            Embedding of the nodes.
        """
        self.fit(*args, **kwargs)
        return self.embedding_

    def backward(self, features: np.ndarray, labels: np.ndarray, loss: str):
        """Compute backpropagation.

        Parameters
        ----------
        features : np.ndarray
            Features, array of shape (n_nodes, n_features).
        labels : np.ndarray
            Labels, array of shape (n_nodes,).
        loss : str
            Loss function name.
        """
        n = len(labels)
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
        output = self.layers[-1].embedding
        if self.layers[-1].activation == 'softmax':
            n_channels = self.layers[-1].out_channels
            y_true_ohe = np.eye(n_channels)[labels]
            prime_update = (output - y_true_ohe).T
        else:
            prime_output = prime_loss_function(labels, output.T)
            prime_update = prime_output * activation_primes[-1](self.layers[self.nb_layers - 1].update.T)

        if self.nb_layers == 1:
            output_prev = features
        else:
            output_prev = self.layers[self.nb_layers - 2].embedding

        prime_weight = output_prev.T.dot(prime_update.T)  # SpM on left-hand side
        prime_bias = np.sum(prime_update, axis=1, keepdims=True) / n
        prime_output_prev = self.layers[self.nb_layers - 1].weight.dot(prime_update)

        self.prime_weight[self.nb_layers - 1] = prime_weight
        self.prime_bias[self.nb_layers - 1] = prime_bias.T

        for layer_idx in range(self.nb_layers - 1, 0, -1):
            if self.layers[layer_idx - 1].activation == 'softmax':
                jacobian = activation_primes[layer_idx - 1](self.layers[layer_idx - 1].update.T)
                prime_update = np.einsum('mnr,mrr->mr', prime_output_prev[:, None, :], jacobian)
            else:
                prime_update = prime_output_prev * activation_primes[layer_idx - 1](self.layers[layer_idx - 1].update.T)
            if layer_idx == 1:
                output_prev = features
            else:
                output_prev = self.layers[layer_idx - 2].embedding
            prime_weight = output_prev.T.dot(prime_update.T)  # SpM on left-hand side
            prime_bias = np.sum(prime_update, axis=1, keepdims=True) / n
            if layer_idx > 1:
                prime_output_prev = self.layers[layer_idx - 1].weight.dot(prime_update)

            self.prime_weight[layer_idx - 1] = prime_weight
            self.prime_bias[layer_idx - 1] = prime_bias.T

    def _check_fitted(self):
        if self.embedding_ is None:
            raise ValueError("This embedding instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments before using this method.")
        else:
            return self

    def _get_layers(self) -> list:
        """Get layers objects in model.

        Returns
        -------
        List of layers objects.
        """
        available_types = [GCNConv]

        return [layer for layer in list(self.__dict__.values()) if any([isinstance(layer, t) for t in available_types])]

    @staticmethod
    def _get_activations(layers: list) -> list:
        """Get activation functions for each layers in model.

        Parameters
        ----------
        layers: list
            list of layers objects in model.

        Returns
        -------
        list of activation functions as strings.
        """
        return [layer.activation for layer in layers]

    def __repr__(self) -> str:
        """String representation of the `GNN`, layers by layers.

        Returns
        -------
        str
            String representation of object.
        """

        lines = ''

        lines += f'{self.__class__.__name__}(\n'

        for layer in self.layers:
            lines += f'  {layer}\n'
        lines += ')'

        return lines
