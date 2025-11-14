#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in April 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Iterable, Optional, Union
from collections import defaultdict

import numpy as np
from scipy import sparse

from sknetwork.classification.metrics import get_accuracy_score
from sknetwork.gnn.base import BaseGNN
from sknetwork.gnn.loss import BaseLoss
from sknetwork.gnn.layer import get_layer
from sknetwork.gnn.neighbor_sampler import UniformNeighborSampler
from sknetwork.gnn.optimizer import BaseOptimizer
from sknetwork.gnn.utils import check_output, check_early_stopping, check_loss, get_layers
from sknetwork.utils.check import check_format, check_nonnegative, check_square
from sknetwork.utils.values import get_values


class GNNClassifier(BaseGNN):
    """Graph Neural Network for node classification.

    Parameters
    ----------
    dims : iterable or int
        Dimension of the output of each layer (in forward direction).
        If an integer, dimension of the output layer (no hidden layer).
        Optional if ``layers`` is specified.
    layer_types : iterable or str
        Layer types (in forward direction).
        If a string, the same type is used at each layer.
        Can be ``'Conv'``, graph convolutional layer (default) or ``'Sage'`` (GraphSage).
    activations : iterable or str
        Activation functions (in forward direction).
        If a string, the same activation function is used at each layer.
        Can be either ``'Identity'``, ``'Relu'``, ``'Sigmoid'`` or ``'Softmax'`` (default = ``'Relu'``).
    use_bias : iterable or bool
        Whether to add a bias term at each layer (in forward direction).
        If ``True``, use a bias term at each layer.
    normalizations : iterable or str
        Normalizations of the adjacency matrix for message passing (in forward direction).
        If a string, the same type of normalization is used at each layer.
        Can be either ``'left'`` (left normalization by the degrees), ``'right'`` (right normalization by the degrees),
        ``'both'`` (symmetric normalization by the square root of degrees, default) or ``None`` (no normalization).
    self_embeddings : iterable or str
        Whether to add the embedding to each node for message passing (in forward direction).
        If ``True``, add a self-embedding at each layer.
    sample_sizes : iterable or int
        Sizes of neighborhood sampled for each node (in forward direction).
        If an integer, the same sampling size is used at each layer.
        Used only for ``'Sage'`` layer type.
    loss : str (default = ``'CrossEntropy'``) or BaseLoss
        Name of loss function or custom loss function.
    layers : iterable or None
        Custom layers (in forward directions). If used, previous parameters are ignored.
    optimizer : str or optimizer
        * ``'Adam'``, stochastic gradient-based optimizer (default).
        * ``'GD'``, gradient descent.
    learning_rate : float
        Learning rate.
    early_stopping : bool (default = ``True``)
        Whether to use early stopping to end training.
        If ``True``, training terminates when validation score is not improving for `patience` number of epochs.
    patience : int (default = 10)
        Number of iterations with no improvement to wait before stopping fitting.
    verbose : bool
        Verbose mode.

    Attributes
    ----------
    layers : list
        Graph convolutional layers.
    output\_ : np.ndarray
        Output of the GNN.
    labels\_: np.ndarray
        Predicted node labels.
    history\_: dict
        Training history per epoch: {``'embedding'``, ``'loss'``, ``'train_accuracy'``, ``'val_accuracy'``}.

    Example
    -------
    >>> from sknetwork.gnn.gnn_classifier import GNNClassifier
    >>> from sknetwork.data import karate_club
    >>> from numpy.random import randint
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> labels = {i: labels_true[i] for i in [0, 1, 33]}
    >>> features = adjacency.copy()
    >>> gnn = GNNClassifier(dims=1, early_stopping=False)
    >>> labels_pred = gnn.fit_predict(adjacency, features, labels, random_state=42)
    >>> float(round(np.mean(labels_pred == labels_true), 2))
    0.88
    """

    def __init__(self, dims: Optional[Union[int, Iterable]] = None, layer_types: Union[str, Iterable] = 'Conv',
                 activations: Union[str, Iterable] = 'ReLu', use_bias: Union[bool, list] = True,
                 normalizations: Union[str, Iterable] = 'both', self_embeddings: Union[bool, Iterable] = True,
                 sample_sizes: Union[int, list] = 25, loss: Union[BaseLoss, str] = 'CrossEntropy',
                 layers: Optional[Iterable] = None, optimizer: Union[BaseOptimizer, str] = 'Adam',
                 learning_rate: float = 0.01, early_stopping: bool = True, patience: int = 10, verbose: bool = False):
        super(GNNClassifier, self).__init__(loss, optimizer, learning_rate, verbose)
        if layers is not None:
            layers = [get_layer(layer) for layer in layers]
        else:
            layers = get_layers(dims, layer_types, activations, use_bias, normalizations, self_embeddings, sample_sizes,
                                loss)
        self.loss = check_loss(layers[-1])
        self.layers = layers
        self.early_stopping = early_stopping
        self.patience = patience
        self.history_ = defaultdict(list)

    def forward(self, adjacency: Union[list, sparse.csr_matrix], features: Union[sparse.csr_matrix, np.ndarray]) \
            -> np.ndarray:
        """Perform a forward pass on the graph and return the output.

        Parameters
        ----------
        adjacency : Union[list, sparse.csr_matrix]
            Adjacency matrix or list of sampled adjacency matrices.
        features : sparse.csr_matrix, np.ndarray
            Features, array of shape (n_nodes, n_features).

        Returns
        -------
        output : np.ndarray
            Output of the GNN.
        """
        h = features.copy()
        for i, layer in enumerate(self.layers):
            if isinstance(adjacency, list):
                h = layer(adjacency[i], h)
            else:
                h = layer(adjacency, h)
        return h

    @staticmethod
    def _compute_predictions(output: np.ndarray) -> np.ndarray:
        """Compute predictions from the output of the GNN.

        Parameters
        ----------
        output : np.ndarray
            Output of the GNN.

        Returns
        -------
        labels : np.ndarray
            Predicted labels.
        """
        if output.shape[1] == 1:
            labels = (output.ravel() > 0.5).astype(int)
        else:
            labels = output.argmax(axis=1)
        return labels

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], features: Union[sparse.csr_matrix, np.ndarray],
            labels: np.ndarray, n_epochs: int = 100, validation: float = 0, reinit: bool = False,
            random_state: Optional[int] = None) -> 'GNNClassifier':
        """ Fit model to data and store trained parameters.

        Parameters
        ----------
        adjacency : sparse.csr_matrix
            Adjacency matrix of the graph.
        features : sparse.csr_matrix, np.ndarray
            Input feature of shape :math:`(n, d)` with :math:`n` the number of nodes in the graph and :math:`d`
            the size of feature space.
        labels : dict, np.ndarray
            Known labels. Negative values ignored.
        n_epochs : int (default = 100)
            Number of epochs (iterations over the whole graph).
        validation : float
            Proportion of the training set used for validation (between 0 and 1).
        reinit: bool  (default = ``False``)
            If ``True``, reinit the trainable parameters of the GNN (weights and biases).
        random_state : int
            Random seed, used for reproducible results across multiple runs.
        """
        if reinit:
            for layer in self.layers:
                layer.weights_initialized = False
            self.history_ = defaultdict(list)

        if random_state is not None:
            np.random.seed(random_state)

        check_format(adjacency, allow_empty=True)
        check_format(features, allow_empty=True)

        labels = get_values(adjacency.shape, labels)
        labels = labels.astype(int)
        if (labels < 0).all():
            raise ValueError('At least one node must have a non-negative label.')
        check_output(self.layers[-1].out_channels, labels)

        self.train_mask = labels >= 0
        if self.val_mask is None and 0 < validation < 1:
            mask = np.random.random(size=len(labels)) < validation
            self.val_mask = self.train_mask & mask
            self.train_mask &= ~mask

        early_stopping = check_early_stopping(self.early_stopping, self.val_mask, self.patience)

        # List of sampled adjacencies (one per layer)
        adjacencies = self._sample_nodes(adjacency)

        best_val_accuracy = 0
        count = 0

        for epoch in range(n_epochs):

            # Forward
            output = self.forward(adjacencies, features)

            # Compute predictions
            labels_pred = self._compute_predictions(output)

            # Loss
            loss_value = self.loss.loss(output[self.train_mask], labels[self.train_mask])

            # Accuracy
            train_accuracy = get_accuracy_score(labels[self.train_mask], labels_pred[self.train_mask])
            if self.val_mask is not None and any(self.val_mask):
                val_accuracy = get_accuracy_score(labels[self.val_mask], labels_pred[self.val_mask])
            else:
                val_accuracy = None

            # Backpropagation
            self.backward(features, labels, self.train_mask)

            # Update weights using optimizer
            self.optimizer.step(self)

            # Save results
            self.history_['loss'].append(loss_value)
            self.history_['train_accuracy'].append(train_accuracy)
            if val_accuracy is not None:
                self.history_['val_accuracy'].append(val_accuracy)

            if n_epochs > 10 and epoch % int(n_epochs / 10) == 0:
                if val_accuracy is not None:
                    self.print_log(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, train accuracy: {train_accuracy:.3f}, '
                        f'val accuracy: {val_accuracy:.3f}')
                else:
                    self.print_log(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, train accuracy: {train_accuracy:.3f}')
            elif n_epochs <= 10:
                if val_accuracy is not None:
                    self.print_log(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, train accuracy: {train_accuracy:.3f}, '
                        f'val accuracy: {val_accuracy:.3f}')
                else:
                    self.print_log(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, train accuracy: {train_accuracy:.3f}')

            # Early stopping
            if early_stopping:
                if val_accuracy > best_val_accuracy:
                    count = 0
                    best_val_accuracy = val_accuracy
                else:
                    count += 1
                    if count >= self.patience:
                        self.print_log('Early stopping.')
                        break

        output = self.forward(adjacencies, features)
        labels_pred = self._compute_predictions(output)

        self.embedding_ = self.layers[-1].embedding
        self.output_ = self.layers[-1].output
        self.labels_ = labels_pred

        return self

    def _sample_nodes(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> list:
        """Perform node sampling on adjacency matrix for GraphSAGE layers. For other layers, the
        adjacency matrix remains unchanged.

        Parameters
        ----------
        adjacency : sparse.csr_matrix
            Adjacency matrix of the graph.

        Returns
        -------
        List of (sampled) adjacency matrices.
        """
        adjacencies = []

        for layer in self.layers:
            if layer.layer_type == 'sage':
                sampler = UniformNeighborSampler(sample_size=layer.sample_size)
                adjacencies.append(sampler(adjacency))
            else:
                adjacencies.append(adjacency)

        return adjacencies
