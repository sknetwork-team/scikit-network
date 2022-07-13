#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Optional, Tuple, Union

import numpy as np
from scipy import sparse

from sknetwork.classification.metrics import accuracy_score
from sknetwork.gnn.base import BaseGNNClassifier
from sknetwork.gnn.loss import get_loss_function
from sknetwork.gnn.utils import check_existing_masks, get_layers_parameters, check_early_stopping
from sknetwork.utils.check import check_format, check_adjacency_vector, check_nonnegative, is_square


class GNNClassifier(BaseGNNClassifier):
    """Graph Neural Network for node classification.

    Parameters
    ----------
    dims: list or int
        Dimensions of the layers (in forward direction, last is the output).
        If an integer, use a single layer (no hidden layer).
    layers: list or str
        Layers (in forward direction).
        If a string, use the same type of layer for all layers.
        Can be ``'GCNConv'``, graph convolutional layers (default).
    activations: list or str
        Activation functions (in forward direction).
        If a string, use the same activation function for all layers.
        Can be either ``'Relu'``, ``'Sigmoid'`` or ``'Softmax'``.
    use_bias: list or bool
        Whether to use a bias term at each layer.
        If ``True``, use a bias term at all layers.
    normalizations: list or str
        Normalization of the adjacency matrix for message passing.
        If a string, use the same normalization for all layers.
        Can be either:

        * ``'left'``, left normalization by the vector of degrees
        * ``'right'``, right normalization by the vector of degrees
        * ``'both'``,  symmetric normalization by the square root of degrees
    self_loops: list or str
        Whether to add a self loop at each node of the graph for message passing.
        If ``True``, add a self-loop for message passing at all layers.
    optimizer: str
        Can be either:

        * ``'Adam'``, stochastic gradient-based optimizer (default).
        * ``'None'``, gradient descent.
    early_stopping: bool (default = ``True``)
        Wether to use early stopping to end training when validation score is not improving anymore.
        If ``True``, training terminates when validation score is not improving for `patience` number of epochs.
    patience: int (default = 10)
        Number of iterations with no improvement to wait before stopping fitting.
    verbose :
        Verbose mode.

    Attributes
    ----------
    conv2, ..., conv1: :class:'GCNConv'
        Graph convolutional layers.
    embedding_ : array
        Embedding of the nodes.
    labels_: np.ndarray
        Predicted node labels.
    history_: dict
        Training history per epoch: {``'embedding'``, ``'loss'``, ``'train_accuracy'``, ``'test_accuracy'``}.

    Example
    -------
    >>> from sknetwork.gnn.gnn_classifier import GNNClassifier
    >>> from sknetwork.data import karate_club
    >>> from numpy.random import randint
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels = graph.labels
    >>> features = adjacency.copy()
    >>> gnn = GNNClassifier(dims=1)
    >>> _ = gnn.fit(adjacency, features, labels, random_state=42)
    >>> gnn.predict()[:3]
    array([0, 0, 0])
    """

    def __init__(self, dims: Union[int, list], layers: Union[str, list] = 'GCNConv',
                 activations: Union[str, list] = 'Sigmoid', use_bias: Union[bool, list] = True,
                 normalizations: Union[str, list] = 'Both', self_loops: Union[bool, list] = True,
                 optimizer: str = 'Adam', early_stopping: bool = True, patience: int = 10, **kwargs):
        super(GNNClassifier, self).__init__(optimizer, **kwargs)
        parameters = get_layers_parameters(dims, layers, activations, use_bias, normalizations, self_loops)
        self.early_stopping = early_stopping
        self.patience = patience
        for layer_idx, params in enumerate(zip(*parameters)):
            layer = params[1]
            args = params[:1] + params[2:]
            setattr(self, f'conv{layer_idx + 1}', self._init_layer(layer, *args))

    def forward(self, adjacency: sparse.csr_matrix, features: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """ Performs a forward pass on the graph and returns embedding of nodes.

            Note that the non-linearity is integrated in the :class:`GCNConv` layers and hence is not applied
            after each layers in the forward method.

        Parameters
        ----------
        adjacency : sparse.csr_matrix
            Adjacency matrix.
        features : sparse.csr_matrix, np.ndarray
            Features, array of shape (n_nodes, n_features).

        Returns
        -------
        np.ndarray
            Nodes embedding.
        """

        layers = self._get_layers()
        h = features.copy()
        for layer in layers:
            h = layer(adjacency, h)

        return h

    @staticmethod
    def _compute_predictions(embedding: np.ndarray) -> Tuple:
        """Compute model predictions.

        Parameters
        ----------
        embedding : np.ndarray
            Embedding of nodes.

        Returns
        -------
        np.ndarray : logits, predicted probabilities.
        np.ndarray : labels_pred, predicted class.
        """

        if embedding.shape[1] == 1:
            labels_pred = np.where(embedding.ravel() < 0.5, 0, 1)
            logits = embedding.ravel()
        else:
            logits = embedding
            labels_pred = embedding.argmax(1)

        return logits, labels_pred

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], features: Union[sparse.csr_matrix, np.ndarray],
            labels: np.ndarray, max_iter: int = 30, loss: str = 'CrossEntropyLoss',
            train_mask: Optional[np.ndarray] = None, val_mask: Optional[np.ndarray] = None,
            test_mask: Optional[np.ndarray] = None, train_size: Optional[float] = 0.8,
            val_size: Optional[float] = 0.1, test_size: Optional[float] = 0.2, random_state: Optional[int] = None,
            shuffle: Optional[bool] = True) -> 'GNNClassifier':
        """ Fits model to data and store trained parameters.

        Parameters
        ----------
        adjacency : sparse.csr_matrix
            Adjacency matrix of the graph.
        features : sparse.csr_matrix, np.ndarray
            Input feature of shape :math:`(n, d)` with :math:`n` the number of nodes in the graph and :math:`d`
            the size of feature space.
        labels : np.ndarray
            Label vectors of length :math:`n`, with :math:`n` the number of nodes in `adjacency`. A value of `labels`
            equals `-1` means no label. The associated nodes are not considered in training steps.
        max_iter: int (default = 30)
            Maximum number of iterations for the solver. Corresponds to the number of epochs.
        loss: str (default = ``'CrossEntropyLoss'``)
            Loss function name.
        train_mask, val_mask, test_mask : np.ndarray
            Boolean array indicating whether nodes are in training/validation/test set.
        train_size, val_size, test_size: float
            Proportion of the nodes in the training/validation/test set (between 0 and 1).
            Only used if when corresponding masks are ``None``.
        random_state : int
            Pass an int for reproducible results across multiple runs. Used if either `train_size` or `test_size`
            is not `None`.
        shuffle : bool (default = ``True``)
            If ``True``, shuffle samples before split. Only used if either `train_size` or `test_size` is not ``None``.
        """
        labels_pred = None
        exists_mask, self.train_mask, self.val_mask, self.test_mask = \
            check_existing_masks(labels, train_mask, val_mask, test_mask, train_size, val_size, test_size)
        if not exists_mask:
            self._generate_masks(train_size, val_size, test_size, random_state, shuffle)

        check_format(adjacency)
        check_format(features)

        check_early_stopping(self.early_stopping, self.val_mask, self.patience)

        best_val_acc = 0
        trigger_times = 0

        for epoch in range(max_iter):

            # Forward
            embedding = self.forward(adjacency, features)

            # Compute predictions
            logits, labels_pred = self._compute_predictions(embedding)

            # Loss
            loss_function = get_loss_function(loss)
            loss_value = loss_function(labels[self.train_mask], logits[self.train_mask])

            # Accuracy
            train_acc = accuracy_score(labels[self.train_mask], labels_pred[self.train_mask])
            if self.val_mask is not None and any(self.val_mask):
                val_acc = accuracy_score(labels[self.val_mask], labels_pred[self.val_mask])

            # Backpropagation
            self.backward(features, labels, loss)

            # Update weights using optimizer
            self.opt.step()

            # Save results
            self.history_['embedding'].append(embedding)
            self.history_['loss'].append(loss_value)
            self.history_['train_accuracy'].append(train_acc)
            if self.val_mask is not None and any(self.val_mask):
                self.history_['val_accuracy'].append(val_acc)

            if max_iter > 10 and epoch % int(max_iter / 10) == 0:
                if self.val_mask is not None and any(self.val_mask):
                    self.log.print(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, training acc: {train_acc:.3f}, '
                        f'val acc: {val_acc:.3f}')
                else:
                    self.log.print(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, training acc: {train_acc:.3f}')
            elif max_iter <= 10:
                if self.val_mask is not None and any(self.val_mask):
                    self.log.print(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, training acc: {train_acc:.3f}, '
                        f'val acc: {val_acc:.3f}')
                else:
                    self.log.print(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, training acc: {train_acc:.3f}')

            # Early stopping
            if self.early_stopping:
                if val_acc > best_val_acc:
                    trigger_times = 0
                    best_val_acc = val_acc
                else:
                    trigger_times += 1
                    if trigger_times >= self.patience:
                        self.log.print('Early stopping.')
                        break

        self.labels_ = labels_pred
        self.embedding_ = self.history_.get('embedding')[-1]

        return self

    def _generate_masks(self, train_size: float = 0.7, val_size: float = 0.1,
                        test_size: float = 0.2, random_state: int = None, shuffle: bool = True):
        """ Create training, validation and test masks.

        Parameters
        ----------
        train_size : float (default = 0.7)
            Proportion of nodes in the training set (between 0 and 1).
        val_size : float (default = 0.1)
            Proportion of nodes in the validation set (between 0 and 1).
        test_size : float (default = 0.2)
            Proportion of nodes in the test set (between 0 and 1).
        random_state : int
            Pass an int for reproducible results across multiple runs.
        shuffle : bool (default = ``True``)
            If ``True``, shuffles samples before split.
        """
        is_negative_labels = self.test_mask

        train_mask = np.zeros(len(self.train_mask)).astype(bool)
        val_mask = np.zeros(len(self.train_mask)).astype(bool)
        test_mask = np.zeros(len(self.train_mask)).astype(bool)

        if random_state is not None:
            np.random.seed(random_state)

        n = len(self.train_mask)
        indices = list(range(n))
        test_split = int(np.floor(train_size * n))

        if shuffle:
            np.random.shuffle(indices)

        train_indices, test_indices = indices[:test_split], indices[test_split:]
        val_split = int(np.floor((1 - val_size) * len(train_indices)))

        train_indices, val_indices = train_indices[:val_split], train_indices[val_split:]
        train_mask[train_indices], val_mask[val_indices], test_mask[test_indices] = True, True, True
        self.train_mask = np.logical_and(train_mask, ~is_negative_labels)
        self.val_mask = np.logical_and(val_mask, ~is_negative_labels)
        self.test_mask = np.logical_or(test_mask, is_negative_labels)

    def predict(self, adjacency_vectors: Union[sparse.csr_matrix, np.ndarray] = None,
                feature_vectors: Union[sparse.csr_matrix, np.ndarray] = None) -> np.ndarray:
        """Predict class labels for new nodes. If `predict()` is called without parameters, class labels are returned
        for all nodes in the graph.

        Parameters
        ----------
        adjacency_vectors
            Adjacency row vectors. Array of shape (n_col,) (single vector) or (n_vectors, n_col).
        feature_vectors
            Features row vectors. Array of shape (n_feat,) (single vector) or (n_vectors, n_feat).

        Returns
        -------
        np.ndarray
            Array containing the class labels for each node in the graph.
        """
        self._check_fitted()

        n = len(self.embedding_)

        if adjacency_vectors is None and feature_vectors is None:
            return self.labels_
        else:
            adjacency_vectors = check_adjacency_vector(adjacency_vectors, n)
            check_nonnegative(adjacency_vectors)

            n_row, n_col = adjacency_vectors.shape
            n_feat = feature_vectors.shape[1]

            if not is_square(adjacency_vectors):
                max_n = max(n_row, n_col)
                adjacency_vectors.resize(max_n, max_n)
                feature_vectors.resize(max_n, n_feat)

            h = self.forward(adjacency_vectors, feature_vectors)
            _, labels = self._compute_predictions(h)

            return labels[:n_row]
