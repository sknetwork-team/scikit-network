#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in April 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Optional, Tuple, Union

import numpy as np
from scipy import sparse

from sknetwork.classification.metrics import get_accuracy_score
from sknetwork.gnn.base import BaseGNNClassifier
from sknetwork.gnn.loss import get_loss_function
from sknetwork.gnn.utils import filter_mask, check_existing_masks, check_output, get_layers_parameters, \
    check_early_stopping
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
        Can be either ``'Identity'``, ``'Relu'``, ``'Sigmoid'`` or ``'Softmax'``.
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
        Training history per epoch: {``'embedding'``, ``'loss'``, ``'train_accuracy'``, ``'val_accuracy'``}.

    Example
    -------
    >>> from sknetwork.gnn.gnn_classifier import GNNClassifier
    >>> from sknetwork.data import karate_club
    >>> from numpy.random import randint
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels = graph.labels
    >>> features = adjacency.copy()
    >>> gnn = GNNClassifier(dims=1, early_stopping=False)
    >>> labels_pred = gnn.fit_predict(adjacency, features, labels, random_state=42)
    >>> np.round(np.mean(labels_pred == labels), 2)
    0.88
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
        self.layers = self._get_layers()

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

        h = features.copy()
        for layer in self.layers:
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
            labels: np.ndarray, loss: str = 'CrossEntropyLoss', n_epochs: int = 100,
            train_mask: Optional[np.ndarray] = None, val_mask: Optional[np.ndarray] = None,
            test_mask: Optional[np.ndarray] = None, train_size: Optional[float] = 0.8,
            val_size: Optional[float] = 0.1, test_size: Optional[float] = 0.1, resample: bool = False,
            random_state: Optional[int] = None) -> 'GNNClassifier':
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
        loss : str (default = ``'CrossEntropyLoss'``)
            Loss function name.
        n_epochs : int (default = 100)
            Number of epochs (iterations over the whole graph).
        train_mask, val_mask, test_mask : np.ndarray
            Boolean array indicating whether nodes are in training/validation/test set.
        train_size, test_size : float
            Proportion of the nodes in the training/test set (between 0 and 1).
            Only used if the corresponding masks are ``None``.
        val_size : float
            Proportion of the training set used for validation (between 0 and 1).
            Only used if the corresponding mask is ``None``.
        resample : bool (default = ``False``)
            Resample the train/test/validation sets before fitting.
            Otherwise, the train/test/validation sets remain the same after the first fit.
        random_state : int
            Pass an int for reproducible results across multiple runs.
        """
        labels_pred = None

        if resample or self.embedding_ is None:
            exists_mask, self.train_mask, self.val_mask, self.test_mask = \
                check_existing_masks(labels, train_mask, val_mask, test_mask, train_size, val_size, test_size)
            if not exists_mask:
                self._generate_masks(train_size, val_size, test_size, random_state)

        check_format(adjacency)
        check_format(features)

        check_output(self.layers[-1].out_channels, labels)

        early_stopping = check_early_stopping(self.early_stopping, self.val_mask, self.patience)

        best_val_acc = 0
        trigger_times = 0

        for epoch in range(n_epochs):

            # Forward
            embedding = self.forward(adjacency, features)

            # Compute predictions
            logits, labels_pred = self._compute_predictions(embedding)

            # Loss
            loss_function = get_loss_function(loss)
            loss_value = loss_function(labels[self.train_mask], logits[self.train_mask])

            # Accuracy
            train_acc = get_accuracy_score(labels[self.train_mask], labels_pred[self.train_mask])
            if self.val_mask is not None and any(self.val_mask):
                val_acc = get_accuracy_score(labels[self.val_mask], labels_pred[self.val_mask])
            else:
                val_acc = None

            # Backpropagation
            self.backward(features, labels, loss)

            # Update weights using optimizer
            self.opt.step()

            # Save results
            self.history_['embedding'].append(embedding)
            self.history_['loss'].append(loss_value)
            self.history_['train_accuracy'].append(train_acc)
            if val_acc is not None:
                self.history_['val_accuracy'].append(val_acc)

            if n_epochs > 10 and epoch % int(n_epochs / 10) == 0:
                if val_acc is not None:
                    self.log.print(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, training acc: {train_acc:.3f}, '
                        f'val acc: {val_acc:.3f}')
                else:
                    self.log.print(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, training acc: {train_acc:.3f}')
            elif n_epochs <= 10:
                if val_acc is not None:
                    self.log.print(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, training acc: {train_acc:.3f}, '
                        f'val acc: {val_acc:.3f}')
                else:
                    self.log.print(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, training acc: {train_acc:.3f}')

            # Early stopping
            if early_stopping:
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

    def _generate_masks(self, train_size: Optional[float] = None, val_size: Optional[float] = None,
                        test_size: Optional[float] = None, random_state: int = None):
        """ Create training, validation and test masks.

        Parameters
        ----------
        train_size : float
            Proportion of nodes in the training set (between 0 and 1).
        val_size : float
            Proportion of nodes in the validation set (between 0 and 1).
        test_size : float
            Proportion of nodes in the test set (between 0 and 1).
        random_state : int
            Pass an int for reproducible results across multiple runs.
        """
        is_negative_labels = self.test_mask

        if random_state is not None:
            np.random.seed(random_state)

        if train_size is None:
            train_size = 1 - test_size

        self.train_mask = filter_mask(~is_negative_labels, train_size)
        self.val_mask = filter_mask(np.logical_and(~self.train_mask, ~is_negative_labels), val_size)
        self.test_mask = np.logical_and(~self.train_mask, ~self.val_mask)

    def predict(self, adjacency_vectors: Union[sparse.csr_matrix, np.ndarray] = None,
                feature_vectors: Union[sparse.csr_matrix, np.ndarray] = None) -> np.ndarray:
        """Predict labels for new nodes. If `predict()` is called without parameters, labels are returned
        for all nodes.

        Parameters
        ----------
        adjacency_vectors
            Adjacency row vectors. Array of shape (n_col,) (single vector) or (n_vectors, n_col).
        feature_vectors
            Features row vectors. Array of shape (n_feat,) (single vector) or (n_vectors, n_feat).

        Returns
        -------
        np.ndarray
            Array containing the labels of each node in the graph.
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
