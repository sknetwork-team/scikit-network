#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in April 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Optional, Union
from collections import defaultdict

import numpy as np
from scipy import sparse

from sknetwork.classification.metrics import get_accuracy_score
from sknetwork.gnn.base import BaseGNN
from sknetwork.gnn.loss import BaseLoss
from sknetwork.gnn.layer import get_layer
from sknetwork.gnn.optimizer import BaseOptimizer
from sknetwork.gnn.utils import filter_mask, check_existing_masks, check_output, check_early_stopping, check_loss, \
    get_layers
from sknetwork.utils.check import check_format, check_nonnegative, check_square


class GNNClassifier(BaseGNN):
    """Graph Neural Network for node classification.

    Parameters
    ----------
    dims : list or int
        Dimensions of the output of each layer (in forward direction).
        If an integer, dimension of the output layer (no hidden layer).
        Optional if ``layers`` is specified.
    layer_types : list or str
        Layer types (in forward direction).
        If a string, use the same type of layer for all layers.
        Can be ``'Conv'``, graph convolutional layer (default) or ``'Sage'`` (GraphSage).
    activations : list or str
        Activation functions (in forward direction).
        If a string, use the same activation function for all layers.
        Can be either ``'Identity'``, ``'Relu'``, ``'Sigmoid'`` or ``'Softmax'`` (default = ``'Relu'``).
    use_bias : list or bool
        Whether to use a bias term at each layer.
        If ``True``, use a bias term at all layers.
    normalizations : list or str
        Normalization of the adjacency matrix for message passing.
        If a string, use the same normalization for all layers.
        Can be either `'left'`` (left normalization by the degrees), ``'right'`` (right normalization by the degrees),
        ``'both'`` (symmetric normalization by the square root of degrees, default) or ``None`` (no normalization).
    self_embeddings : list or str
        Whether to add a self embeddings to each node of the graph for message passing.
        If ``True``, add self-embeddings at all layers.
    sample_sizes : list or int
        Size of neighborhood sampled for each node. Used only for ``'Sage'`` layer type.
    loss : str (default = ``'CrossEntropy'``) or BaseLoss
        Loss function name or custom loss.
    layers : list or None
        Custom layers. If used, previous parameters are ignored.
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
    conv2, ..., conv1: :class:'GCNConv'
        Graph convolutional layers.
    output_ : array
        Output of the GNN.
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
    0.94
    """

    def __init__(self, dims: Optional[Union[int, list]] = None, layer_types: Union[str, list] = 'Conv',
                 activations: Union[str, list] = 'ReLu', use_bias: Union[bool, list] = True,
                 normalizations: Union[str, list] = 'both', self_embeddings: Union[bool, list] = True,
                 sample_sizes: Union[int, list] = 25, loss: Union[BaseLoss, str] = 'CrossEntropy',
                 layers: Optional[list] = None, optimizer: Union[BaseOptimizer, str] = 'Adam',
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

    def forward(self, adjacency: sparse.csr_matrix, features: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Perform a forward pass on the graph and return the output.

        Parameters
        ----------
        adjacency : sparse.csr_matrix
            Adjacency matrix.
        features : sparse.csr_matrix, np.ndarray
            Features, array of shape (n_nodes, n_features).

        Returns
        -------
        output : np.ndarray
            Output of the GNN.
        """
        h = features.copy()
        for layer in self.layers:
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
            labels: np.ndarray, n_epochs: int = 100, train_mask: Optional[np.ndarray] = None,
            val_mask: Optional[np.ndarray] = None,
            test_mask: Optional[np.ndarray] = None, train_size: Optional[float] = 0.8,
            val_size: Optional[float] = 0.1, test_size: Optional[float] = 0.1, resample: bool = False,
            reinit: bool = False, random_state: Optional[int] = None, history: bool = False) -> 'GNNClassifier':
        """ Fit model to data and store trained parameters.

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
            If ``True``, resample the train/test/validation sets before fitting.
            Otherwise, the train/test/validation sets remain the same after the first fit.
        reinit: bool  (default = ``False``)
            If ``True``, reinit the trainable parameters of the GNN (weights and biases).
        random_state : int
            Pass an int for reproducible results across multiple runs.
        history : bool (default = ``False``)
            If ``True``, save training history.
        """
        labels_pred = None

        if reinit:
            for layer in self.layers:
                layer.weights_initialized = False

        if resample or self.output_ is None:
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
            output = self.forward(adjacency, features)

            # Compute predictions
            labels_pred = self._compute_predictions(output)

            # Loss
            loss_value = self.loss.loss(output[self.train_mask], labels[self.train_mask])

            # Accuracy
            train_acc = get_accuracy_score(labels[self.train_mask], labels_pred[self.train_mask])
            if self.val_mask is not None and any(self.val_mask):
                val_acc = get_accuracy_score(labels[self.val_mask], labels_pred[self.val_mask])
            else:
                val_acc = None

            # Backpropagation
            self.backward(features, labels)

            # Update weights using optimizer
            self.optimizer.step(self)

            # Save results
            if history:
                self.history_['embedding'].append(self.layers[-1].embedding)
                self.history_['loss'].append(loss_value)
                self.history_['train_accuracy'].append(train_acc)
                if val_acc is not None:
                    self.history_['val_accuracy'].append(val_acc)

            if n_epochs > 10 and epoch % int(n_epochs / 10) == 0:
                if val_acc is not None:
                    self.log.print(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, train accuracy: {train_acc:.3f}, '
                        f'val accuracy: {val_acc:.3f}')
                else:
                    self.log.print(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, train accuracy: {train_acc:.3f}')
            elif n_epochs <= 10:
                if val_acc is not None:
                    self.log.print(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, train accuracy: {train_acc:.3f}, '
                        f'val accuracy: {val_acc:.3f}')
                else:
                    self.log.print(
                        f'In epoch {epoch:>3}, loss: {loss_value:.3f}, train accuracy: {train_acc:.3f}')

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

        output = self.forward(adjacency, features)
        labels_pred = self._compute_predictions(output)

        self.embedding_ = self.layers[-1].embedding
        self.output_ = self.layers[-1].output
        self.labels_ = labels_pred

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
        """Predict labels for new nodes. If called without parameters, labels are returned for all nodes.

        Parameters
        ----------
        adjacency_vectors : np.ndarray
            Square adjacency matrix. Array of shape (n, n).
        feature_vectors : np.ndarray
            Features row vectors. Array of shape (n, n_feat). The number of features n_feat must match with the one
            used during training.

        Returns
        -------
        labels : np.ndarray
            Label of each node of the graph.
        """
        self._check_fitted()

        if adjacency_vectors is None and feature_vectors is None:
            return self.labels_
        elif adjacency_vectors is None:
            adjacency_vectors = sparse.identity(feature_vectors.shape[0], format='csr')

        check_square(adjacency_vectors)
        check_nonnegative(adjacency_vectors)
        feature_vectors = check_format(feature_vectors)

        n_row, n_col = adjacency_vectors.shape
        feat_row, feat_col = feature_vectors.shape

        if n_col != feat_row:
            raise ValueError(f'Dimension mismatch: dim0={n_col} != dim1={feat_row}.')
        elif feat_col != self.layers[0].weight.shape[0]:
            raise ValueError(f'Dimension mismatch: current number of features is {feat_col} whereas GNN has been '
                             f'trained with '
                             f'{self.layers[0].weight.shape[0]} features.')

        h = self.forward(adjacency_vectors, feature_vectors)
        labels = self._compute_predictions(h)

        return labels
