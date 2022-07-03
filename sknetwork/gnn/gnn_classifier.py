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
from sknetwork.gnn.layers import get_layer
from sknetwork.gnn.loss import get_loss_function
from sknetwork.gnn.utils import check_existing_masks, check_layers_parameters
from sknetwork.utils.check import check_format, check_adjacency_vector, check_nonnegative, is_square


class GNNClassifier(BaseGNNClassifier):
    """ Graph Convolutional Network node classifier.

    Parameters
    ----------
    layer:
        Layer name (for multi-layers GNN, use a `list`). Layer name can be either:
        * ``'GCNConv'``, graph convolutional layer.
    n_hidden:
        Size of hidden layer (for multi-layers GNN, use a list).
    activation:
        Activation function (for multi-layers GNN, use a list). Can be either ``'Relu'``, ``'Sigmoid'`` or
        ``'Softmax'``.
    use_bias:
        If `True`, use bias vector (for multi-layers GNN, use a list).
    normalization:
        How to normalize the adjacency matrix (for multi-layers GNN, use a list). Can be either:
        *  ``'Both'`` (default), equivalent to symmetric normalization.
    self_loops:
        If `True`, add self loops to each node in the graph (for multi-layers GNN, use a list).
    optimizer: str (default = ``'Adam'``)
        Optimizer name:
        * ``'Adam'``, stochastic gradient-based optimizer.
        * ``'None'``, gradient descent.
    verbose :
        Verbose mode.

    Attributes
    ----------
    conv1, conv2: :class:`GCNConv``
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
    >>> gnn = GNNClassifier(layer='GCNConv', n_hidden=2, activation='Softmax', optimizer='Adam', verbose=False)
    >>> _ = gnn.fit(adjacency, features, labels, max_iter=10, val_size=0.2, random_state=42)
    >>> gnn.predict(adjacency[:3], features[:3])
    array([0, 0, 0])
    """

    def __init__(self, layer: Union[str, list], n_hidden: Union[int, list], activation: Union[str, list] = 'Sigmoid',
                 use_bias: Union[bool, list] = True, normalization: Union[str, list] = 'Both',
                 self_loops: Union[bool, list] = True,
                 optimizer: str = 'Adam', **kwargs):
        super(GNNClassifier, self).__init__(optimizer, **kwargs)
        parameters = check_layers_parameters(layer, n_hidden, activation, use_bias, normalization, self_loops)
        for layer_idx, params in enumerate(zip(*parameters)):
            setattr(self, f'conv{layer_idx + 1}', get_layer(params[0], *params[1:]))

    def forward(self, adjacency: sparse.csr_matrix, feat: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """ Performs a forward pass on the graph and returns embedding of nodes.

            Note that the non-linearity is integrated in the :class:`GCNConv` layer and hence is not applied
            after each layer in the forward method.

        Parameters
        ----------
        adjacency : sparse.csr_matrix
            Adjacency matrix.
        feat : sparse.csr_matrix, np.ndarray
            Input feature of shape :math:`(n, d)` with :math:`n` the number of nodes in the graph and :math:`d`
            the size of the embedding.

        Returns
        -------
        np.ndarray
            Nodes embedding.
        """

        layers = self._get_layers()
        h = feat.copy()
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
        np.ndarray : logits corresponding predicted probabilities for each class.
        np.ndarray : y_pred corresponding to predicted class.
        """

        if embedding.shape[1] == 1:
            y_pred = np.where(embedding.ravel() < 0.5, 0, 1)
            logits = embedding.ravel()
        else:
            logits = embedding
            y_pred = embedding.argmax(1)

        return logits, y_pred

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], features: Union[sparse.csr_matrix, np.ndarray],
            labels: np.ndarray, max_iter: int = 30, loss: str = 'CrossEntropyLoss',
            train_mask: Optional[np.ndarray] = None, val_size: Optional[float] = None,
            random_state: Optional[int] = None, shuffle: Optional[bool] = True) -> 'GNNClassifier':
        """ Fits model to data and store trained parameters.

        Parameters
        ----------
        adjacency
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
        train_mask: np.ndarray
            Boolean array indicating whether nodes are in training set.
        val_size: float
            Should be between 0 and 1 and represents the proportion of the nodes to include in validation set. Only
            used if `train_mask` is `None`.
        random_state : int
            Pass an int for reproducible results across multiple runs. Used if `val_size` is not `None`.
        shuffle : bool (default = `True`)
            If `True`, shuffles samples before split. Used if `val_size` is not `None`.
        """
        y_pred = None
        exists_mask, self.train_mask, self.val_mask, self.test_mask = \
            check_existing_masks(labels, train_mask, val_size)
        if not exists_mask:
            self._generate_masks(adjacency.shape[0], self.train_mask, val_size, random_state, shuffle)

        check_format(adjacency)
        check_format(features)

        for epoch in range(max_iter):

            # Forward
            emb = self.forward(adjacency, features)

            # Compute predictions
            logits, y_pred = self._compute_predictions(emb)

            # Loss
            loss_function = get_loss_function(loss)
            loss_value = loss_function(labels[self.train_mask], logits[self.train_mask])

            # Accuracy
            train_acc = accuracy_score(labels[self.train_mask], y_pred[self.train_mask])
            val_acc = accuracy_score(labels[self.val_mask], y_pred[self.val_mask])

            # Backpropagation
            self.backward(features, labels, loss)

            # Update weights using optimizer
            self.opt.step()

            # Save results
            self.history_['embedding'].append(emb)
            self.history_['loss'].append(loss_value)
            self.history_['train_accuracy'].append(train_acc)
            self.history_['val_accuracy'].append(val_acc)

            if max_iter > 10 and epoch % int(max_iter / 10) == 0:
                self.log.print(
                    f'In epoch {epoch:>3}, loss: {loss_value:.3f}, training acc: {train_acc:.3f}, '
                    f'val acc: {val_acc:.3f}')
            elif max_iter <= 10:
                self.log.print(
                    f'In epoch {epoch:>3}, loss: {loss_value:.3f}, training acc: {train_acc:.3f}, '
                    f'val acc: {val_acc:.3f}')

        self.labels_ = y_pred
        self.embedding_ = self.history_.get('embedding')[-1]

        return self

    def _generate_masks(self, n: int, train_mask: np.ndarray, val_size: float = 0.2, random_state: int = None,
                        shuffle: bool = True):
        """ Create training, validation and test masks.

        Parameters
        ----------
        train_mask : np.ndarray
            Training mask. This mask will be split into training and validation masks.
        n : int
            Number of nodes in graph.
        val_size : float (default = 0.2)
            Proportion of nodes in the validation set (between 0 and 1).
        random_state : int
            Pass an int for reproducible results across multiple runs.
        shuffle : bool (default = `True`)
            If `True`, shuffles samples before split.
        """
        if random_state is not None:
            np.random.seed(random_state)

        self.test_mask = ~train_mask

        # Useful in case of -1 in labels
        available_indexes = np.where(~self.test_mask)[0]

        if shuffle:
            val_indexes = np.random.choice(available_indexes, int(val_size * n))
        else:
            val_indexes = available_indexes[-int(val_size * n):]

        train_mask[val_indexes] = False
        self.val_mask = np.logical_and(~train_mask, ~self.test_mask)
        self.train_mask = train_mask.copy()

    def predict(self, adjacency_vectors: Union[sparse.csr_matrix, np.ndarray] = None,
                feat_vectors: Union[sparse.csr_matrix, np.ndarray] = None) -> np.ndarray:
        """Predict class labels for nodes in `nodes`.

        Parameters
        ----------
        adjacency_vectors
            Adjacency row vectors. Array of shape (n_col,) (single vector) or (n_vectors, n_col).
        feat_vectors
            Features row vectors. Array of shape (n_feat,) (single vector) or (n_vectors, n_feat).

        Returns
        -------
        np.ndarray
            Array containing the class labels for each node in the graph.
        """
        self._check_fitted()

        n = len(self.embedding_)
        adjacency_vectors = check_adjacency_vector(adjacency_vectors, n)
        check_nonnegative(adjacency_vectors)

        n_row, n_col = adjacency_vectors.shape
        n_feat = feat_vectors.shape[1]

        if not is_square(adjacency_vectors):
            max_n = max(n_row, n_col)
            adjacency_vectors.resize(max_n, max_n)
            feat_vectors.resize(max_n, n_feat)

        h = self.forward(adjacency_vectors, feat_vectors)
        _, labels = self._compute_predictions(h)

        return labels[:n_row]
