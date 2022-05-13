#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""

import numpy as np
from scipy import sparse

from typing import Optional, Tuple, Union

from sknetwork.gnn.utils import check_existing_masks
from sknetwork.gnn.base_gnn import BaseGNNClassifier
from sknetwork.gnn.layers import GCNConv
from sknetwork.gnn.activation import ACTIVATIONS
from sknetwork.gnn.loss import *
from sknetwork.classification.metrics import accuracy_score
from sknetwork.utils.check import check_format, check_is_proba
from sknetwork.utils.verbose import VerboseMixin


class GNNClassifier(BaseGNNClassifier, VerboseMixin):
    """ Graph Convolutional Network node classifier.

    Parameters
    ----------
    in_channel: int
        Size of each input sample.
    h_channel: int
        Size hidden layer.
    num_classes: int
        Number of classes.
    opt: str (default='Adam')
        Optimizer name:
        - 'SGD', stochastic gradient descent.
        - 'Adam', refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba.
        - 'none', gradient descent.
    verbose :
        Verbose mode.

    Attributes
    ----------
    conv1, conv2: `GCNConv``
        Graph convolutional layers.
    labels_: np.ndarray
        Predicted node labels.
    history_: dict
        Training history per epoch: {'embedding', 'loss', 'train_accuracy', 'test_accuracy'}.

    Example
    -------
    >>> from sknetwork.gnn.gnn_classifier import GNNClassifier
    >>> from sknetwork.data import karate_club
    >>> import numpy as np
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels = graph.labels
    >>> features = adjacency.copy()
    >>> gnn = GNNClassifier(features.shape[1], 4, 1, opt='none')
    >>> y_pred = gnn.fit_transform(adjacency, features, labels, max_iter=30, loss='CrossEntropyLoss')
    >>> gnn.history_['train_accuracy'][-1], gnn.history_['test_accuracy'][-1]
    (1.0, 0.928)

    >>> new_n = np.random.randint(2, size=adjacency.shape[0])
    >>> gnn.predict(new_n)
    array([1])
    """

    def __init__(self, in_channels: int, h_channels: int, num_classes: int, opt: str = 'Adam', verbose: bool = False,
                 **kwargs):
        super(GNNClassifier, self).__init__(opt, **kwargs)
        VerboseMixin.__init__(self, verbose)
        self.conv1 = GCNConv(in_channels, h_channels)
        if num_classes > 1:
            self.conv2 = GCNConv(h_channels, num_classes, activation='softmax')
        else:
            self.conv2 = GCNConv(h_channels, num_classes)

    def forward(self, adjacency: sparse.csr_matrix, feat: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """ Performs a forward pass on the graph and returns embedding of nodes.

            Note that the non-linearity is integrated in the `GCNConv` layer and hence is not applied
            after each layer in the forward method.

        Parameters
        ----------
        adjacency : sparse.csr_matrix
            Adjacency matrix.
        feat : sparse.csr_matrix, np.ndarray
            Input feature of shape (n, d) with n the number of nodes in the graph and d the size of the embedding.

        Returns
        -------
        np.ndarray
            Nodes embedding.
        """

        h = self.conv1(adjacency, feat)
        h = self.conv2(adjacency, h)

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

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], feat: Union[sparse.csr_matrix, np.ndarray],
            y_true: np.ndarray, max_iter: int = 30, loss: str = 'CrossEntropyLoss',
            train_mask: Optional[np.ndarray] = None, test_mask: Optional[np.ndarray] = None,
            test_size: Optional[float] = None, random_state: Optional[int] = None, shuffle: Optional[bool] = True):
        """ Fits model to data and store trained parameters.

        Parameters
        ----------
        adjacency
            Adjacency matrix of the graph.
        feat : sparse.csr_matrix, np.ndarray
            Input feature of shape (n, d) with n the number of nodes in the graph and d the size of feature space.
        y_true : np.ndarray
            Label vectors of length n, with n the number of nodes in `adjacency`
        max_iter: int (default=30)
            Maximum number of iterations for the solver. Corresponds to the number of epochs.
        loss: str (default="CrossEntropyLoss")
            Loss function name.
        train_mask, test_mask: np.ndarray, np.ndarray
            Boolean array indicating wether nodes are in training or test sets.
        test_size: float
            Should be between 0 and 1 and represents the proportion of the nodes to include in test set. Only used if
            `train_mask` and `test_mask` are not provided.
        random_state : int
            Pass an int for reproducible results across multiple runs. Used if `test_size` is not None.
        shuffle : bool (default=True)
            If True, shuffles samples before split. Used if `test_size` is not None.
        """

        if not check_existing_masks(train_mask, test_mask, test_size):
            train_mask, test_mask = self._generate_masks(adjacency.shape[0], test_size, random_state, shuffle)

        check_format(adjacency)
        check_format(feat)

        for epoch in range(max_iter):

            # Forward
            emb = self.forward(adjacency, feat)

            # Compute predictions
            logits, y_pred = self._compute_predictions(emb)

            # Loss
            loss_function = get_loss_function(loss)
            loss_value = loss_function(y_true[train_mask], logits[train_mask])

            # Accuracy
            train_acc = accuracy_score(y_true[train_mask], y_pred[train_mask])
            test_acc = accuracy_score(y_true[test_mask], y_pred[test_mask])

            # Backpropagation
            self.backward(feat, y_true, loss)

            # Update weights using optimizer
            self.opt.step()

            # Save results
            self.history_['embedding'].append(emb)
            self.history_['loss'].append(loss_value)
            self.history_['train_accuracy'].append(train_acc)
            self.history_['test_accuracy'].append(test_acc)

            if max_iter > 10 and epoch % int(max_iter / 10) == 0:
                self.log.print(
                    f'In epoch {epoch:>3}, loss: {loss_value:.3f}, training acc: {train_acc:.3f}, test acc: {test_acc:.3f}')
            elif max_iter <= 10:
                self.log.print(
                    f'In epoch {epoch:>3}, loss: {loss_value:.3f}, training acc: {train_acc:.3f}, test acc: {test_acc:.3f}')

        self.labels_ = y_pred

    def _generate_masks(self, n: int, test_size: float = 0.2, random_state: int = None, shuffle: bool = True) -> tuple:
        """ Generate train and test masks.

        Parameters
        ----------
        n : int
            Number of nodes in graph.
        test_size : float (default=0.2)
            Should be between 0 and 1 and represents the proportion of the nodes to include in test set.
        random_state : int
            Pass an int for reproducible results across multiple runs.
        shuffle : bool (default=True)
            If True, shuffles samples before split.

        Returns
        -------
        tuple
            train_mask, test_maks: Arrays of booleans used as masks to filter graph data.
        """

        check_is_proba(test_size)

        if random_state is not None:
            np.random.seed(random_state)

        mask = np.zeros(n, dtype=bool)
        self.train_mask = mask.copy()
        self.train_mask[:int(test_size * n)] = True

        if shuffle:
            np.random.shuffle(self.train_mask)

        self.test_mask = np.invert(self.train_mask)

        return self.train_mask, self.test_mask

    def predict(self, nodes: np.ndarray) -> np.ndarray:
        """Predict class labels for nodes in `nodes`.

        Parameters
        ----------
        nodes : np.ndarray
            Data matrix for which we want to get predictions. Each row in `nodes` corresponds to the feature vector
            of a node. `nodes` as shape :math:`(x \times n_feat)`, with :math:`x` the number of nodes and
            :math:`n_feat` the number of features used in model training (i.e `in_channels`).

        Returns
        -------
        np.ndarray
            Array containing the class labels for each node in `nodes`.
        """
        check_format(nodes)

        if nodes.ndim < 2:
            nodes = nodes.reshape(1, -1)

        layers = self._get_layers()
        h = nodes

        for l in layers:
            activation_function = ACTIVATIONS[l.activation]
            h = h.dot(l.W)
            if l.use_bias:
                h += l.bias
            h = activation_function(h)

        logits, y_pred = self._compute_predictions(h)

        return y_pred
