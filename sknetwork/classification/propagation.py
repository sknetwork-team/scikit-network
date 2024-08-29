#!/usr/bin/env python3
# coding: utf-8
"""
Created in April 2020
@author: Thomas Bonald <tbonald@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.classification.base import BaseClassifier
from sknetwork.classification.vote import vote_update
from sknetwork.linalg.normalizer import normalize
from sknetwork.utils.format import get_adjacency_values
from sknetwork.utils.membership import get_membership


class Propagation(BaseClassifier):
    """Node classification by label propagation.

    Parameters
    ----------
    n_iter : float
        Maximum number of iterations (-1 for infinity).
    node_order : str
        * ``'random'``: node labels are updated in random order.
        * ``'increasing'``: node labels are updated by increasing order of (in-) weight.
        * ``'decreasing'``: node labels are updated by decreasing order of (in-) weight.
        * Otherwise, node labels are updated by index order.
    weighted : bool
        If ``True``, the vote of each neighbor is proportional to the edge weight.
        Otherwise, all votes have weight 1.

    Attributes
    ----------
    labels_ : np.ndarray, shape (n_labels,)
        Labels of nodes.
    probs_ : sparse.csr_matrix, shape (n_row, n_labels)
        Probability distribution over labels.
    labels_row_ : np.ndarray
        Labels of rows, for bipartite graphs.
    labels_col_ : np.ndarray
        Labels of columns, for bipartite graphs.
    probs_row_ : sparse.csr_matrix, shape (n_row, n_labels)
        Probability distributions over labels of rows, for bipartite graphs.
    probs_col_ : sparse.csr_matrix, shape (n_col, n_labels)
        Probability distributions over labels of columns, for bipartite graphs.

    Example
    -------
    >>> from sknetwork.classification import Propagation
    >>> from sknetwork.data import karate_club
    >>> propagation = Propagation()
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> labels = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = propagation.fit_predict(adjacency, labels)
    >>> float(np.round(np.mean(labels_pred == labels_true), 2))
    0.94

    References
    ----------
    Raghavan, U. N., Albert, R., & Kumara, S. (2007).
    `Near linear time algorithm to detect community structures in large-scale networks.
    <https://arxiv.org/pdf/0709.2938.pdf>`_
    Physical review E, 76(3), 036106.
    """
    def __init__(self, n_iter: float = -1, node_order: str = None, weighted: bool = True):
        super(Propagation, self).__init__()

        if n_iter < 0:
            self.n_iter = np.inf
        else:
            self.n_iter = n_iter
        self.node_order = node_order
        self.weighted = weighted

    @staticmethod
    def _instantiate_vars(labels: np.ndarray):
        """Instantiate variables for label propagation."""
        n = len(labels)
        if len(set(labels)) == n:
            index_seed = np.arange(n)
            index_remain = np.arange(n)
        else:
            index_seed = np.argwhere(labels >= 0).ravel()
            index_remain = np.argwhere(labels < 0).ravel()
            labels = labels[index_seed]
        return index_seed.astype(np.int32), index_remain.astype(np.int32), labels.astype(np.int32)

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], labels: Union[np.ndarray, list, dict] = None,
            labels_row: Union[np.ndarray, list, dict] = None,
            labels_col: Union[np.ndarray, list, dict] = None) -> 'Propagation':
        """Node classification by label propagation.

        Parameters
        ----------
        input_matrix : sparse.csr_matrix, np.ndarray
            Adjacency matrix or biadjacency matrix of the graph.
        labels : array, list or dict
            Known labels. Negative values ignored.
        labels_row : array, list or dict
            Known labels of rows, for bipartite graphs.
        labels_col : array, list or dict
            Known labels of columns, for bipartite graphs.

        Returns
        -------
        self: :class:`Propagation`
        """
        adjacency, seeds, self.bipartite = get_adjacency_values(input_matrix, values=labels, values_row=labels_row,
                                                                values_col=labels_col, which='labels')
        n = adjacency.shape[0]
        index_seed, index_remain, labels_seed = self._instantiate_vars(seeds)

        if self.node_order == 'random':
            np.random.shuffle(index_remain)
        elif self.node_order == 'decreasing':
            index = np.argsort(-adjacency.T.dot(np.ones(n))).astype(np.int32)
            index_remain = index[index_remain]
        elif self.node_order == 'increasing':
            index = np.argsort(adjacency.T.dot(np.ones(n))).astype(np.int32)
            index_remain = index[index_remain]

        labels = -np.ones(n, dtype=np.int32)
        labels[index_seed] = labels_seed
        labels_remain = np.zeros_like(index_remain, dtype=np.int32)

        indptr = adjacency.indptr.astype(np.int32)
        indices = adjacency.indices.astype(np.int32)
        if self.weighted:
            data = adjacency.data.astype(np.float32)
        else:
            data = np.ones(n, dtype=np.float32)

        t = 0
        while t < self.n_iter and not np.array_equal(labels_remain, labels[index_remain]):
            t += 1
            labels_remain = labels[index_remain].copy()
            labels = np.asarray(vote_update(indptr, indices, data, labels, index_remain))

        probs = get_membership(labels)
        probs = normalize(adjacency.dot(probs))

        self.labels_ = labels
        self.probs_ = probs
        self._split_vars(input_matrix.shape)

        return self
