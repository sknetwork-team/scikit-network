#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April, 2020
@author: Thomas Bonald <tbonald@enst.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.classification import BaseClassifier
from sknetwork.utils.check import check_seeds
from sknetwork.utils.seeds import stack_seeds
from sknetwork.utils.check import check_format
from sknetwork.utils.format import bipartite2undirected
from sknetwork.utils.membership import membership_matrix
from sknetwork.linalg import normalize


class Propagation(BaseClassifier):
    """Node classification by label propagation.

    * Graphs
    * Digraphs

    Parameters
    ----------
    n_iter : int
        Maximum number of iterations (-1 for infinity).

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.
    membership_ : sparse.csr_matrix
        Membership matrix (columns = labels).

    Example
    -------
    >>> from sknetwork.classification import Propagation
    >>> from sknetwork.data import karate_club
    >>> propagation = Propagation()
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = propagation.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.94
    """
    def __init__(self, n_iter: int = -1):
        super(Propagation, self).__init__()

        if n_iter < 0:
            self.n_iter = np.inf
        else:
            self.n_iter = n_iter

    def _instanciate_vars(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]):
        n = adjacency.shape[0]
        labels = check_seeds(seeds, n).astype(int)
        index_seed = np.argwhere(labels >= 0).ravel()
        index_remain = np.argwhere(labels < 0).ravel()
        labels_seed = labels[index_seed]
        return index_seed, index_remain, labels_seed

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]) \
            -> 'Propagation':
        """Node classification by label propagation.

        Parameters
        ----------
        adjacency :
            Adjacency or biadjacency matrix of the graph.
        seeds :
            Seed nodes. Can be a dict {node: label} or an array where "-1" means no label.

        Returns
        -------
        self: :class:`Propagation`
        """
        adjacency = check_format(adjacency)
        n = adjacency.shape[0]
        index_seed, index_remain, labels_seed = self._instanciate_vars(adjacency, seeds)

        labels = -np.ones(n, dtype=int)
        labels[index_seed] = labels_seed
        labels_remain = np.zeros_like(index_remain, dtype=int)
        t = 0
        while t < self.n_iter and not np.array_equal(labels_remain, labels[index_remain]):
            t += 1
            labels_remain = labels[index_remain].copy()
            for i in index_remain:
                labels_ = labels[adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]]
                labels_ = labels_[labels_ >= 0]
                if len(labels_):
                    labels_unique, counts = np.unique(labels_, return_counts=True)
                    labels[i] = labels_unique[np.argmax(counts)]

        membership = membership_matrix(labels)
        membership = normalize(adjacency.dot(membership))

        self.labels_ = labels
        self.membership_ = membership

        return self


class BiPropagation(Propagation):
    """Node classification by label propagation in bipartite graphs.

    * Bigraphs

    Parameters
    ----------
    n_iter :
        Maximum number of iteration (-1 for infinity).

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each row.
    labels_row_ : np.ndarray
        Label of each row (copy of **labels_**).
    labels_col_ : np.ndarray
        Label of each column.
    membership_ : sparse.csr_matrix
        Membership matrix of rows.
    membership_row_ : sparse.csr_matrix
        Membership matrix of rows (copy of **membership_**).
    membership_col_ : sparse.csr_matrix
        Membership matrix of columns.

    Example
    -------
    >>> from sknetwork.classification import BiPropagation
    >>> from sknetwork.data import movie_actor
    >>> bipropagation = BiPropagation()
    >>> graph = movie_actor(metadata=True)
    >>> biadjacency = graph.biadjacency
    >>> seeds_row = {0: 0, 1: 2, 2: 1}
    >>> len(bipropagation.fit_transform(biadjacency, seeds_row))
    15
    >>> len(bipropagation.labels_col_)
    16


    """
    def __init__(self, n_iter: int = -1):
        super(BiPropagation, self).__init__(n_iter)

        self.label_propagation = Propagation(n_iter)

        self.labels_row_ = None
        self.labels_col_ = None
        self.membership_row_ = None
        self.membership_col_ = None

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray], seeds_row: Union[np.ndarray, dict],
            seeds_col: Optional[Union[np.ndarray, dict]] = None) -> 'BiPropagation':
        """Node classification by k-nearest neighbors in the embedding space.

        Parameters
        ----------
        biadjacency :
            Biadjacency matrix of the graph.
        seeds_row :
            Seed rows. Can be a dict {node: label} or an array where "-1" means no label.
        seeds_col :
            Seed columns (optional). Same format.

        Returns
        -------
        self: :class:`BiPropagation`
        """
        n_row, n_col = biadjacency.shape

        biadjacency = check_format(biadjacency)
        adjacency = bipartite2undirected(biadjacency)
        seeds = stack_seeds(n_row, n_col, seeds_row, seeds_col).astype(int)
        self.label_propagation.fit(adjacency, seeds)

        labels = self.label_propagation.labels_
        membership = self.label_propagation.membership_

        self.labels_row_ = labels[:n_row]
        self.labels_col_ = labels[n_row:]
        self.labels_ = self.labels_row_
        self.membership_row_ = membership[:n_row]
        self.membership_col_ = membership[n_row:]
        self.membership_ = self.membership_row_

        return self
