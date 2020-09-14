#!/usr/bin/env python3
# coding: utf-8
"""
Created on Sep 2020
@author: Quentin Lutz <qlutz@enst.fr>
"""
from typing import Optional, Union

from scipy import sparse
import numpy as np

from sknetwork.clustering.louvain import BiLouvain
from sknetwork.embedding.base import BaseBiEmbedding
from sknetwork.utils.check import check_random_state


class BiLouvainEmbedding(BaseBiEmbedding):
    """Embedding of bipartite graphs from a clustering obtained with Louvain.

    Parameters
    ----------
    resolution :
        Resolution parameter.
    modularity : str
        Which objective function to maximize. Can be ``'dugue'``, ``'newman'`` or ``'potts'``.
    tol_optimization :
        Minimum increase in the objective function to enter a new optimization pass.
    tol_aggregation :
        Minimum increase in the objective function to enter a new aggregation pass.
    n_aggregations :
        Maximum number of aggregations.
        A negative value is interpreted as no limit.
    shuffle_nodes :
        Enables node shuffling before optimization.
    random_state :
        Random number generator or random seed. If None, numpy.random is used.

    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.
    embedding_row_ : array, shape = (n_row, n_components)
        Embedding of the rows (copy of **embedding_**).
    embedding_col_ : array, shape = (n_col, n_components)
        Embedding of the columns.

    Example
    -------
    >>> from sknetwork.embedding import BiLouvainEmbedding
    >>> from sknetwork.data import star_wars
    >>> bilouvain = BiLouvainEmbedding()
    >>> biadjacency = star_wars()
    >>> embedding = bilouvain.fit_transform(biadjacency)
    >>> embedding.shape
    (34, 2)

    References
    ----------
    Belkin, M. & Niyogi, P. (2003). Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
    Neural computation.
    """
    def __init__(self, resolution: float = 1, merge_isolated: bool = True, modularity: str = 'dugue',
                 tol_optimization: float = 1e-3, tol_aggregation: float = 1e-3, n_aggregations: int = -1,
                 shuffle_nodes: bool = False, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super(BiLouvainEmbedding, self).__init__()
        self.resolution = np.float32(resolution)
        self.modularity = modularity.lower()
        self.tol_optimization = np.float32(tol_optimization)
        self.tol_aggregation = tol_aggregation
        self.n_aggregations = n_aggregations
        self.shuffle_nodes = shuffle_nodes
        self.random_state = check_random_state(random_state)
        self.merge_isolated = merge_isolated

    def fit(self, biadjacency):
        bilouvain = BiLouvain(resolution=self.resolution, modularity=self.modularity,
                              tol_optimization=self.tol_optimization, tol_aggregation=self.tol_aggregation,
                              n_aggregations=self.n_aggregations, shuffle_nodes=self.shuffle_nodes, sort_clusters=True,
                              return_membership=True, return_aggregate=True, random_state=self.random_state)
        bilouvain.fit(biadjacency)

        embedding_row = bilouvain.membership_row_
        embedding_col = bilouvain.membership_col_

        if self.merge_isolated:
            _, counts_row = np.unique(bilouvain.labels_row_, return_counts=True)

            n_clusters = len(counts_row)

            n_isolated_nodes_row = (counts_row == 1).sum()
            if n_isolated_nodes_row:
                n_remaining_row = n_clusters - n_isolated_nodes_row
                indptr_row = np.zeros(n_remaining_row + 2, dtype=int)
                indptr_row[-1] = n_isolated_nodes_row
                combiner_row = sparse.vstack([sparse.eye(n_remaining_row, n_remaining_row + 1, format='csr'),
                                              sparse.csr_matrix((np.ones(n_isolated_nodes_row, dtype=int),
                                                                 np.full(n_isolated_nodes_row, n_remaining_row,
                                                                         dtype=int),
                                                                 np.arange(n_isolated_nodes_row + 1, dtype=int)
                                                                 ))])
                embedding_row = embedding_row.dot(combiner_row)

            _, counts_col = np.unique(bilouvain.labels_col_, return_counts=True)
            n_isolated_nodes_col = (counts_col == 1).sum()
            if n_isolated_nodes_col:
                n_remaining_col = n_clusters - n_isolated_nodes_col
                indptr_col = np.zeros(n_remaining_col + 2, dtype=int)
                indptr_col[-1] = n_isolated_nodes_col
                combiner_col = sparse.vstack([sparse.eye(n_remaining_col, n_remaining_col + 1, format='csr'),
                                              sparse.csr_matrix((np.ones(n_isolated_nodes_col, dtype=int),
                                                                 np.full(n_isolated_nodes_col, n_remaining_col,
                                                                         dtype=int),
                                                                 np.arange(n_isolated_nodes_col + 1, dtype=int)
                                                                 ))])
                embedding_col = embedding_col.dot(combiner_col)

        cluster_degrees_row = np.array(embedding_row.sum(axis=1)).ravel()
        reindex_row = np.argsort(cluster_degrees_row)[::-1]

        cluster_degrees_col = np.array(embedding_col.sum(axis=1)).ravel()
        reindex_col = np.argsort(cluster_degrees_col)[::-1]

        self.embedding_row_ = embedding_row[reindex_row]
        self.embedding_col_ = embedding_col[reindex_col]
        self.embedding_ = self.embedding_row_
