#!/usr/bin/env python3
# coding: utf-8
"""
Created on Dec 2020
@author: Quentin Lutz <qlutz@enst.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.utils.check import check_format, check_random_state
from sknetwork.utils.format import get_adjacency
from sknetwork.clustering.louvain import Louvain
from sknetwork.embedding.base import BaseEmbedding


class LouvainNE(BaseEmbedding):
    """Embedding of graphs based on the hierarchical Louvain algorithm with random scattering per level.

    Parameters
    ----------
    n_components : int
        Dimension of the embedding.
    scale : float
        Dilution factor to be applied on the random vector to be added at each iteration of the clustering method.
    resolution :
        Resolution parameter.
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
        Embedding of the rows, for bipartite graphs.
    embedding_col_ : array, shape = (n_col, n_components)
        Embedding of the columns, for bipartite graphs.
    Example
    -------
    >>> from sknetwork.embedding import LouvainNE
    >>> from sknetwork.data import karate_club
    >>> louvain = LouvainNE(n_components=3)
    >>> adjacency = karate_club()
    >>> embedding = louvain.fit_transform(adjacency)
    >>> embedding.shape
    (34, 3)

    References
    ----------
    Bhowmick, A. K., Meneni, K., Danisch, M., Guillaume, J. L., & Mitra, B. (2020, January).
    `LouvainNE: Hierarchical Louvain Method for High Quality and Scalable Network Embedding.
    <https://hal.archives-ouvertes.fr/hal-02999888/document>`_
    In Proceedings of the 13th International Conference on Web Search and Data Mining (pp. 43-51).
    """
    def __init__(self, n_components: int = 2, scale: float = .1, resolution: float = 1, tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, n_aggregations: int = -1, shuffle_nodes: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, verbose: bool = False):
        super(LouvainNE, self).__init__()

        self.n_components = n_components
        self.scale = scale
        self._clustering_method = Louvain(resolution=resolution, tol_optimization=tol_optimization,
                                          tol_aggregation=tol_aggregation, n_aggregations=n_aggregations,
                                          shuffle_nodes=shuffle_nodes, random_state=random_state, verbose=verbose)
        self.random_state = check_random_state(random_state)
        self.bipartite = None

    def _recursive_louvain(self, adjacency: Union[sparse.csr_matrix, np.ndarray], depth: int,
                           nodes: Optional[np.ndarray] = None):
        """Recursive function for fit, modifies the embedding in place.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.
        depth :
            Depth of the recursion.
        nodes :
            The indices of the current nodes in the original graph.
        """
        n = adjacency.shape[0]
        if nodes is None:
            nodes = np.arange(n)

        if adjacency.nnz:
            labels = self._clustering_method.fit_transform(adjacency)
        else:
            labels = np.zeros(n)

        clusters = np.unique(labels)

        if len(clusters) != 1:
            random_vectors = (self.scale ** depth) * self.random_state.rand(self.n_components, len(clusters))
            for index, cluster in enumerate(clusters):
                mask = (labels == cluster)
                nodes_cluster = nodes[mask]
                self.embedding_[nodes_cluster, :] += random_vectors[:, index]
                n_row = len(mask)
                indptr = np.zeros(n_row + 1, dtype=int)
                indptr[1:] = np.cumsum(mask)
                n_col = indptr[-1]
                combiner = sparse.csr_matrix((np.ones(n_col), np.arange(n_col, dtype=int), indptr),
                                             shape=(n_row, n_col))
                adjacency_cluster = adjacency[mask, :].dot(combiner)
                self._recursive_louvain(adjacency_cluster, depth + 1, nodes_cluster)

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], force_bipartite: bool = False):
        """Embedding of graphs from a clustering obtained with Louvain.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        force_bipartite :
            If ``True``, force the input matrix to be considered as a biadjacency matrix even if square.
        Returns
        -------
        self: :class:`LouvainNE`
        """
        # input
        input_matrix = check_format(input_matrix)
        adjacency, self.bipartite = get_adjacency(input_matrix, force_bipartite=force_bipartite)
        n = adjacency.shape[0]

        # embedding
        self.embedding_ = np.zeros((n, self.n_components))
        self._recursive_louvain(adjacency, 0)

        if self.bipartite:
            self._split_vars(input_matrix.shape)
        return self
