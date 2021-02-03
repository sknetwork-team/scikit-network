#!/usr/bin/env python3
# coding: utf-8
"""
Created on Dec 2020
@author: Quentin Lutz <qlutz@enst.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.utils.check import check_random_state, check_format
from sknetwork.utils.format import bipartite2undirected
from sknetwork.clustering.louvain import Louvain
from sknetwork.embedding.base import BaseBiEmbedding, BaseEmbedding


class HLouvainEmbedding(BaseEmbedding):
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

    Example
    -------
    >>> from sknetwork.embedding import HLouvainEmbedding
    >>> from sknetwork.data import karate_club
    >>> hlouvain = HLouvainEmbedding(n_components=3)
    >>> adjacency = karate_club()
    >>> embedding = hlouvain.fit_transform(adjacency)
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
        super(HLouvainEmbedding, self).__init__()

        self.n_components = n_components
        self.scale = scale
        self._clustering_method = Louvain(resolution=resolution, tol_optimization=tol_optimization,
                                          tol_aggregation=tol_aggregation, n_aggregations=n_aggregations,
                                          shuffle_nodes=shuffle_nodes, random_state=random_state, verbose=verbose)
        self.random_state = check_random_state(random_state)

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

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]):
        """Embedding of graphs from a clustering obtained with Louvain.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`LouvainNE`
        """
        n = adjacency.shape[0]
        self.embedding_ = np.zeros((n, self.n_components))
        self._recursive_louvain(adjacency, 0)
        return self


class BiHLouvainEmbedding(HLouvainEmbedding, BaseBiEmbedding):
    """Embedding of graphs based on random vectors and clustering by the Louvain method.

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
    embedding_ : array, shape = (n_row, n_components)
        Embedding of the rows.
    embedding_row_ : array, shape = (n_row, n_components)
        Embedding of the rows (copy of **embedding_**).
    embedding_col_ : array, shape = (n_col, n_components)
        Embedding of the columns.
    eigenvalues_ : array, shape = (n_components)
        Eigenvalues in increasing order (first eigenvalue ignored).
    eigenvectors_ : array, shape = (n, n_components)
        Corresponding eigenvectors.
    regularization_ : ``None`` or float
        Regularization factor added to all pairs of nodes.

    Example
    -------
    >>> from sknetwork.embedding import BiHLouvainEmbedding
    >>> from sknetwork.data import movie_actor
    >>> bihlouvain = BiHLouvainEmbedding()
    >>> biadjacency = movie_actor()
    >>> embedding = bihlouvain.fit_transform(biadjacency)
    >>> embedding.shape
    (15, 2)

    References
    ----------
    Belkin, M. & Niyogi, P. (2003). Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
    Neural computation.
    """
    def __init__(self, n_components: int = 2, scale: float = .1, resolution: float = 1, tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, n_aggregations: int = -1, shuffle_nodes: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, verbose: bool = False):
        super(BiHLouvainEmbedding, self).__init__(n_components, scale=scale, resolution=resolution,
                                                  tol_optimization=tol_optimization, tol_aggregation=tol_aggregation,
                                                  n_aggregations=n_aggregations, shuffle_nodes=shuffle_nodes,
                                                  random_state=random_state, verbose=verbose)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiHLouvainEmbedding':
        """Embedding of graphs from a clustering obtained with Louvain.

        Parameters
        ----------
        biadjacency:
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiLouvainNE`
        """
        biadjacency = check_format(biadjacency)
        n_row, _ = biadjacency.shape
        HLouvainEmbedding.fit(self, bipartite2undirected(biadjacency))
        self._split_vars(n_row)

        return self


