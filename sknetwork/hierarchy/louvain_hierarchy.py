#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in March 2020
@author: Quentin Lutz <qlutz@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.clustering.louvain import Louvain
from sknetwork.hierarchy.base import BaseHierarchy
from sknetwork.hierarchy.postprocess import get_dendrogram, reorder_dendrogram
from sknetwork.utils.check import check_format
from sknetwork.utils.format import get_adjacency


class LouvainIteration(BaseHierarchy):
    """Hierarchical clustering by successive instances of Louvain (top-down).

    Parameters
    ----------
    depth : int
        Depth of the tree.
        A negative value is interpreted as no limit (return a tree of maximum depth).
    resolution : float
        Resolution parameter.
    tol_optimization : float
        Minimum increase in the objective function to enter a new optimization pass.
    tol_aggregation : float
        Minimum increase in the objective function to enter a new aggregation pass.
    n_aggregations : int
        Maximum number of aggregations.
        A negative value is interpreted as no limit.
    shuffle_nodes : bool
        If ``True``,  shuffle nodes before optimization.
    random_state : int
        Random number generator or random seed. If ``None``, numpy.random is used.
    verbose : bool
        Verbose mode.

    Attributes
    ----------
    dendrogram\_ : np.ndarray
        Dendrogram of the graph.

    Example
    -------
    >>> from sknetwork.hierarchy import LouvainIteration
    >>> from sknetwork.data import house
    >>> louvain = LouvainIteration()
    >>> adjacency = house()
    >>> louvain.fit_predict(adjacency)
    array([[3., 2., 1., 2.],
           [4., 1., 1., 2.],
           [6., 0., 1., 3.],
           [5., 7., 2., 5.]])

    Notes
    -----
    Each row of the dendrogram = merge nodes, distance, size of cluster.

    See Also
    --------
    scipy.cluster.hierarchy.dendrogram
    sknetwork.clustering.Louvain
    """

    def __init__(self, depth: int = 3, resolution: float = 1, tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, n_aggregations: int = -1, shuffle_nodes: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, verbose: bool = False):
        super(LouvainIteration, self).__init__()

        self.dendrogram_ = None
        self.depth = depth
        self._clustering_method = Louvain(resolution=resolution, tol_optimization=tol_optimization,
                                          tol_aggregation=tol_aggregation, n_aggregations=n_aggregations,
                                          shuffle_nodes=shuffle_nodes, random_state=random_state, verbose=verbose)
        self.bipartite = None

    def _recursive_louvain(self, adjacency: Union[sparse.csr_matrix, np.ndarray], depth: int,
                           nodes: Optional[np.ndarray] = None):
        """Recursive function for fit.

        Parameters
        ----------
        adjacency : sparse.csr_matrix, np.ndarray
            Adjacency matrix of the graph.
        depth : int
            Depth of the recursion.
        nodes : np.ndarray
            The indices of the current nodes in the original graph.

        Returns
        -------
        tree: recursive list of list of nodes.
        """
        n = adjacency.shape[0]
        if nodes is None:
            nodes = np.arange(n)

        if adjacency.nnz and depth:
            labels = self._clustering_method.fit_predict(adjacency)
        else:
            labels = np.zeros(n)

        clusters = np.unique(labels)

        tree = []
        if len(clusters) == 1:
            if len(nodes) > 1:
                return [[node] for node in nodes]
            else:
                return [nodes[0]]
        else:
            for cluster in clusters:
                mask = (labels == cluster)
                nodes_cluster = nodes[mask]
                adjacency_cluster = adjacency[mask, :][:, mask]
                tree.append(self._recursive_louvain(adjacency_cluster, depth - 1, nodes_cluster))
            return tree

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], force_bipartite: bool = False) \
            -> 'LouvainIteration':
        """Fit algorithm to data.

        Parameters
        ----------
        input_matrix : sparse.csr_matrix, np.ndarray
            Adjacency matrix or biadjacency matrix of the graph.
        force_bipartite :
            If ``True``, force the input matrix to be considered as a biadjacency matrix.

        Returns
        -------
        self: :class:`LouvainIteration`
        """
        self._init_vars()
        adjacency, self.bipartite = get_adjacency(input_matrix, force_bipartite=force_bipartite)
        tree = self._recursive_louvain(adjacency, self.depth)
        dendrogram, _ = get_dendrogram(tree)
        dendrogram = np.array(dendrogram)
        dendrogram[:, 2] += 1 - min(dendrogram[:, 2])
        self.dendrogram_ = reorder_dendrogram(dendrogram)
        if self.bipartite:
            self._split_vars(input_matrix.shape)
        return self


class LouvainHierarchy(BaseHierarchy):
    """Hierarchical clustering by Louvain (bottom-up).

    Each level corresponds to an aggregation step of the Louvain algorithm.

    Parameters
    ----------
    resolution : float
        Resolution parameter.
    tol_optimization : float
        Minimum increase in the objective function to enter a new optimization pass.
    tol_aggregation : float
        Minimum increase in the objective function to enter a new aggregation pass.
    shuffle_nodes : bool
        If ``True``, shuffle nodes before optimization.
    random_state : int
        Random number generator or random seed. If ``None``, numpy.random is used.
    verbose : bool
        Verbose mode.

    Attributes
    ----------
    dendrogram\_ : np.ndarray
        Dendrogram of the graph.

    Example
    -------
    >>> from sknetwork.hierarchy import LouvainHierarchy
    >>> from sknetwork.data import house
    >>> louvain = LouvainHierarchy()
    >>> adjacency = house()
    >>> louvain.fit_predict(adjacency)
    array([[3., 2., 1., 2.],
           [4., 1., 1., 2.],
           [6., 0., 1., 3.],
           [5., 7., 2., 5.]])

    Notes
    -----
    Each row of the dendrogram = merge nodes, distance, size of cluster.

    See Also
    --------
    scipy.cluster.hierarchy.dendrogram
    sknetwork.clustering.Louvain
    """

    def __init__(self, resolution: float = 1, tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, shuffle_nodes: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, verbose: bool = False):
        super(LouvainHierarchy, self).__init__()

        self.dendrogram_ = None
        self._clustering_method = Louvain(resolution=resolution, tol_optimization=tol_optimization,
                                          tol_aggregation=tol_aggregation, n_aggregations=1,
                                          shuffle_nodes=shuffle_nodes, random_state=random_state, verbose=verbose)
        self.bipartite = None

    def _get_hierarchy(self, adjacency: Union[sparse.csr_matrix, np.ndarray]):
        """Get the hierarchy from Louvain.

        Parameters
        ----------
        adjacency : sparse.csr_matrix, np.ndarray
            Adjacency matrix of the graph.

        Returns
        -------
        tree: recursive list of list of nodes
        """
        tree = [[node] for node in range(adjacency.shape[0])]
        labels = self._clustering_method.fit_predict(adjacency)
        labels_unique = np.unique(labels)
        while 1:
            tree = [[tree[node] for node in np.flatnonzero(labels == label)] for label in labels_unique]
            tree = [cluster[0] if len(cluster) == 1 else cluster for cluster in tree]
            aggregate = self._clustering_method.aggregate_
            labels = self._clustering_method.fit_predict(aggregate)
            if len(labels_unique) == len(np.unique(labels)):
                break
            else:
                labels_unique = np.unique(labels)
        return tree

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], force_bipartite: bool = False) \
            -> 'LouvainHierarchy':
        """Fit algorithm to data.

        Parameters
        ----------
        input_matrix : sparse.csr_matrix, np.ndarray
            Adjacency matrix or biadjacency matrix of the graph.
        force_bipartite :
            If ``True``, force the input matrix to be considered as a biadjacency matrix.

        Returns
        -------
        self: :class:`LouvainHierarchy`
        """
        self._init_vars()
        adjacency, self.bipartite = get_adjacency(input_matrix, force_bipartite=force_bipartite)
        tree = self._get_hierarchy(adjacency)
        dendrogram, _ = get_dendrogram(tree)
        dendrogram = np.array(dendrogram)
        dendrogram[:, 2] += 1 - min(dendrogram[:, 2])
        self.dendrogram_ = reorder_dendrogram(dendrogram)
        if self.bipartite:
            self._split_vars(input_matrix.shape)
        return self
