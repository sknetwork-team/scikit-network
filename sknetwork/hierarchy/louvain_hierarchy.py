#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2020
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


class LouvainHierarchy(BaseHierarchy):
    """Hierarchical clustering by successive instances of Louvain (top-down).

    Parameters
    ----------
    depth :
        Depth of the tree.
        A negative value is interpreted as no limit (return a tree of maximum depth).
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
        Random number generator or random seed. If ``None``, numpy.random is used.
    verbose :
        Verbose mode.

    Attributes
    ----------
    dendrogram_ :
        Dendrogram of the graph.
    dendrogram_row_ :
        Dendrogram for the rows, for bipartite graphs.
    dendrogram_col_ :
        Dendrogram for the columns, for bipartite graphs.
    dendrogram_full_ :
        Dendrogram for both rows and columns, indexed in this order, for bipartite graphs.

    Example
    -------
    >>> from sknetwork.hierarchy import LouvainHierarchy
    >>> from sknetwork.data import house
    >>> louvain = LouvainHierarchy()
    >>> adjacency = house()
    >>> louvain.fit_transform(adjacency)
    array([[3., 2., 0., 2.],
           [4., 1., 0., 2.],
           [6., 0., 0., 3.],
           [5., 7., 1., 5.]])

    Notes
    -----
    Each row of the dendrogram = merge nodes, distance, size of cluster.

    See Also
    --------
    scipy.cluster.hierarchy.dendrogram
    """

    def __init__(self, depth: int = 3, resolution: float = 1, tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3, n_aggregations: int = -1, shuffle_nodes: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None, verbose: bool = False):
        super(LouvainHierarchy, self).__init__()

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
        adjacency :
            Adjacency matrix of the graph.
        depth :
            Depth of the recursion.
        nodes :
            The indices of the current nodes in the original graph.

        Returns
        -------
        result: list of list of nodes by cluster
        """
        n = adjacency.shape[0]
        if nodes is None:
            nodes = np.arange(n)

        if adjacency.nnz and depth:
            labels = self._clustering_method.fit_transform(adjacency)
        else:
            labels = np.zeros(n)

        clusters = np.unique(labels)

        result = []
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
                result.append(self._recursive_louvain(adjacency_cluster, depth - 1, nodes_cluster))
            return result

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray]) -> 'LouvainHierarchy':
        """Fit algorithm to data.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`LouvainHierarchy`
        """
        self._init_vars()
        input_matrix = check_format(input_matrix)
        adjacency, self.bipartite = get_adjacency(input_matrix)
        tree = self._recursive_louvain(adjacency, self.depth)
        dendrogram, _ = get_dendrogram(tree)
        dendrogram = np.array(dendrogram)
        dendrogram[:, 2] -= min(dendrogram[:, 2])
        self.dendrogram_ = reorder_dendrogram(dendrogram)
        if self.bipartite:
            self._split_vars(input_matrix.shape)
        return self
