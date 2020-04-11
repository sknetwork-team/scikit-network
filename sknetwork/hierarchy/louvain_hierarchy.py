#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on January, 22nd 2020
@author: Quentin Lutz <qlutz@enst.fr>
"""

from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.clustering.louvain import Louvain
from sknetwork.hierarchy.base import BaseHierarchy
from sknetwork.hierarchy.postprocess import get_dendrogram
from sknetwork.utils.check import check_format, is_square


class LouvainHierarchy(BaseHierarchy):
    """Hierarchical clustering by successive instances of Louvain (top-down).

    * Graphs
    * Digraphs

    Parameters
    ----------
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
    verbose :
        Verbose mode.

    Attributes
    ----------
    dendrogram_ : np.ndarray
        Dendrogram.

    Example
    -------
    >>> from sknetwork.hierarchy import LouvainHierarchy
    >>> from sknetwork.data import karate_club
    >>> louvain = LouvainHierarchy()
    >>> adjacency = karate_club()
    >>> dendrogram = louvain.fit_transform(adjacency)
    >>> dendrogram.shape
    (33, 4)

    Notes
    -----
    Each row of the dendrogram = merge nodes, distance, size of cluster.

    See Also
    --------
    scipy.cluster.hierarchy.dendrogram
    """

    def __init__(self, **kwargs):
        super(LouvainHierarchy, self).__init__()

        self._clustering_method = Louvain(**kwargs)

    def _recursive_louvain(self, adjacency: Union[sparse.csr_matrix, np.ndarray],
                           nodes: Optional[np.ndarray] = None):
        """Recursive function for fit.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.
        nodes :
            The current nodes index in the original graph.

        Returns
        -------
        tree: :class:`Tree`
        """
        labels = self._clustering_method.fit_transform(adjacency)

        clusters = np.unique(labels)

        if nodes is None:
            nodes = np.arange(adjacency.shape[0])

        result = []

        if len(clusters) == 1:
            if len(nodes) > 1:
                return [[node] for node in nodes]
            else:
                return [nodes[0]]

        else:
            for cluster in clusters:
                mask = (labels == cluster)
                subgraph_nodes = nodes[mask]
                subgraph = adjacency[mask, :][:, mask]
                result.append(self._recursive_louvain(subgraph, subgraph_nodes))
            return result

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'LouvainHierarchy':
        """Fit algorithm to data.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`LouvainHierarchy`
        """
        adjacency = check_format(adjacency)
        if not is_square(adjacency):
            raise ValueError('The adjacency matrix is not square.')

        tree = self._recursive_louvain(adjacency)
        dendrogram, _ = get_dendrogram(tree)
        dendrogram = np.array(dendrogram)
        dendrogram[:, 2] -= min(dendrogram[:, 2])

        self.dendrogram_ = dendrogram

        return self
