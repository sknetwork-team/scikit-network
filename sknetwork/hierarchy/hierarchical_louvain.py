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
from sknetwork.utils.checks import check_format, is_square, is_symmetric


class HierarchicalLouvain(BaseHierarchy):
    """
    Iterative Louvain for hierarchical clustering

    Parameters
    ----------
    engine : str
        ``'default'``, ``'python'`` or ``'numba'``. If ``'default'``, tests if numba is available.

    Attributes
    ----------
    dendrogram_ : numpy array of shape (total number of nodes - 1, 4)
        Dendrogram.

    Notes
    -----
    Each row of the dendrogram = children, distance, size of cluster :math:`i + j`.


    See Also
    --------
    scipy.cluster.hierarchy.dendrogram

    """

    def __init__(self, engine: str = 'default'):
        super(HierarchicalLouvain, self).__init__()

        self.louvain = Louvain(engine=engine)

    def recursive_louvain(self, adjacency: Union[sparse.csr_matrix, np.ndarray], depth: int = 0,
                          nodes: Optional[np.ndarray] = None, next_node: Optional[int] = None):
        """
        Recursive function for fit.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.
        depth :
            The current depth.
        nodes :
            The current nodes index in the original graph.
        next_node :
            The next available node index.

        Returns
        -------
        dendrogram: np.ndarray
        """
        labels = self.louvain.fit_transform(adjacency)

        clusters = np.unique(labels)

        if nodes is None:
            nodes = np.arange(adjacency.shape[0])

        if next_node is None:
            next_node = adjacency.shape[0]

        dendrogram = []

        if len(clusters) == 1:
            return []

        else:
            for cluster in clusters:
                mask = (labels == cluster)
                children = nodes[mask]
                subgraph = adjacency[mask, :][:, mask]
                dendrogram.append([set(children), depth, next_node])
                child_dendrogram = self.recursive_louvain(subgraph, depth + 1, children, next_node + 1)
                dendrogram += child_dendrogram
                next_node += len(child_dendrogram) + 1
            return dendrogram

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'HierarchicalLouvain':
        """
        Hierarchical clustering using several Louvain instances.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`HierarchicalLouvain`
        """
        adjacency = check_format(adjacency)
        if not is_square(adjacency):
            raise ValueError('The adjacency matrix is not square.')
        if not is_symmetric(adjacency):
            raise ValueError('The adjacency matrix is not symmetric.')

        dendrogram = np.array(self.recursive_louvain(adjacency))
        dendrogram = dendrogram[dendrogram[:, 1].argsort()]

        max_depth = dendrogram[-1, 1]

        for row in range(dendrogram.shape[0]):
            dendrogram[row, 1] = max_depth - dendrogram[row, 1]

        self.dendrogram_ = dendrogram

        return self
