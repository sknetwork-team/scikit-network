#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on January, 22nd 2020
@author: Quentin Lutz <qlutz@enst.fr>
"""

from typing import List, Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.clustering.base import BaseClustering
from sknetwork.clustering.louvain import Louvain
from sknetwork.hierarchy.base import BaseHierarchy
from sknetwork.utils.checks import check_format, is_square, is_symmetric


class Tree:
    """
    n-ary tree.

    Parameters
    ----------
    label: Optional[int]
        Label of the node.
    children: list
        The list of children of this node.

    Attributes
    ----------
    leaf_count: int
        The number of leaves under this node.
    """
    def __init__(self, label: Optional[int] = None, children: Optional[List['Tree']] = None,
                 leaf_count: Optional[int] = None):
        self.label = label
        if children is None:
            self.children = []
        else:
            self.children = children
        if leaf_count is None:
            self.leaf_count = sum([child.leaf_count for child in self.children])
        else:
            self.leaf_count = leaf_count

    def add_child(self, child: 'Tree'):
        self.children.append(child)
        self.leaf_count += child.leaf_count

    def to_dendrogram(self, depth=0, current_count=None):
        if current_count is None:
            current_count = self.leaf_count
        full_dendrogram = []
        last_child = None
        for index, child in enumerate(self.children):
            print(depth * '\t', 'Before:', current_count)
            dendrogram, current_count = child.to_dendrogram(depth=depth + 1, current_count=current_count)
            full_dendrogram += dendrogram
            if index > 0:
                full_dendrogram.append([last_child, self.children[index].label,
                                        depth + 1, self.leaf_count])
                current_count += 1
                last_child = current_count
            else:
                last_child = self.children[0].label
            print(depth * '\t', 'After:', current_count)
        if self.label is None:
            self.label = current_count
        return full_dendrogram, current_count


class HierarchicalLouvain(BaseHierarchy):
    """
    Iterative clustering algorithm for hierarchical clustering.

    Using a standard clustering algorithm on each previously obtained cluster yields a tree-like structure.

    Defaults to using Louvain.

    Parameters
    ----------
    engine : str
        ``'default'``, ``'python'`` or ``'numba'``. If ``'default'``, tests if numba is available.

    Attributes
    ----------
    algorithm : :class:`BaseClustering`
        Sub-algorithm for clustering.
    dendrogram_ : numpy array of shape (n, 3)
        Dendrogram.

    Notes
    -----
    Each row of the dendrogram = children, distance, size of cluster :math:`\\sum` nodes.


    See Also
    --------
    scipy.cluster.hierarchy.dendrogram

    """

    def __init__(self, engine: str = 'default', algorithm: Optional['BaseClustering'] = None):
        super(HierarchicalLouvain, self).__init__()

        if algorithm is None:
            self.algorithm = Louvain(engine=engine)
        else:
            self.algorithm = algorithm

    def recursive_louvain(self, adjacency: Union[sparse.csr_matrix, np.ndarray],
                          nodes: Optional[np.ndarray] = None) -> 'Tree':
        """
        Recursive function for fit. Returns a tree rather than a dendrogram.

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
        labels = self.algorithm.fit_transform(adjacency)

        clusters = np.unique(labels)

        if nodes is None:
            nodes = np.arange(adjacency.shape[0])

        result = Tree()

        if len(clusters) == 1:
            if len(nodes) > 1:
                return Tree(children=[Tree(label=node, leaf_count=1) for node in nodes])
            else:
                return Tree(label=nodes[0], leaf_count=1)

        else:
            for cluster in clusters:
                mask = (labels == cluster)
                subgraph_nodes = nodes[mask]
                subgraph = adjacency[mask, :][:, mask]
                result.add_child(self.recursive_louvain(subgraph, subgraph_nodes))
            return result

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

        tree = self.recursive_louvain(adjacency)
        dendrogram = np.array(tree.to_dendrogram()[0])
        max_depth = max(dendrogram[:, 2])
        for row in range(dendrogram.shape[0]):
            dendrogram[row, 2] = max_depth - dendrogram[row, 2]

        dendrogram = dendrogram[
            np.array([(depth, max(n1, n2)) for n1, n2, depth in dendrogram[:, :3]],
                     dtype=[('depth', int), ('max_node', int)]).argsort(order=['depth', 'max_node'])]

        self.dendrogram_ = dendrogram

        return self
