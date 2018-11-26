#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 25, 2018
@author: Thomas Bonald <tbonald@enst.fr>
@author: Bertrand Charpentier <bertrand.charpentier@live.fr>
"""

try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

import numpy as np
from scipy import sparse


class AggregateGraph:
    """
    A class of graphs suitable for aggregation.

    Attributes
    ----------
    n_nodes: number of nodes in the aggregate graph
    agg_adj: aggregated adjacency matrix
    active_nodes: vector of active nodes (not yet aggregated)
    node_sizes: vector of node sizes
    node_weights: vector of node weights
    """

    def __init__(self, adj_matrix, node_weights='degree'):
        """

        Parameters
        ----------
        adj_matrix: adjacency matrix of the graph as SciPy sparse matrix
        node_weights: node weights to be used in the aggregation
        """
        self.n_nodes = adj_matrix.shape[0]
        if type(node_weights) == np.ndarray:
            if len(node_weights) != self.n_nodes:
                raise ValueError('The number of node weights must match the number of nodes.')
            if any(node_weights <= np.zeros(self.n_nodes)):
                raise ValueError('All node weights must be positive.')
            else:
                node_weights_vec = node_weights
        elif type(node_weights) == str:
            if node_weights == 'degree':
                node_weights_vec = adj_matrix.dot(np.ones(self.n_nodes))
            elif node_weights == 'uniform':
                node_weights_vec = np.ones(self.n_nodes) / self.n_nodes
            else:
                raise ValueError('Unknown distribution type.')
        else:
            raise TypeError(
                'Node weights must be a known distribution ("degree" or "uniform" string) or a custom NumPy array.')
        zero_block = sparse.csr_matrix((self.n_nodes - 1, self.n_nodes - 1))
        self.agg_adj = sparse.bmat([(adj_matrix, None), (None, zero_block)])
        self.active_nodes = np.zeros(2 * self.n_nodes - 1)
        self.active_nodes[:self.n_nodes] = np.ones(self.n_nodes)
        self.node_sizes = self.active_nodes
        self.node_weights = np.zeros(2 * self.n_nodes - 1)
        self.node_weights[:self.n_nodes] = node_weights_vec

    def merge(self, nodes):
        """
        Merges two nodes.
        Parameters
        ----------
        nodes: t-uple

        Returns
        -------
        the aggregated graph
        """
        self.active_nodes[1 - self.n_nodes] = 1
        to_add_mat = sparse.lil_matrix(self.agg_adj.shape)
        to_add_vec = sparse.lil_matrix(self.agg_adj[nodes, :].sum(axis = 0))
        to_add_mat[1 - self.n_nodes] = to_add_vec
        to_add_mat.T[1 - self.n_nodes] = to_add_vec
        self.agg_adj += sparse.csr_matrix(to_add_mat)
        self.node_weights[1 - self.n_nodes] = self.node_weights[[nodes]].sum()
        self.node_sizes[1 - self.n_nodes] = self.node_sizes[[nodes]].sum()
        self.active_nodes[[nodes]] = (0, 0)
        self.n_nodes -= 1
        return self


class Paris:
    """
    Agglomerative algorithm.

    Attributes
    ----------
    dendrogram_: dendrogram of the nodes as numpy array

    See Also
    --------
    scipy.cluster.hierarchy.dendrogram

    References
    ----------
    T. Bonald, B. Charpentier, A. Galland, A. Hollocou (2018).
    Hierarchical Graph Clustering using Node Pair Sampling.
    KDD Workshop, 2018.
    """

    def __init__(self):
        self.dendrogram_ = None

    def fit(self, adj_matrix: sparse.csr_matrix, node_weights="degree"):
        """
        Agglomerative clustering using the nearest neighbor chain

        Parameters
        ----------
        adj_matrix: adjacency matrix of the graph to cluster
        node_weights: node weights to be used in the linkage

        Returns
        -------
        self
        """
        if type(adj_matrix) != sparse.csr_matrix:
            raise TypeError('The adjacency matrix must be in a scipy compressed sparse row (csr) format.')
        # check that the graph is not directed
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError('The adjacency matrix must be square.')
        if (adj_matrix != adj_matrix.T).nnz != 0:
            raise ValueError('The graph cannot be directed. Please fit a symmetric adjacency matrix.')
        graph = AggregateGraph(adj_matrix, node_weights)

        return self
