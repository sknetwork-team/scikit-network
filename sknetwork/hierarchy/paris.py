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
    current_node: current node index
    agg_adj: aggregated adjacency matrix
    active_nodes: vector of active nodes (not yet aggregated)
    n_active_nodes: number of active nodes
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
        n_nodes = adj_matrix.shape[0]
        if type(node_weights) == np.ndarray:
            if len(node_weights) != n_nodes:
                raise ValueError('The number of node weights must match the number of nodes.')
            if any(node_weights <= np.zeros(n_nodes)):
                raise ValueError('All node weights must be positive.')
            else:
                node_weights_vec = node_weights
        elif type(node_weights) == str:
            if node_weights == 'degree':
                node_weights_vec = adj_matrix.dot(np.ones(n_nodes))
            elif node_weights == 'uniform':
                node_weights_vec = np.ones(n_nodes) / n_nodes
            else:
                raise ValueError('Unknown distribution type.')
        else:
            raise TypeError(
                'Node weights must be a known distribution ("degree" or "uniform" string) or a custom NumPy array.')
        self.current_node = n_nodes
        # extension of the adjacency matrix to include future nodes (resulting from successive merges)
        zero_block = sparse.csr_matrix((n_nodes - 1, n_nodes - 1))
        self.agg_adj = sparse.csr_matrix(sparse.bmat([(adj_matrix, None), (None, zero_block)]))
        self.active_nodes = np.zeros(2 * n_nodes - 1)
        self.active_nodes[:n_nodes] = np.ones(n_nodes)
        self.n_active_nodes = n_nodes
        self.node_sizes = self.active_nodes.copy()
        self.node_weights = np.zeros(2 * n_nodes - 1)
        self.node_weights[:n_nodes] = node_weights_vec

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
        # new row / column to be added to the aggregate adjacency matrix
        to_add_vec = sparse.lil_matrix(self.agg_adj[nodes, :].sum(axis=0))
        to_add_mat = sparse.lil_matrix(self.agg_adj.shape)
        to_add_mat[self.current_node] = to_add_vec
        to_add_mat[:, self.current_node] = to_add_vec.T
        self.agg_adj += sparse.csr_matrix(to_add_mat)
        self.node_sizes[self.current_node] = self.node_sizes[[nodes]].sum()
        self.node_weights[self.current_node] = self.node_weights[[nodes]].sum()
        self.active_nodes[[nodes]] = (0, 0)
        self.active_nodes[self.current_node] = 1
        self.n_active_nodes -= 1
        self.current_node += 1
        return self


def reorder_dendrogram(dendrogram: np.ndarray):
    """
    Reorder the rows of a dendrogram in increasing order of heights (third column), with proper node reindexing
    Parameters
    ----------
    dendrogram: numpy array

    Returns
    -------
    reordered dendrogram
    """
    n_nodes = np.shape(dendrogram)[0] + 1
    order = np.zeros((2, n_nodes - 1), float)
    order[0] = np.arange(n_nodes - 1)
    order[1] = np.array(dendrogram)[:, 2]
    index = np.lexsort(order)
    node_index = {i: i for i in range(n_nodes)}
    node_index.update({n_nodes + index[t]: n_nodes + t for t in range(n_nodes - 1)})
    return np.array([[node_index[int(dendrogram[t][0])], node_index[int(dendrogram[t][1])],
                      dendrogram[t][2], dendrogram[t][3]] for t in range(n_nodes - 1)])[index,:]


class Paris:
    """
    Agglomerative algorithm.

    Attributes
    ----------
    dendrogram_: dendrogram of the nodes as numpy array of shape number of nodes x 4

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
        node_weights: {'degree', 'uniform'} or numpy array.
            node weights to be used in the similarity between nodes i and j:
             adj_matrix[i,j] / node_weights[i] / node_weights[j]

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

        connected_components = []
        dendrogram = []

        while graph.n_active_nodes > 0:
            node = np.where(graph.active_nodes)[0][0]
            chain = [node]
            while chain:
                node = chain.pop()
                similarity = graph.agg_adj[node].toarray()[0] * graph.active_nodes
                index = np.where(graph.active_nodes)[0]
                similarity[index] /= graph.node_weights[index] * graph.node_weights[node]
                # exclude self-loop
                similarity[node] = 0
                nearest_neighbor = np.argmax(similarity)
                if similarity[nearest_neighbor] > 0:
                    if chain:
                        nearest_neighbor_reverse = chain.pop()
                        if nearest_neighbor_reverse == nearest_neighbor:
                            dendrogram.append([node, nearest_neighbor, 1. / similarity[nearest_neighbor],
                                               graph.node_sizes[node] + graph.node_sizes[nearest_neighbor]])
                            graph.merge((node, nearest_neighbor))
                        else:
                            chain.append(nearest_neighbor_reverse)
                            chain.append(node)
                            chain.append(nearest_neighbor)
                    else:
                        chain.append(node)
                        chain.append(nearest_neighbor)
                else:
                    connected_components.append((node, graph.node_sizes[node]))
                    graph.active_nodes[node] = 0
                    graph.n_active_nodes -= 1

        node, node_size = connected_components.pop()
        for next_node, next_node_size in connected_components:
            node_size += next_node_size
            dendrogram.append([node, next_node, float("inf"), node_size])
            node = graph.current_node
            graph.current_node += 1

        self.dendrogram_ = reorder_dendrogram(np.array(dendrogram))

        return self

