#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Bertrand Charpentier <bertrand.charpentier@live.fr>
"""


import numpy as np
from scipy import sparse
from typing import Union


class AggregateGraph:
    """
    A class of graph suitable for aggregation. Each node represents a cluster.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    node_probs :
        Distribution of node weights.

    Attributes
    ----------
    graph : dict[dict]
        Dictionary of dictionary of edge weights.
    next_cluster : int
        Index of the next cluster (resulting from aggregation).
    cluster_sizes : dict
        Dictionary of cluster sizes.
    cluster_probs : dict
        Dictionary of cluster probabilities.
    """

    def __init__(self, adjacency: sparse.csr_matrix, node_probs: np.ndarray):
        n_nodes = adjacency.shape[0]
        total_weight = adjacency.data.sum()

        self.next_cluster = n_nodes
        self.graph = {}
        for node in range(n_nodes):
            # normalize so that the total weight is equal to 1
            # remove self-loops
            self.graph[node] = {adjacency.indices[i]: adjacency.data[i] / total_weight for i in
                                range(adjacency.indptr[node], adjacency.indptr[node + 1])
                                if adjacency.indices[i] != node}
        self.cluster_sizes = {node: 1 for node in range(n_nodes)}
        self.cluster_probs = {node: node_probs[node] for node in range(n_nodes)}

    def merge(self, node1: int, node2: int) -> 'AggregateGraph':
        """Merges two nodes.

        Parameters
        ----------
        node1, node2 :
            The two nodes to merge.

        Returns
        -------
        self: :class:`AggregateGraph`
            The aggregated graph (without self-loop).
        """
        new_node = self.next_cluster
        self.graph[new_node] = {}
        common_neighbors = set(self.graph[node1]) & set(self.graph[node2]) - {node1, node2}
        for node in common_neighbors:
            self.graph[new_node][node] = self.graph[node1][node] + self.graph[node2][node]
            self.graph[node][new_node] = self.graph[node].pop(node1) + self.graph[node].pop(node2)
        node1_neighbors = set(self.graph[node1]) - set(self.graph[node2]) - {node2}
        for node in node1_neighbors:
            self.graph[new_node][node] = self.graph[node1][node]
            self.graph[node][new_node] = self.graph[node].pop(node1)
        node2_neighbors = set(self.graph[node2]) - set(self.graph[node1]) - {node1}
        for node in node2_neighbors:
            self.graph[new_node][node] = self.graph[node2][node]
            self.graph[node][new_node] = self.graph[node].pop(node2)
        del self.graph[node1]
        del self.graph[node2]
        self.cluster_sizes[new_node] = self.cluster_sizes.pop(node1) + self.cluster_sizes.pop(node2)
        self.cluster_probs[new_node] = self.cluster_probs.pop(node1) + self.cluster_probs.pop(node2)
        self.next_cluster += 1
        return self


def reorder_dendrogram(dendrogram: np.ndarray) -> np.ndarray:
    """
    Get the dendrogram in increasing order of height.

    Parameters
    ----------
    dendrogram:
        Original dendrogram.

    Returns
    -------
    dendrogram: np.ndarray
        Reordered dendrogram.
    """
    n_nodes = np.shape(dendrogram)[0] + 1
    order = np.zeros((2, n_nodes - 1), float)
    order[0] = np.arange(n_nodes - 1)
    order[1] = np.array(dendrogram)[:, 2]
    index = np.lexsort(order)
    node_index = np.arange(2 * n_nodes - 1)
    for t in range(n_nodes - 1):
        node_index[n_nodes + index[t]] = n_nodes + t
    return np.array([[node_index[int(dendrogram[t][0])], node_index[int(dendrogram[t][1])],
                      dendrogram[t][2], dendrogram[t][3]] for t in range(n_nodes - 1)])[index, :]


class Paris:
    """Agglomerative algorithm.

    Attributes
    ----------
    dendrogram_ : numpy array of shape (n_nodes - 1, 4)
        Dendrogram.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse

    >>> # House graph
    >>> row = np.array([0, 0, 1, 1, 2, 3])
    >>> col = np.array([1, 4, 2, 4, 3, 4])
    >>> adjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)), shape=(5, 5))
    >>> adjacency = adjacency + adjacency.T

    >>> paris = Paris()
    >>> paris.fit(adjacency).predict()
    array([1, 1, 0, 0, 1])

    Notes
    -----
    Each row of the dendrogram = i, j, height, size of cluster i + j.

    The similarity between clusters i,j is :math:`\\dfrac{w_{ij}}{w_i w_j}` where

    * :math:`w_{ij}` is the weight of edge i,j, if any, and 0 otherwise,

    * :math:`w_{i}` is the weight of cluster i

    * :math:`w_{j}` is the weight of cluster j

    See Also
    --------
    scipy.cluster.hierarchy.dendrogram

    References
    ----------
    T. Bonald, B. Charpentier, A. Galland, A. Hollocou (2018).
    Hierarchical Graph Clustering using Node Pair Sampling.
    Workshop on Mining and Learning with Graphs.
    https://arxiv.org/abs/1806.01664

    """

    def __init__(self):
        self.dendrogram_ = None

    def fit(self, adjacency: sparse.csr_matrix, node_weights: Union[str, np.ndarray] = 'degree', reorder: bool = True):
        """
        Agglomerative clustering using the nearest neighbor chain.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph to cluster.
        node_weights :
            Node weights used in the linkage.
        reorder :
            If True, reorder the dendrogram in increasing order of heights.

        Returns
        -------
        self: :class:`Paris`
        """
        if type(adjacency) != sparse.csr_matrix:
            raise TypeError('The adjacency matrix must be in a scipy compressed sparse row (csr) format.')
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError('The adjacency matrix must be square.')
        if adjacency.shape[0] <= 1:
            raise ValueError('The graph must contain at least two nodes.')
        if (adjacency != adjacency.T).nnz != 0:
            raise ValueError('The graph cannot be directed. Please fit a symmetric adjacency matrix.')

        n_nodes = adjacency.shape[0]

        if type(node_weights) == np.ndarray:
            if len(node_weights) != n_nodes:
                raise ValueError('The number of node weights must match the number of nodes.')
            else:
                node_probs = node_weights
        elif type(node_weights) == str:
            if node_weights == 'degree':
                node_probs = adjacency.dot(np.ones(n_nodes))
            elif node_weights == 'uniform':
                node_probs = np.ones(n_nodes)
            else:
                raise ValueError('Unknown distribution of node weights.')
        else:
            raise TypeError(
                'Node weights must be a known distribution ("degree" or "uniform" string) or a custom NumPy array.')

        if np.any(node_probs <= 0):
            raise ValueError('All node weights must be positive.')
        else:
            node_probs = node_probs / np.sum(node_probs)

        aggregate_graph = AggregateGraph(adjacency, node_probs)

        connected_components = []
        dendrogram = []

        while len(aggregate_graph.cluster_sizes) > 0:
            node = None
            for node in aggregate_graph.cluster_sizes:
                break
            chain = [node]
            while chain:
                node = chain.pop()
                if aggregate_graph.graph[node]:
                    max_sim = -float("inf")
                    nearest_neighbor = None
                    for neighbor in aggregate_graph.graph[node]:
                        sim = aggregate_graph.graph[node][neighbor] / aggregate_graph.cluster_probs[node] / \
                              aggregate_graph.cluster_probs[neighbor]
                        if sim > max_sim:
                            nearest_neighbor = neighbor
                            max_sim = sim
                        elif sim == max_sim:
                            nearest_neighbor = min(neighbor, nearest_neighbor)
                    if chain:
                        nearest_neighbor_last = chain.pop()
                        if nearest_neighbor_last == nearest_neighbor:
                            dendrogram.append([node, nearest_neighbor, 1. / max_sim,
                                               aggregate_graph.cluster_sizes[node]
                                               + aggregate_graph.cluster_sizes[nearest_neighbor]])
                            aggregate_graph.merge(node, nearest_neighbor)
                        else:
                            chain.append(nearest_neighbor_last)
                            chain.append(node)
                            chain.append(nearest_neighbor)
                    else:
                        chain.append(node)
                        chain.append(nearest_neighbor)
                else:
                    connected_components.append((node, aggregate_graph.cluster_sizes[node]))
                    del aggregate_graph.cluster_sizes[node]

        node, cluster_size = connected_components.pop()
        for next_node, next_cluster_size in connected_components:
            cluster_size += next_cluster_size
            dendrogram.append([node, next_node, float("inf"), cluster_size])
            node = aggregate_graph.next_cluster
            aggregate_graph.next_cluster += 1

        dendrogram = np.array(dendrogram)
        if reorder:
            dendrogram = reorder_dendrogram(dendrogram)

        self.dendrogram_ = dendrogram

        return self

    def predict(self, n_clusters: int = 2, sorted_clusters: bool = False) -> np.ndarray:
        """
        Extract the clustering with given number of clusters from the dendrogram.

        Parameters
        ----------
        n_clusters :
            Number of clusters.
        sorted_clusters :
            If True, sort labels in decreasing order of cluster size.

        Returns
        -------
        labels : np.ndarray
            Cluster index of each node.
        """
        if self.dendrogram_ is None:
            raise ValueError("First fit data using the fit method.")
        n_nodes = np.shape(self.dendrogram_)[0] + 1
        if n_clusters < 1 or n_clusters > n_nodes:
            raise ValueError("The number of clusters must be between 1 and the number of nodes.")

        labels = np.zeros(n_nodes, dtype=int)
        clusters = {node: [node] for node in range(n_nodes)}
        for t in range(n_nodes - n_clusters):
            clusters[n_nodes + t] = clusters.pop(int(self.dendrogram_[t][0])) + \
                                    clusters.pop(int(self.dendrogram_[t][1]))
        clusters = list(clusters.values())
        if sorted_clusters:
            clusters = sorted(clusters, key=len, reverse=True)
        for label, cluster in enumerate(clusters):
            labels[cluster] = label
        return labels
