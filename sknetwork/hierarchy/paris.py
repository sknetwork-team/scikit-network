#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Bertrand Charpentier <bertrand.charpentier@live.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

from sknetwork.utils.checks import *
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork import njit, types, TypedDict


class AggregateGraph:
    """
    A class of graph suitable for aggregation. Each node represents a cluster.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    node_probs :
        Probability distribution of node weights.

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


@njit
def ints2int(first: np.int32, second: np.int32):
    """Merge two int32 into one single int64

    Parameters
    ----------
    first:
        An int32 making up the first 32 bits of the result
    second:
        An int32 making up the last 32 bits of the result

    Returns
    -------
    result: np.int64
        An int64.
    """
    return (first << 32) | second


@njit
def fit_core(n_nodes: int, node_probs: np.ndarray, data: np.ndarray,
             indices: np.ndarray, indptr: np.ndarray):
    """

    Parameters
    ----------
    n_nodes:
        Number of nodes.
    node_probs:
        Distribution of node weights (sums to 1).
    data:
        CSR format data array of the normalized adjacency matrix.
    indices:
        CSR format index array of the normalized adjacency matrix.
    indptr:
        CSR format index pointer array of the normalized adjacency matrix.

    Returns
    -------
    dendrogram:
        The dendrogram associated with the obtained clustering.
    """
    maxfloat = 1.7976931348623157e+308
    total_weight = data.sum()
    next_cluster = n_nodes
    graph = TypedDict.empty(
        key_type=types.int64,
        value_type=types.float64,
    )

    cluster_sizes = {}
    cluster_probs = {}
    neighbors = [[types.int32(-1)] for _ in range(n_nodes)]
    for node in range(n_nodes):
        node = types.int32(node)
        # normalize so that the total weight is equal to 1
        # remove self-loops
        for i in range(indptr[types.int32(node)], indptr[node + 1]):
            if i == indptr[types.int32(node)]:
                neighbors[node][0] = indices[i]
            else:
                neighbors[node].append(indices[i])
            if indices[i] != node:
                graph[ints2int(node, indices[i])] = data[i] / total_weight
        if node in neighbors[node]:
            neighbors[node].remove(node)
        cluster_sizes[node] = 1
        cluster_probs[node] = node_probs[types.int32(node)]

    connected_components = []
    dendrogram = []

    while len(cluster_sizes) > 0:
        node = None
        for node in cluster_sizes:
            break
        chain = [node]
        while chain:
            node = chain.pop()
            if neighbors[node][0] != -1:
                max_sim = -maxfloat
                nearest_neighbor = None
                for neighbor in neighbors[node]:
                    sim = graph[ints2int(node, neighbor)] / \
                          (cluster_probs[node] * cluster_probs[neighbor])
                    if sim > max_sim:
                        nearest_neighbor = neighbor
                        max_sim = sim
                    elif sim == max_sim:
                        nearest_neighbor = min(neighbor, nearest_neighbor)
                if chain:
                    nearest_neighbor_last = chain.pop()
                    if nearest_neighbor_last == nearest_neighbor:
                        dendrogram.append([node, nearest_neighbor, 1. / max_sim,
                                           cluster_sizes[node]
                                           + cluster_sizes[nearest_neighbor]])
                        new_node = types.int32(next_cluster)
                        neighbors.append([types.int32(-1)])
                        common_neighbors = set(neighbors[node]) & set(neighbors[nearest_neighbor]) - {types.int32(node),
                                                                                                      types.int32(
                                                                                                      nearest_neighbor)}
                        for curr_node in common_neighbors:
                            graph[ints2int(new_node, curr_node)] = graph[ints2int(node, curr_node)] + graph[
                                ints2int(nearest_neighbor, curr_node)]
                            graph[ints2int(curr_node, new_node)] = graph.pop(ints2int(curr_node, node)) + graph.pop(
                                ints2int(curr_node, nearest_neighbor))
                            if neighbors[new_node][0] != -1:
                                neighbors[new_node].append(curr_node)
                            else:
                                neighbors[new_node][0] = curr_node
                            neighbors[curr_node].append(new_node)
                            neighbors[curr_node].remove(node)
                            neighbors[curr_node].remove(nearest_neighbor)
                        node_neighbors = set(neighbors[node]) - set(neighbors[nearest_neighbor]) - {types.int32(
                            nearest_neighbor)}
                        for curr_node in node_neighbors:
                            graph[ints2int(new_node, curr_node)] = graph[ints2int(node, curr_node)]
                            graph[ints2int(curr_node, new_node)] = graph.pop(ints2int(curr_node, node))
                            if neighbors[new_node][0] != -1:
                                neighbors[new_node].append(curr_node)
                            else:
                                neighbors[new_node][0] = curr_node
                            neighbors[curr_node].append(new_node)
                            neighbors[curr_node].remove(node)
                        nearest_neighbor_neighbors = set(neighbors[nearest_neighbor]) - set(neighbors[node]) - {
                            types.int32(node)}
                        for curr_node in nearest_neighbor_neighbors:
                            graph[ints2int(new_node, curr_node)] = graph[ints2int(nearest_neighbor, curr_node)]
                            graph[ints2int(curr_node, new_node)] = graph.pop(ints2int(curr_node, nearest_neighbor))
                            if neighbors[new_node][0] != -1:
                                neighbors[new_node].append(curr_node)
                            else:
                                neighbors[new_node][0] = curr_node
                            neighbors[curr_node].append(new_node)
                            neighbors[curr_node].remove(nearest_neighbor)
                        neighbors[node] = [types.int32(-1)]
                        neighbors[nearest_neighbor] = [types.int32(-1)]
                        cluster_sizes[new_node] = cluster_sizes.pop(node) + cluster_sizes.pop(nearest_neighbor)
                        cluster_probs[new_node] = cluster_probs.pop(node) + cluster_probs.pop(nearest_neighbor)
                        next_cluster += 1
                    else:
                        chain.append(nearest_neighbor_last)
                        chain.append(node)
                        chain.append(nearest_neighbor)
                else:
                    chain.append(node)
                    chain.append(nearest_neighbor)
            else:
                connected_components.append((node, cluster_sizes[node]))
                del cluster_sizes[node]

    node, cluster_size = connected_components.pop()
    for next_node, next_cluster_size in connected_components:
        cluster_size += next_cluster_size
        dendrogram.append([node, next_node, maxfloat, cluster_size])
        node = next_cluster
        next_cluster += 1
    return dendrogram


class Paris(Algorithm):
    """
    Agglomerative clustering algorithm that performs greedy merge of clusters based on their similarity.

    The similarity between clusters i,j is :math:`\\dfrac{A_{ij}}{w_i w_j}` where

    * :math:`A_{ij}` is the weight of edge i,j in the aggregate graph

    * :math:`w_{i}` is the weight of cluster i

    * :math:`w_{j}` is the weight of cluster j.


    Attributes
    ----------
    dendrogram_ : numpy array of shape (n_nodes - 1, 4)
        Dendrogram.

    Examples
    --------
    >>> from sknetwork.toy_graphs import house_graph
    >>> adjacency = house_graph()
    >>> paris = Paris()
    >>> paris.fit(adjacency)
    Paris(engine='numba')
    >>> paris.dendrogram_
    array([[3.        , 2.        , 0.33333333, 2.        ],
           [1.        , 0.        , 0.5       , 2.        ],
           [6.        , 4.        , 0.625     , 3.        ],
           [7.        , 5.        , 1.33333333, 5.        ]])

    Notes
    -----
    Each row of the dendrogram = i, j, height, size of cluster i + j.


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

    def __init__(self, engine: str = 'default'):
        self.dendrogram_ = None
        self.engine = check_engine(engine)

    def fit(self, adjacency: sparse.csr_matrix, weights: Union[str, np.ndarray] = 'degree', reorder: bool = True):
        """
        Agglomerative clustering using the nearest neighbor chain.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph to cluster.
        weights :
            Node weights used in the linkage.
        reorder :
            If True, reorder the dendrogram in increasing order of heights.

        Returns
        -------
        self: :class:`Paris`
        """
        adjacency = check_format(adjacency)

        if not is_square(adjacency):
            raise ValueError('The adjacency matrix must be square.')
        if adjacency.shape[0] <= 1:
            raise ValueError('The graph must contain at least two nodes.')
        if not is_symmetric(adjacency):
            raise ValueError('The graph must be undirected. Please fit a symmetric adjacency matrix.')

        node_probs = check_probs(weights, adjacency, positive_entries=True)

        if self.engine == 'python':
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
                            sim = aggregate_graph.graph[node][neighbor] / \
                                  (aggregate_graph.cluster_probs[node] * aggregate_graph.cluster_probs[neighbor])
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

        elif self.engine == 'numba':

            n_nodes = np.int32(adjacency.shape[0])
            indices, indptr, data = adjacency.indices, adjacency.indptr, adjacency.data

            dendrogram = fit_core(n_nodes, node_probs, data, indices, indptr)
            dendrogram = np.array(dendrogram)
            if reorder:
                dendrogram = reorder_dendrogram(dendrogram)

            self.dendrogram_ = dendrogram

            return self

        else:
            raise ValueError('Unknown engine.')
