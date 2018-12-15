# -*- coding: utf-8 -*-
# paris.py - agglomerative clustering algorithm
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Thomas Bonald <bonald@enst.fr>
# Copyright 2018 Bertrand Charpentier <bertrand.charpentier@live.fr>
#
# This file is part of Scikit-network.
#
# NetworkX is distributed under a BSD license; see LICENSE.txt for more information.

try:
    # noinspection PyPackageRequirements
    from numba import njit
except ImportError:
    def njit(func):
        return func

import numpy as np
from scipy import sparse


class SimilarityGraph:
    """
    A class of similarity graph suitable for aggregation.

    Attributes
    ----------
    current_node: int
        current node index
    neighbor_sim: np.ndarray of lists of tuples
        list of (neighbor, similarity) for each node
    active_nodes: set
        set of active nodes
    node_sizes: np.ndarray(dtype=int)
        vector of node sizes

    """

    def __init__(self, adj_matrix, node_weights='degree'):
        """

        Parameters
        ----------
        adj_matrix: scipy.csr_matrix
            adjacency matrix of the graph
        node_weights: Union[str,np.ndarray(dtype=float)]
            vector of node weights
    """
        n_nodes = adj_matrix.shape[0]
        if type(node_weights) == np.ndarray:
            if len(node_weights) != n_nodes:
                raise ValueError('The number of node weights must match the number of nodes.')
            if np.any(node_weights <= 0):
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
        # arrays have size 2 * n_nodes - 1 for the future nodes (resulting from merges)
        self.neighbor_sim = np.empty((2 * n_nodes - 1,), dtype=object)
        inv_weights_diag = sparse.spdiags(1 / node_weights_vec, [0], n_nodes, n_nodes, format='csr')
        sim_matrix = inv_weights_diag.dot(adj_matrix.dot(inv_weights_diag))
        for node in range(n_nodes):
            start = sim_matrix.indptr[node]
            end = sim_matrix.indptr[node + 1]
            self.neighbor_sim[node] = [(sim_matrix.indices[i], sim_matrix.data[i]) for i in range(start, end) if
                                       sim_matrix.indices[i] != node]
        self.active_nodes = set(range(n_nodes))
        self.node_sizes = np.zeros(2 * n_nodes - 1, dtype=int)
        self.node_sizes[:n_nodes] = np.ones(n_nodes, dtype=int)
        self.node_weights = np.zeros(2 * n_nodes - 1, dtype=float)
        self.node_weights[:n_nodes] = node_weights_vec

    def merge(self, nodes):
        """
        Merges two nodes.

        Parameters
        ----------
        nodes: tuple
            The two nodes
        Returns
        -------
        The aggregated graph
        """
        first_node = nodes[0]
        second_node = nodes[1]
        first_weight = self.node_weights[first_node]
        second_weight = self.node_weights[second_node]
        new_weight = first_weight + second_weight
        first_list = self.neighbor_sim[first_node]
        second_list = self.neighbor_sim[second_node]
        new_list = []
        while first_list and second_list:
            if first_list[-1][0] > second_list[-1][0]:
                node, sim = first_list.pop()
                if node != second_node:
                    new_sim = sim * first_weight / new_weight
                    new_list.append((node, new_sim))
                    self.neighbor_sim[node].append((self.current_node, new_sim))
            elif first_list[-1][0] < second_list[-1][0]:
                node, sim = second_list.pop()
                if node != first_node:
                    new_sim = sim * second_weight / new_weight
                    new_list.append((node, new_sim))
                    self.neighbor_sim[node].append((self.current_node, new_sim))
            else:
                node, first_sim = first_list.pop()
                _, second_sim = second_list.pop()
                # weighted sum of similarities
                new_sim = (first_sim * first_weight + second_sim * second_weight) / new_weight
                new_list.append((node, new_sim))
                self.neighbor_sim[node].append((self.current_node, new_sim))
        while first_list:
            node, sim = first_list.pop()
            if node != second_node:
                new_sim = sim * first_weight / new_weight
                new_list.append((node, new_sim))
                self.neighbor_sim[node].append((self.current_node, new_sim))
        while second_list:
            node, sim = second_list.pop()
            if node != first_node:
                new_sim = sim * second_weight / new_weight
                new_list.append((node, new_sim))
                self.neighbor_sim[node].append((self.current_node, new_sim))
        self.neighbor_sim[self.current_node] = new_list
        self.active_nodes -= set(nodes)
        self.active_nodes.add(self.current_node)
        self.node_sizes[self.current_node] = self.node_sizes[[nodes]].sum()
        self.node_sizes[[nodes]] = (0, 0)
        self.node_weights[self.current_node] = self.node_weights[[nodes]].sum()
        self.current_node += 1
        return self


def reorder_dendrogram(dendrogram: np.ndarray):
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
    """
    Agglomerative algorithm.

    Attributes
    ----------
    dendrogram_: numpy array of shape (n_nodes - 1) x 4
        dendrogram of the nodes

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

    def fit(self, adj_matrix: sparse.csr_matrix, node_weights="degree", reorder=False):
        """
        Agglomerative clustering using the nearest neighbor chain

        Parameters
        ----------
        adj_matrix: scipy.csr_matrix
            adjacency matrix of the graph to cluster
        node_weights: np.ndarray(dtype=float)
            node weights to be used in the linkage
        reorder: boolean
            reorder the dendrogram in increasing order of heights

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

        sim_graph = SimilarityGraph(adj_matrix, node_weights)

        connected_components = []
        dendrogram = []

        while len(sim_graph.active_nodes) > 0:
            node = None
            for node in sim_graph.active_nodes:
                break
            chain = [node]
            while chain:
                node = chain.pop()
                # filter active nodes
                sim_graph.neighbor_sim[node] = [element for element in sim_graph.neighbor_sim[node] if
                                                (sim_graph.node_sizes[element[0]] > 0)]
                if sim_graph.neighbor_sim[node]:
                    max_sim = -float("inf")
                    nearest_neighbor = None
                    for neighbor, sim in sim_graph.neighbor_sim[node]:
                        if sim > max_sim:
                            nearest_neighbor = neighbor
                            max_sim = sim
                        elif sim == max_sim:
                            nearest_neighbor = min(neighbor, nearest_neighbor)
                    if chain:
                        nearest_neighbor_last = chain.pop()
                        if nearest_neighbor_last == nearest_neighbor:
                            dendrogram.append([node, nearest_neighbor, 1. / max_sim,
                                               sim_graph.node_sizes[node] + sim_graph.node_sizes[nearest_neighbor]])
                            sim_graph.merge((node, nearest_neighbor))
                        else:
                            chain.append(nearest_neighbor_last)
                            chain.append(node)
                            chain.append(nearest_neighbor)
                    else:
                        chain.append(node)
                        chain.append(nearest_neighbor)
                else:
                    connected_components.append((node, sim_graph.node_sizes[node]))
                    sim_graph.active_nodes -= {node}
                    sim_graph.node_sizes[node] = 0

        node, node_size = connected_components.pop()
        for next_node, next_node_size in connected_components:
            node_size += next_node_size
            dendrogram.append([node, next_node, float("inf"), node_size])
            node = sim_graph.current_node
            sim_graph.current_node += 1

        dendrogram = np.array(dendrogram)
        if reorder:
            dendrogram = reorder_dendrogram(dendrogram)

        self.dendrogram_ = dendrogram

        return self
