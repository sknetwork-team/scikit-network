#!/usr/bin/env python3
# coding: utf-8
"""
Metrics for hierarchy
"""

import numpy as np
from scipy import sparse
from sknetwork.hierarchy.paris import SimilarityGraph


def relative_entropy(adj_matrix: sparse.csr_matrix, dendrogram: np.ndarray, node_weights='degree'):
    """Relative entropy of a hierarchy (quality metric)
     Parameters
     ----------
     adj_matrix :
        Adjacency matrix of the graph.
     dendrogram : array of shape (n_nodes,4)
        Each row contains the two merged nodes, the height in the dendrogram, and the size of the corresponding cluster
     node_weights : Union[str,np.ndarray(dtype=float)]
        Vector of node weights. Default = 'degree', weight of each node in the graph.

     Returns
     -------
     quality : float
         The relative entropy of the hierarchy (quality metric).

     Reference
     ---------
     T. Bonald, B. Charpentier (2018), Learning Graph Representations by Dendrograms, https://arxiv.org/abs/1807.05087
    """

    if type(adj_matrix) != sparse.csr_matrix:
        raise TypeError('The adjacency matrix must be in a scipy compressed sparse row (csr) format.')
    # check that the graph is not directed
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError('The adjacency matrix must be square.')
    if (adj_matrix != adj_matrix.T).nnz != 0:
        raise ValueError('The graph cannot be directed. Please fit a symmetric adjacency matrix.')

    sim_graph = SimilarityGraph(adj_matrix, node_weights)

    quality = 0.
    for t in range(dendrogram.shape[0]):
        first_node = int(dendrogram[t][0])
        second_node = int(dendrogram[t][1])
        first_weight = sim_graph.node_weights[first_node]
        second_weight = sim_graph.node_weights[second_node]
        first_list = sim_graph.neighbor_sim[first_node]
        for neighbor, sim in first_list:
            if neighbor == second_node:
                quality += sim * first_weight * second_weight * np.log(sim)
        sim_graph.merge((first_node, second_node))
    return quality
