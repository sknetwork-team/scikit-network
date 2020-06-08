#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from scipy import sparse

from sknetwork.hierarchy.paris import AggregateGraph
from sknetwork.utils.check import check_format, check_probs, check_square
from sknetwork.utils.format import directed2undirected
from sknetwork.utils.check import check_min_size, check_min_nnz


def _instanciate_vars(adjacency: sparse.csr_matrix, weights: str = 'uniform'):
    """Initialize standard variables for metrics."""
    n = adjacency.shape[0]
    weights_row = check_probs(weights, adjacency)
    weights_col = check_probs(weights, adjacency.T)
    sym_adjacency = directed2undirected(adjacency)

    aggregate_graph = AggregateGraph(weights_row, weights_col, sym_adjacency.data.astype(np.float),
                                     sym_adjacency.indices, sym_adjacency.indptr)

    height = np.zeros(n - 1)
    cluster_weight = np.zeros(n - 1)
    edge_sampling = np.zeros(n - 1)

    return aggregate_graph, height, cluster_weight, edge_sampling, weights_row, weights_col


def dasgupta_cost(adjacency: sparse.csr_matrix, dendrogram: np.ndarray, weights: str = 'uniform',
                  normalized: bool = False) -> float:
    """Dasgupta's cost of a hierarchy.

    * Graphs
    * Digraphs

    Expected size (weights = ``'uniform'``) or expected weight (weights = ``'degree'``) of the cluster induced by
    random edge sampling (closest ancestor of the two nodes in the hierarchy).

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    dendrogram :
        Dendrogram.
    weights :
        Weights of nodes.
        ``'degree'`` or ``'uniform'`` (default).
    normalized :
        If ``True``, normalized cost (between 0 and 1).

    Returns
    -------
    cost : float
        Cost.

    Example
    -------
    >>> from sknetwork.hierarchy import dasgupta_score, Paris
    >>> from sknetwork.data import house
    >>> paris = Paris()
    >>> adjacency = house()
    >>> dendrogram = paris.fit_transform(adjacency)
    >>> cost = dasgupta_cost(adjacency, dendrogram)
    >>> np.round(cost, 2)
    3.33

    References
    ----------
    Dasgupta, S. (2016). A cost function for similarity-based hierarchical clustering.
    Proceedings of ACM symposium on Theory of Computing.
    """
    adjacency = check_format(adjacency)
    check_square(adjacency)

    n = adjacency.shape[0]
    check_min_size(n, 2)

    aggregate_graph, height, edge_sampling, cluster_weight, _, _ = _instanciate_vars(adjacency, weights)

    for t in range(n - 1):
        i = int(dendrogram[t][0])
        j = int(dendrogram[t][1])
        if i >= n and height[i - n] == dendrogram[t][2]:
            edge_sampling[t] = edge_sampling[i - n]
            edge_sampling[i - n] = 0
        elif j >= n and height[j - n] == dendrogram[t][2]:
            edge_sampling[t] = edge_sampling[j - n]
            edge_sampling[j - n] = 0
        height[t] = dendrogram[t][2]
        if j in aggregate_graph.neighbors[i]:
            edge_sampling[t] += aggregate_graph.neighbors[i][j]
        cluster_weight[t] = aggregate_graph.cluster_out_weights[i] + aggregate_graph.cluster_out_weights[j] \
            + aggregate_graph.cluster_in_weights[i] + aggregate_graph.cluster_in_weights[j]
        aggregate_graph.merge(i, j)

    cost: float = edge_sampling.dot(cluster_weight) / 2

    if not normalized:
        if weights == 'degree':
            cost *= adjacency.data.sum()
        else:
            cost *= n

    return cost


def dasgupta_score(adjacency: sparse.csr_matrix, dendrogram: np.ndarray, weights: str = 'uniform') -> float:
    """Dasgupta's score of a hierarchy (quality metric, between 0 and 1).

    * Graphs
    * Digraphs

    Defined as 1 - normalized Dasgupta's cost.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    dendrogram :
        Dendrogram.
    weights :
        Weights of nodes.
        ``'degree'`` or ``'uniform'`` (default).

    Returns
    -------
    score : float
        Score.

    Example
    -------
    >>> from sknetwork.hierarchy import dasgupta_score, Paris
    >>> from sknetwork.data import house
    >>> paris = Paris()
    >>> adjacency = house()
    >>> dendrogram = paris.fit_transform(adjacency)
    >>> score = dasgupta_score(adjacency, dendrogram)
    >>> np.round(score, 2)
    0.33

    References
    ----------
    Dasgupta, S. (2016). A cost function for similarity-based hierarchical clustering.
    Proceedings of ACM symposium on Theory of Computing.
    """
    return 1 - dasgupta_cost(adjacency, dendrogram, weights, normalized=True)


def tree_sampling_divergence(adjacency: sparse.csr_matrix, dendrogram: np.ndarray, weights: str = 'degree',
                             normalized: bool = True) -> float:
    """Tree sampling divergence of a hierarchy (quality metric).

    * Graphs
    * Digraphs

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    dendrogram :
        Dendrogram.
    weights :
        Weights of nodes.
        ``'degree'`` (default) or ``'uniform'``.
    normalized :
        If ``True``, normalized score (between 0 and 1).

    Returns
    -------
    score : float
        Score.

    Example
    -------
    >>> from sknetwork.hierarchy import tree_sampling_divergence, Paris
    >>> from sknetwork.data import house
    >>> paris = Paris()
    >>> adjacency = house()
    >>> dendrogram = paris.fit_transform(adjacency)
    >>> score = tree_sampling_divergence(adjacency, dendrogram)
    >>> np.round(score, 2)
    0.52

    References
    ----------
    Charpentier, B. & Bonald, T. (2019).
    `Tree Sampling Divergence: An Information-Theoretic Metric for
    Hierarchical Graph Clustering.
    <https://hal.telecom-paristech.fr/hal-02144394/document>`_
    Proceedings of IJCAI.
    """
    adjacency = check_format(adjacency)
    check_square(adjacency)
    check_min_nnz(adjacency.nnz, 1)
    adjacency = adjacency.astype(float)
    n = adjacency.shape[0]
    check_min_size(n, 2)

    adjacency.data /= adjacency.data.sum()

    aggregate_graph, height, cluster_weight, edge_sampling, weights_row, weights_col = _instanciate_vars(adjacency,
                                                                                                         weights)
    node_sampling = np.zeros(n - 1)

    for t in range(n - 1):
        i = int(dendrogram[t][0])
        j = int(dendrogram[t][1])
        if i >= n and height[i - n] == dendrogram[t][2]:
            edge_sampling[t] = edge_sampling[i - n]
            edge_sampling[i - n] = 0
            node_sampling[t] = node_sampling[i - n]
        elif j >= n and height[j - n] == dendrogram[t][2]:
            edge_sampling[t] = edge_sampling[j - n]
            edge_sampling[j - n] = 0
            node_sampling[t] = node_sampling[j - n]
        if j in aggregate_graph.neighbors[i]:
            edge_sampling[t] += aggregate_graph.neighbors[i][j]
        node_sampling[t] += aggregate_graph.cluster_out_weights[i] * aggregate_graph.cluster_in_weights[j] + \
            aggregate_graph.cluster_out_weights[j] * aggregate_graph.cluster_in_weights[i]
        height[t] = dendrogram[t][2]
        aggregate_graph.merge(i, j)

    index = np.where(edge_sampling)[0]
    score = edge_sampling[index].dot(np.log(edge_sampling[index] / node_sampling[index]))
    if normalized:
        inv_out_weights = sparse.diags(weights_row, shape=(n, n), format='csr')
        inv_out_weights.data = 1 / inv_out_weights.data
        inv_in_weights = sparse.diags(weights_col, shape=(n, n), format='csr')
        inv_in_weights.data = 1 / inv_in_weights.data
        sampling_ratio = inv_out_weights.dot(adjacency.dot(inv_in_weights))
        inv_out_weights.data = np.ones(len(inv_out_weights.data))
        inv_in_weights.data = np.ones(len(inv_in_weights.data))
        edge_sampling = inv_out_weights.dot(adjacency.dot(inv_in_weights))
        mutual_information = edge_sampling.data.dot(np.log(sampling_ratio.data))
        score /= mutual_information
    return score
