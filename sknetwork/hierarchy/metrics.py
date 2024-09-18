#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from scipy import sparse
from sknetwork.hierarchy.paris import AggregateGraph

from sknetwork.utils.check import check_format, get_probs, check_square
from sknetwork.utils.check import check_min_size, check_min_nnz
from sknetwork.utils.format import directed2undirected


def _instantiate_vars(adjacency: sparse.csr_matrix, weights: str = 'uniform'):
    """Initialize standard variables for metrics."""
    weights_row = get_probs(weights, adjacency)
    weights_col = get_probs(weights, adjacency.T)
    sym_adjacency = directed2undirected(adjacency)
    aggregate_graph = AggregateGraph(weights_row, weights_col, sym_adjacency.data.astype(float),
                                     sym_adjacency.indices, sym_adjacency.indptr)
    return aggregate_graph, weights_row, weights_col


def get_sampling_distributions(adjacency: sparse.csr_matrix, dendrogram: np.ndarray, weights: str = 'uniform'):
    """Get sampling distributions over each internal node of the tree.
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
    edge_sampling: np.ndarray
        Edge sampling distribution.
    node_sampling: np.ndarray
        Node sampling distribution.
    cluster_weights: np.ndarray
        Cluster weights.
    """
    n = adjacency.shape[0]
    aggregate_graph, weights_row, weights_col = _instantiate_vars(adjacency, weights)
    cluster_weight = np.zeros(n-1)
    edge_sampling = np.zeros(n-1)
    node_sampling = np.zeros(n-1)

    for t in range(n - 1):
        i = int(dendrogram[t][0])
        j = int(dendrogram[t][1])
        if j in aggregate_graph.neighbors[i]:
            edge_sampling[t] += 2 * aggregate_graph.neighbors[i][j]
        node_sampling[t] += aggregate_graph.cluster_out_weights[i] * aggregate_graph.cluster_in_weights[j] + \
            aggregate_graph.cluster_out_weights[j] * aggregate_graph.cluster_in_weights[i]
        cluster_weight[t] = aggregate_graph.cluster_out_weights[i] + aggregate_graph.cluster_out_weights[j] + \
            aggregate_graph.cluster_in_weights[i] + aggregate_graph.cluster_in_weights[j]
        for node in {i, j}:
            if node < n:
                # self-loop
                node_sampling[t] += aggregate_graph.cluster_out_weights[node] * aggregate_graph.cluster_in_weights[node]
                if node in aggregate_graph.neighbors[node]:
                    edge_sampling[t] += aggregate_graph.neighbors[node][node]
        aggregate_graph.merge(i, j)
    return edge_sampling, node_sampling, cluster_weight / 2


def dasgupta_cost(adjacency: sparse.csr_matrix, dendrogram: np.ndarray, weights: str = 'uniform',
                  normalized: bool = False) -> float:
    """Dasgupta's cost of a hierarchy.

    Expected size (weights = ``'uniform'``) or expected volume (weights = ``'degree'``) of the cluster induced by
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
    >>> float(np.round(cost, 2))
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

    edge_sampling, _, cluster_weight = get_sampling_distributions(adjacency, dendrogram, weights)
    cost = edge_sampling.dot(cluster_weight)

    if not normalized:
        if weights == 'degree':
            cost *= adjacency.data.sum()
        else:
            cost *= n

    return cost


def dasgupta_score(adjacency: sparse.csr_matrix, dendrogram: np.ndarray, weights: str = 'uniform') -> float:
    """Dasgupta's score of a hierarchy (quality metric, between 0 and 1).

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
    >>> float(np.round(score, 2))
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
    >>> float(np.round(score, 2))
    0.05

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
    edge_sampling, node_sampling, _ = get_sampling_distributions(adjacency, dendrogram, weights)

    index = np.where(edge_sampling)[0]
    score = edge_sampling[index].dot(np.log(edge_sampling[index] / node_sampling[index]))
    if normalized:
        weights_row = get_probs(weights, adjacency)
        weights_col = get_probs(weights, adjacency.T)
        inv_out_weights = sparse.diags(weights_row, shape=(n, n), format='csr')
        inv_out_weights.data = 1 / inv_out_weights.data
        inv_in_weights = sparse.diags(weights_col, shape=(n, n), format='csr')
        inv_in_weights.data = 1 / inv_in_weights.data
        sampling_ratio = inv_out_weights.dot(adjacency.dot(inv_in_weights))
        inv_out_weights.data = np.ones(len(inv_out_weights.data))
        inv_in_weights.data = np.ones(len(inv_in_weights.data))
        edge_sampling = inv_out_weights.dot(adjacency.dot(inv_in_weights))
        mutual_information = edge_sampling.data.dot(np.log(sampling_ratio.data))
        if mutual_information > 0:
            score /= mutual_information
    return score
