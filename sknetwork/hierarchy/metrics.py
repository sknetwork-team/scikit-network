# -*- coding: utf-8 -*-
# metrics.py - functions for computing metrics on hierarchy
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Bertrand Charpentier <bertrand.charpentier@live.fr>
#
# This file is part of Scikit-network.
#
# NetworkX is distributed under a BSD license; see LICENSE.txt for more information.
"""
Metrics for hierarchy
"""

import numpy as np
import networkx as nx

_AFFINITY = {'unitary', 'weighted'}
_LINKAGE = {'classic'}
_ADMISSIBLE_FUNCTION = {'dasgupta'}


def hierarchical_cost(graph, dendrogram, affinity='weighted', linkage='classic', g=lambda a, b: a + b, check=True):
    """Compute the hierarchichal cost of an undirected graph  hierarchy given a linkage and a admissible function g.
    The graph can be weighted or unweighted

    Parameters
    ----------
    graph : NetworkX graph
        An undirected graph.
    dendrogram : numpy array
        A dendrogram
    affinity : string (default: 'weighted')
        The affinity  can be either 'weighted' or 'unitary'.
        Value 'weighted' takes the attribute 'weight' on the edges.
        Value 'unitary' consider that the edges have a weight equal to 1
    linkage : string, optional (default: 'modular')
        The parameter linkage can be 'single', 'average', 'complete' or 'modular'.

        =============== ========================================
        Value           Linkage
        =============== ========================================
        'single'        max wij
        'average'       min wij
        'complete'      n^2/w*wij/(|i||j|)
        'modular'       w wij/(wi wj)
        =============== ========================================

    g : function, optional (default: lambda a, b: a + b)
        The function g must be admissible. The classic choice is the dasgupta function.
    check : bool, optional (default: True)
        If True, reorder the node labels and check that the edges have the
        'weight' attribute corresponding to the given affinity.

    Returns
    -------
    dendrogram : numpy array
        dendrogram.

    Raises
    ------
    ValueError
        If the affinity or the linkage is not known.
    KeyError
        If all the edges do not have the 'weight' attribute with the 'weighted' affinity.

    Notes
    -----
    This is a classic implementation of the hierarchical cost function

    See Also
    --------
    laplacian_matrix
    """

    if affinity not in _AFFINITY:
        raise ValueError("Unknown affinity type %s."
                         "Valid options are %s" % (affinity, _AFFINITY))

    if linkage not in _LINKAGE:
        raise ValueError("Unknown linkage type %s."
                         "Valid options are %s" % (linkage, _LINKAGE))

    graph_copy = graph.copy()

    if check:

        graph_copy = nx.convert_node_labels_to_integers(graph_copy)

        if affinity == 'unitary':
            for e in graph_copy.edges:
                graph_copy.add_edge(e[0], e[1], weight=1)

        n_edges = len(list(graph_copy.edges()))
        n_weighted_edges = len(nx.get_edge_attributes(graph_copy, 'weight'))
        if affinity == 'weighted' and not n_weighted_edges == n_edges:
            raise KeyError("%s edges among %s do not have the attribute/key \'weigth\'."
                           % (n_edges - n_weighted_edges, n_edges))

    if linkage == 'classic':
        cost = classic_linkage_hierarchical_cost(graph_copy, dendrogram, g)
    else:
        cost = 0.

    return cost


def classic_linkage_hierarchical_cost(graph, dendrogram, g):
    n_nodes = np.shape(dendrogram)[0] + 1

    cost = 0

    u = n_nodes

    cluster_size = {t: 1 for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        linkage = graph[a][b]['weight']
        cost += linkage * g(cluster_size[a], cluster_size[b])

        graph.add_node(u)
        neighbors_a = list(graph.neighbors(a))
        neighbors_b = list(graph.neighbors(b))
        for v in neighbors_a:
            graph.add_edge(u, v, weight=graph[a][v]['weight'])
        for v in neighbors_b:
            if graph.has_edge(u, v):
                graph[u][v]['weight'] += graph[b][v]['weight']
            else:
                graph.add_edge(u, v, weight=graph[b][v]['weight'])
        graph.remove_node(a)
        graph.remove_node(b)
        cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
        u += 1

    return cost
