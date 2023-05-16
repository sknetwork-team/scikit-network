#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2023
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Iterable, Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.path.distances import get_distances
from sknetwork.utils.check import check_format, check_square


def get_dag(adjacency: sparse.csr_matrix, source: Optional[Union[int, Iterable]] = None,
            order: Optional[np.ndarray] = None) -> sparse.csr_matrix:
    """Get a Directed Acyclic Graph (DAG) from a graph.
    If the order is specified, keep only edges i -> j such that 0 <= order[i] < order[j].
    Otherwise, use the distances from the specified source in number of hops to order nodes;
    if the source is not specified, use node 0.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    source :
        Source node (or set of source nodes) of the graph, used if the order is not specified.
    order :
        Order of nodes. Negative values ignored.

    Returns
    -------
    dag :
        Adjacency matrix of the directed acyclic graph.
    """
    adjacency = check_format(adjacency, allow_empty=True)
    check_square(adjacency)

    if order is None:
        if source is None:
            source = 0
        order = get_distances(adjacency, source)

    dag = adjacency.astype(bool).tocoo()
    for value in np.unique(order):
        if value < 0:
            dag.data[order[dag.row] == value] = 0
        else:
            dag.data[(order[dag.row] == value) & (order[dag.col] <= value)] = 0
    dag.eliminate_zeros()

    return dag.tocsr()
