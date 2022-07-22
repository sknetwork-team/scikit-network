#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 7, 2022.
@author: Henry L. Carscadden <hcarscad@gmail.com>
"""
from scipy import sparse
import numpy as np

from sknetwork.flow.utils import get_residual_graph
from sknetwork.flow.push_relabel_core import push


def residual_to_flow(adjacency: sparse.csr_matrix, residual: sparse.csr_matrix, preflow: sparse.csr_matrix):
    rows, cols = adjacency.nonzero()
    for i in range(rows.size):
        row, col = rows[i], cols[i]
        preflow[row, col] = residual[col, row]
    return preflow


def get_max_flow(adjacency: sparse.csr_matrix, src: int, sink: int):
    """ This algorithm finds a maximum flow following the classic push-relabel algorithm.

    Parameters
    ----------
    adjacency : sparse.csr_matrix
        The adjacency matrix of the graph with weights containing the edge capacities.
    src : int
        The node with the flow source.
    sink : int
        The node with that receives the flow.

    Returns
    -------
    flow : sparse.csr_matrix
        A maximum flow for the graph.

    Reference
    ---------
    Goldberg, A V; Tarjan, R E (1986). A new approach to the maximum flow problem.
    Proceedings of the eighteenth annual ACM symposium on Theory of computing – STOC 86.
    """
    # Initialize preflow.
    preflow_vals = np.zeros(dtype=np.int32, shape=(adjacency.nnz,))
    preflow = sparse.csr_matrix((preflow_vals, adjacency.indices, adjacency.indptr), shape=adjacency.shape)
    # Initialize the heights to 0.
    heights = np.zeros(shape=(adjacency.shape[0],), dtype=np.int32)
    # Set the height of the src to n.
    heights[src] = adjacency.shape[0]
    # Create the excess flow array.
    excess_flow = np.zeros(shape=(adjacency.shape[0],), dtype=np.int32)
    # Send all the possible flow out of the source.
    for dest_node in adjacency.indices[adjacency.indptr[src]:adjacency.indptr[src + 1]]:
        edge_capacity = adjacency[src, dest_node]
        preflow[src, dest_node] = edge_capacity
        if edge_capacity > 0:
            excess_flow[dest_node] = edge_capacity
    # Initialize the residual graph.
    residual = get_residual_graph(adjacency, preflow)
    non_zero_indices = excess_flow.nonzero()[0]
    # While there are nodes with excess, push or relabel.
    while non_zero_indices.size > 0:
        # while nodes have excess
        # Get nodes from active nodes (excess > 0)
        for curr_node in non_zero_indices:
            pushed = push(residual, curr_node, src, sink, heights, excess_flow)
            # Relabel step
            if not pushed:
                heights[curr_node] = heights[curr_node] + 1
        non_zero_indices = excess_flow.nonzero()[0]

    return residual_to_flow(adjacency, residual, preflow)
