#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 4, 2022.
@author: Henry L. Carscadden <hcarscad@gmail.com>
"""
from scipy import sparse
import numpy as np


def find_excess(flow: sparse.csr_matrix, node: int):
    """
    This function computes the excess flow for a node in a preflow.
    Parameters
    ----------
    flow: sparse.csr_matrix
        The matrix showing the flows in the proposed solution.
    node : int
        Node to compute the excess on.

    Returns
    -------
    excess: int
        The amount of excess flow.
    """
    return flow[:, node].sum() - flow[node, :].sum()


def flow_is_feasible(adjacency: sparse.csr_matrix, flow: sparse.csr_matrix, src: int, sink: int):
    """ This utility is used to check if a provided flow satisfies the capacity and flow conservation constraints.

    Parameters
    ----------
    adjacency : sparse.csr_matrix
        The adjacency matrix of the graph with weights containing the edge capacities.
    flow: sparse.csr_matrix
        The matrix showing the flows in the proposed solution.
    src : int
        The node with the flow source.
    sink : int
        The node with that receives the flow.


    Returns
    -------
    feasible: bool
        Whether the follow is feasible, i.e., both the capacity and flow constraints are satisfied.
    """
    if adjacency.shape != flow.shape:
        raise ValueError("The flow has the incorrect shape.")

    rows, cols = adjacency.nonzero()
    for row in rows:
        if row != src and row != sink:
            if flow[row, :].sum() != flow[:, row].sum():
                return False
    for i in range(rows.size):
        row, col = rows[i], cols[i]
        if adjacency[row, col] < flow[row, col]:
            return False
    return True


def get_residual_graph(adjacency: sparse.csr_matrix, flow: sparse.csr_matrix):
    """ This utility is used for maximum flow algorithms to find the residual graph given a flow.

    Parameters
    ----------
    adjacency : sparse.csr_matrix
        The adjacency matrix of the graph with weights containing the edge capacities.
    flow: sparse.csr_matrix
        The matrix showing the flows in the proposed solution.

    Returns
    -------
    residual: sparse.csr_matrix
        The adjacency matrix of the residual graph.
    """
    rows, cols = adjacency.nonzero()
    row_ind = np.zeros(shape=(adjacency.nnz * 2), dtype=np.int)
    col_ind = np.zeros(shape=(adjacency.nnz * 2), dtype=np.int)
    data = np.zeros(shape=(adjacency.nnz * 2), dtype=np.int)
    for i in range(rows.size):
        row, col = rows[i], cols[i]
        curr_ind = i * 2
        row_ind[curr_ind], col_ind[curr_ind] = row, col
        row_ind[curr_ind + 1], col_ind[curr_ind + 1] = col, row
        data[curr_ind] = adjacency[row, col] - flow[row, col]
        data[curr_ind + 1] = flow[row, col]
    residual = sparse.csr_matrix((data, (row_ind, col_ind)), shape=adjacency.shape)
    return residual
