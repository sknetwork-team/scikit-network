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
    source : int
        Node to compute the excess on.

    Returns
    -------
    excess: int
        The amount of excess flow.
    """
    return flow[:, node].sum() - flow[node, :].sum()

def flow_is_feasible(adjacency: sparse.csr_matrix, flow: sparse.csr_matrix, src: int, sink: int):
    """ This utility is used to check if a provided flow satifisies the capacity and flow conservation constraints.

    * Digraphs


    Parameters
    ----------
    adjacency : sparse.csr_matrix
        The adjacency matrix of the graph with weights containing the edge capacities.
    flow: sparse.csr_matrix
        The matrix showing the flows in the proposed solution.
    source : int
        The node with the flow source.
    sink : int
        The node with that receives the flow.


    Returns
    -------
    feasible: bool
        This states whether or not the follow is feasible. i.e., both the capacity and flow constraints are satisfied.
    """
    if adjacency.shape != flow.shape:
        raise ValueError("The flow has the incorrect shape.")


    rows, cols = adjacency.nonzero()
    for row in rows:
        if row != src and row != sink:
            if flow[row, :].sum() != flow[:, row].sum():
                return False
    print('here')
    for i in range(rows.size):
        row, col = rows[i], cols[i]
        if adjacency[row, col] < flow[row, col]:
            return False
    return True



def get_residual_graph(adjacency: sparse.csr_matrix, flow: sparse.csr_matrix, src: int, sink: int):
    """ This utility is used for maximum flow algorithms to find the residual graph given a flow.

    * Digraphs


    Parameters
    ----------
    adjacency : sparse.csr_matrix
        The adjacency matrix of the graph with weights containing the edge capacities.
    flow: sparse.csr_matrix
        The matrix showing the flows in the proposed solution.
    source : int
        The node with the flow source.
    sink : int
        The node with that receives the flow.


    Returns
    -------
    residual: sparse.csr_matrix
        The adjacency matrix of the residual graph.
    """
    residual = adjacency.copy()
    rows, cols = adjacency.nonzero()
    for i in range(rows.size):
            row, col = rows[i], cols[i]
            if row == src or col == sink:
                residual[row, col] = residual[row, col] - flow[row, col]
            else:
                residual[row, col] += flow[col, row] - flow[row, col]
    return residual
