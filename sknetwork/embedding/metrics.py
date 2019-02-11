#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 6 2018

Authors:
Nathan De Lara <ndelara@enst.fr>

Quality metrics for graph embeddings
"""

import numpy as np

from scipy import sparse


def dot_modularity(adjacency_matrix, embedding: np.ndarray, return_all: bool=False):
    """
    Computes the difference of the weighted average dot product between embeddings of pairs of neighbors in the graph
    (fit term) and pairs of nodes in the graph (diversity term). This is metric normalized to lie between -1 and 1.
    If the embeddings are normalized, this reduces to the cosine modularity.
    Parameters
    ----------
    adjacency_matrix: sparse.csr_matrix or np.ndarray
    the adjacency matrix of the graph
    embedding: np.ndarray
    the embedding to evaluate, embedding[i] must represent the embedding of node i
    return_all: whether to return (fit, diversity) or fit - diversity

    Returns
    -------
    a float or a tuple of floats.
    """
    if type(adjacency_matrix) == sparse.csr_matrix:
        adj_matrix = adjacency_matrix
    elif sparse.isspmatrix(adjacency_matrix) or type(adjacency_matrix) == np.ndarray:
        adj_matrix = sparse.csr_matrix(adjacency_matrix)
    else:
        raise TypeError(
            "The argument must be a NumPy array or a SciPy Sparse matrix.")
    n_nodes, m_nodes = adj_matrix.shape
    if n_nodes != m_nodes:
        raise ValueError("The adjacency matrix must be a square matrix.")
    if (adj_matrix != adj_matrix.maximum(adj_matrix.T)).nnz != 0:
        raise ValueError("The adjacency matrix is not symmetric.")

    total_weight: float = adjacency_matrix.data.sum()
    normalization = np.linalg.norm(embedding) ** 2 / embedding.shape[0]

    fit = (np.multiply(embedding, adjacency_matrix.dot(embedding))).sum() / (total_weight * normalization)
    diversity = np.linalg.norm(np.mean(embedding, axis=0)) ** 2 / normalization

    if return_all:
        return fit, diversity
    else:
        return fit - diversity
