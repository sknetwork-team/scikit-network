#!/usr/bin/env python3
# coding: utf-8

"""
Created on Nov 6 2018

Authors:
Nathan De Lara <ndelara@enst.fr>

Quality metrics for graph embeddings
"""

import numpy as np

from scipy import sparse


def cosine_modularity(adjacency_matrix, embedding: np.ndarray, return_all: bool=False):
    """
    Computes the difference of the weighted average cosine similarity between pairs of neighbors in the graph
    and pairs of nodes in the graph.
    Parameters
    ----------
    adjacency_matrix: sparse.csr_matrix or np.ndarray
    the adjacency matrix of the graph
    embedding: np.ndarray
    the embedding to evaluate, embedding[i] must represent the embedding of node i
    return_all: whether to return both means besides their difference, or only the difference

    Returns
    -------
    the difference of means or (the difference of means, mean_neighbors, mean_pairs)

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
    normalized_embedding = embedding / embedding.sum(axis=1)[:, np.newaxis]

    neighbors_mean = (np.multiply(normalized_embedding, adjacency_matrix.dot(normalized_embedding))).sum()
    neighbors_mean /= total_weight
    pairs_mean = np.linalg.norm(normalized_embedding.mean(axis=0))**2

    if return_all:
        return neighbors_mean - pairs_mean, neighbors_mean, pairs_mean
    else:
        return neighbors_mean - pairs_mean
