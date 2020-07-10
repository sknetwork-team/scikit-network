#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Union, Iterable, Tuple

import numpy as np
from scipy import sparse


def is_edge(adjacency: sparse.csr_matrix, query: Union[int, Iterable, Tuple]) -> Union[bool, np.ndarray]:
    """Given a query, return whether each edge is actually in the adjacency.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    query : int, Iterable or Tuple
            * If int i, queries (i, j) for all j.
            * If Iterable of integers, return queries (i, j) for i in query, for all j.
            * If tuple (i, j), queries (i, j).
            * If list of tuples or array of shape (n_queries, 2), queries (i, j) in for each line in query.

    Returns
    -------
    y_true : Union[bool, np.ndarray]
        For each element in the query, returns ``True`` if the edge exists in the adjacency and ``False`` otherwise.

    Examples
    --------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> is_edge(adjacency, 0)
    array([False,  True, False, False,  True])
    >>> is_edge(adjacency, [0, 1])
    array([[False,  True, False, False,  True],
           [ True, False,  True, False,  True]])
    >>> is_edge(adjacency, (0, 1))
    True
    >>> is_edge(adjacency, [(0, 1), (0, 2)])
    array([ True, False])
    """
    if np.issubdtype(type(query), np.integer):
        return adjacency[query].toarray().astype(bool).ravel()
    if isinstance(query, Tuple):
        source, target = query
        neighbors = adjacency.indices[adjacency.indptr[source]:adjacency.indptr[source + 1]]
        return bool(np.isin(target, neighbors, assume_unique=True))
    if isinstance(query, list):
        query = np.array(query)
    if isinstance(query, np.ndarray):
        if query.ndim == 1:
            return adjacency[query].toarray().astype(bool)
        elif query.ndim == 2 and query.shape[1] == 2:
            y_true = []
            for edge in query:
                y_true.append(is_edge(adjacency, (edge[0], edge[1])))
            return np.array(y_true)
        else:
            raise ValueError("Query not understood.")
    else:
        raise ValueError("Query not understood.")


def whitened_sigmoid(scores: np.ndarray):
    """Map the entries of a score array to probabilities through

    :math:`\\dfrac{1}{1 + \\exp(-(x - \\mu)/\\sigma)}`,

    where :math:`\\mu` and :math:`\\sigma` are respectively the mean and standard deviation of x.

    Parameters
    ----------
    scores : np.ndarray
        The input array

    Returns
    -------
    probas : np.ndarray
        Array with entries between 0 and 1.

    Examples
    --------
    >>> probas = whitened_sigmoid(np.array([1, 5, 0.25]))
    >>> probas.round(2)
    array([0.37, 0.8 , 0.29])
    >>> probas = whitened_sigmoid(np.array([2, 2, 2]))
    >>> probas
    array([1, 1, 1])
    """
    mu = scores.mean()
    sigma = scores.std()
    if sigma > 0:
        return 1 / (1 + np.exp(-(scores - mu) / sigma))
    else:
        return np.ones_like(scores)
