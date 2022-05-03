#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.path.shortest_path import get_distances


def get_diameter(adjacency: Union[sparse.csr_matrix, np.ndarray],
                 n_sources: Optional[Union[int, float]] = None,
                 unweighted: bool = False, n_jobs: Optional[int] = None) -> int:
    """Lower bound on the diameter of a graph which is the length of the longest
    shortest path between two nodes.


    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    n_sources :
        Number of node sources to use for approximation.

        * If None, compute exact diameter.
        * If int, sample n_sample source nodes at random.
        * If float, sample (n_samples * n) source nodes at random.
    unweighted:
        Whether or not the graph is unweighted.
    n_jobs :
        If an integer value is given, denotes the number of workers to use (-1
        means the maximum number will be used).
        If ``None``, no parallel computations are made.

    Returns
    -------
    diameter : int

    Examples
    --------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> d_exact = get_diameter(adjacency)
    >>> d_exact
    2
    >>> d_approx = get_diameter(adjacency, 2)
    >>> d_approx <= d_exact
    True
    >>> d_approx = get_diameter(adjacency, 0.5)
    >>> d_approx <= d_exact
    True

    Notes
    -----
    This is a basic implementation that computes distances between nodes and
    returns the maximum.
    """
    n = adjacency.shape[0]
    if n_sources is None or n_sources == n:
        sources = np.arange(n)
    else:
        if np.issubdtype(type(n_sources), np.floating) and n_sources < 1.:
            n_sources = int(n_sources * n)
        if np.issubdtype(type(n_sources), np.integer) and n_sources <= n:
            sources = np.random.choice(n, n_sources, replace=False)
        else:
            raise ValueError("n_sources must be either None, an integer smaller"
                             "than the number of nodes or a float"
                             "smaller than 1.")
    dists = get_distances(adjacency, sources, method='D', return_predecessors=False,
                          unweighted=unweighted, n_jobs=n_jobs).astype(int)
    return dists.max()


def get_radius(adjacency: Union[sparse.csr_matrix, np.ndarray],
               n_sources: Optional[Union[int, float]] = None,
               unweighted: bool = False, n_jobs: Optional[int] = None) -> int:
    """Computes the radius of the graph which. The radius of the graph is the
    minimum eccentricity of the graph.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    n_sources :
        Number of node sources to use for approximation.

        * If None, compute exact diameter.
        * If int, sample n_sample source nodes at random.
        * If float, sample (n_samples * n) source nodes at random.
    unweighted:
        Whether the graph is unweighted.
    n_jobs :
        If an integer value is given, denotes the number of workers to use (-1
        means the maximum number will be used).
        If ``None``, no parallel computations are made.

    Returns
    -------
    radius : int

    Notes
    -----
    This is a basic implementation that computes distances between nodes and
    returns the maximum.
    """

    # Get the nodes.
    dists = get_distances(adjacency, sources=n_sources, method='D',
                          return_predecessors=False,
                          unweighted=unweighted, n_jobs=n_jobs).astype(int)
    # Get the eccentricities of each node.
    eccentricities = dists.max(axis=1)

    return eccentricities.min()


def get_eccentricity(adjacency: Union[sparse.csr_matrix, np.ndarray], node: int,
                     unweighted: bool = False,
                     n_jobs: Optional[int] = None) -> int:
    """Computes the eccentricity of a node. The eccentricity of a node, u, is the
    maximum length of the shortest paths from u to the other nodes in the graph.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    node:
        The node to compute the eccentricity for.
    unweighted:
        Whether or not the graph is unweighted.
    n_jobs :
        If an integer value is given, denotes the number of workers to use (-1
        means the maximum number will be used).
        If ``None``, no parallel computations are made.

    Returns
    -------
    eccentricity : int
    """

    dists = get_distances(adjacency, node, method='D', return_predecessors=False,
                          unweighted=unweighted, n_jobs=n_jobs).astype(int)
    return dists.max()
