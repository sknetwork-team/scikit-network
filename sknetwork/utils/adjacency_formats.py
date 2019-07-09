#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 8, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse
from sknetwork.utils.sparse_lowrank import SparseLR
from typing import Union


def directed2undirected(adjacency: Union[sparse.csr_matrix, SparseLR],
                        weighted: bool = True) -> Union[sparse.csr_matrix, SparseLR]:
    """

    Parameters
    ----------
    adjacency
    weighted

    Returns
    -------

    """
    if type(adjacency) == sparse.csr_matrix:
        if weighted:
            return sparse.csr_matrix(adjacency + adjacency.T)
        else:
            return adjacency.maximum(adjacency.T)
    elif type(adjacency) == SparseLR:
        if weighted:
            new_tuples = [(y, x) for (x, y) in adjacency.low_rank_tuples]
            return SparseLR(directed2undirected(adjacency.sparse_mat), adjacency.low_rank_tuples + new_tuples)
        else:
            raise ValueError('This function only works with ``weighted==True`` for SparseLR objects.')
    else:
        raise TypeError('Input must be a scipy CSR matrix or a SparseLR object.')


def bipartite2directed(biadjacency: Union[sparse.csr_matrix, SparseLR]) -> Union[sparse.csr_matrix, SparseLR]:
    """

    Parameters
    ----------
    biadjacency

    Returns
    -------

    """
    n, m = biadjacency.shape
    if type(biadjacency) == sparse.csr_matrix:
        return sparse.bmat([[None, biadjacency], [sparse.csr_matrix((m, n)), None]], format='csr')
    elif type(biadjacency) == SparseLR:
        new_tuples = [(np.hstack((x, np.zeros(m))), np.hstack((np.zeros(n), y)))
                      for (x, y) in biadjacency.low_rank_tuples]
        return SparseLR(bipartite2directed(biadjacency.sparse_mat), new_tuples)
    else:
        raise TypeError('Input must be a scipy CSR matrix or a SparseLR object.')


def bipartite2undirected(biadjacency: Union[sparse.csr_matrix, SparseLR]) -> Union[sparse.csr_matrix, SparseLR]:
    """

    Parameters
    ----------
    biadjacency

    Returns
    -------

    """
    if type(biadjacency) == sparse.csr_matrix:
        return sparse.bmat([[None, biadjacency], [biadjacency.T, None]], format='csr')
    elif type(biadjacency) == SparseLR:
        n, m = biadjacency.shape
        new_tuples = []
        for (x, y) in biadjacency.low_rank_tuples:
            new_tuples.append((np.hstack((x, np.zeros(m))), np.hstack((np.zeros(n), y))))
            new_tuples.append((np.hstack((np.zeros(n), y)), np.hstack((x, np.zeros(m)))))
        return SparseLR(bipartite2undirected(biadjacency.sparse_mat), new_tuples)
    else:
        raise TypeError('Input must be a scipy CSR matrix or a SparseLR object.')
