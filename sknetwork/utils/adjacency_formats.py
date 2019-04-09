#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 8, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from scipy import sparse


def directed2undirected(adjacency: sparse.csr_matrix, weighted: bool = True) -> sparse.csr_matrix:
    if weighted:
        return sparse.csr_matrix(adjacency + adjacency.T)
    else:
        return adjacency.maximum(adjacency.T)


def bipartite2directed(biadjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    n, m = biadjacency.shape
    return sparse.bmat([[None, biadjacency], [sparse.csr_matrix((m, n)), None]], format='csr')


def bipartite2undirected(biadjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    return sparse.bmat([[None, biadjacency], [biadjacency.T, None]], format='csr')
