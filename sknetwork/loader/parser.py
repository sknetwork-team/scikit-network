#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 5, 2018
@author: Quentin Lutz <qlutz@enst.fr>, Nathan de Lara <ndelara@enst.fr>
"""

from scipy import sparse
import numpy as np


class Parser:
    """
    A collection of parsers to import datasets into a SciPy format

    """

    @staticmethod
    def parse_tsv(path, directed=False, bipartite=False, weighted=False, labeled=False, offset=1):
        """
        A parser for Tabulation-Separated Values (TSV) datasets.

        Parameters
        ----------
        path : str, the path to the dataset in TSV format
        directed : bool, ensures the adjacency matrix is symmetric
        bipartite : bool, if True, returns the biadjacency matrix of shape (n1, n2)
        weighted : bool, retrieves the weights in the third field of the file, raises an error if no such field exists
        labeled : bool, retrieves the names given to the nodes and renumbers them. Returns an additional array
        offset : int, renumbers the nodes (useful if the nodes are indexed from a given value other than 0)

        Returns
        -------
        adj_matrix : csr_matrix, the adjacency_matrix of the graph
        labels : numpy.array, optional, an array such that labels[k] is the label given to the k-th node

        """
        parsed_file = np.loadtxt(path, dtype=str, unpack=True, comments='%')
        n_edges = len(parsed_file[0])
        if len(parsed_file) < 2:
            raise ValueError('Unknown format')
        if labeled:
            nodes = np.concatenate(parsed_file[0:2], axis=None)
            labels, new_nodes = np.unique(nodes, return_inverse=True)
            row = new_nodes[:n_edges]
            col = new_nodes[n_edges:]
        else:
            labels = None
            row = parsed_file[0].astype(int) - offset * np.ones(n_edges, dtype=int)
            col = parsed_file[1].astype(int) - offset * np.ones(n_edges, dtype=int)
        if weighted:
            if len(parsed_file) != 3:
                raise ValueError('No weight data could be derived from the TSV file.')
            else:
                data = parsed_file[2].astype(float)
        else:
            data = np.ones(n_edges, dtype=bool)

        n_nodes = max(max(row), max(col)) + 1
        if n_nodes < 2 * 10e9:
            dtype = np.int32
        else:
            dtype = np.int64

        if bipartite:
            adj_matrix = sparse.csr_matrix((data, (row, col)), dtype=dtype)
        else:
            adj_matrix = sparse.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes), dtype=dtype)
            if not directed:
                adj_matrix += adj_matrix.transpose()
        if labeled:
            return adj_matrix, labels
        else:
            return adj_matrix
