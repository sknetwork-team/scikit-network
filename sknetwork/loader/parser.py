#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 5, 2018
@author: Quentin Lutz <qlutz@enst.fr>
"""

from scipy import sparse
import numpy as np


class Parser:
    """
    A collection of parsers to import datasets into a SciPy format

    """
    @staticmethod
    def parse_tsv(path, directed=False, weighted=False, labeled=False, offset=1):
        """
        A parser for Tabulation-Separated Values (TSV) datasets.

        Parameters
        ----------
        path : str, the path to the dataset in TSV format
        directed : bool, ensures the adjacency matrix is symmetric
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
            row = parsed_file[0].astype(int) - offset * np.ones(n_edges, dtype=int)
            col = parsed_file[1].astype(int) - offset * np.ones(n_edges, dtype=int)
        if weighted:
            if len(parsed_file) != 3:
                raise ValueError('No weight data could be derived from the TSV file.')
            else:
                data = parsed_file[2].astype(float)
        else:
            data = np.ones(n_edges, dtype=int)
        adj_matrix = sparse.csr_matrix((data, (row, col)))
        n_nodes = max(adj_matrix.shape)
        adj_matrix.resize((n_nodes, n_nodes))
        if not directed:
            adj_matrix += adj_matrix.transpose()
        if labeled:
            return adj_matrix, labels
        return adj_matrix
