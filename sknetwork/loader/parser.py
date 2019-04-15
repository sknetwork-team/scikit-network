#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 5, 2018
@author: Quentin Lutz <qlutz@enst.fr>, Nathan de Lara <ndelara@enst.fr>
"""

from numpy import zeros, unique, argmax, int32, int64, ones, concatenate, fromfile
from scipy import sparse
from typing import Union
from csv import reader


def parse_tsv(file: str, directed: bool = False, bipartite: bool = False, weighted: bool = None,
              labeled: bool = None, comment: str = '%#', delimiter: str = None) -> Union[sparse.csr_matrix, tuple]:
    """
    A parser for Tabulation-Separated, Comma-Separated or Space-Separated (or other) Values datasets.

    Parameters
    ----------
    file : str
        the path to the dataset in TSV format
    directed : bool
        ensures the adjacency matrix is symmetric if False
    bipartite : bool
        if True, returns the biadjacency matrix of shape (n1, n2)
    weighted : Union[bool, str]
        retrieves the weights in the third field of the file. None makes a guess based on the first lines
    labeled : Union[bool, str]
        retrieves the names given to the nodes and renumbers them. Returns an additional array. None makes a guess
        based on the first lines
    comment : str
        set of characters denoting lines to ignore
    delimiter : str
        delimiter used in the file. None makes a guess

    Returns
    -------
    adj_matrix : csr_matrix
        the adjacency_matrix of the graph
    labels : numpy.array
        optional, an array such that labels[k] is the label given to the k-th node

    """
    header_len = -1
    possible_delimiters = ['\t', ',', ' ']
    del_count = zeros(3, dtype=int)
    lines = []
    row = comment
    with open(file) as f:
        while row[0] in comment:
            row = f.readline()
            header_len += 1
        for line in range(3):
            for i, poss_del in enumerate(possible_delimiters):
                if poss_del in row:
                    del_count[i] += 1
            lines.append(row.rstrip())
            row = f.readline()
        guess_delimiter = possible_delimiters[int(argmax(del_count))]
        guess_weighted = bool(min([line.count(guess_delimiter) for line in lines]) - 1)
        guess_labeled = not all([all([el.isdigit() for el in line.split(guess_delimiter)][0:2]) for line in lines])
    if weighted is None:
        weighted = guess_weighted
    if labeled is None:
        labeled = guess_labeled
    if delimiter is None:
        delimiter = guess_delimiter
    rows, cols, dat = [], [], []
    with open(file, 'r') as f:
        for i in range(header_len):
            f.readline()
        csv_reader = reader(f, delimiter=delimiter)
        for row in csv_reader:
            if row[0] not in comment:
                if labeled:
                    rows.append(row[0])
                    cols.append(row[1])
                else:
                    rows.append(int(row[0]))
                    cols.append(int(row[1]))
                if weighted:
                    dat.append(float(row[2]))
    n_edges = len(rows)
    if labeled:
        nodes = concatenate((rows, cols), axis=None)
        labels, new_nodes = unique(nodes, return_inverse=True)
        rows = new_nodes[:n_edges]
        cols = new_nodes[n_edges:]
    else:
        labels = None
    if not weighted:
        dat = ones(n_edges, dtype=bool)

    n_nodes = max(max(rows), max(cols)) + 1
    if n_nodes < 2 * 10e9:
        dtype = int32
    else:
        dtype = int64

    if bipartite:
        adj_matrix = sparse.csr_matrix((dat, (rows, cols)), dtype=dtype)
    else:
        adj_matrix = sparse.csr_matrix((dat, (rows, cols)), shape=(n_nodes, n_nodes), dtype=dtype)
        if not directed:
            adj_matrix += adj_matrix.transpose()
    if labeled:
        return adj_matrix, labels
    else:
        return adj_matrix


def fast_parse_tsv(file: str, directed: bool = False, bipartite: bool = False, weighted: bool = None,
                   comment: str = '%#', delimiter: str = None) -> Union[sparse.csr_matrix, tuple]:
    """
    A faster parser for Tabulation-Separated or Space-Separated Values datasets.

    It requires that the data is made of integers and floats only and that the indexing need not be adjusted.

    Parameters
    ----------
    file : str
        the path to the dataset in TSV format
    directed : bool
        ensures the adjacency matrix is symmetric if False
    bipartite : bool
        if True, returns the biadjacency matrix of shape (n1, n2)
    weighted : Union[bool, str]
        retrieves the weights in the third field of the file. None makes a guess based on the first lines
    comment : str
        set of characters denoting lines to ignore
    delimiter : str
        delimiter used in the file. None makes a guess

    Returns
    -------
    adj_matrix : csr_matrix
        the adjacency_matrix of the graph
        
    """
    header_len = -1
    possible_delimiters = ['\t', ' ']
    del_count = zeros(2, dtype=int)
    lines = []
    row = comment
    with open(file) as f:
        while row[0] in comment:
            row = f.readline()
            header_len += 1
        for line in range(3):
            for i, poss_del in enumerate(possible_delimiters):
                if poss_del in row:
                    del_count[i] += 1
            lines.append(row.rstrip())
            row = f.readline()
        guess_delimiter = possible_delimiters[int(argmax(del_count))]
        guess_weighted = bool(min([line.count(guess_delimiter) for line in lines]) - 1)
    if weighted is None:
        weighted = guess_weighted
    if delimiter is None:
        delimiter = guess_delimiter
    parsed = fromfile(file, sep=delimiter)
    if weighted:
        rows = parsed[0::3]
        cols = parsed[1::3]
        dat = parsed[2::3]
    else:
        rows = parsed[0::2]
        cols = parsed[1::2]
        dat = ones(len(rows), dtype=int)

    n_nodes = max(max(rows), max(cols)) + 1
    if n_nodes < 2 * 10e9:
        dtype = int32
    else:
        dtype = int64

    if bipartite:
        adj_matrix = sparse.csr_matrix((dat, (rows, cols)), dtype=dtype)
    else:
        adj_matrix = sparse.csr_matrix((dat, (rows, cols)), shape=(n_nodes, n_nodes), dtype=dtype)
        if not directed:
            adj_matrix += adj_matrix.transpose()
    return adj_matrix
