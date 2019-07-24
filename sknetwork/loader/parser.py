#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 5, 2018
@author: Quentin Lutz <qlutz@enst.fr>, Nathan de Lara <ndelara@enst.fr>
"""

from numpy import zeros, unique, argmax, ones, concatenate
from scipy import sparse
from typing import Tuple, Union
from csv import reader


def parse_tsv(file: str, directed: bool = False, bipartite: bool = False, weighted: bool = None,
              labeled: bool = None, comment: str = '%#', delimiter: str = None) -> Union[sparse.csr_matrix,
                                                                                         Tuple[sparse.csr_matrix, dict],
                                                                                         Tuple[sparse.csr_matrix, dict,
                                                                                               dict]]:
    """
    A parser for Tabulation-Separated, Comma-Separated or Space-Separated (or other) Values datasets.

    Parameters
    ----------
    file : str
        The path to the dataset in TSV format
    directed : bool
        If False, considers the graph as undirected.
    bipartite : bool
        If True, returns a biadjacency matrix of shape (n, p).
    weighted : Union[NoneType, bool]
        Retrieves the weights in the third field of the file. None makes a guess based on the first lines.
    labeled : Union[NoneType, bool]
        Retrieves the names given to the nodes and renumbers them. Returns an additional array. None makes a guess
        based on the first lines.
    comment : str
        Set of characters denoting lines to ignore.
    delimiter : str
        delimiter used in the file. None makes a guess

    Returns
    -------
    adjacency : csr_matrix
        Adjacency or biadjacency matrix of the graph.
    labels : dict, optional
        Label of each node.
    feature_labels : dict, optional
        Label of each feature node (for bipartite graph).
    """
    reindex = False
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
        guess_labeled = not all([all([el.strip().isdigit() for el in line.split(guess_delimiter)][0:2]) for line
                                 in lines])
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
    if bipartite:
        labels, rows = unique(rows, return_inverse=True)
        feature_labels, cols = unique(cols, return_inverse=True)
        n_nodes = len(labels)
        n_feature_nodes = len(feature_labels)
        if not weighted:
            dat = ones(n_edges, dtype=bool)
        biadjacency = sparse.csr_matrix((dat, (rows, cols)), shape=(n_nodes, n_feature_nodes))
        if labeled:
            labels = {i: l for i, l in enumerate(labels)}
            feature_labels = {i: l for i, l in enumerate(feature_labels)}
            return biadjacency, labels, feature_labels
        else:
            return biadjacency
    else:
        nodes = concatenate((rows, cols), axis=None)
        labels, new_nodes = unique(nodes, return_inverse=True)
        n_nodes = len(labels)
        if labeled:
            rows = new_nodes[:n_edges]
            cols = new_nodes[n_edges:]
        else:
            if not all(labels == range(len(labels))):
                reindex = True
                rows = new_nodes[:n_edges]
                cols = new_nodes[n_edges:]
        if not weighted:
            dat = ones(n_edges, dtype=bool)
        adjacency = sparse.csr_matrix((dat, (rows, cols)), shape=(n_nodes, n_nodes))
        if not directed:
            adjacency += adjacency.transpose()
        if labeled or reindex:
            labels = {i: l for i, l in enumerate(labels)}
            return adjacency, labels
        else:
            return adjacency
