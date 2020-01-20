#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 5, 2018
@author: Quentin Lutz <qlutz@enst.fr>, Nathan de Lara <ndelara@enst.fr>
"""

from csv import reader
from typing import Tuple, Union, Optional

from numpy import zeros, unique, argmax, ones, concatenate, array, ndarray
from scipy import sparse

from sknetwork.utils import Bunch


def parse_tsv(file: str, directed: bool = False, bipartite: bool = False, weighted: Optional[bool] = None,
              labeled: Optional[bool] = None, comment: str = '%#', delimiter: str = None, reindex: bool = True)\
                -> Union[sparse.csr_matrix, Tuple[sparse.csr_matrix, dict], Tuple[sparse.csr_matrix, dict, dict]]:
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
    weighted : Optional[bool]
        Retrieves the weights in the third field of the file. None makes a guess based on the first lines.
    labeled : Optional[bool]
        Retrieves the names given to the nodes and renumbers them. Returns an additional array. None makes a guess
        based on the first lines.
    comment : str
        Set of characters denoting lines to ignore.
    delimiter : str
        delimiter used in the file. None makes a guess
    reindex : bool
        If True and the graph nodes have numeric values, the size of the returned adjacency will be determined by the
        maximum of those values. Does not work for bipartite graphs.

    Returns
    -------
    adjacency : csr_matrix
        Adjacency or biadjacency matrix of the graph.
    labels : dict, optional
        Label of each node.
    feature_labels : dict, optional
        Label of each feature node (for bipartite graphs).
    """
    reindexed = False
    header_len = -1
    possible_delimiters = ['\t', ',', ' ']
    del_count = zeros(3, dtype=int)
    lines = []
    row = comment
    with open(file, 'r', encoding='utf-8') as f:
        while row[0] in comment:
            row = f.readline()
            header_len += 1
        for line in range(3):
            for i, poss_del in enumerate(possible_delimiters):
                if poss_del in row:
                    del_count[i] += 1
            lines.append(row.rstrip())
            row = f.readline()
        lines = [line for line in lines if line != '']
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
    with open(file, 'r', encoding='utf-8') as f:
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
        if not reindex:
            n_nodes = max(labels) + 1
            n_feature_nodes = max(feature_labels) + 1
        else:
            n_nodes = len(labels)
            n_feature_nodes = len(feature_labels)
        if not weighted:
            dat = ones(n_edges, dtype=bool)
        biadjacency = sparse.csr_matrix((dat, (rows, cols)), shape=(n_nodes, n_feature_nodes))
        if labeled or reindex:
            labels = {i: l for i, l in enumerate(labels)}
            feature_labels = {i: l for i, l in enumerate(feature_labels)}
            return biadjacency, labels, feature_labels
        else:
            return biadjacency
    else:
        nodes = concatenate((rows, cols), axis=None)
        labels, new_nodes = unique(nodes, return_inverse=True)
        if not reindex:
            n_nodes = max(labels) + 1
        else:
            n_nodes = len(labels)
        if labeled:
            rows = new_nodes[:n_edges]
            cols = new_nodes[n_edges:]
        else:
            if not all(labels == range(len(labels))) and reindex:
                reindexed = True
                rows = new_nodes[:n_edges]
                cols = new_nodes[n_edges:]
        if not weighted:
            dat = ones(n_edges, dtype=bool)
        adjacency = sparse.csr_matrix((dat, (rows, cols)), shape=(n_nodes, n_nodes))
        if not directed:
            adjacency += adjacency.transpose()
        if labeled or reindexed:
            labels = {i: l for i, l in enumerate(labels)}
            return adjacency, labels
        else:
            return adjacency


def parse_labels(file: str) -> ndarray:
    """
    A parser for files with a single entry on each row.

    Parameters
    ----------
    file : str
        The path to the dataset

    Returns
    -------
    labels:
        The labels on each row.
    """
    rows = []
    with open(file, 'r', encoding='utf-8') as f:
        for row in f:
            rows.append(row.strip())
    return array(rows)


def parse_hierarchical_labels(file: str, depth: int, full_path: bool = True, delimiter: str = '|||'):
    """
    A parser for files with a single entry of the form ``'String1'<delimiter>...<delimiter>'StringN'`` on each row.

    Parameters
    ----------
    file : str
        The path to the dataset
    depth: int
        The maximum depth to search
    full_path: bool
        Denotes if only the deepest label possible should be returned or if all super categories should be considered
    delimiter: str
        The delimiter on each row

    Returns
    -------
    labels:
        An array of the labels.
    """
    rows = []
    with open(file, 'r', encoding='utf-8') as f:
        for row in f:
            parts = row.strip().split(delimiter)
            if full_path:
                rows.append(".".join(parts[:min(depth, len(parts))]))
            else:
                rows.append(parts[:min(depth, len(parts))][-1])
    return array(rows)


def parse_header(file: str):
    directed, bipartite, weighted = False, False, True
    with open(file, 'r', encoding='utf-8') as f:
        row = f.readline()
        if 'bip' in row:
            bipartite = True
        if 'unweighted' in row:
            weighted = False
        if 'asym' in row:
            directed = True
    return directed, bipartite, weighted


def parse_metadata(file: str, delimiter: str = ': ') -> 'Bunch':
    metadata = Bunch()
    with open(file, 'r', encoding='utf-8') as f:
        for row in f:
            parts = row.split(delimiter)
            key, value = parts[0], ': '.join(parts[1:]).strip('\n')
            metadata[key] = value
    return metadata
