#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 5, 2018
@author: Quentin Lutz <qlutz@enst.fr>
Nathan de Lara <ndelara@enst.fr>
"""
from csv import reader
from typing import Optional
from xml.etree import ElementTree


import numpy as np
from scipy import sparse

from sknetwork.utils import Bunch


def parse_tsv(file: str, directed: bool = False, bipartite: bool = False, weighted: Optional[bool] = None,
              named: Optional[bool] = None, comment: str = '%#', delimiter: str = None, reindex: bool = True) -> Bunch:
    """Parser for Tabulation-Separated, Comma-Separated or Space-Separated (or other) Values datasets.

    Parameters
    ----------
    file : str
        The path to the dataset in TSV format
    directed : bool
        If ``True``, considers the graph as directed.
    bipartite : bool
        If ``True``, returns a biadjacency matrix of shape (n1, n2).
    weighted : Optional[bool]
        Retrieves the weights in the third field of the file. None makes a guess based on the first lines.
    named : Optional[bool]
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
    graph: :class:`Bunch`
    """
    reindexed = False
    header_len = -1
    possible_delimiters = ['\t', ',', ' ']
    del_count = np.zeros(3, dtype=int)
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
        guess_delimiter = possible_delimiters[int(np.argmax(del_count))]
        guess_weighted = bool(min([line.count(guess_delimiter) for line in lines]) - 1)
        guess_named = not all([all([el.strip().isdigit() for el in line.split(guess_delimiter)][0:2])
                               for line in lines])
    if weighted is None:
        weighted = guess_weighted
    if named is None:
        named = guess_named
    if delimiter is None:
        delimiter = guess_delimiter

    row, col, data = [], [], []
    with open(file, 'r', encoding='utf-8') as f:
        for i in range(header_len):
            f.readline()
        csv_reader = reader(f, delimiter=delimiter)
        for line in csv_reader:
            if line[0] not in comment:
                if named:
                    row.append(line[0])
                    col.append(line[1])
                else:
                    row.append(int(line[0]))
                    col.append(int(line[1]))
                if weighted:
                    data.append(float(line[2]))
    n_edges = len(row)

    graph = Bunch()
    if bipartite:
        names_row, row = np.unique(row, return_inverse=True)
        names_col, col = np.unique(col, return_inverse=True)
        if not reindex:
            n_row = max(names_row) + 1
            n_col = max(names_col) + 1
        else:
            n_row = len(names_row)
            n_col = len(names_col)
        if not weighted:
            data = np.ones(n_edges, dtype=bool)
        biadjacency = sparse.csr_matrix((data, (row, col)), shape=(n_row, n_col))
        graph.biadjacency = biadjacency
        if named or reindex:
            graph.names = names_row
            graph.names_row = names_row
            graph.names_col = names_col
    else:
        nodes = np.concatenate((row, col), axis=None)
        names, new_nodes = np.unique(nodes, return_inverse=True)
        if not reindex:
            n_nodes = max(names) + 1
        else:
            n_nodes = len(names)
        if named:
            row = new_nodes[:n_edges]
            col = new_nodes[n_edges:]
        else:
            if not all(names == range(len(names))) and reindex:
                reindexed = True
                row = new_nodes[:n_edges]
                col = new_nodes[n_edges:]
        if not weighted:
            data = np.ones(n_edges, dtype=int)
        adjacency = sparse.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        if not directed:
            adjacency += adjacency.T
        graph.adjacency = adjacency
        if named or reindexed:
            graph.names = names

    return graph


def parse_labels(file: str) -> np.ndarray:
    """Parser for files with a single entry on each row.

    Parameters
    ----------
    file : str
        The path to the dataset

    Returns
    -------
    labels: np.ndarray
        Labels.
    """
    rows = []
    with open(file, 'r', encoding='utf-8') as f:
        for row in f:
            rows.append(row.strip())
    return np.array(rows)


def parse_header(file: str):
    """Check if the graph is directed, bipartite, weighted."""
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


def parse_metadata(file: str, delimiter: str = ': ') -> Bunch:
    """Extract metadata from the file."""
    metadata = Bunch()
    with open(file, 'r', encoding='utf-8') as f:
        for row in f:
            parts = row.split(delimiter)
            key, value = parts[0], ': '.join(parts[1:]).strip('\n')
            metadata[key] = value
    return metadata


def parse_graphml(file: str) -> Bunch:
    # see http://graphml.graphdrawing.org/primer/graphml-primer.html
    # and http://graphml.graphdrawing.org/specification/dtd.html#top
    # hyperedges and nested graphs (graphs inside nodes or edges) are not supported
    tree = ElementTree.parse(file)
    optimizers = {'n_nodes': None, 'n_edges': None, 'order': 'none',
                  'naming_nodes': True, 'max_out_degree': None, 'symetrize': None}
    graph = None
    for file_element in tree.getroot():
        if file_element.tag == 'graph':
            graph = file_element
            try:
                optimizers['symetrize'] = (graph.attrib['edgedefault'] == 'undirected')
            except KeyError:
                raise ValueError(f'{file} is an invalid GraphML file. edgedefault is unspecified.')
            if 'parse.nodes' in graph.attrib:
                optimizers['n_nodes'] = graph.attrib['parse.nodes']
            else:
                optimizers['n_nodes'] = len([node for node in graph if node.tag == 'node'])
            if 'parse.edges' in graph.attrib:
                optimizers['n_edges'] = graph.attrib['parse.edges']
            else:
                optimizers['n_edges'] = len([edge for edge in graph if edge.tag == 'edges'])
            if 'parse.maxoutdegree' in graph.attrib:
                optimizers['max_out_degree'] = graph.attrib['parse.maxoutdegree']
            if 'parse.nodeids' in graph.attrib:
                optimizers['naming_nodes'] = not (graph.attrib['parse.nodeids'] == 'canonical')
            if 'parse.order' in graph.attrib:
                optimizers['order'] = graph.attrib['parse.order']
            break
    if graph:
        if optimizers['n_nodes'] and optimizers['n_edges'] and optimizers['order'] == 'nodesfirst':
            # parse to CSR directly
            raise NotImplementedError('Efficient to CSR parser not yet implemented')
        elif optimizers['order'] == 'adjacencylist':
            # parse to CSR directly
            raise NotImplementedError('To CSR parser not yet implemented')
        else:
            # parse to LIL then CSR
            raise NotImplementedError('To LIL parser not yet implemented')
    else:
        raise ValueError(f'No graph defined in {file}.')


def parse_xml_to_csr(tree: ElementTree.ElementTree, optimizers: dict) -> Bunch:
    data = Bunch()
    for file_element in tree.getroot():
        keys = {}
        attribute_descriptions = {}
        if file_element.tag == 'key':
            attribute_name = file_element.attrib['attr.name']
            attribute_type = java_type_to_python_type(file_element.attrib['attr.type'])
            default_value = None
            for key_element in file_element:
                if key_element.tag == 'desc':
                    attribute_descriptions[attribute_name] = key_element.text
                elif key_element.tag == 'default':
                    default_value = attribute_type(key_element.text)
            if default_value:
                data[attribute_name] = np.full(optimizers['n_nodes'], default_value, dtype=attribute_type)
            else:
                data[attribute_name] = np.zeros(optimizers['n_nodes'], dtype=attribute_type)
            keys[file_element.attrib['id']] = [attribute_name, attribute_type, default_value]

        elif file_element.tag == 'graph':
            graph = file_element

    pass


def java_type_to_python_type(input: str) -> type:
    if input == 'boolean':
        return bool
    elif input == 'int':
        return int
    elif input == 'string':
        return str
    elif input in ('long', 'float', 'double'):
        return float
