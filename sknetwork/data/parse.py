#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 5, 2018
@author: Quentin Lutz <qlutz@enst.fr>
Nathan de Lara <ndelara@enst.fr>
"""
import warnings
from csv import reader
from typing import Optional, List, Tuple, Union
from xml.etree import ElementTree

import numpy as np
from scipy import sparse

from sknetwork.utils import Bunch
from sknetwork.utils.format import directed2undirected


def load_edge_list(file: str, directed: bool = False, bipartite: bool = False, weighted: Optional[bool] = None,
                   named: Optional[bool] = None, comment: str = '%#', delimiter: str = None, reindex: bool = True,
                   fast_format: bool = True) -> Bunch:
    """Parse Tabulation-Separated, Comma-Separated or Space-Separated (or other) Values datasets in the form of
    edge lists.

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
    fast_format : bool
        If True, assumes that the file is well-formatted:

        * no comments except for the header
        * only 2 or 3 columns
        * only int or float values

    Returns
    -------
    graph: :class:`Bunch`
    """
    header_len, guess_delimiter, guess_weighted, guess_named, guess_string_present, guess_type = scan_header(file,
                                                                                                             comment)

    if weighted is None:
        weighted = guess_weighted
    if named is None:
        named = guess_named
    if delimiter is None:
        delimiter = guess_delimiter

    with open(file, 'r', encoding='utf-8') as f:
        for i in range(header_len):
            f.readline()
        if fast_format and not guess_string_present:
            # fromfile raises a DeprecationWarning on fail. This should be changed to ValueError in the future.
            warnings.filterwarnings("error")
            try:
                parsed = np.fromfile(f, sep=guess_delimiter, dtype=guess_type)
            except (DeprecationWarning, ValueError):
                raise ValueError('File not suitable for fast parsing. Set fast_format to False.')
            warnings.filterwarnings("default")
            n_entries = len(parsed)
            if weighted:
                parsed.resize((n_entries//3, 3))
                row, col, data = parsed[:, 0], parsed[:, 1], parsed[:, 2]
            else:
                parsed.resize((n_entries//2, 2))
                row, col = parsed[:, 0], parsed[:, 1]
                data = np.ones(row.shape[0], dtype=bool)
        else:
            row, col, data = [], [], []
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
            row, col, data = np.array(row), np.array(col), np.array(data)

    return from_edge_list(row=row, col=col, data=data, directed=directed, bipartite=bipartite,
                          reindex=reindex, named=named)


def convert_edge_list(edge_list: Union[np.ndarray, List[Tuple], List[List]], directed: bool = False,
                      bipartite: bool = False, reindex: bool = True, named: Optional[bool] = None) -> Bunch:
    """Turn an edge list into a :class:`Bunch`.

    Parameters
    ----------
    edge_list : Union[np.ndarray, List[Tuple], List[List]]
        The edge list to convert, given as a NumPy array of size (n, 2) or (n, 3) or a list of either lists or tuples of
        length 2 or 3.
    directed : bool
        If ``True``, considers the graph as directed.
    bipartite : bool
        If ``True``, returns a biadjacency matrix of shape (n1, n2).
    reindex : bool
        If True and the graph nodes have numeric values, the size of the returned adjacency will be determined by the
        maximum of those values. Does not work for bipartite graphs.
    named : Optional[bool]
        Retrieves the names given to the nodes and renumbers them. Returns an additional array. None makes a guess
        based on the first lines.

    Returns
    -------
    graph: :class:`Bunch`
    """
    if isinstance(edge_list, list):
        if edge_list:
            if isinstance(edge_list[0], tuple) or isinstance(edge_list[0], list):
                edge_list = np.array(edge_list)
    if isinstance(edge_list, np.ndarray):
        if edge_list.ndim == 2:
            if edge_list.shape[1] == 2:
                row, col, data = edge_list[:, 0], edge_list[:, 1], np.array([])
            elif edge_list.shape[1] == 3:
                row, col, data = edge_list[:, 0], edge_list[:, 1], edge_list[:, 2].astype(float)
            else:
                raise ValueError('Edges must be given as couples or triplets.')
        else:
            raise ValueError('Too many dimensions.')
    else:
        raise TypeError('Edge lists must be given as NumPy arrays or lists of lists or lists of tuples.')
    return from_edge_list(row=row, col=col, data=data, directed=directed, bipartite=bipartite,
                          reindex=reindex, named=named)


def from_edge_list(row: np.ndarray, col: np.ndarray, data: np.ndarray, directed: bool = False, bipartite: bool = False,
                   reindex: bool = True, named: Optional[bool] = None) -> Bunch:
    """Turn an edge list given as a triplet of NumPy arrays into a :class:`Bunch`.

    Parameters
    ----------
    row : np.ndarray
        The array of sources in the graph.
    col : np.ndarray
        The array of targets in the graph.
    data : np.ndarray
        The array of weights in the graph. Pass an empty array for unweighted graphs.
    directed : bool
        If ``True``, considers the graph as directed.
    bipartite : bool
        If ``True``, returns a biadjacency matrix of shape (n1, n2).
    reindex : bool
        If True and the graph nodes have numeric values, the size of the returned adjacency will be determined by the
        maximum of those values. Does not work for bipartite graphs.
    named : Optional[bool]
        Retrieves the names given to the nodes and renumbers them. Returns an additional array. None makes a guess
        based on the first lines.

    Returns
    -------
    graph: :class:`Bunch`
    """
    reindexed = False
    if named is None:
        named = (row.dtype != int) or (col.dtype != int)
    weighted = bool(len(data))
    n_edges = len(row)
    graph = Bunch()
    if bipartite:
        names_row, row = np.unique(row, return_inverse=True)
        names_col, col = np.unique(col, return_inverse=True)
        if not reindex:
            n_row = names_row.max() + 1
            n_col = names_col.max() + 1
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
            n_nodes = names.max() + 1
        else:
            n_nodes = len(names)
        if named:
            row = new_nodes[:n_edges]
            col = new_nodes[n_edges:]
        else:
            should_reindex = not (names[0] == 0 and names[-1] == n_nodes - 1)
            if should_reindex and reindex:
                reindexed = True
                row = new_nodes[:n_edges]
                col = new_nodes[n_edges:]
        if not weighted:
            data = np.ones(n_edges, dtype=bool)
        adjacency = sparse.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        if not directed:
            adjacency = directed2undirected(adjacency, weighted=weighted)
        graph.adjacency = adjacency
        if named or reindexed:
            graph.names = names

    return graph


def load_adjacency_list(file: str, bipartite: bool = False, comment: str = '%#',
                        delimiter: str = None, ) -> Bunch:
    """Parse Tabulation-Separated, Comma-Separated or Space-Separated (or other) Values datasets in the form of
    adjacency lists.

    Parameters
    ----------
    file : str
        The path to the dataset in TSV format
    bipartite : bool
        If ``True``, returns a biadjacency matrix of shape (n1, n2).
    comment : str
        Set of characters denoting lines to ignore.
    delimiter : str
        delimiter used in the file. None makes a guess

    Returns
    -------
    graph: :class:`Bunch`
    """
    header_len, guess_delimiter, _, _, _, _ = scan_header(file, comment)
    if delimiter is None:
        delimiter = guess_delimiter
    indptr, indices = [0], []
    with open(file, 'r', encoding='utf-8') as f:
        for i in range(header_len):
            f.readline()
        for row in f:
            neighbors = [int(el) for el in row.split(delimiter)]
            indices += neighbors
            indptr.append(indptr[-1] + len(neighbors))
    indices = np.array(indices)
    n_rows = len(indptr) - 1
    min_index = np.min(indices)
    n_cols = np.max(indices) + 1 - min_index
    indices -= min_index
    graph = Bunch()
    if not bipartite:
        max_dim = max(n_rows, n_cols)
        new_indptr = np.full(max_dim + 1, indptr[-1])
        new_indptr[:len(indptr)] = indptr
        graph.adjacency = sparse.csr_matrix((np.ones_like(indices, dtype=bool), indices, new_indptr),
                                            shape=(max_dim, max_dim))
    else:
        indptr = np.array(indptr)
        graph.biadjacency = sparse.csr_matrix((np.ones_like(indices, dtype=bool), indices, indptr),
                                              shape=(n_rows, n_cols))
    return graph


def scan_header(file: str, comment: str):
    """Infer some properties of the graph in a TSV file from the first few lines."""
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
        guess_string_present = not all([all([isnumber(el.strip()) for el in line.split(guess_delimiter)][0:2])
                                        for line in lines])
        if not guess_named:
            guess_type = np.int32
        else:
            guess_type = np.float32
    return header_len, guess_delimiter, guess_weighted, guess_named, guess_string_present, guess_type


def load_labels(file: str) -> np.ndarray:
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


def load_header(file: str):
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


def load_metadata(file: str, delimiter: str = ': ') -> Bunch:
    """Extract metadata from the file."""
    metadata = Bunch()
    with open(file, 'r', encoding='utf-8') as f:
        for row in f:
            parts = row.split(delimiter)
            key, value = parts[0], ': '.join(parts[1:]).strip('\n')
            metadata[key] = value
    return metadata


def load_graphml(file: str, weight_key: str = 'weight', max_string_size: int = 512) -> Bunch:
    """Parse GraphML datasets.

    Hyperedges and nested graphs are not supported.

    Parameters
    ----------
    file: str
        The path to the dataset
    weight_key: str
        The key to be used as a value for edge weights
    max_string_size: int
        The maximum size for string features of the data

    Returns
    -------
    data: :class:`Bunch`
        The dataset in a bunch with the adjacency as a CSR matrix.
    """
    # see http://graphml.graphdrawing.org/primer/graphml-primer.html
    # and http://graphml.graphdrawing.org/specification/dtd.html#top
    tree = ElementTree.parse(file)
    n_nodes = 0
    n_edges = 0
    symmetrize = None
    naming_nodes = True
    default_weight = 1
    weight_type = bool
    weight_id = None
    # indices in the graph tree
    node_indices = []
    edge_indices = []
    data = Bunch()
    graph = None
    file_description = None
    attribute_descriptions = Bunch()
    attribute_descriptions.node = Bunch()
    attribute_descriptions.edge = Bunch()
    keys = {}
    for file_element in tree.getroot():
        if file_element.tag.endswith('graph'):
            graph = file_element
            symmetrize = (graph.attrib['edgedefault'] == 'undirected')
            for index, element in enumerate(graph):
                if element.tag.endswith('node'):
                    node_indices.append(index)
                    n_nodes += 1
                elif element.tag.endswith('edge'):
                    edge_indices.append(index)
                    if 'directed' in element.attrib:
                        if element.attrib['directed'] == 'true':
                            n_edges += 1
                        else:
                            n_edges += 2
                    elif symmetrize:
                        n_edges += 2
                    else:
                        n_edges += 1
            if 'parse.nodeids' in graph.attrib:
                naming_nodes = not (graph.attrib['parse.nodeids'] == 'canonical')
    for file_element in tree.getroot():
        if file_element.tag.endswith('key'):
            attribute_name = file_element.attrib['attr.name']
            attribute_type = java_type_to_python_type(file_element.attrib['attr.type'])
            if attribute_name == weight_key:
                weight_type = java_type_to_python_type(file_element.attrib['attr.type'])
                weight_id = file_element.attrib['id']
                for key_element in file_element:
                    if key_element.tag == 'default':
                        default_weight = attribute_type(key_element.text)
            else:
                default_value = None
                if file_element.attrib['for'] == 'node':
                    size = n_nodes
                    if 'node_attribute' not in data:
                        data.node_attribute = Bunch()
                    for key_element in file_element:
                        if key_element.tag.endswith('desc'):
                            attribute_descriptions.node[attribute_name] = key_element.text
                        elif key_element.tag.endswith('default'):
                            default_value = attribute_type(key_element.text)
                    if attribute_type == str:
                        local_type = '<U' + str(max_string_size)
                    else:
                        local_type = attribute_type
                    if default_value:
                        data.node_attribute[attribute_name] = np.full(size, default_value, dtype=local_type)
                    else:
                        data.node_attribute[attribute_name] = np.zeros(size, dtype=local_type)
                elif file_element.attrib['for'] == 'edge':
                    size = n_edges
                    if 'edge_attribute' not in data:
                        data.edge_attribute = Bunch()
                    for key_element in file_element:
                        if key_element.tag.endswith('desc'):
                            attribute_descriptions.edge[attribute_name] = key_element.text
                        elif key_element.tag.endswith('default'):
                            default_value = attribute_type(key_element.text)
                    if attribute_type == str:
                        local_type = '<U' + str(max_string_size)
                    else:
                        local_type = attribute_type
                    if default_value:
                        data.edge_attribute[attribute_name] = np.full(size, default_value, dtype=local_type)
                    else:
                        data.edge_attribute[attribute_name] = np.zeros(size, dtype=local_type)
                keys[file_element.attrib['id']] = [attribute_name, attribute_type]
        elif file_element.tag.endswith('desc'):
            file_description = file_element.text
    if file_description or attribute_descriptions.node or attribute_descriptions.edge:
        data.meta = Bunch()
        if file_description:
            data.meta['description'] = file_description
        if attribute_descriptions.node or attribute_descriptions.edge:
            data.meta['attributes'] = attribute_descriptions
    if graph is not None:
        row = np.zeros(n_edges, dtype=int)
        col = np.zeros(n_edges, dtype=int)
        dat = np.full(n_edges, default_weight, dtype=weight_type)
        data.names = None
        if naming_nodes:
            data.names = np.zeros(n_nodes, dtype='<U512')

        node_map = {}
        # deal with nodes first
        for number, index in enumerate(node_indices):
            node = graph[index]
            if naming_nodes:
                name = node.attrib['id']
                data.names[number] = name
                node_map[name] = number
            for node_attribute in node:
                if node_attribute.tag.endswith('data'):
                    data.node_attribute[keys[node_attribute.attrib['key']][0]][number] = \
                        keys[node_attribute.attrib['key']][1](node_attribute.text)
        # deal with edges
        edge_index = -1
        for index in edge_indices:
            edge_index += 1
            duplicate = False
            edge = graph[index]
            if naming_nodes:
                node1 = node_map[edge.attrib['source']]
                node2 = node_map[edge.attrib['target']]
            else:
                node1 = int(edge.attrib['source'][1:])
                node2 = int(edge.attrib['target'][1:])
            row[edge_index] = node1
            col[edge_index] = node2
            for edge_attribute in edge:
                if edge_attribute.tag.endswith('data'):
                    if edge_attribute.attrib['key'] == weight_id:
                        dat[edge_index] = weight_type(edge_attribute.text)
                    else:
                        data.edge_attribute[keys[edge_attribute.attrib['key']][0]][edge_index] = \
                            keys[edge_attribute.attrib['key']][1](edge_attribute.text)
            if 'directed' in edge.attrib:
                if edge.attrib['directed'] != 'true':
                    duplicate = True
            elif symmetrize:
                duplicate = True
            if duplicate:
                edge_index += 1
                row[edge_index] = node2
                col[edge_index] = node1
                for edge_attribute in edge:
                    if edge_attribute.tag.endswith('data'):
                        if edge_attribute.attrib['key'] == weight_id:
                            dat[edge_index] = weight_type(edge_attribute.text)
                        else:
                            data.edge_attribute[keys[edge_attribute.attrib['key']][0]][edge_index] = \
                                keys[edge_attribute.attrib['key']][1](edge_attribute.text)
        data.adjacency = sparse.csr_matrix((dat, (row, col)), shape=(n_nodes, n_nodes))
        if data.names is None:
            data.pop('names')
        return data
    else:
        raise ValueError(f'No graph defined in {file}.')


def java_type_to_python_type(value: str) -> type:
    if value == 'boolean':
        return bool
    elif value == 'int':
        return int
    elif value == 'string':
        return str
    elif value in ('long', 'float', 'double'):
        return float


def isnumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
