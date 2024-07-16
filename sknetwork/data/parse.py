#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in December 2018
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <bonald@enst.fr>
"""

from csv import reader
from typing import Dict, List, Tuple, Union, Optional
from xml.etree import ElementTree

import numpy as np
from scipy import sparse

from sknetwork.data.base import Dataset
from sknetwork.utils.format import directed2undirected


def from_edge_list(edge_list: Union[np.ndarray, List[Tuple]], directed: bool = False,
                   bipartite: bool = False, weighted: bool = True, reindex: bool = False, shape: Optional[tuple] = None,
                   sum_duplicates: bool = True, matrix_only: bool = None) -> Union[Dataset, sparse.csr_matrix]:
    """Load a graph from an edge list.

    Parameters
    ----------
    edge_list : Union[np.ndarray, List[Tuple]]
        The edge list to convert, given as a NumPy array of size (n, 2) or (n, 3) or a list of tuples of
        length 2 or 3.
    directed : bool
        If ``True``, considers the graph as directed.
    bipartite : bool
        If ``True``, returns a biadjacency matrix.
    weighted : bool
        If ``True``, returns a weighted graph.
    reindex : bool
        If ``True``, reindex nodes and returns the original node indices as names.
        Reindexing is enforced if nodes are not integers.
    shape : tuple
        Shape of the adjacency or biadjacency matrix.
        If not specified or if nodes are reindexed, the shape is the smallest compatible with node indices.
    sum_duplicates : bool
        If ``True`` (default), sums weights of duplicate edges.
        Otherwise, the weight of each edge is that of the first occurrence of this edge.
    matrix_only : bool
        If ``True``, returns only the adjacency or biadjacency matrix.
        Otherwise, returns a ``Dataset`` object with graph attributes (e.g., node names).
        If not specified (default), selects the most appropriate format.
    Returns
    -------
    graph : :class:`Dataset` (including node names) or sparse matrix

    Examples
    --------
    >>> edges = [(0, 1), (1, 2), (2, 0)]
    >>> adjacency = from_edge_list(edges)
    >>> adjacency.shape
    (3, 3)
    >>> edges = [('Alice', 'Bob'), ('Bob', 'Carol'), ('Carol', 'Alice')]
    >>> graph = from_edge_list(edges)
    >>> adjacency = graph.adjacency
    >>> adjacency.shape
    (3, 3)
    >>> print(graph.names)
    ['Alice' 'Bob' 'Carol']
    """
    edge_array = np.array([])
    weights = None
    if isinstance(edge_list, list):
        try:
            edge_array = np.array([[edge[0], edge[1]] for edge in edge_list])
            if len(edge_list) and len(edge_list[0]) == 3:
                weights = np.array([edge[2] for edge in edge_list])
            else:
                raise ValueError()
        except ValueError:
            ValueError('Edges must be given as tuples of fixed size (2 or 3).')
    elif isinstance(edge_list, np.ndarray):
        if edge_list.ndim != 2 or edge_list.shape[1] not in [2, 3]:
            raise ValueError('The edge list must be given as an array of shape (n_edges, 2) or '
                             '(n_edges, 3).')
        edge_array = edge_list[:, :2]
        if edge_list.shape[1] == 3:
            weights = edge_list[:, 2]
    else:
        raise TypeError('The edge list must be given as a NumPy array or a list of tuples.')
    return from_edge_array(edge_array=edge_array, weights=weights, directed=directed, bipartite=bipartite,
                           weighted=weighted, reindex=reindex, shape=shape, sum_duplicates=sum_duplicates,
                           matrix_only=matrix_only)


def from_adjacency_list(adjacency_list: Union[List[List], Dict[str, List]], directed: bool = False,
                        bipartite: bool = False, weighted: bool = True, reindex: bool = False,
                        shape: Optional[tuple] = None, sum_duplicates: bool = True, matrix_only: bool = None) \
                        -> Union[Dataset, sparse.csr_matrix]:
    """Load a graph from an adjacency list.

    Parameters
    ----------
    adjacency_list : Union[List[List], Dict[str, List]]
        Adjacency list (neighbors of each node) or dictionary (node: neighbors).
    directed : bool
        If ``True``, considers the graph as directed.
    bipartite : bool
        If ``True``, returns a biadjacency matrix.
    weighted : bool
        If ``True``, returns a weighted graph.
    reindex : bool
        If ``True``, reindex nodes and returns the original node indices as names.
        Reindexing is enforced if nodes are not integers.
    shape : tuple
        Shape of the adjacency or biadjacency matrix.
        If not specified or if nodes are reindexed, the shape is the smallest compatible with node indices.
    sum_duplicates : bool
        If ``True`` (default), sums weights of duplicate edges.
        Otherwise, the weight of each edge is that of the first occurrence of this edge.
    matrix_only : bool
        If ``True``, returns only the adjacency or biadjacency matrix.
        Otherwise, returns a ``Dataset`` object with graph attributes (e.g., node names).
        If not specified (default), selects the most appropriate format.
    Returns
    -------
    graph : :class:`Dataset` or sparse matrix

    Example
    -------
    >>> edges = [[1, 2], [0, 2, 3], [0, 1]]
    >>> adjacency = from_adjacency_list(edges)
    >>> adjacency.shape
    (4, 4)
    """
    edge_list = []
    if isinstance(adjacency_list, list):
        for i, neighbors in enumerate(adjacency_list):
            for j in neighbors:
                edge_list.append((i, j))
    elif isinstance(adjacency_list, dict):
        for i, neighbors in adjacency_list.items():
            for j in neighbors:
                edge_list.append((i, j))
    else:
        raise TypeError('The adjacency list must be given as a list of lists or a dict of lists.')
    return from_edge_list(edge_list=edge_list, directed=directed, bipartite=bipartite, weighted=weighted,
                          reindex=reindex, shape=shape, sum_duplicates=sum_duplicates, matrix_only=matrix_only)


def from_edge_array(edge_array: np.ndarray, weights: np.ndarray = None, directed: bool = False, bipartite: bool = False,
                    weighted: bool = True, reindex: bool = False, shape: Optional[tuple] = None,
                    sum_duplicates: bool = True, matrix_only: bool = None) -> Union[Dataset, sparse.csr_matrix]:
    """Load a graph from an edge array of shape (n_edges, 2) and weights (optional).

    Parameters
    ----------
    edge_array : np.ndarray
        Array of edges.
    weights : np.ndarray
        Array of weights.
    directed : bool
        If ``True``, considers the graph as directed.
    bipartite : bool
        If ``True``, returns a biadjacency matrix.
    weighted : bool
        If ``True``, returns a weighted graph.
    reindex : bool
        If ``True``, reindex nodes and returns the original node indices as names.
        Reindexing is enforced if nodes are not integers.
    shape : tuple
        Shape of the adjacency or biadjacency matrix.
        If not specified or if nodes are reindexed, the shape is the smallest compatible with node indices.
    sum_duplicates : bool
        If ``True`` (default), sums weights of duplicate edges.
        Otherwise, the weight of each edge is that of the first occurrence of this edge.
    matrix_only : bool
        If ``True``, returns only the adjacency or biadjacency matrix.
        Otherwise, returns a ``Dataset`` object with graph attributes (e.g., node names).
        If not specified (default), selects the most appropriate format.

    Returns
    -------
    graph : :class:`Dataset` or sparse matrix
    """
    try:
        edge_array = edge_array.astype(float)
    except ValueError:
        pass
    if edge_array.dtype == float and (edge_array == edge_array.astype(int)).all():
        edge_array = edge_array.astype(int)
    if weights is None:
        weights = np.ones(len(edge_array))
    if weights.dtype not in [bool, int, float]:
        try:
            weights = weights.astype(float)
        except ValueError:
            raise ValueError('Weights must be numeric.')
    if all(weights == weights.astype(int)):
        weights = weights.astype(int)
    if not weighted:
        weights = weights.astype(bool)

    if not sum_duplicates:
        _, index = np.unique(edge_array, axis=0, return_index=True)
        edge_array = edge_array[index]
        weights = weights[index]
    graph = Dataset()
    if bipartite:
        row = edge_array[:, 0]
        col = edge_array[:, 1]
        if row.dtype != int or reindex:
            names_row, row = np.unique(row, return_inverse=True)
            graph.names_row = names_row
            graph.names = names_row
            n_row = len(names_row)
        elif shape is not None:
            n_row = max(shape[0], max(row) + 1)
        else:
            n_row = max(row) + 1
        if col.dtype != int or reindex:
            names_col, col = np.unique(col, return_inverse=True)
            graph.names_col = names_col
            n_col = len(names_col)
        elif shape is not None:
            n_col = max(shape[1], max(col) + 1)
        else:
            n_col = max(col) + 1
        matrix = sparse.csr_matrix((weights, (row, col)), shape=(n_row, n_col))
        matrix.sum_duplicates()
        graph.biadjacency = matrix
    else:
        nodes = edge_array.ravel()
        if nodes.dtype != int or reindex:
            names, nodes = np.unique(nodes, return_inverse=True)
            graph.names = names
            n = len(names)
            edge_array = nodes.reshape(-1, 2)
        elif shape is not None:
            n = max(shape[0], max(nodes) + 1)
        else:
            n = max(nodes) + 1
        row = edge_array[:, 0]
        col = edge_array[:, 1]
        matrix = sparse.csr_matrix((weights, (row, col)), shape=(n, n))
        if not directed:
            matrix = directed2undirected(matrix)
        matrix.sum_duplicates()
        graph.adjacency = matrix
    if matrix_only or (matrix_only is None and len(graph) == 1):
        return matrix
    else:
        return graph


def from_csv(file_path: str, delimiter: str = None, sep: str = None, comments: str = '#%',
             data_structure: str = None, directed: bool = False, bipartite: bool = False, weighted: bool = True,
             reindex: bool = False, shape: Optional[tuple] = None, sum_duplicates: bool = True,
             matrix_only: bool = None) -> Union[Dataset, sparse.csr_matrix]:
    """Load a graph from a CSV or TSV file.
    The delimiter can be specified (e.g., ' ' for space-separated values).

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    delimiter : str
        Delimiter used in the file. Guessed if not specified.
    sep : str
        Alias for delimiter.
    comments : str
        Characters for comment lines.
    data_structure : str
        If 'edge_list', consider each row of the file as an edge (tuple of size 2 or 3).
        If 'adjacency_list', consider each row of the file as an adjacency list (list of neighbors,
        in the order of node indices; an empty line means no neighbor).
        If 'adjacency_dict', consider each row of the file as an adjacency dictionary with key
        given by the first column (node: list of neighbors).
        If ``None`` (default), data_structure is guessed from the first rows of the file.
    directed : bool
        If ``True``, considers the graph as directed.
    bipartite : bool
        If ``True``, returns a biadjacency matrix of shape (n1, n2).
    weighted : bool
        If ``True``, returns a weighted graph (e.g., counts the number of occurrences of each edge).
    reindex : bool
        If ``True``, reindex nodes and returns the original node indices as names.
        Reindexing is enforced if nodes are not integers.
    shape : tuple
        Shape of the adjacency or biadjacency matrix.
        If not specified or if nodes are reindexed, the shape is the smallest compatible with node indices.
    sum_duplicates : bool
        If ``True`` (default), sums weights of duplicate edges.
        Otherwise, the weight of each edge is that of the first occurrence of this edge.
    matrix_only : bool
        If ``True``, returns only the adjacency or biadjacency matrix.
        Otherwise, returns a ``Dataset`` object with graph attributes (e.g., node names).
        If not specified (default), selects the most appropriate format.

    Returns
    -------
    graph: :class:`Dataset` or sparse matrix
    """
    header_length, delimiter_guess, comment_guess, data_structure_guess = scan_header(file_path, delimiters=delimiter,
                                                                                      comments=comments)
    if delimiter is None:
        if sep is not None:
            delimiter = sep
        else:
            delimiter = delimiter_guess
    if data_structure is None:
        data_structure = data_structure_guess
    if data_structure == 'edge_list':
        try:
            array = np.genfromtxt(file_path, delimiter=delimiter, comments=comment_guess)
            if np.isnan(array).any():
                raise TypeError()
            edge_array = array[:, :2].astype(int)
            if array.shape[1] == 3:
                weights = array[:, 2]
            else:
                weights = None
            return from_edge_array(edge_array=edge_array, weights=weights, directed=directed, bipartite=bipartite,
                                   weighted=weighted, reindex=reindex, shape=shape, sum_duplicates=sum_duplicates,
                                   matrix_only=matrix_only)
        except TypeError:
            pass
    with open(file_path, 'r', encoding='utf-8') as f:
        for i in range(header_length):
            f.readline()
        csv_reader = reader(f, delimiter=delimiter)
        if data_structure == 'edge_list':
            edge_list = [tuple(row) for row in csv_reader]
            return from_edge_list(edge_list=edge_list, directed=directed, bipartite=bipartite,
                                  weighted=weighted, reindex=reindex, shape=shape, sum_duplicates=sum_duplicates,
                                  matrix_only=matrix_only)
        elif data_structure == 'adjacency_list':
            adjacency_list = [row for row in csv_reader]
            return from_adjacency_list(adjacency_list=adjacency_list, directed=directed, bipartite=bipartite,
                                       weighted=weighted, reindex=reindex, shape=shape, sum_duplicates=sum_duplicates,
                                       matrix_only=matrix_only)
        elif data_structure == 'adjacency_dict':
            adjacency_list = {row[0]: row[1:] for row in csv_reader}
            return from_adjacency_list(adjacency_list=adjacency_list, directed=directed, bipartite=bipartite,
                                       weighted=weighted, reindex=reindex, shape=shape, sum_duplicates=sum_duplicates,
                                       matrix_only=matrix_only)


def scan_header(file_path: str, delimiters: str = None, comments: str = '#%', n_scan: int = 100):
    """Infer some properties of the graph from the first lines of a CSV file .
    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    delimiters : str
        Possible delimiters.
    comments : str
        Possible comment characters.
    n_scan : int
        Number of rows scanned for inference.

    Returns
    -------
    header_length : int
        Length of the header (comments and blank lines)
    delimiter_guess : str
        Guessed delimiter.
    comment_guess : str
        Guessed comment character.
    data_structure_guess : str
        Either 'edge_list' or 'adjacency_list'.
    """
    header_length = 0
    if delimiters is None:
        delimiters = '\t,; '
    comment_guess = comments[0]
    count = {delimiter: [] for delimiter in delimiters}
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for row in f.readlines():
            if row.startswith(tuple(comments)) or row == '':
                if len(row):
                    comment_guess = row[0]
                header_length += 1
            else:
                rows.append(row.rstrip())
                for delimiter in delimiters:
                    count[delimiter].append(row.count(delimiter))
                if len(rows) == n_scan:
                    break
    means = [np.mean(count[delimiter]) for delimiter in delimiters]
    stds = [np.std(count[delimiter]) for delimiter in delimiters]
    index = np.argwhere((np.array(means) > 0) * (np.array(stds) == 0)).ravel()
    if len(index) == 1:
        delimiter_guess = delimiters[int(index)]
    else:
        delimiter_guess = delimiters[int(np.argmax(means))]
    length = {len(row.split(delimiter_guess)) for row in rows}
    if length == {2} or length == {3}:
        data_structure_guess = 'edge_list'
    else:
        data_structure_guess = 'adjacency_list'
    return header_length, delimiter_guess, comment_guess, data_structure_guess


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


def load_metadata(file: str, delimiter: str = ': ') -> Dataset:
    """Extract metadata from the file."""
    metadata = Dataset()
    with open(file, 'r', encoding='utf-8') as f:
        for row in f:
            parts = row.split(delimiter)
            key, value = parts[0], ': '.join(parts[1:]).strip('\n')
            metadata[key] = value
    return metadata


def from_graphml(file_path: str, weight_key: str = 'weight', max_string_size: int = 512) -> Dataset:
    """Load graph from GraphML file.

    Hyperedges and nested graphs are not supported.

    Parameters
    ----------
    file_path: str
        Path to the GraphML file.
    weight_key: str
        The key to be used as a value for edge weights
    max_string_size: int
        The maximum size for string features of the data

    Returns
    -------
    data: :class:`Dataset`
        The dataset in a Dataset with the adjacency as a CSR matrix.
    """
    # see http://graphml.graphdrawing.org/primer/graphml-primer.html
    # and http://graphml.graphdrawing.org/specification/dtd.html#top
    tree = ElementTree.parse(file_path)
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
    data = Dataset()
    graph = None
    file_description = None
    attribute_descriptions = Dataset()
    attribute_descriptions.node = Dataset()
    attribute_descriptions.edge = Dataset()
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
                        data.node_attribute = Dataset()
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
                        data.edge_attribute = Dataset()
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
        data.meta = Dataset()
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
        raise ValueError(f'No graph defined in {file_path}.')


def java_type_to_python_type(value: str) -> type:
    if value == 'boolean':
        return bool
    elif value == 'int':
        return int
    elif value == 'string':
        return str
    elif value in ('long', 'float', 'double'):
        return float


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
