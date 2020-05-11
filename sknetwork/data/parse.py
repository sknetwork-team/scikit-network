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


def parse_graphml(file: str, weight_key: str = 'weight', max_string_size: int = 512) -> Bunch:
    # see http://graphml.graphdrawing.org/primer/graphml-primer.html
    # and http://graphml.graphdrawing.org/specification/dtd.html#top
    # hyperedges and nested graphs (graphs inside nodes or edges) are not supported
    tree = ElementTree.parse(file)
    optimizers = {'n_nodes': None, 'n_edges': None, 'order': 'none',
                  'naming_nodes': True, 'symetrize': None,
                  'weighted': False, 'weight_type': None, 'weight_id': None, 'nodes': [], 'edges': []}
    graph = None
    for file_element in tree.getroot():
        if file_element.tag.endswith('key'):
            if file_element.attrib['attr.name'] == weight_key:
                optimizers['weighted'] = True
                optimizers['weight_type'] = java_type_to_python_type(file_element.attrib['attr.type'])
                optimizers['weight_id'] = file_element.attrib['id']
        elif file_element.tag.endswith('graph'):
            graph = file_element
            optimizers['symetrize'] = (graph.attrib['edgedefault'] == 'undirected')
            if 'parse.order' in graph.attrib:
                optimizers['order'] = graph.attrib['parse.order']
            if 'parse.nodes' in graph.attrib:
                optimizers['n_nodes'] = graph.attrib['parse.nodes']
            else:
                if optimizers['order'] != 'adjacencylist':
                    optimizers['nodes'] = [index for index, node in enumerate(graph) if node.tag.endswith('node')]
                    optimizers['n_nodes'] = len(optimizers['nodes'])
                else:
                    optimizers['n_nodes'] = len([node for node in graph if node.tag.endswith('node')])
            if 'parse.edges' in graph.attrib:
                optimizers['n_edges'] = graph.attrib['parse.edges']
            else:
                count = 0
                for index, edge in enumerate(graph):
                    if edge.tag.endswith('edge'):
                        if optimizers['order'] != 'adjacencylist':
                            optimizers['edges'].append(index)
                        if 'directed' in edge.attrib:
                            if edge.attrib['directed'] == 'true':
                                count += 1
                            else:
                                count += 2
                        elif optimizers['symetrize']:
                            count += 2
                        else:
                            count += 1
                optimizers['n_edges'] = count
            if 'parse.nodeids' in graph.attrib:
                optimizers['naming_nodes'] = not (graph.attrib['parse.nodeids'] == 'canonical')
    if graph is not None:
        if optimizers['order'] == 'adjacencylist':
            # efficient parse to CSR directly
            raise NotImplementedError('To CSR parser not yet implemented')
        else:
            # parse to CSR
            return parse_xml_to_csr(tree, optimizers, weight_key, max_string_size)
    else:
        raise ValueError(f'No graph defined in {file}.')


def parse_xml_to_csr(tree: ElementTree.ElementTree, optimizers: dict,
                                 weight_key: str, max_string_size: int) -> Bunch:
    data = Bunch()
    graph = None
    file_description = None
    attribute_descriptions = Bunch()
    attribute_descriptions.node = Bunch()
    attribute_descriptions.edge = Bunch()
    keys = {}
    row = np.zeros(optimizers['n_edges'], dtype=int)
    col = np.zeros(optimizers['n_edges'], dtype=int)
    dat = np.ones(optimizers['n_edges'], dtype=optimizers['weight_type'])
    for file_element in tree.getroot():
        if file_element.tag.endswith('key'):
            attribute_name = file_element.attrib['attr.name']
            if attribute_name != weight_key:
                attribute_type = java_type_to_python_type(file_element.attrib['attr.type'])
                default_value = None
                if file_element.attrib['for'] == 'node':
                    size = optimizers['n_nodes']
                    if 'node_info' not in data:
                        data.node_info = Bunch()
                    for key_element in file_element:
                        if key_element.tag == 'desc':
                            attribute_descriptions[attribute_name] = key_element.text
                        elif key_element.tag == 'default':
                            default_value = attribute_type(key_element.text)
                    if attribute_type == str:
                        local_type = '<U' + str(max_string_size)
                    else:
                        local_type = attribute_type
                    if default_value:
                        data.node_info[attribute_name] = np.full(size, default_value, dtype=local_type)
                    else:
                        data.node_info[attribute_name] = np.zeros(size, dtype=local_type)
                elif file_element.attrib['for'] == 'edge':
                    size = optimizers['n_edges']
                    if 'edge_info' not in data:
                        data.edge_info = Bunch()
                    for key_element in file_element:
                        if key_element.tag == 'desc':
                            attribute_descriptions[attribute_name] = key_element.text
                        elif key_element.tag == 'default':
                            default_value = attribute_type(key_element.text)
                    if attribute_type == str:
                        local_type = '<U' + str(max_string_size)
                    else:
                        local_type = attribute_type
                    if default_value:
                        data.edge_info[attribute_name] = np.full(size, default_value, dtype=local_type)
                    else:
                        data.edge_info[attribute_name] = np.zeros(size, dtype=local_type)
                keys[file_element.attrib['id']] = [attribute_name, attribute_type]
        elif file_element.tag.endswith('desc'):
            file_description = file_element.text
        elif file_element.tag.endswith('graph'):
            graph = file_element
    if file_description or attribute_descriptions.node or attribute_descriptions.edge:
        data.meta = {}
        if file_description:
            data.meta['description'] = file_description
        if attribute_descriptions.node or attribute_descriptions.edge:
            data.meta['attributes'] = attribute_descriptions

    data.names = None
    if optimizers['naming_nodes']:
        data.names = np.zeros(optimizers['n_nodes'], dtype='<U512')

    node_map = {}
    # deal with nodes first
    for number, index in enumerate(optimizers['nodes']):
        graph_element = graph[index]
        if optimizers['naming_nodes']:
            name = graph_element.attrib['id']
            data.names[number] = name
            node_map[name] = number
        for node_attribute in graph_element:
            if node_attribute.tag.endswith('data'):
                data.node_info[keys[node_attribute.attrib['key']][0]][number] = \
                    keys[node_attribute.attrib['key']][1](node_attribute.text)
    # deal with edges
    for number, index in enumerate(optimizers['edges']):
        graph_element = graph[index]
        if optimizers['naming_nodes']:
            try:
                node1 = node_map[graph_element.attrib['source']]
                node2 = node_map[graph_element.attrib['target']]
            except:
                print(index, number, graph_element)
        else:
            node1 = int(graph_element.attrib['source'][1:])
            node2 = int(graph_element.attrib['target'][1:])
        row[number] = node1
        col[number] = node2
        for edge_attribute in graph_element:
            if edge_attribute.tag.endswith('data'):
                if edge_attribute.attrib['key'] == optimizers['weight_id']:
                    dat[number] = optimizers['weight_type'](edge_attribute.text)
                else:
                    data.edge_info[keys[edge_attribute.attrib['key']][0]][number] = \
                        keys[edge_attribute.attrib['key']][1](edge_attribute.text)
    data.adjacency = sparse.csr_matrix((dat, (row, col)), shape=(optimizers['n_nodes'], optimizers['n_nodes']))
    return data


def java_type_to_python_type(value: str) -> type:
    if value == 'boolean':
        return bool
    elif value == 'int':
        return int
    elif value == 'string':
        return str
    elif value in ('long', 'float', 'double'):
        return float
