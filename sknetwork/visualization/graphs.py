#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2020
@authors:
Thomas Bonald <bonald@enst.fr>
Quentin Lutz <qlutz@enst.fr>
"""

from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.clustering import BiLouvain
from sknetwork.visualization.colors import STANDARD_COLORS, COOLWARM_RGB


def min_max_scaling(x: np.ndarray) -> np.ndarray:
    """Shift and scale vector to be between 0 and 1."""
    x = x.astype(float)
    x -= np.min(x)
    if np.max(x):
        x /= np.max(x)
    else:
        x = .5 * np.ones_like(x)
    return x


def max_min_scaling(y: np.ndarray) -> np.ndarray:
    """Shift and scale vector to be between 0 and 1."""
    y = y.astype(float)
    y = np.max(y) - y
    if np.max(y):
        y /= np.max(y)
    else:
        y = .5 * np.ones_like(y)
    return y


def rescale(position, width, height, margin, node_size_max, node_weight):
    """Rescale position and adjust parameters"""
    x = min_max_scaling(position[:, 0])
    y = max_min_scaling(position[:, 1])
    position = np.vstack((x, y)).T
    position = position * np.array([width, height])

    # margins
    margin = max(margin, 4 * node_size_max * node_weight)
    position += margin
    width += 2 * margin
    height += 2 * margin
    return position, width, height


def get_colors(n: int, labels: Union[dict, np.ndarray, None], scores: np.ndarray, color: str) -> np.ndarray:
    """Return the colors using either labels or scores or default color."""
    colors = np.array(n * [color]).astype('U64')
    if labels is not None:
        colors_label = STANDARD_COLORS.copy()
        if isinstance(labels, dict):
            index = np.array(list(labels.keys()))
            values = np.array(list(labels.values())).astype(int)
            labels = -np.ones(n, dtype=int)
            labels[index] = values
        index = labels >= 0
        colors[index] = colors_label[labels[index] % len(colors_label)]
    elif scores is not None:
        colors_score = COOLWARM_RGB.copy()
        n_colors = colors_score.shape[0]
        colors_score_svg = np.array(['rgb' + str(tuple(colors_score[i])) for i in range(n_colors)])
        scores = (min_max_scaling(scores) * (n_colors - 1)).astype(int)
        colors = colors_score_svg[scores]
    return colors


def get_node_widths(n: int, seeds: Union[dict, list], node_width: float, node_width_max: float) -> np.ndarray:
    """Return the node widths."""
    node_widths = node_width * np.ones(n)
    if seeds is not None:
        if type(seeds) == dict:
            seeds = list(seeds.keys())
        if len(seeds):
            node_widths[np.array(seeds)] = node_width_max
    return node_widths


def get_node_sizes(weights: np.ndarray, node_size: float, node_size_min: float, node_size_max: float, node_weight) \
        -> np.ndarray:
    """Return the node sizes."""
    if node_weight and np.min(weights) < np.max(weights):
        node_sizes = node_size_min + np.abs(node_size_max - node_size_min) * weights / np.max(weights)
    else:
        node_sizes = node_size * np.ones_like(weights)
    return node_sizes


def get_node_sizes_bipartite(weights_row: np.ndarray, weights_col: np.ndarray, node_size: float, node_size_min: float,
                             node_size_max: float, node_weight) -> (np.ndarray, np.ndarray):
    """Return the node sizes for bipartite graphs."""
    weights = np.hstack((weights_row, weights_col))
    if node_weight and np.min(weights) < np.max(weights):
        node_sizes_row = node_size_min + np.abs(node_size_max - node_size_min) * weights_row / np.max(weights)
        node_sizes_col = node_size_min + np.abs(node_size_max - node_size_min) * weights_col / np.max(weights)
    else:
        node_sizes_row = node_size * np.ones_like(weights_row)
        node_sizes_col = node_size * np.ones_like(weights_col)
    return node_sizes_row, node_sizes_col


def get_edge_widths(weights: np.ndarray, edge_width: float, edge_width_min: float, edge_width_max: float,
                    edge_weight: bool) -> np.ndarray:
    """Return the edge widths."""
    if edge_weight and np.min(weights) < np.max(weights):
        edge_widths = edge_width_min + np.abs(edge_width_max - edge_width_min) * weights / np.max(weights)
    else:
        edge_widths = edge_width * np.ones_like(weights)
    return edge_widths


def svg_node(pos_node: np.ndarray, size: float, color: str, stroke_width: float = 1, stroke_color: str = 'black') \
        -> str:
    """Return svg code for a node."""
    x, y = pos_node.astype(int)
    return """<circle cx="{}" cy="{}" r="{}" style="fill:{};stroke:{};stroke-width:{}"/>"""\
        .format(x, y, size, color, stroke_color, stroke_width)


def svg_edge(pos_1: np.ndarray, pos_2: np.ndarray, stroke_width: float = 1, stroke_color: str = 'black') -> str:
    """Return svg code for an edge."""
    x1, y1 = pos_1.astype(int)
    x2, y2 = pos_2.astype(int)
    return """<path stroke-width="{}" stroke="{}" d="M {} {} {} {}" />"""\
        .format(stroke_width, stroke_color, x1, y1, x2, y2)


def svg_edge_directed(pos_1: np.ndarray, pos_2: np.ndarray, stroke_width: float = 1, stroke_color: str = 'black',
                      node_size: float = 1.) -> str:
    """Return svg code for a directed edge."""
    vec = pos_2 - pos_1
    norm = np.linalg.norm(vec)
    if norm:
        x, y = ((vec / norm) * node_size).astype(int)
        x1, y1 = pos_1.astype(int)
        x2, y2 = pos_2.astype(int)
        return """<path stroke-width="{}" stroke="{}" d="M {} {} {} {}" marker-end="url(#arrow)"/>"""\
            .format(stroke_width, stroke_color, x1, y1, x2 - x, y2 - y)
    else:
        return ""


def svg_text(pos, text, font_size=12, align_right=False):
    """Return svg code for text."""
    x, y = pos.astype(int)
    if align_right:
        return """<text text-anchor="end" x="{}" y="{}" font-size="{}">{}</text>"""\
            .format(x, y, font_size, str(text))
    else:
        return """<text x="{}" y="{}" font-size="{}">{}</text>"""\
            .format(x, y, font_size, str(text))


def svg_graph(adjacency: sparse.csr_matrix, position: np.ndarray, names: Optional[np.ndarray] = None,
              labels: Optional[Union[dict, np.ndarray]] = None, scores: Optional[np.ndarray] = None,
              seeds: Union[list, dict] = None, width: float = 400, height: float = 300,
              margin: float = 20, margin_text: float = 3, scale: float = 1,
              node_size: float = 7, node_size_min: float = 1, node_size_max: float = 20,
              node_weight: bool = False, node_weights: Optional[np.ndarray] = None, node_width: float = 1,
              node_width_max: float = 3, node_color: str = 'gray', edge_width: float = 1, edge_width_min: float = 0.5,
              edge_width_max: float = 20, edge_weight: bool = True, edge_color: Optional[str] = None,
              font_size: int = 12, directed: bool = False, filename: Optional[str] = None) -> str:
    """Return SVG image of a graph.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    position :
        Positions of the nodes.
    names :
        Names of the nodes.
    labels :
        Labels of the nodes (negative values mean no label).
    scores :
        Scores of the nodes (measure of importance).
    seeds :
        Nodes to be highlighted (if dict, only keys are considered).
    width :
        Width of the image.
    height :
        Height of the image.
    margin :
        Margin of the image.
    margin_text :
        Margin between node and text.
    scale :
        Multiplicative factor on the dimensions of the image.
    node_size :
        Size of nodes.
    node_size_min :
        Minimum size of a node.
    node_size_max:
        Maximum size of a node.
    node_width :
        Width of node circle.
    node_width_max :
        Maximum width of node circle.
    node_color :
        Default color of nodes (svg color).
    node_weight :
        Display node weights by node size.
    node_weights :
        Node weights (used only if **node_weight** is ``True``).
    edge_width :
        Width of edges.
    edge_width_min :
        Minimum width of edges.
    edge_width_max :
        Maximum width of edges.
    edge_weight :
        Display edge weights with edge widths.
    edge_color :
        Default color of edges (svg color).
    font_size :
        Font size.
    directed :
        If ``True``, considers the graph as directed.
    filename :
        Filename for saving image (optional).

    Returns
    -------
    image : str
        SVG image.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> graph = karate_club(True)
    >>> adjacency = graph.adjacency
    >>> position = graph.position
    >>> from sknetwork.visualization import svg_graph
    >>> image = svg_graph(adjacency, position)
    >>> image[1:4]
    'svg'
    """
    n = adjacency.shape[0]
    adjacency = sparse.coo_matrix(adjacency)
    n_edges = len(adjacency.row)

    # colors
    colors = get_colors(n, labels, scores, node_color)
    if edge_color is None:
        if names is None:
            edge_color = 'black'
        else:
            edge_color = 'gray'

    # node sizes
    if node_weights is None:
        node_weights = adjacency.dot(np.ones(n))
    node_sizes = get_node_sizes(node_weights, node_size, node_size_min, node_size_max, node_weight)

    # node widths
    node_widths = get_node_widths(n, seeds, node_width, node_width_max)

    # edge widths
    edge_widths = get_edge_widths(adjacency.data, edge_width, edge_width_min, edge_width_max, edge_weight)

    # rescaling
    position, width, height = rescale(position, width, height, margin, node_size_max, node_weight)

    if names is not None:
        text_length = np.max(np.array([len(str(name)) for name in names]))
        width += text_length * font_size * .5

    # scaling
    position *= scale
    height *= scale
    width *= scale

    svg = """<svg width="{}" height="{}">""".format(width, height)
    if directed:
        svg += """<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" >"""
        svg += """<path d="M0,0 L0,6 L9,3 z" fill="{}"/></marker></defs>""".format(edge_color)

    # edges
    for ix in range(n_edges):
        i = adjacency.row[ix]
        j = adjacency.col[ix]
        if directed:
            svg += svg_edge_directed(pos_1=position[i], pos_2=position[j], stroke_width=edge_widths[ix],
                                     stroke_color=edge_color, node_size=node_sizes[j])
        else:
            svg += svg_edge(pos_1=position[i], pos_2=position[j], stroke_width=edge_widths[ix], stroke_color=edge_color)

    # nodes
    for i in range(n):
        svg += svg_node(position[i], node_sizes[i], colors[i], node_widths[i])
    # text
    if names is not None:
        for i in range(n):
            svg += svg_text(position[i] + node_sizes[i] + (margin_text, 0), names[i], font_size)
    svg += """</svg>"""

    if filename is not None:
        with open(filename + '.svg', 'w') as f:
            f.write(svg)
    else:
        return svg


def svg_digraph(adjacency: sparse.csr_matrix, position: np.ndarray, names: Optional[np.ndarray] = None,
                labels: Optional[Union[dict, np.ndarray]] = None, scores: Optional[np.ndarray] = None,
                seeds: Union[list, dict] = None, width: float = 400, height: float = 300,
                margin: float = 20, margin_text: float = 10, scale: float = 1,
                node_size: float = 7, node_size_min: float = 1, node_size_max: float = 20, node_weight: bool = False,
                node_weights: Optional[np.ndarray] = None, node_width: float = 1, node_width_max: float = 3,
                node_color: str = 'gray', edge_width: float = 1, edge_width_min: float = 0.5,
                edge_width_max: float = 3, edge_weight: bool = True, edge_color: Optional[str] = None,
                font_size: int = 12, filename: Optional[str] = None) -> str:
    """Return SVG image of a digraph.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    position :
        Positions of the nodes.
    names :
        Names of the nodes.
    labels :
        Labels of the nodes (negative values mean no label).
    scores :
        Scores of the nodes (measure of importance).
    seeds :
        Nodes to be highlighted (if dict, only keys are considered).
    width :
        Width of the image.
    height :
        Height of the image.
    margin :
        Margin of the image.
    margin_text :
        Margin between node and text.
    scale :
        Multiplicative factor on the dimensions of the image.
    node_size :
        Size of nodes.
    node_size_min :
        Minimum size of a node.
    node_size_max :
        Maximum size of a node.
    node_weight :
        Display node weights by node size.
    node_weights :
        Node weights (used only if **node_weight** is ``True``).
    node_width :
        Width of node circle.
    node_width_max :
        Maximum width of node circle.
    node_color :
        Default color of nodes (svg color).
    edge_width :
        Width of edges.
    edge_width_min :
        Minimum width of edges.
    edge_width_max :
        Maximum width of edges.
    edge_weight :
        Display edge weights with edge widths.
    edge_color :
        Default color of edges (svg color).
    font_size :
        Font size.
    filename :
        Filename for saving image (optional).

    Returns
    -------
    image : str
        SVG image.

    Example
    -------
    >>> from sknetwork.data import painters
    >>> graph = painters(True)
    >>> adjacency = graph.adjacency
    >>> position = graph.position
    >>> from sknetwork.visualization import svg_digraph
    >>> image = svg_graph(adjacency, position)
    >>> image[1:4]
    'svg'
    """
    return svg_graph(adjacency=adjacency, position=position, names=names, labels=labels, scores=scores, seeds=seeds,
                     width=width, height=height, margin=margin, margin_text=margin_text, scale=scale,
                     node_size=node_size, node_size_min=node_size_min, node_size_max=node_size_max,
                     node_weight=node_weight, node_weights=node_weights, node_width=node_width,
                     node_width_max=node_width_max, node_color=node_color, edge_width=edge_width,
                     edge_width_min=edge_width_min, edge_width_max=edge_width_max, edge_weight=edge_weight,
                     edge_color=edge_color, font_size=font_size, directed=True, filename=filename)


def svg_bigraph(biadjacency: sparse.csr_matrix,
                names_row: Optional[np.ndarray] = None, names_col: Optional[np.ndarray] = None,
                labels_row: Optional[Union[dict, np.ndarray]] = None,
                labels_col: Optional[Union[dict, np.ndarray]] = None,
                scores_row: Optional[np.ndarray] = None, scores_col: Optional[np.ndarray] = None,
                seeds_row: Union[list, dict] = None, seeds_col: Union[list, dict] = None,
                position_row: Optional[np.ndarray] = None, position_col: Optional[np.ndarray] = None,
                cluster: bool = True, width: float = 400,
                height: float = 300, margin: float = 20, margin_text: float = 3, scale: float = 1,
                node_size: float = 7, node_size_min: float = 1, node_size_max: float = 20, node_weight: bool = False,
                node_weights_row: Optional[np.ndarray] = None, node_weights_col: Optional[np.ndarray] = None,
                node_width: float = 1, node_width_max: float = 3,
                color_row: str = 'gray', color_col: str = 'gray', edge_width: float = 1, edge_width_min: float = 0.5,
                edge_width_max: float = 10, edge_color: str = 'black', edge_weight: bool = True,
                font_size: int = 12, filename: Optional[str] = None) -> str:
    """Return SVG image of a bigraph.

    Parameters
    ----------
    biadjacency :
        Biadjacency matrix of the graph.
    names_row :
        Names of the rows.
    names_col :
        Names of the columns.
    labels_row :
        Labels of the rows (negative values mean no label).
    labels_col :
        Labels of the columns (negative values mean no label).
    scores_row :
        Scores of the rows (measure of importance).
    scores_col :
        Scores of the rows (measure of importance).
    seeds_row :
        Rows to be highlighted (if dict, only keys are considered).
    seeds_col :
        Columns to be highlighted (if dict, only keys are considered).
    position_row :
        Positions of the rows.
    position_col :
        Positions of the columns.
    cluster :
        Use clustering to order nodes.
    width :
        Width of the image.
    height :
        Height of the image.
    margin :
        Margin of the image.
    margin_text :
        Margin between node and text.
    scale :
        Multiplicative factor on the dimensions of the image.
    node_size :
        Size of nodes.
    node_size_min :
        Minimum size of nodes.
    node_size_max :
        Maximum size of nodes.
    node_weight :
        Display node weights by node size.
    node_weights_row :
        Weights of rows (used only if **node_weight** is ``True``).
    node_weights_col :
        Weights of columns (used only if **node_weight** is ``True``).
    node_width :
        Width of node circle.
    node_width_max :
        Maximum width of node circle.
    color_row :
        Default color of rows (svg color).
    color_col :
        Default color of cols (svg color).
    edge_width :
        Width of edges.
    edge_width_min :
        Minimum width of edges.
    edge_width_max :
        Maximum width of edges.
    edge_weight :
        Display edge weights with edge widths.
    edge_color :
        Default color of edges (svg color).
    font_size :
        Font size.
    filename :
        Filename for saving image (optional).

    Returns
    -------
    image : str
        SVG image.

    Example
    -------
    >>> from sknetwork.data import movie_actor
    >>> biadjacency = movie_actor()
    >>> from sknetwork.visualization import svg_bigraph
    >>> image = svg_bigraph(biadjacency)
    >>> image[1:4]
    'svg'
    """
    n_row, n_col = biadjacency.shape

    if position_row is None or position_col is None:
        position_row = np.zeros((n_row, 2))
        position_col = np.ones((n_col, 2))
        if cluster:
            bilouvain = BiLouvain()
            bilouvain.fit(biadjacency)
            index_row = np.argsort(bilouvain.labels_row_)
            index_col = np.argsort(bilouvain.labels_col_)
        else:
            index_row = np.arange(n_row)
            index_col = np.arange(n_col)
        position_row[index_row, 1] = np.arange(n_row)
        position_col[index_col, 1] = np.arange(n_col) + .5 * (n_row - n_col)

    biadjacency = sparse.coo_matrix(biadjacency)

    # colors
    colors_row = get_colors(n_row, labels_row, scores_row, color_row)
    colors_col = get_colors(n_col, labels_col, scores_col, color_col)

    # node sizes
    if node_weights_row is None:
        node_weights_row = biadjacency.dot(np.ones(n_col))
    if node_weights_col is None:
        node_weights_col = biadjacency.T.dot(np.ones(n_row))
    node_sizes_row, node_sizes_col = get_node_sizes_bipartite(node_weights_row, node_weights_col,
                                                              node_size, node_size_min, node_size_max, node_weight)

    # node widths
    node_widths_row = get_node_widths(n_row, seeds_row, node_width, node_width_max)
    node_widths_col = get_node_widths(n_col, seeds_col, node_width, node_width_max)

    # edge widths
    edge_widths = get_edge_widths(biadjacency.data, edge_width, edge_width_min, edge_width_max, edge_weight)

    position = np.vstack((position_row, position_col))

    # rescaling
    position, width, height = rescale(position, width, height, margin, node_size_max, node_weight)

    if names_row is not None:
        text_length = np.max(np.array([len(str(name)) for name in names_row]))
        position[:, 0] += text_length * font_size * .5
        width += text_length * font_size * .5
    if names_col is not None:
        text_length = np.max(np.array([len(str(name)) for name in names_col]))
        width += text_length * font_size * .5

    # scaling
    position *= scale
    height *= scale
    width *= scale
    position_row = position[:n_row]
    position_col = position[n_row:]

    svg = """<svg width="{}" height="{}">""".format(width, height)
    # edges
    for i in range(len(biadjacency.row)):
        svg += svg_edge(position_row[biadjacency.row[i]], position_col[biadjacency.col[i]], edge_widths[i], edge_color)
    # nodes
    for i in range(n_row):
        svg += svg_node(position_row[i], node_sizes_row[i], colors_row[i], node_widths_row[i])
    for i in range(n_col):
        svg += svg_node(position_col[i], node_sizes_col[i], colors_col[i], node_widths_col[i])
    # text
    if names_row is not None:
        for i in range(n_row):
            svg += svg_text(position_row[i] - (margin_text + node_sizes_row[i], 0), names_row[i], font_size, True)
    if names_col is not None:
        for i in range(n_col):
            svg += svg_text(position_col[i] + (margin_text + node_sizes_col[i], 0), names_col[i], font_size)
    svg += """</svg>"""

    if filename is not None:
        with open(filename + '.svg', 'w') as f:
            f.write(svg)
    else:
        return svg
