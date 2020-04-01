#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2020
@authors:
Thomas Bonald <bonald@enst.fr>
Quentin Lutz <qlutz@enst.fr>
"""

from typing import Optional

import numpy as np
from scipy import sparse

from sknetwork.visualization.colors import get_standard_colors, get_coolwarm_rgb


def min_max_scaling(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    x -= np.min(x)
    if np.max(x):
        x /= np.max(x)
    else:
        x = .5 * np.ones_like(x)
    return x


def max_min_scaling(y: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    y = np.max(y) - y
    if np.max(y):
        y /= np.max(y)
    else:
        y = .5 * np.ones_like(y)
    return y


def get_colors(n: int, labels: np.ndarray, scores: np.ndarray, color: str) -> np.ndarray:
    """Return the colors using either labels or scores or default color."""
    if labels is not None:
        colors_label = get_standard_colors()
        colors = colors_label[labels % len(colors_label)]
    elif scores is not None:
        colors_score = get_coolwarm_rgb()
        scores = (min_max_scaling(scores) * 99).astype(int)
        colors = ['rgb' + str(tuple(colors_score[s])) for s in scores]
    else:
        colors = n * [color]
    return np.array(colors)


def svg_node(pos_node: np.ndarray, size: float, color: str, stroke_width: float = 1, stroke_color: str = 'black') \
        -> str:
    """Return svg code for a node."""
    return """<circle cx="{}" cy="{}" r="{}" style="fill:{};stroke:{};stroke-width:{}"/>"""\
        .format(pos_node[0], pos_node[1], size, color, stroke_color, stroke_width)


def svg_edge(pos_1: np.ndarray, pos_2: np.ndarray, stroke_width: float = 1, stroke_color: str = 'black') -> str:
    """Return svg code for an edge."""
    return """<path stroke-width="{}" stroke="{}" d="M {} {} {} {}" />"""\
        .format(stroke_width, stroke_color, pos_1[0], pos_1[1], pos_2[0], pos_2[1])


def svg_edge_directed(pos_1: np.ndarray, pos_2: np.ndarray, node_size: float, stroke_width: float = 1,
                      stroke_color: str = 'black') -> str:
    """Return svg code for a directed edge."""
    vec = pos_2 - pos_1
    norm = np.linalg.norm(vec)
    if norm:
        vec = vec / norm * node_size * 2
        return """<path stroke-width="{}" stroke="{}" d="M {} {} {} {}" marker-end="url(#arrow)"/>"""\
            .format(stroke_width, stroke_color, pos_1[0], pos_1[1], pos_2[0] - vec[0], pos_2[1] - vec[1])
    else:
        return ""


def svg_text(pos, text, font_size=12, align_right=False):
    """Return svg code for text."""
    if align_right:
        return """<text text-anchor="end" x="{}" y="{}" font-size="{}">{}</text>"""\
            .format(pos[0], pos[1], font_size, text)
    else:
        return """<text x="{}" y="{}" font-size="{}">{}</text>""".format(pos[0], pos[1], font_size, text)


def svg_graph(adjacency: sparse.csr_matrix, position: np.ndarray, names: Optional[np.ndarray] = None,
              labels: Optional[np.ndarray] = None, scores: Optional[np.ndarray] = None, color: str = 'blue',
              width: float = 400, height: float = 300, margin: float = 20, margin_text: float = 10,
              scale: float = 1, node_size: float = 5, edge_width: float = 1, font_size: int = 12) -> str:
    """Return svg code for a graph.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    position :
        Positions of the nodes.
    names :
        Names of the nodes.
    labels :
        Labels of the nodes.
    scores :
        Scores of the nodes (measure of importance).
    color :
        Default color (svg color).
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
    edge_width :
        Width of edges.
    font_size :
        Font size.

    Returns
    -------
    image : str
        SVG image.

    Example
    -------
    >>> adjacency = sparse.csr_matrix(np.ones((3,3)))
    >>> position = np.random.random((3,2))
    >>> image = svg_graph(adjacency, position)
    >>> image[1:4]
    'svg'
    """
    n = adjacency.shape[0]

    # colors
    colors = get_colors(n, labels, scores, color)

    # rescaling
    x = min_max_scaling(position[:, 0])
    y = max_min_scaling(position[:, 1])
    position = np.vstack((x, y)).T
    position = position * np.array([width, height])

    # margins
    position += margin
    height += 2 * margin
    width += 2 * margin
    if names is not None:
        text_length = np.max(np.array([len(name) for name in names]))
        width += text_length * font_size * .5

    # scaling
    position *= scale
    height *= scale
    width *= scale

    svg = """<svg width="{}" height="{}">""".format(width, height)
    # edges
    for i in range(n):
        for j in adjacency[i].indices:
            svg += svg_edge(position[i], position[j], edge_width)
    # nodes
    for i in range(n):
        svg += svg_node(position[i], node_size, colors[i])
    # text
    if names is not None:
        for i in range(n):
            svg += svg_text(position[i] + (margin_text, node_size), names[i], font_size)
    svg += '</svg>'
    return svg


def svg_digraph(adjacency: sparse.csr_matrix, position: np.ndarray, names: Optional[np.ndarray] = None,
                labels: Optional[np.ndarray] = None, scores: Optional[np.ndarray] = None, color: str = 'blue',
                width: float = 400, height: float = 300, margin: float = 20, margin_text: float = 10,
                scale: float = 1, node_size: float = 5, edge_width: float = 1, font_size: int = 12) -> str:
    """Return svg code for a directed graph.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    position :
        Positions of the nodes.
    names :
        Names of the nodes.
    labels :
        Labels of the nodes.
    scores :
        Scores of the nodes (measure of importance).
    color :
        Default color (svg color).
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
    edge_width :
        Width of edges.
    font_size :
        Font size.

    Returns
    -------
    image : str
        SVG image.

    Example
    -------
    >>> adjacency = sparse.csr_matrix(np.ones((3,3)))
    >>> position = np.random.random((3,2))
    >>> image = svg_digraph(adjacency, position)
    >>> image[1:4]
    'svg'
    """
    n = adjacency.shape[0]

    # colors
    colors = get_colors(n, labels, scores, color)

    # rescaling
    x = min_max_scaling(position[:, 0])
    y = max_min_scaling(position[:, 1])
    position = np.vstack((x, y)).T
    position = position * np.array([width, height])

    # margins
    position += margin
    height += 2 * margin
    width += 2 * margin
    if names is not None:
        text_length = np.max(np.array([len(name) for name in names]))
        width += text_length * font_size * .5

    # scaling
    position *= scale
    height *= scale
    width *= scale

    svg = """<svg width="{}" height="{}">""".format(width, height)
    # arrow
    svg += """<defs><marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="10" """
    svg += """markerHeight="10" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" /></marker></defs>"""
    # edges
    for i in range(n):
        for j in adjacency[i].indices:
            svg += svg_edge_directed(position[i], position[j], node_size, edge_width)
    # nodes
    for i in range(n):
        svg += svg_node(position[i], node_size, colors[i])
    # text
    if names is not None:
        for i in range(n):
            svg += svg_text(position[i] + (margin_text, node_size), names[i], font_size)
    svg += '</svg>'
    return svg


def svg_bigraph(biadjacency: sparse.csr_matrix, position_row: np.ndarray, position_col: np.ndarray,
                names_row: Optional[np.ndarray] = None, names_col: Optional[np.ndarray] = None,
                labels_row: Optional[np.ndarray] = None, labels_col: Optional[np.ndarray] = None,
                scores_row: Optional[np.ndarray] = None, scores_col: Optional[np.ndarray] = None,
                color: str = 'blue', width: float = 400, height: float = 300, margin: float = 20,
                margin_text: float = 10, scale: float = 1, node_size: float = 5, edge_width: float = 1,
                font_size: int = 12) -> str:
    """Return svg code for a bipartite graph.

    Parameters
    ----------
    biadjacency :
        Adjacency matrix of the graph.
    position_row :
        Positions of the rows.
    position_col :
        Positions of the columns.
    names_row :
        Names of the rows.
    names_col :
        Names of the columns.
    labels_row :
        Labels of the rows.
    labels_col :
        Labels of the columns.
    scores_row :
        Scores of the rows (measure of importance).
    scores_col :
        Scores of the rows (measure of importance).
    color :
        Default color (svg color).
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
    edge_width :
        Width of edges.
    font_size :
        Font size.

    Returns
    -------
    image : str
        SVG image.

    Example
    -------
    >>> biadjacency = sparse.csr_matrix(np.ones((4,3)))
    >>> position_row = np.random.random((4,2))
    >>> position_col = np.random.random((3,2))
    >>> image = svg_bigraph(biadjacency, position_row, position_col)
    >>> image[1:4]
    'svg'
    """
    n_row, n_col = biadjacency.shape

    # colors
    colors_row = get_colors(n_row, labels_row, scores_row, color)
    colors_col = get_colors(n_col, labels_col, scores_col, color)

    # rescaling
    position = np.vstack((position_row, position_col))
    x = min_max_scaling(position[:, 0])
    y = max_min_scaling(position[:, 1])
    position = np.vstack((x, y)).T
    position = position * np.array([width, height])

    # margins
    position += margin
    height += 2 * margin
    width += 2 * margin
    if names_row is not None:
        text_length = np.max(np.array([len(name) for name in names_row]))
        position[:, 0] += text_length * font_size * .5
        width += text_length * font_size * .5
    if names_col is not None:
        text_length = np.max(np.array([len(name) for name in names_col]))
        width += text_length * font_size * .5

        # scaling
    position *= scale
    height *= scale
    width *= scale
    position_row = position[:n_row]
    position_col = position[n_row:]

    svg = """<svg width="{}" height="{}">""".format(width, height)
    # edges
    for i in range(n_row):
        for j in biadjacency[i].indices:
            svg += svg_edge(position_row[i], position_col[j], edge_width)
    # nodes
    for i in range(n_row):
        svg += svg_node(position_row[i], node_size, colors_row[i])
    for i in range(n_col):
        svg += svg_node(position_col[i], node_size, colors_col[i])
    # text
    if names_row is not None:
        for i in range(n_row):
            svg += svg_text(position_row[i] - (margin_text, 0), names_row[i], font_size, True)
    if names_col is not None:
        for i in range(n_col):
            svg += svg_text(position_col[i] + (margin_text, 0), names_col[i], font_size)
    svg += '</svg>'
    return svg




