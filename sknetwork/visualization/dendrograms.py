#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in April 2020
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Iterable, Optional

import numpy as np

from sknetwork.hierarchy.postprocess import cut_straight
from sknetwork.visualization.colors import STANDARD_COLORS


def get_index(dendrogram, reorder=True):
    """Index nodes for pretty dendrogram."""
    n = dendrogram.shape[0] + 1
    tree = {i: [i] for i in range(n)}
    for t in range(n - 1):
        i = int(dendrogram[t, 0])
        j = int(dendrogram[t, 1])
        left: list = tree.pop(i)
        right: list = tree.pop(j)
        if reorder and len(left) < len(right):
            tree[n + t] = right + left
        else:
            tree[n + t] = left + right
    return list(tree.values())[0]


def svg_dendrogram_top(dendrogram, names, width, height, margin, margin_text, scale, line_width, n_clusters,
                       color, colors, font_size, reorder, rotate_names):
    """Dendrogram as SVG image with root on top."""

    # scaling
    height *= scale
    width *= scale

    # positioning
    labels = cut_straight(dendrogram, n_clusters, return_dendrogram=False)
    index = get_index(dendrogram, reorder)
    n = len(index)
    unit_height = height / dendrogram[-1, 2]
    unit_width = width / n
    height_basis = margin + height
    position = {index[i]: (margin + i * unit_width, height_basis) for i in range(n)}
    label = {i: l for i, l in enumerate(labels)}
    width += 2 * margin
    height += 2 * margin
    if names is not None:
        text_length = np.max(np.array([len(str(name)) for name in names]))
        height += text_length * font_size * .5 + margin_text

    svg = """<svg width="{}" height="{}"  xmlns="http://www.w3.org/2000/svg">""".format(width, height)

    # text
    if names is not None:
        for i in range(n):
            x, y = position[i]
            x -= margin_text
            y += margin_text
            text = str(names[i]).replace('&', ' ')
            if rotate_names:
                svg += """<text x="{}" y="{}"  transform="rotate(60, {}, {})" font-size="{}">{}</text>""" \
                    .format(x, y, x, y, font_size, text)
            else:
                y += margin_text
                svg += """<text x="{}" y="{}"  font-size="{}">{}</text>""" \
                    .format(x, y, font_size, text)

    # tree
    for t in range(n - 1):
        i = int(dendrogram[t, 0])
        j = int(dendrogram[t, 1])
        x1, y1 = position.pop(i)
        x2, y2 = position.pop(j)
        l1 = label.pop(i)
        l2 = label.pop(j)
        if l1 == l2:
            line_color = colors[l1 % len(colors)]
        else:
            line_color = color
        x = .5 * (x1 + x2)
        y = height_basis - dendrogram[t, 2] * unit_height
        svg += """<path stroke-width="{}" stroke="{}" d="M {} {} {} {}" />"""\
            .format(line_width, line_color, x1, y1, x1, y)
        svg += """<path stroke-width="{}" stroke="{}" d="M {} {} {} {}" />"""\
            .format(line_width, line_color, x2, y2, x2, y)
        svg += """<path stroke-width="{}" stroke="{}" d="M {} {} {} {}" />"""\
            .format(line_width, line_color, x1, y, x2, y)
        position[n + t] = (x, y)
        label[n + t] = l1

    svg += '</svg>'
    return svg


def svg_dendrogram_left(dendrogram, names, width, height, margin, margin_text, scale, line_width, n_clusters,
                        color, colors, font_size, reorder):
    """Dendrogram as SVG image with root on left side."""

    # scaling
    height *= scale
    width *= scale

    # positioning
    labels = cut_straight(dendrogram, n_clusters, return_dendrogram=False)
    index = get_index(dendrogram, reorder)
    n = len(index)
    unit_height = height / n
    unit_width = width / dendrogram[-1, 2]
    width_basis = width + margin
    position = {index[i]: (width_basis, margin + i * unit_height) for i in range(n)}
    label = {i: l for i, l in enumerate(labels)}
    width += 2 * margin
    height += 2 * margin
    if names is not None:
        text_length = np.max(np.array([len(str(name)) for name in names]))
        width += text_length * font_size * .5 + margin_text

    svg = """<svg width="{}" height="{}"  xmlns="http://www.w3.org/2000/svg">""".format(width, height)

    # text
    if names is not None:
        for i in range(n):
            x, y = position[i]
            x += margin_text
            y += unit_height / 3
            text = str(names[i]).replace('&', ' ')
            svg += """<text x="{}" y="{}" font-size="{}">{}</text>""" \
                .format(x, y, font_size, text)

    # tree
    for t in range(n - 1):
        i = int(dendrogram[t, 0])
        j = int(dendrogram[t, 1])
        x1, y1 = position.pop(i)
        x2, y2 = position.pop(j)
        l1 = label.pop(i)
        l2 = label.pop(j)
        if l1 == l2:
            line_color = colors[l1 % len(colors)]
        else:
            line_color = color
        y = .5 * (y1 + y2)
        x = width_basis - dendrogram[t, 2] * unit_width
        svg += """<path stroke-width="{}" stroke="{}" d="M {} {} {} {}" />"""\
            .format(line_width, line_color, x1, y1, x, y1)
        svg += """<path stroke-width="{}" stroke="{}" d="M {} {} {} {}" />"""\
            .format(line_width, line_color, x2, y2, x, y2)
        svg += """<path stroke-width="{}" stroke="{}" d="M {} {} {} {}" />"""\
            .format(line_width, line_color, x, y1, x, y2)
        position[n + t] = (x, y)
        label[n + t] = l1

    svg += '</svg>'

    return svg


def visualize_dendrogram(dendrogram: np.ndarray, names: Optional[np.ndarray] = None, rotate: bool = False,
                         width: float = 400, height: float = 300, margin: float = 10, margin_text: float = 5,
                         scale: float = 1, line_width: float = 2, n_clusters: int = 2, color: str = 'black',
                         colors: Optional[Iterable] = None, font_size: int = 12, reorder: bool = False,
                         rotate_names: bool = True, filename: Optional[str] = None):
    """Return the image of a dendrogram in SVG format.

    Parameters
    ----------
    dendrogram :
        Dendrogram to display.
    names :
        Names of leaves.
    rotate :
        If ``True``, rotate the tree so that the root is on the left.
    width :
        Width of the image (margins excluded).
    height :
        Height of the image (margins excluded).
    margin :
        Margin.
    margin_text :
        Margin between leaves and their names, if any.
    scale :
        Scaling factor.
    line_width :
        Line width.
    n_clusters :
        Number of coloured clusters to display.
    color :
        Default SVG color for the dendrogram.
    colors :
        SVG colors of the clusters of the dendrogram (optional).
    font_size :
        Font size.
    reorder :
        If ``True``, reorder leaves so that left subtree has more leaves than right subtree.
    rotate_names :
        If ``True``, rotate names of leaves (only valid if **rotate** is ``False``).
    filename :
        Filename for saving image (optional).

    Example
    -------
    >>> dendrogram = np.array([[0, 1, 1, 2], [2, 3, 2, 3]])
    >>> from sknetwork.visualization import svg_dendrogram
    >>> image = svg_dendrogram(dendrogram)
    >>> image[1:4]
    'svg'
    """
    if colors is None:
        colors = STANDARD_COLORS
    elif isinstance(colors, dict):
        colors = np.array(list(colors.values()))
    elif isinstance(colors, list):
        colors = np.array(colors)

    if rotate:
        svg = svg_dendrogram_left(dendrogram, names, width, height, margin, margin_text, scale, line_width, n_clusters,
                                  color, colors, font_size, reorder)
    else:
        svg = svg_dendrogram_top(dendrogram, names, width, height, margin, margin_text, scale, line_width, n_clusters,
                                 color, colors, font_size, reorder, rotate_names)

    if filename is not None:
        with open(filename + '.svg', 'w') as f:
            f.write(svg)

    return svg


def svg_dendrogram(dendrogram: np.ndarray, names: Optional[np.ndarray] = None, rotate: bool = False, width: float = 400,
                   height: float = 300, margin: float = 10, margin_text: float = 5, scale: float = 1,
                   line_width: float = 2, n_clusters: int = 2, color: str = 'black', colors: Optional[Iterable] = None,
                   font_size: int = 12, reorder: bool = False, rotate_names: bool = True,
                   filename: Optional[str] = None):
    """Return the image of a dendrogram in SVG format.

    Alias for visualize_dendrogram.

    Parameters
    ----------
    dendrogram :
        Dendrogram to display.
    names :
        Names of leaves.
    rotate :
        If ``True``, rotate the tree so that the root is on the left.
    width :
        Width of the image (margins excluded).
    height :
        Height of the image (margins excluded).
    margin :
        Margin.
    margin_text :
        Margin between leaves and their names, if any.
    scale :
        Scaling factor.
    line_width :
        Line width.
    n_clusters :
        Number of coloured clusters to display.
    color :
        Default SVG color for the dendrogram.
    colors :
        SVG colors of the clusters of the dendrogram (optional).
    font_size :
        Font size.
    reorder :
        If ``True``, reorder leaves so that left subtree has more leaves than right subtree.
    rotate_names :
        If ``True``, rotate names of leaves (only valid if **rotate** is ``False``).
    filename :
        Filename for saving image (optional).
    """
    return visualize_dendrogram(dendrogram, names, rotate, width, height, margin, margin_text, scale, line_width,
                                n_clusters, color, colors, font_size, reorder, rotate_names, filename)
