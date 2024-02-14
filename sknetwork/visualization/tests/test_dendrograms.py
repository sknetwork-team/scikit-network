#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for visualization of dendrograms"""

import tempfile
import unittest

import numpy as np

from sknetwork.data.toy_graphs import karate_club, painters
from sknetwork.hierarchy import Paris
from sknetwork.visualization.dendrograms import visualize_dendrogram, svg_dendrogram, svg_dendrogram_top


# noinspection DuplicatedCode
class TestVisualization(unittest.TestCase):

    def test_undirected(self):
        adjacency = karate_club()
        paris = Paris()
        dendrogram = paris.fit_transform(adjacency)
        image = svg_dendrogram(dendrogram)
        self.assertEqual(image[1:4], 'svg')
        n = adjacency.shape[0]
        image = visualize_dendrogram(dendrogram, names=np.arange(n), width=200, height=200, margin=10, margin_text=5,
                                     scale=3, n_clusters=3, color='green', colors=['red', 'blue'], font_size=14,
                                     reorder=True, rotate=True)
        self.assertEqual(image[1:4], 'svg')
        image = svg_dendrogram(dendrogram, names=np.arange(n), width=200, height=200, margin=10, margin_text=5, scale=3,
                               n_clusters=3, color='green', colors={0: 'red', 1: 'blue'}, font_size=14, reorder=False,
                               rotate=True)
        self.assertEqual(image[1:4], 'svg')
        svg_dendrogram_top(dendrogram, names=np.arange(n), width=200, height=200, margin=10, margin_text=5, scale=3,
                           n_clusters=3, color='green', colors=np.array(['red', 'black', 'blue']), font_size=14,
                           reorder=False, rotate_names=True, line_width=0.1)

    def test_directed(self):
        graph = painters(True)
        adjacency = graph.adjacency
        names = graph.names
        paris = Paris()
        dendrogram = paris.fit_transform(adjacency)
        image = visualize_dendrogram(dendrogram)
        self.assertEqual(image[1:4], 'svg')
        image = visualize_dendrogram(dendrogram, names=names, width=200, height=200, margin=10, margin_text=5, scale=3,
                                     n_clusters=3, color='green', font_size=14, reorder=True, rotate=True)
        self.assertEqual(image[1:4], 'svg')

        filename = tempfile.gettempdir() + '/image'
        _ = visualize_dendrogram(dendrogram, filename=filename)
        with open(filename + '.svg', 'r') as f:
            row = f.readline()
            self.assertEqual(row[1:4], 'svg')
