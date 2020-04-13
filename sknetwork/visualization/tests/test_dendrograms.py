#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for visualization/dendrograms.py"""

import unittest

import numpy as np

from sknetwork.data.toy_graphs import karate_club, painters
from sknetwork.hierarchy import Paris
from sknetwork.visualization.dendrograms import svg_dendrogram


# noinspection DuplicatedCode
class TestVisualization(unittest.TestCase):

    def test_undirected(self):
        adjacency = karate_club()
        paris = Paris()
        dendrogram = paris.fit_transform(adjacency)
        image = svg_dendrogram(dendrogram)
        self.assertEqual(image[1:4], 'svg')
        n = adjacency.shape[0]
        image = svg_dendrogram(dendrogram, names=np.arange(n), width=200, height=200, margin=10, margin_text=5, scale=3,
                               n_clusters=3, color='green', font_size=14, reorder=True, rotate=True)
        self.assertEqual(image[1:4], 'svg')

    def test_directed(self):
        graph = painters(True)
        adjacency = graph.adjacency
        names = graph.names
        paris = Paris()
        dendrogram = paris.fit_transform(adjacency)
        image = svg_dendrogram(dendrogram)
        self.assertEqual(image[1:4], 'svg')
        image = svg_dendrogram(dendrogram, names=names, width=200, height=200, margin=10, margin_text=5, scale=3,
                               n_clusters=3, color='green', font_size=14, reorder=True, rotate=True)
        self.assertEqual(image[1:4], 'svg')

