#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for hits.py"""

import unittest

from sknetwork.data.toy_graphs import karate_club, painters, movie_actor
from sknetwork.visualization.svg import svg_graph, svg_digraph, svg_bigraph


# noinspection DuplicatedCode,PyMissingOrEmptyDocstring
class TestVisualization(unittest.TestCase):

    def test_undirected(self):
        graph = karate_club(True)
        adjacency = graph.adjacency
        position = graph.position
        labels = graph.labels
        image = svg_graph(adjacency, position, labels=labels)
        self.assertEqual(image[1:4], 'svg')

    def test_directed(self):
        graph = painters(True)
        adjacency = graph.adjacency
        position = graph.position
        names = graph.names
        image = svg_digraph(adjacency, position, names=names)
        self.assertEqual(image[1:4], 'svg')

    def test_bipartite(self):
        graph = movie_actor(True)
        biadjacency = graph.biadjacency
        position_row = graph.position_row
        position_col = graph.position_col
        names_row = graph.names_row
        names_col = graph.names_col
        image = svg_bigraph(biadjacency, position_row, position_col, names_row, names_col)
        self.assertEqual(image[1:4], 'svg')

