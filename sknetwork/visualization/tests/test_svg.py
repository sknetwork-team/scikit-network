#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for hits.py"""

import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph_disconnect, test_bigraph_disconnect
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
        n = adjacency.shape[0]
        image = svg_graph(adjacency, position, names=np.arange(n), labels=np.arange(n), scores=np.arange(n),
                          seeds=[0, 1], color='red', width=200, height=200, margin=10, margin_text=5, scale=3,
                          node_size=5, node_width=2, node_width_max=5, edge_width=2, edge_width_max=4,
                          edge_weight=True, font_size=14)
        self.assertEqual(image[1:4], 'svg')

    def test_directed(self):
        graph = painters(True)
        adjacency = graph.adjacency
        position = graph.position
        names = graph.names
        image = svg_digraph(adjacency, position, names=names)
        self.assertEqual(image[1:4], 'svg')
        n = adjacency.shape[0]
        image = svg_digraph(adjacency, position, names=np.arange(n), labels=np.arange(n), scores=np.arange(n),
                            seeds=[0, 1], color='red', width=200, height=200, margin=10, margin_text=5, scale=3,
                            node_size=5, node_width=2, node_width_max=5, edge_width=2, edge_width_max=4,
                            edge_weight=True, font_size=14)
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
        n_row, n_col = biadjacency.shape
        image = svg_bigraph(biadjacency, position_row, position_col, np.arange(n_row), np.arange(n_col),
                            np.arange(n_row), np.arange(n_col), np.arange(n_row), np.arange(n_col), seeds_row=[0, 1],
                            seeds_col=[1, 2], color_row='red', color_col='white', width=200, height=200, margin=10,
                            margin_text=5, scale=3, node_size=5, node_width=2, node_width_max=5, edge_width=2,
                            edge_width_max=4, edge_weight=True, font_size=14)
        self.assertEqual(image[1:4], 'svg')

    def test_disconnect(self):
        adjacency = test_graph_disconnect()
        position = np.random.random((adjacency.shape[0], 2))
        image = svg_graph(adjacency, position)
        self.assertEqual(image[1:4], 'svg')
        biadjacency = test_bigraph_disconnect()
        position_row = np.random.random((biadjacency.shape[0], 2))
        position_col = np.random.random((biadjacency.shape[1], 2))
        image = svg_bigraph(biadjacency, position_row, position_col)
        self.assertEqual(image[1:4], 'svg')
