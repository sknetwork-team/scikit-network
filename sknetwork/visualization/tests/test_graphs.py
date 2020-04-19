#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for visualization/graphs.py"""

import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph_disconnect, test_bigraph_disconnect
from sknetwork.data.toy_graphs import karate_club, painters, movie_actor
from sknetwork.visualization.graphs import svg_graph, svg_digraph, svg_bigraph


# noinspection DuplicatedCode
class TestVisualization(unittest.TestCase):

    def test_undirected(self):
        graph = karate_club(True)
        adjacency = graph.adjacency
        position = graph.position
        labels = graph.labels
        image = svg_graph(adjacency, position, labels=labels)
        self.assertEqual(image[1:4], 'svg')
        image = svg_graph(adjacency, position, display_edges=False)
        self.assertEqual(image[1:4], 'svg')
        n = adjacency.shape[0]
        image = svg_graph(adjacency, position=None, names=np.arange(n), labels=np.arange(n), scores=np.arange(n),
                          seeds=[0, 1], width=200, height=200, margin=10, margin_text=5, scale=3,
                          node_order=np.flip(np.arange(n)),
                          node_size=5, node_size_min=2, node_size_max=6, display_node_weight=True,
                          node_weights=np.arange(n),
                          node_width=2, node_width_max=5, node_color='red', edge_width=2, edge_width_min=2,
                          edge_width_max=4, edge_color='blue', edge_weight=True, font_size=14)
        self.assertEqual(image[1:4], 'svg')

    def test_directed(self):
        graph = painters(True)
        adjacency = graph.adjacency
        position = graph.position
        names = graph.names
        image = svg_digraph(adjacency, position, names=names)
        self.assertEqual(image[1:4], 'svg')
        image = svg_graph(adjacency, position, display_edges=False)
        self.assertEqual(image[1:4], 'svg')
        n = adjacency.shape[0]
        image = svg_digraph(adjacency, position=None, names=np.arange(n), labels=np.arange(n), scores=np.arange(n),
                            seeds=[0, 1], width=200, height=200, margin=10, margin_text=5, scale=3,
                            node_order=np.flip(np.arange(n)),
                            node_size=5, node_size_min=2, node_size_max=6, display_node_weight=True,
                            node_weights=np.arange(n),
                            node_width=2, node_width_max=5, node_color='red', edge_width=2, edge_width_min=2,
                            edge_width_max=4, edge_color='blue', edge_weight=True, font_size=14)
        self.assertEqual(image[1:4], 'svg')

    def test_bipartite(self):
        graph = movie_actor(True)
        biadjacency = graph.biadjacency
        names_row = graph.names_row
        names_col = graph.names_col
        image = svg_bigraph(biadjacency, names_row, names_col)
        self.assertEqual(image[1:4], 'svg')
        image = svg_bigraph(biadjacency, display_edges=False)
        self.assertEqual(image[1:4], 'svg')
        image = svg_bigraph(biadjacency, reorder=False)
        self.assertEqual(image[1:4], 'svg')
        n_row, n_col = biadjacency.shape
        position_row = np.random.random((n_row, 2))
        position_col = np.random.random((n_col, 2))
        image = svg_bigraph(biadjacency, np.arange(n_row), np.arange(n_col),
                            np.arange(n_row), np.arange(n_col), np.arange(n_row), np.arange(n_col), [0, 1],
                            [1, 2], position_row, position_col, color_row='red', color_col='white',
                            width=200, height=200, margin=10, margin_text=5, scale=3, node_size=5,
                            node_size_min=1, node_size_max=30, node_weights_row=np.arange(n_row),
                            node_weights_col=np.arange(n_col), display_node_weight=True, node_width=2, node_width_max=5,
                            edge_width=2, edge_width_min=0.3, edge_width_max=4, edge_color='red', edge_weight=True,
                            font_size=14)
        self.assertEqual(image[1:4], 'svg')

    def test_disconnect(self):
        adjacency = test_graph_disconnect()
        position = np.random.random((adjacency.shape[0], 2))
        image = svg_graph(adjacency, position)
        self.assertEqual(image[1:4], 'svg')
        biadjacency = test_bigraph_disconnect()
        image = svg_bigraph(biadjacency)
        self.assertEqual(image[1:4], 'svg')
