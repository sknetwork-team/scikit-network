#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for triangle counting"""

import unittest

from scipy.special import comb

from sknetwork.data import karate_club
from sknetwork.data.parse import from_edge_list
from sknetwork.data.test_graphs import *
from sknetwork.topology.triangles import count_triangles, get_clustering_coefficient


class TestTriangle(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        self.assertEqual(count_triangles(adjacency), 0)

    def test_disconnected(self):
        adjacency = test_disconnected_graph()
        self.assertEqual(count_triangles(adjacency), 1)

    def test_cliques(self):
        adjacency = test_clique()
        n = adjacency.shape[0]
        self.assertEqual(count_triangles(adjacency), comb(n, 3, exact=True))

    def test_clustering_coefficient(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        adjacency = from_edge_list(edges, directed=False, matrix_only=True)
        self.assertEqual(0.75, get_clustering_coefficient(adjacency))

    def test_options(self):
        adjacency = karate_club()
        self.assertEqual(count_triangles(adjacency, parallelize=False), 45)
        self.assertEqual(count_triangles(adjacency, parallelize=True), 45)
