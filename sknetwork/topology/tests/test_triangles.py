#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for label propagation"""

import unittest

from scipy.special import comb

from sknetwork.data import karate_club
from sknetwork.data.parse import from_edge_list
from sknetwork.data.test_graphs import *
from sknetwork.topology import Triangles


class TestTriangleListing(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        self.assertEqual(Triangles().fit_transform(adjacency), 0)

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        self.assertEqual(Triangles().fit_transform(adjacency), 1)

    def test_cliques(self):
        adjacency = test_graph_clique()
        n = adjacency.shape[0]
        nb = Triangles().fit_transform(adjacency)
        self.assertEqual(nb, comb(n, 3, exact=True))

    def test_clustering_coefficient(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        adjacency = from_edge_list(edges, directed=False, matrix_only=True)

        triangles = Triangles().fit(adjacency)
        self.assertEqual(0.75, triangles.clustering_coef_)

    def test_options(self):
        adjacency = karate_club()

        self.assertEqual(Triangles(parallelize=False).fit_transform(adjacency), 45)
        self.assertEqual(Triangles(parallelize=True).fit_transform(adjacency), 45)
