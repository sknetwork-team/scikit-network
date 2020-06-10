#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for label propagation"""

import unittest

from sknetwork.topology import TriangleListing
from sknetwork.data.test_graphs import *
from sknetwork.data import karate_club

from scipy.special import comb


class TestTriangleListing(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        n = adjacency.shape[0]
        nb = TriangleListing().fit_transform(adjacency)
        self.assertEqual(nb, 0)

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        n = adjacency.shape[0]
        nb = TriangleListing().fit_transform(adjacency)
        self.assertEqual(nb, 1)

    def test_cliques(self):
        adjacency = test_graph_clique()
        n = adjacency.shape[0]
        nb = TriangleListing().fit_transform(adjacency)
        self.assertEqual(nb, comb(n, 3, exact=True))

    def test_options(self):
        adjacency = karate_club()

        tri = TriangleListing()
        nb = tri.fit_transform(adjacency)
        self.assertEqual(nb, 45)

        tri = TriangleListing(parallelize=True)
        nb = tri.fit_transform(adjacency)
        self.assertEqual(nb, 45)
