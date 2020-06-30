#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Weisfeiler-Lehman kernels"""

import unittest
from sknetwork.topology import WLKernel
from sknetwork.data.test_graphs import *
from sknetwork.data import house


class TestWLKernel(unittest.TestCase):

    def test_empty(self):
        adjacency_1 = test_graph_empty()
        adjacency_2 = test_graph_empty()
        wlc = WLKernel()
        similarity = wlc.fit(adjacency_1, adjacency_2, "subtree")
        self.assertTrue(similarity == 1000)

    def test_cliques(self):
        adjacency_1 = test_graph_empty()
        adjacency_2 = test_graph_empty()
        wlc = WLKernel()
        similarity = wlc.fit(adjacency_1, adjacency_2, "subtree")
        self.assertTrue(similarity == 1000)

    def test_house(self):
        adjacency_1 = house()
        adjacency_2 = house()
        wlc = WLKernel()
        similarity = wlc.fit(adjacency_1, adjacency_2, "subtree")
        self.assertTrue(similarity == 49)

    def test_iso(self):
        adjacency_1 = house()
        adjacency_2 = house()
        wlc = WLKernel()
        similarity_1 = wlc.fit(adjacency_1, adjacency_2, "subtree")
        n = adjacency_1.indptr.shape[0] - 1
        reorder = list(range(n))
        np.random.shuffle(reorder)
        adjacency_2 = adjacency_2[reorder][:, reorder]
        similarity_2 = wlc.fit(adjacency_1, adjacency_2, "subtree")
        self.assertTrue(similarity_1 == similarity_2)

    def test_different(self):
        adjacency_1 = house()
        adjacency_2 = test_graph_empty()
        wlc = WLKernel()
        similarity = wlc.fit(adjacency_1, adjacency_2, "subtree")
        self.assertTrue(similarity == 0)
