#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Weisfeiler-Lehman kernels"""

import unittest
import numpy as np
from sknetwork.topology import wl_kernel
from sknetwork.data.test_graphs import test_graph_clique, test_graph_empty
from sknetwork.data import house


class TestWLKernel(unittest.TestCase):

    # Isomorphism tests.

    def test_iso_and_house_iso(self):
        adjacency_1 = house()
        adjacency_2 = house()
        similarity_1 = wl_kernel(adjacency_1, adjacency_2, -1, "isomorphism")
        n = adjacency_1.indptr.shape[0] - 1
        reorder = list(range(n))
        np.random.shuffle(reorder)
        adjacency_2 = adjacency_2[reorder][:, reorder]
        similarity_2 = wl_kernel(adjacency_1, adjacency_2, -1, "isomorphism")
        self.assertTrue(similarity_1 == similarity_2 and similarity_1 == 1)

    # Subtree tests.

    def test_iso_and_house_sub(self):
        adjacency_1 = house()
        adjacency_2 = house()
        similarity_1 = wl_kernel(adjacency_1, adjacency_2, -1, "subtree")
        n = adjacency_1.indptr.shape[0] - 1
        reorder = list(range(n))
        np.random.shuffle(reorder)
        adjacency_2 = adjacency_2[reorder][:, reorder]
        similarity_2 = wl_kernel(adjacency_1, adjacency_2, -1, "subtree")
        self.assertTrue(similarity_1 == similarity_2 and similarity_1 == 45)

    # Edge tests.

    def test_clique_edg(self):
        adjacency_1 = test_graph_clique()
        adjacency_2 = test_graph_clique()
        similarity = wl_kernel(adjacency_1, adjacency_2, -1, "edge")
        self.assertTrue(similarity == 20250)

    def test_iso_and_house_edg(self):
        adjacency_1 = house()
        adjacency_2 = house()
        similarity_1 = wl_kernel(adjacency_1, adjacency_2, -1, "edge")
        n = adjacency_1.indptr.shape[0] - 1
        reorder = list(range(n))
        np.random.shuffle(reorder)
        adjacency_2 = adjacency_2[reorder][:, reorder]
        similarity_2 = wl_kernel(adjacency_1, adjacency_2, -1, "edge")
        self.assertTrue(similarity_1 == similarity_2 and similarity_1 == 50)
