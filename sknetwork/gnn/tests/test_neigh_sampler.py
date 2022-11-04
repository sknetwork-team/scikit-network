#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for neighbor sampler"""

import unittest

from sknetwork.data.test_graphs import test_graph
from sknetwork.gnn.neighbor_sampler import *
from sknetwork.utils import get_degrees


class TestNeighSampler(unittest.TestCase):

    def setUp(self) -> None:
        """Test graph for tests."""
        self.adjacency = test_graph()
        self.n = self.adjacency.shape[0]

    def test_uni_node_sampler(self):
        uni_sampler = UniformNeighborSampler(sample_size=2)
        sampled_adj = uni_sampler(self.adjacency)
        self.assertTrue(sampled_adj.shape == self.adjacency.shape)
        self.assertTrue(all(get_degrees(sampled_adj) <= 2))
