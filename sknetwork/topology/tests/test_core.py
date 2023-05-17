#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for k-core decomposition"""
import unittest

from sknetwork.data.test_graphs import *
from sknetwork.topology.core import get_core_decomposition


class TestCoreDecomposition(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        self.assertEqual(max(get_core_decomposition(adjacency)), 0)

    def test_cliques(self):
        adjacency = test_clique()
        n = adjacency.shape[0]
        self.assertEqual(max(get_core_decomposition(adjacency)), n - 1)
