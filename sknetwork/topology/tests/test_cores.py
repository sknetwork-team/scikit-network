#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for k-core decimposition"""
import unittest

from sknetwork.data.test_graphs import *
from sknetwork.topology import CoreDecomposition


class TestCoreDecomposition(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        kcore = CoreDecomposition().fit(adjacency)
        self.assertEqual(kcore.core_value_, 0)

    def test_cliques(self):
        adjacency = test_graph_clique()
        n = adjacency.shape[0]
        kcore = CoreDecomposition().fit(adjacency)
        self.assertEqual(kcore.core_value_, n - 1)
