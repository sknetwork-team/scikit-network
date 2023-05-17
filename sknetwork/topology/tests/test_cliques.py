#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for cliques"""
import unittest

from scipy.special import comb

from sknetwork.data.test_graphs import *
from sknetwork.topology.cliques import count_cliques


class TestClique(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        self.assertEqual(count_cliques(adjacency), 0)
        with self.assertRaises(ValueError):
            count_cliques(adjacency, 1)

    def test_disconnected(self):
        adjacency = test_disconnected_graph()
        self.assertEqual(count_cliques(adjacency), 1)

    def test_cliques(self):
        adjacency = test_clique()
        n = adjacency.shape[0]
        self.assertEqual(count_cliques(adjacency), comb(n, 3, exact=True))
        self.assertEqual(count_cliques(adjacency, 4), comb(n, 4, exact=True))
