#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Weisfeiler-Lehman coloring"""

import unittest

from sknetwork.topology import WLColoring
from sknetwork.data.test_graphs import *
from sknetwork.data import house

class TestWLColoring(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        labels = WLColoring().fit(adjacency).labels_
        self.assertEqual(labels, np.zeros(10))

    def test_cliques(self):
        adjacency = test_graph_clique()
        labels = WLColoring().fit(adjacency).labels_
        self.assertEqual(labels, np.zeros(10))

    def test_house(self):
        adjacency = house()
        labels = WLColoring().fit(adjacency).labels_
        self.assertEqual(labels, np.array([1, 2, 0, 0, 2]))
