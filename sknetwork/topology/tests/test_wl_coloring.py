#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Weisfeiler-Lehman coloring"""

import unittest

from sknetwork.topology import WLColoring
from sknetwork.data.test_graphs import *


class TestWLColoring(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        labels = WLColoring().fit(adjacency).labels_
        self.assertEqual(labels, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_cliques(self):
        adjacency = test_graph_clique()
        labels = WLColoring().fit(adjacency).labels_
        self.assertEqual(labels, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        labels = WLColoring().fit(adjacency).labels_
        labels_array = np.array(labels)
        self.assertEqual(labels_array.max(), 6)
        self.assertTrue(labels[1] == labels[2] and labels[2] == labels[3])
        self.assertTrue(labels[4] == labels[5])
