#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for closeness.py"""

import unittest

from sknetwork.ranking.closeness import Closeness
from sknetwork.data.test_graphs import *


# noinspection DuplicatedCode,PyMissingOrEmptyDocstring
class TestDiffusion(unittest.TestCase):

    def test_undirected(self):
        adjacency = test_graph()
        closeness = Closeness(method='approximate', n_jobs=-1)
        scores = closeness.fit_transform(adjacency)
        self.assertEqual(len(scores), 10)

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        closeness = Closeness()
        with self.assertRaises(ValueError):
            closeness.fit(adjacency)
