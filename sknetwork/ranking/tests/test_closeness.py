#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for closeness.py"""

import unittest

from sknetwork.data.test_graphs import *
from sknetwork.ranking.closeness import Closeness


class TestDiffusion(unittest.TestCase):

    def test_params(self):
        with self.assertRaises(ValueError):
            adjacency = test_graph()
            Closeness(method='toto').fit(adjacency)

    def test_parallel(self):
        adjacency = test_graph()
        n = adjacency.shape[0]

        closeness = Closeness(method='approximate')
        scores1 = closeness.fit_transform(adjacency)
        closeness = Closeness(method='approximate', n_jobs=-1)
        scores2 = closeness.fit_transform(adjacency)

        self.assertEqual(scores1.shape, (n,))
        self.assertAlmostEqual(np.linalg.norm(scores1 - scores2), 0)

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        closeness = Closeness()
        with self.assertRaises(ValueError):
            closeness.fit(adjacency)
