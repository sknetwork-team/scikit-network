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
        scores = closeness.fit_predict(adjacency)
        self.assertEqual(scores.shape, (n,))

    def test_disconnected(self):
        adjacency = test_disconnected_graph()
        closeness = Closeness()
        with self.assertRaises(ValueError):
            closeness.fit(adjacency)
