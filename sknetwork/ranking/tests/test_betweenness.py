#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for betweenness.py"""

import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph
from sknetwork.ranking import Betweenness


class TestBetweenness(unittest.TestCase):

    def test_betweenness(self):
        adjacency = test_graph()
        n = adjacency.shape[0]

        betweenness = Betweenness()
        scores = betweenness.fit_transform(adjacency)
        ans = np.array([36., 28.,  0., 16., 40.,  0., 36., 28.,  0., 16.])
        self.assertEqual(scores.tolist(), ans.tolist())
