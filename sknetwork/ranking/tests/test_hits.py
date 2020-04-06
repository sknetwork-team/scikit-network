#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for hits.py"""

import unittest

from sknetwork.ranking.hits import HITS
from sknetwork.data.test_graphs import *


# noinspection DuplicatedCode
class TestHITS(unittest.TestCase):

    def test_hits(self):
        hits = HITS()

        adjacency = test_graph()
        n = adjacency.shape[0]
        hits.fit(adjacency)
        self.assertEqual(hits.scores_.shape, (n,))
        self.assertEqual(hits.scores_col_.shape, (n,))

        adjacency = test_digraph()
        n = adjacency.shape[0]
        hits.fit(adjacency)
        self.assertEqual(hits.scores_.shape, (n,))
        self.assertEqual(hits.scores_col_.shape, (n,))

        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape
        hits.fit(biadjacency)
        self.assertEqual(hits.scores_.shape, (n_row,))
        self.assertEqual(hits.scores_col_.shape, (n_col,))
