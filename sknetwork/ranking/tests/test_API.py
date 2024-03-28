#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for ranking API"""
import unittest

from sknetwork.data.test_graphs import *
from sknetwork.ranking import *


class TestPageRank(unittest.TestCase):

    def test_basic(self):
        methods = [PageRank(), Closeness(), HITS(), Katz()]
        for adjacency in [test_graph(), test_digraph()]:
            n = adjacency.shape[0]
            for method in methods:
                score = method.fit_predict(adjacency)
                self.assertEqual(score.shape, (n, ))
                self.assertTrue(min(score) >= 0)

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape

        methods = [PageRank(), HITS(), Katz()]
        for method in methods:
            method.fit(biadjacency)
            scores_row = method.scores_row_
            scores_col = method.scores_col_

            self.assertEqual(scores_row.shape, (n_row,))
            self.assertEqual(scores_col.shape, (n_col,))
