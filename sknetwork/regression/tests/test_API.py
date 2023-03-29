#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for regression API"""
import unittest

from sknetwork.data.test_graphs import test_bigraph, test_graph, test_digraph
from sknetwork.regression import *


class TestAPI(unittest.TestCase):

    def test_basic(self):
        methods = [Diffusion(), Dirichlet()]
        for adjacency in [test_graph(), test_digraph()]:
            n = adjacency.shape[0]
            for method in methods:
                score = method.fit_predict(adjacency)
                self.assertEqual(score.shape, (n, ))
                self.assertTrue(min(score) >= 0)

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape

        methods = [Diffusion(), Dirichlet()]
        for method in methods:
            method.fit(biadjacency)
            values_row = method.values_row_
            values_col = method.values_col_

            self.assertEqual(values_row.shape, (n_row,))
            self.assertEqual(values_col.shape, (n_col,))
