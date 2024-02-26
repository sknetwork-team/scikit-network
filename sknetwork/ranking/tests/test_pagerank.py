#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for pagerank.py"""

import unittest

import numpy as np

from sknetwork.data.models import cyclic_digraph
from sknetwork.data.test_graphs import test_graph, test_digraph, test_bigraph
from sknetwork.ranking.pagerank import PageRank


class TestPageRank(unittest.TestCase):

    def setUp(self) -> None:
        """Cycle graph for tests."""
        self.n = 5
        self.adjacency = cyclic_digraph(self.n)
        self.truth = np.ones(self.n) / self.n

    def test_params(self):
        with self.assertRaises(ValueError):
            PageRank(damping_factor=1789)

    def test_solvers(self):
        for solver in ['piteration', 'lanczos', 'bicgstab', 'RH']:
            pagerank = PageRank(solver=solver)
            scores = pagerank.fit_predict(self.adjacency)
            self.assertAlmostEqual(0, np.linalg.norm(scores - self.truth))
        with self.assertRaises(ValueError):
            PageRank(solver='toto').fit_predict(self.adjacency)

    def test_seeding(self):
        pagerank = PageRank()
        seeds_array = np.zeros(self.n)
        seeds_array[0] = 1.
        seeds_dict = {0: 1}

        scores1 = pagerank.fit_predict(self.adjacency, seeds_array)
        scores2 = pagerank.fit_predict(self.adjacency, seeds_dict)
        self.assertAlmostEqual(np.linalg.norm(scores1 - scores2), 0.)

    def test_input(self):
        pagerank = PageRank()
        scores = pagerank.fit_predict(self.adjacency, force_bipartite=True)
        self.assertEqual(len(scores), len(pagerank.scores_col_))

    def test_damping(self):
        pagerank = PageRank(damping_factor=0.99)
        scores = pagerank.fit_predict(self.adjacency)
        self.assertAlmostEqual(np.linalg.norm(scores - self.truth), 0.)

        pagerank = PageRank(damping_factor=0.01)
        scores = pagerank.fit_predict(self.adjacency)
        self.assertAlmostEqual(np.linalg.norm(scores - self.truth), 0.)

    def test_bigraph(self):
        pagerank = PageRank()
        for adjacency in [test_graph(), test_digraph(), test_bigraph()]:
            pagerank.fit(adjacency, weights_col={0: 1})
            self.assertAlmostEqual(np.linalg.norm(pagerank.scores_col_ - pagerank.predict(columns=True)), 0.)
