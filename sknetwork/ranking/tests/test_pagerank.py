#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for pagerank.py"""

import unittest

import numpy as np

from sknetwork.ranking.pagerank import PageRank, BiPageRank
from sknetwork.data.test_graphs import test_bigraph
from sknetwork.data.models import cyclic_digraph


class TestPageRank(unittest.TestCase):

    def setUp(self) -> None:
        """Cycle graph for tests."""
        self.n = 5
        self.adjacency = cyclic_digraph(self.n)
        self.truth = np.ones(self.n) / self.n

    def test_solvers(self):
        pr = PageRank()
        scores = pr.fit_transform(self.adjacency)
        self.assertAlmostEqual(np.linalg.norm(scores - self.truth), 0.)

        pr = PageRank(solver='lanczos')
        scores = pr.fit_transform(self.adjacency)
        self.assertAlmostEqual(np.linalg.norm(scores - self.truth), 0.)

        pr = PageRank(solver='lsqr')
        scores = pr.fit_transform(self.adjacency)
        self.assertAlmostEqual(np.linalg.norm(scores - self.truth), 0.)

    def test_seeding(self):
        pr = PageRank()
        seeds_array = np.zeros(self.n)
        seeds_array[0] = 1.
        seeds_dict = {0: 1}

        scores1 = pr.fit_transform(self.adjacency, seeds_array)
        scores2 = pr.fit_transform(self.adjacency, seeds_dict)
        self.assertAlmostEqual(np.linalg.norm(scores1 - scores2), 0.)

    def test_damping(self):
        pr = PageRank(damping_factor=0.99)
        scores = pr.fit_transform(self.adjacency)
        self.assertAlmostEqual(np.linalg.norm(scores - self.truth), 0.)

        pr = PageRank(damping_factor=0.01)
        scores = pr.fit_transform(self.adjacency)
        self.assertAlmostEqual(np.linalg.norm(scores - self.truth), 0.)

    def test_bipartite(self):
        bipagerank = BiPageRank()
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape

        bipagerank.fit(biadjacency, {0: 1})
        scores_row = bipagerank.scores_row_
        scores_col = bipagerank.scores_col_
        scores = bipagerank.scores_

        self.assertEqual(scores_row.shape, (n_row,))
        self.assertEqual(scores_col.shape, (n_col,))
        self.assertEqual(scores.shape, (n_row,))
