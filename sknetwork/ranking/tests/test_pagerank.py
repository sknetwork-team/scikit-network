#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for pagerank.py"""

import unittest

import numpy as np

from sknetwork.ranking.pagerank import PageRank, BiPageRank
from sknetwork.data.test_graphs import *
from sknetwork.data.models import cyclic_digraph


# noinspection PyMissingOrEmptyDocstring
class TestPageRank(unittest.TestCase):

    def test_directed(self):
        adjacency = cyclic_digraph(3)
        ground_truth = np.ones(3) / 3

        pagerank_sps = PageRank(solver='spsolve')
        pagerank_sps.fit(adjacency)
        scores = pagerank_sps.scores_
        self.assertAlmostEqual(np.linalg.norm(scores - ground_truth), 0.)

        pagerank_sps.fit(adjacency, seeds=np.array([0, 1, 0]))
        pagerank_sps.fit(adjacency, seeds={1: 1})

        pagerank_high_damping = PageRank(damping_factor=0.99)
        pagerank_high_damping.fit(adjacency)
        scores = pagerank_high_damping.scores_
        self.assertAlmostEqual(np.linalg.norm(scores - ground_truth), 0., places=1)

        pagerank_lcz = PageRank(solver='lanczos')
        pagerank_lcz.fit(adjacency)
        scores = pagerank_lcz.scores_
        self.assertAlmostEqual(np.linalg.norm(scores - ground_truth), 0.)

        pagerank_lsq = PageRank(solver='lsqr')
        pagerank_lsq.fit(adjacency)
        scores = pagerank_lsq.scores_
        self.assertAlmostEqual(np.linalg.norm(scores - ground_truth), 0.)

        pagerank_naive = PageRank(solver=None)
        pagerank_naive.fit(adjacency)

    def test_bipartite(self):
        bipagerank = BiPageRank()
        biadjacency = test_bigraph()
        n1, n2 = biadjacency.shape

        bipagerank.fit(biadjacency, {0: 1})
        row_scores = bipagerank.scores_row_
        self.assertEqual(len(row_scores), n1)
        col_scores = bipagerank.scores_col_
        self.assertEqual(len(col_scores), n2)
        scores = bipagerank.scores_
        self.assertEqual(len(scores), n1)
