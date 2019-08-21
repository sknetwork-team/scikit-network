#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for pagerank.py"""

import unittest

import numpy as np

from sknetwork.ranking.pagerank import PageRank
from sknetwork.toy_graphs import rock_paper_scissors, random_bipartite_graph


class TestPageRank(unittest.TestCase):

    def setUp(self):
        self.adjacency = rock_paper_scissors()
        self.pagerank = PageRank()
        self.pagerank_high_damping = PageRank(damping_factor=0.99)
        self.bipartite = random_bipartite_graph()

    def test_pagerank(self):
        self.pagerank.fit(self.adjacency)
        score = self.pagerank.score_
        self.assertAlmostEqual(np.linalg.norm(score - np.ones(3) / 3), 0.)
        self.pagerank_high_damping.fit(self.adjacency)
        score = self.pagerank_high_damping.score_
        self.assertAlmostEqual(np.linalg.norm(score - np.ones(3) / 3), 0., places=1)

    def test_bipartite(self):
        biadjacency = self.bipartite
        self.pagerank.fit(biadjacency, {0: 1})
        score = self.pagerank.score_
        self.assertEqual(len(score), 9)
