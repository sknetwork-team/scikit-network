#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for pagerank.py"""

import unittest

import numpy as np

from sknetwork.ranking.pagerank import PageRank
from sknetwork.toy_graphs import miserables, rock_paper_scissors, random_bipartite_graph


class TestPageRank(unittest.TestCase):

    def setUp(self):
        self.adjacency = rock_paper_scissors()
        self.miserables = miserables()
        self.pagerank_sps = PageRank(solver='spsolve')
        self.pagerank_lcz = PageRank(solver='lanczos')
        # self.pagerank_hlk = PageRank(solver='halko')
        self.pagerank_high_damping = PageRank(damping_factor=0.99)
        self.bipartite = random_bipartite_graph()

    def test_pagerank(self):
        ground_truth = np.ones(3) / 3
        self.pagerank_sps.fit(self.adjacency)
        score = self.pagerank_sps.score_
        self.assertAlmostEqual(np.linalg.norm(score - ground_truth), 0.)

        self.pagerank_sps.fit(self.adjacency, personalization=np.array([0, 1, 0]))
        self.pagerank_sps.fit(self.adjacency, personalization={1: 1})

        self.pagerank_high_damping.fit(self.adjacency)
        score = self.pagerank_high_damping.score_
        self.assertAlmostEqual(np.linalg.norm(score - ground_truth), 0., places=1)

        self.pagerank_lcz.fit(self.adjacency)
        score = self.pagerank_lcz.score_
        self.assertAlmostEqual(np.linalg.norm(score - ground_truth), 0.)

        # self.pagerank_hlk.fit(self.adjacency)
        # score = self.pagerank_hlk.score_
        # self.assertAlmostEqual(np.linalg.norm(score - ground_truth), 0.)

        # self.pagerank_sps.fit(self.miserables)
        # self.pagerank_hlk.fit(self.miserables)
        # self.assertAlmostEqual(np.linalg.norm(self.pagerank_sps.score_ - self.pagerank_hlk.score_), 0)

    def test_bipartite(self):
        biadjacency = self.bipartite
        self.pagerank_sps.fit(biadjacency, {0: 1})
        score = self.pagerank_sps.score_
        self.assertEqual(len(score), 9)
