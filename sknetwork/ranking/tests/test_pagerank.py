#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for pagerank.py"""

import unittest

import numpy as np

from sknetwork.ranking.pagerank import PageRank, BiPageRank
from sknetwork.toy_graphs import rock_paper_scissors, movie_actor


class TestPageRank(unittest.TestCase):

    def setUp(self):
        self.bipagerank = BiPageRank()

    def test_pagerank(self):
        ground_truth = np.ones(3) / 3
        self.adjacency = rock_paper_scissors()

        self.pagerank_sps = PageRank(solver='spsolve')
        self.pagerank_sps.fit(self.adjacency)
        score = self.pagerank_sps.score_
        self.assertAlmostEqual(np.linalg.norm(score - ground_truth), 0.)

        self.pagerank_sps.fit(self.adjacency, personalization=np.array([0, 1, 0]))
        self.pagerank_sps.fit(self.adjacency, personalization={1: 1})

        self.pagerank_high_damping = PageRank(damping_factor=0.99)
        self.pagerank_high_damping.fit(self.adjacency)
        score = self.pagerank_high_damping.score_
        self.assertAlmostEqual(np.linalg.norm(score - ground_truth), 0., places=1)

        self.pagerank_lcz = PageRank(solver='lanczos')
        self.pagerank_lcz.fit(self.adjacency)
        score = self.pagerank_lcz.score_
        self.assertAlmostEqual(np.linalg.norm(score - ground_truth), 0.)

        self.pagerank_lsq = PageRank(solver='lsqr')
        self.pagerank_lsq.fit(self.adjacency)
        score = self.pagerank_lsq.score_
        self.assertAlmostEqual(np.linalg.norm(score - ground_truth), 0.)

        self.bipagerank.fit(self.adjacency)
        score = self.bipagerank.score_
        self.assertAlmostEqual(np.linalg.norm(score - ground_truth), 0.)

    def test_bipartite(self):
        biadjacency = movie_actor()
        self.bipagerank.fit(biadjacency, {0: 1})
        score = self.bipagerank.score_
        self.assertEqual(len(score), biadjacency.shape[0])
