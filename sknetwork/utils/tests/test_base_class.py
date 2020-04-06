#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for base.py"""

import unittest

from sknetwork.clustering import Louvain
from sknetwork.hierarchy import Paris
from sknetwork.ranking import PageRank


# noinspection PyMissingOrEmptyDocstring
class TestAlgo(unittest.TestCase):

    def setUp(self):
        self.pagerank = PageRank()
        self.paris = Paris()
        self.louvain = Louvain(tol=0)

    def test_reprs(self):
        self.assertEqual(str(self.pagerank), "PageRank(damping_factor=0.85, solver='lanczos', n_iter=10)")
        self.assertEqual(str(self.paris), "Paris(weights='degree', reorder=True)")
        self.assertEqual(str(self.louvain), "Louvain(algorithm=GreedyModularity(resolution=1, tol=0), "
                                            "agg_tol=0.001, max_agg_iter=-1, shuffle_nodes=False, "
                                            "sorted_cluster=True)")
        self.assertEqual(str(self.louvain.algorithm), "GreedyModularity(resolution=1, tol=0)")
