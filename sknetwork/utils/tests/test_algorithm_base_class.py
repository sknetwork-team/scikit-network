#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for algorithm_base_class.py"""

import unittest
from sknetwork.ranking import PageRank
from sknetwork.hierarchy import Paris
from sknetwork.clustering import Louvain
from sknetwork.utils.checks import check_engine


class TestAlgo(unittest.TestCase):

    def setUp(self):
        self.pagerank = PageRank()
        self.paris = Paris()
        self.louvain = Louvain(tol=0)
        self.engine = check_engine('default')

    def test_reprs(self):
        self.assertEqual(str(self.pagerank), "PageRank(damping_factor=0.85, method='diter', n_iter=25)")
        self.assertEqual(str(self.paris), "Paris(engine='{}')".format(self.engine))
        self.assertEqual(str(self.louvain), "Louvain(algorithm=GreedyModularity(resolution=1, "
                                            "tol=0, engine='{}'), agg_tol=0.001, max_agg_iter=-1, "
                                            "shuffle_nodes=False, verbose=False)".format(self.engine))
        self.assertEqual(str(self.louvain.algorithm), "GreedyModularity(resolution=1, "
                                                      "tol=0, engine='{}')".format(self.engine))
