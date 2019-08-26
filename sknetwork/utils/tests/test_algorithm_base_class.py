#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for algorithm_base_class.py"""

import unittest

from sknetwork.clustering import Louvain
from sknetwork.hierarchy import Paris
from sknetwork.ranking import PageRank
from sknetwork.utils.checks import check_engine


class TestAlgo(unittest.TestCase):

    def setUp(self):
        self.pagerank = PageRank()
        self.paris = Paris()
        self.louvain = Louvain(tol=0)
        self.engine = check_engine('default')

    def test_reprs(self):
        self.assertEqual(str(self.pagerank), "PageRank(damping_factor=0.85, solver='lanczos')")
        self.assertEqual(str(self.paris), "Paris(engine='{}', weights='degree', secondary_weights=None, "
                                          "force_undirected=False, reorder=True)".format(self.engine))
        self.assertEqual(str(self.louvain), "Louvain(algorithm=GreedyModularity(resolution=1, "
                                            "tol=0, engine='{}'), "
                                            "weights='degree', secondary_weights=None, "
                                            "agg_tol=0.001, max_agg_iter=-1, "
                                            "shuffle_nodes=False, "
                                            "force_undirected=False, "
                                            "sorted_cluster=True, verbose=False)".format(self.engine))
        self.assertEqual(str(self.louvain.algorithm), "GreedyModularity(resolution=1, "
                                                      "tol=0, engine='{}')".format(self.engine))
