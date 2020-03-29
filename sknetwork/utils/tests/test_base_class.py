#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for base.py"""

import unittest

from sknetwork.clustering import Louvain
from sknetwork.hierarchy import Paris
from sknetwork.ranking import PageRank
from sknetwork.utils.checks import check_engine


# noinspection PyMissingOrEmptyDocstring
class TestAlgo(unittest.TestCase):

    def setUp(self):
        self.pagerank = PageRank()
        self.paris = Paris()
        self.louvain = Louvain(tol=0)
        self.engine = check_engine('default')

    def test_reprs(self):
        self.assertEqual(str(self.pagerank), "PageRank(damping_factor=0.85, solver='lanczos', n_iter=10)")
        self.assertEqual(str(self.paris), "Paris(engine='{}', weights='degree', reorder=True)".format(self.engine))
        self.assertEqual(str(self.louvain), "Louvain(algorithm=GreedyModularity(resolution=1, tol=0, engine='{}'), "
                                            "tol_aggregation=0.001, n_aggregations=-1, shuffle_nodes=False, "
                                            "sort_clusters=True, return_graph=False)".format(self.engine))
        self.assertEqual(str(self.louvain.algorithm), "GreedyModularity(resolution=1, "
                                                      "tol=0, engine='{}')".format(self.engine))
