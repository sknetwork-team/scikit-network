# -*- coding: utf-8 -*-
# test for metrics.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Bertrand Charpentier <bertrand.charpentier@live.fr>
#
# This file is part of Scikit-network.
#
# NetworkX is distributed under a BSD license; see LICENSE.txt for more information.

import unittest
from sknetwork.hierarchy.paris import Paris
#from sknetwork.hierarchy.metrics import hierarchical_cost


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()

    #def test_unknown_options(self):
        #with self.assertRaises(ValueError):
            #hierarchical_cost(self.weighted_graph, self.d_w, affinity='unknown')

        #with self.assertRaises(ValueError):
            #hierarchical_cost(self.weighted_graph, self.dendrogram, linkage='unknown')

