#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2020
@author: Clément Bonet <cbonet@enst.fr>
"""

import unittest

from sknetwork.clustering import VEM
from sknetwork.data.test_graphs import *

class TestVEM(unittest.TestCase):

    def setUp(self):
        self.vem = VEM(3)

    def test_undirected(self):
        for adjacency in [test_graph()]:
            n = adjacency.shape[0]
            labels = self.vem.fit_transform(adjacency)
            self.assertEqual(len(set(labels)), 3)
            self.assertEqual(self.vem.membership_.shape, (n, 3))
            self.assertEqual(self.vem.adjacency_.shape, (3, 3))

