#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for k-core decimposition"""

import unittest

from sknetwork.topology import CoreDecomposition
from sknetwork.data.test_graphs import *
from sknetwork.data import karate_club


class TestCoreDecomposition(unittest.TestCase):
	
	def test_empty(self):
		adjacency = test_graph_empty()
		n = adjacency.shape[0]
		core_val = CoreDecomposition().fit_transform(adjacency)
		self.assertEqual(core_val, 0)
		
	def test_cliques(self):
		adjacency = test_graph_clique()
		n = adjacency.shape[0]
		core_val = CoreDecomposition().fit_transform(adjacency)
		self.assertEqual(core_val, n-1)
		
	def test_cores(self):
		adjacency = karate_club()
		
		core = CoreDecomposition()
		core_val = core.fit_transform(adjacency)
		self.assertEqual(core_val, 4)