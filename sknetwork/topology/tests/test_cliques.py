#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for k-cliques listing"""

import unittest

from sknetwork.topology import CliqueListing
from sknetwork.data.test_graphs import *
from sknetwork.data import karate_club

from scipy.special import comb

class TestCliqueListing(unittest.TestCase):
	
	def test_empty(self):
		adjacency = test_graph_empty()
		nb = CliqueListing().fit_transform(adjacency, 2)
		self.assertEqual(nb, 0)
		
	def test_disconnected(self):
		adjacency = test_graph_disconnect()
		nb = CliqueListing().fit_transform(adjacency, 3)
		self.assertEqual(nb, 1)
		
	def test_cliques(self):
		adjacency = test_graph_clique()
		n = adjacency.shape[0]
		clique = CliqueListing()
		nb = clique.fit_transform(adjacency, 3)
		self.assertEqual(nb, comb(n, 3))
		
		nb = clique.fit_transform(adjacency, 4)
		self.assertEqual(nb, comb(n, 4))
		
	def test_kcliques(self):
		adjacency = karate_club()
		nb = CliqueListing().fit_transform(adjacency, 3)
		self.assertEqual(nb, 45)