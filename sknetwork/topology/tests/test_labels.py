#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for label propagation"""

import unittest

from sknetwork.topology import LabelPropagation
from sknetwork.data.test_graphs import *
from sknetwork.data import karate_club


class TestLabelPropagation(unittest.TestCase):
	
	def test_empty(self):
		adjacency = test_graph_empty()
		n = adjacency.shape[0]
		labels = LabelPropagation().fit_transform(adjacency)
		self.assertEqual(len(set(labels)), n)
	
	def test_disconnected(self):
		adjacency = test_graph_disconnect()
		n = adjacency.shape[0]
		labels = LabelPropagation(seed=0).fit_transform(adjacency)
		self.assertEqual(len(set(labels)), 5)
		
	def test_cliques(self):
		adjacency = test_graph_clique()
		labels = LabelPropagation(seed=0).fit_transform(adjacency)
		self.assertEqual(len(set(labels)), 1)
		
		
	def test_option(self):
		adjacency = karate_club()
		
		lab = LabelPropagation(seed=10)
		labels = lab.fit_transform(adjacency)
		self.assertEqual(labels[0], 0)
		
		lab = LabelPropagation(seed=32)
		labels = lab.fit_transform(adjacency)
		self.assertEqual(labels[0], 1)
		
		lab = LabelPropagation(seed=10)
		labels = lab.fit_transform(adjacency)
		self.assertEqual(labels[0], 0)