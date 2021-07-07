#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for spectral embedding."""

import unittest

import numpy as np

from sknetwork.data.test_graphs import *
from sknetwork.embedding import Spectral


class TestEmbeddings(unittest.TestCase):

    def setUp(self) -> None:
        """Simple case for testing"""
        self.adjacency = test_graph()
        self.n = self.adjacency.shape[0]
        self.k = 5

    def test_spectral(self):
        spectral = Spectral(3)
        for adjacency in [test_graph()]: #, test_digraph()]:
            embedding = spectral.fit_transform(adjacency)
            weights = adjacency.dot(np.ones(self.n))
            self.assertAlmostEqual(np.linalg.norm(embedding.T.dot(weights)), 0)
        #error = np.abs(spectral.predict(self.adjacency[:4]) - embedding[:4]).sum()
        #self.assertAlmostEqual(error, 0)

    def test_regularization(self):
        adjacency = test_graph_disconnect()
        n = adjacency.shape[0]
        spectral = Spectral(regularization=1.)
        spectral.fit(adjacency)
        spectral.predict(np.random.rand(n))
