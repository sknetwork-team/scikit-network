#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for link prediction by nearest neighbors"""
import unittest

from sknetwork.linkpred import NNLinker
from sknetwork.data.test_graphs import *
from sknetwork.embedding import Spectral
from sknetwork.utils import get_degrees


class TestNNLinker(unittest.TestCase):

    def test_link_prediction(self):
        for input_matrix in [test_graph(), test_digraph(), test_bigraph()]:

            n_neighbors = 5
            threshold = 0.2
            algo = NNLinker(n_neighbors=n_neighbors, threshold=threshold)
            links = algo.fit_predict(input_matrix)
            self.assertTrue(links.shape == input_matrix.shape)
            self.assertTrue(np.all(get_degrees(links) <= n_neighbors))
            self.assertTrue(np.all(links.data >= threshold))

            algo = NNLinker(embedding_method=Spectral(2))
            links = algo.fit_predict(input_matrix)
            self.assertTrue(links.shape == input_matrix.shape)
