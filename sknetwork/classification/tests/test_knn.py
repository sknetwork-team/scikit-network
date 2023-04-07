#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for KNN"""
import unittest

from sknetwork.classification import NNClassifier
from sknetwork.data.test_graphs import *
from sknetwork.embedding import Spectral


class TestKNNClassifier(unittest.TestCase):

    def test_classification(self):
        for adjacency in [test_graph(), test_digraph(), test_bigraph()]:
            labels = {0: 0, 1: 1}

            algo = NNClassifier(n_neighbors=1)
            labels_pred = algo.fit_predict(adjacency, labels)
            self.assertTrue(len(set(labels_pred)) == 2)

            algo = NNClassifier(n_neighbors=1, embedding_method=Spectral(2), normalize=False)
            labels_pred = algo.fit_predict(adjacency, labels)
            self.assertTrue(len(set(labels_pred)) == 2)
