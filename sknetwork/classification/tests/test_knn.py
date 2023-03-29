#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for KNN"""
import unittest

from sknetwork.classification import KNN
from sknetwork.data.test_graphs import *
from sknetwork.embedding import LouvainEmbedding


class TestKNNClassifier(unittest.TestCase):

    def test_classification(self):
        for adjacency in [test_graph(), test_digraph(), test_bigraph()]:
            labels = {0: 0, 1: 1}

            algo = KNN(n_neighbors=1)
            labels_pred = algo.fit_predict(adjacency, labels)
            self.assertTrue(len(set(labels_pred)) == 2)
