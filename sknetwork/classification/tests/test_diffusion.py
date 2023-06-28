#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for DiffusionClassifier"""

import unittest

from sknetwork.classification import DiffusionClassifier
from sknetwork.data.test_graphs import *


class TestDiffusionClassifier(unittest.TestCase):

    def test_graph(self):
        adjacency = test_graph()
        n_nodes = adjacency.shape[0]
        labels = {0: 0, 1: 1}
        algo = DiffusionClassifier()
        algo.fit(adjacency, labels=labels)
        self.assertTrue(len(algo.labels_) == n_nodes)
        adjacency = test_digraph()
        algo = DiffusionClassifier(centering=False)
        algo.fit(adjacency, labels=labels)
        self.assertTrue(len(algo.labels_) == n_nodes)
        with self.assertRaises(ValueError):
            DiffusionClassifier(n_iter=0)
        algo = DiffusionClassifier(centering=True, scale=10)
        probs = algo.fit_predict_proba(adjacency, labels=labels)[:, 1]
        self.assertTrue(max(probs) > 0.99)

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape
        labels_row = {0: 0, 1: 1}
        labels_col = {5: 1}
        algo = DiffusionClassifier()
        algo.fit(biadjacency, labels_row=labels_row, labels_col=labels_col)
        self.assertTrue(len(algo.labels_row_) == n_row)
        self.assertTrue(len(algo.labels_col_) == n_col)
        self.assertTrue(all(algo.labels_col_ == algo.predict(columns=True)))

    def test_predict(self):
        adjacency = test_graph()
        n_nodes = adjacency.shape[0]
        labels = {0: 0, 1: 1}
        algo = DiffusionClassifier()
        labels_pred = algo.fit_predict(adjacency, labels=labels)
        self.assertTrue(len(labels_pred) == n_nodes)
        probs_pred = algo.fit_predict_proba(adjacency, labels=labels)
        self.assertTrue(probs_pred.shape == (n_nodes, 2))
        membership = algo.fit_transform(adjacency, labels=labels)
        self.assertTrue(membership.shape == (n_nodes, 2))

        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape
        labels_row = {0: 0, 1: 1}
        algo = DiffusionClassifier()
        labels_pred = algo.fit_predict(biadjacency, labels_row=labels_row)
        self.assertTrue(len(labels_pred) == n_row)
        labels_pred = algo.predict(columns=True)
        self.assertTrue(len(labels_pred) == n_col)
        probs_pred = algo.fit_predict_proba(biadjacency, labels_row=labels_row)
        self.assertTrue(probs_pred.shape == (n_row, 2))
        probs_pred = algo.predict_proba(columns=True)
        self.assertTrue(probs_pred.shape == (n_col, 2))
        membership = algo.fit_transform(biadjacency, labels_row=labels_row)
        self.assertTrue(membership.shape == (n_row, 2))
        membership = algo.transform(columns=True)
        self.assertTrue(membership.shape == (n_col, 2))

    def test_reindex_label(self):
        adjacency = test_graph()
        n_nodes = adjacency.shape[0]
        labels = {0: 0, 1: 2, 2: 3}
        algo = DiffusionClassifier()
        labels_pred = algo.fit_predict(adjacency, labels=labels)
        self.assertTrue(len(labels_pred) == n_nodes)
        self.assertTrue(set(list(labels_pred)) == {0, 2, 3})
