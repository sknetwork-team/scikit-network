#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for binary GNN probability predictions."""
import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph as make_test_graph
from sknetwork.gnn.gnn_classifier import GNNClassifier


class TestBinaryProbabilities(unittest.TestCase):

    def setUp(self):
        random_state = np.random.get_state()
        self.addCleanup(np.random.set_state, random_state)
        self.adjacency = make_test_graph()
        self.labels = np.array(4 * [0, 1] + 2 * [-1])

    def test_binary_probability_columns(self):
        """Binary prediction APIs return class-zero and class-one probabilities for each node."""
        for dims in [1, [4, 1]]:
            for fit_predict in [False, True]:
                with self.subTest(dims=dims, fit_predict=fit_predict):
                    gnn = GNNClassifier(dims=dims, early_stopping=False)
                    if fit_predict:
                        probs = gnn.fit_predict_proba(self.adjacency, self.adjacency, self.labels,
                                                      n_epochs=3, random_state=42)
                    else:
                        gnn.fit(self.adjacency, self.adjacency, self.labels, n_epochs=3, random_state=42)
                        output = gnn.output_.copy()
                        probs = gnn.predict_proba()
                        np.testing.assert_array_equal(gnn.output_, output)
                    self.assertEqual(probs.shape, (len(self.labels), 2))
                    self.assertTrue(np.isfinite(probs).all())
                    self.assertTrue(((probs >= 0) & (probs <= 1)).all())
                    np.testing.assert_allclose(probs.sum(axis=1), 1)
                    np.testing.assert_array_equal(probs[:, 1], gnn.output_[:, 0])
                    np.testing.assert_array_equal(probs.argmax(axis=1), gnn.predict())

    def test_multiclass_and_unfitted_probability_controls(self):
        """Multiclass outputs are returned directly and an unfitted model still returns None."""
        gnn = GNNClassifier(2, early_stopping=False)
        self.assertIsNone(gnn.predict_proba())
        gnn.fit(self.adjacency, self.adjacency, self.labels, n_epochs=3, random_state=42)
        self.assertIs(gnn.predict_proba(), gnn.output_)
