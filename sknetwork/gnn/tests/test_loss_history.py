#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training losses agree with the probabilities returned by the classifier."""

import unittest

import numpy as np

from sknetwork.data import karate_club
from sknetwork.gnn import GNNClassifier


class TestLossHistory(unittest.TestCase):

    def setUp(self):
        graph = karate_club(metadata=True)
        self.adjacency = graph.adjacency
        self.labels = graph.labels.copy()

    def test_binary_loss_history(self):
        """Binary loss is computed before sigmoid, using only training labels."""
        self.labels[[2, 7]] = -1
        for sparse_features in (False, True):
            with self.subTest(sparse_features=sparse_features):
                features = self.adjacency if sparse_features else self.adjacency.toarray()
                model = GNNClassifier(1, learning_rate=0, early_stopping=False)
                model.fit(self.adjacency, features, self.labels, n_epochs=1,
                          validation=0.25, random_state=42)
                mask = model.train_mask
                self.assertTrue(mask.any() and model.val_mask.any())
                self.assertFalse(mask[[2, 7]].any())
                probabilities = model.output_[mask, 0]
                labels = self.labels[mask]
                expected = -(labels * np.log(probabilities)
                             + (1 - labels) * np.log1p(-probabilities)).mean()
                self.assertAlmostEqual(model.history_['loss'][0], expected)

    def test_categorical_loss_history(self):
        """Categorical loss is the negative log probability of the true class."""
        self.labels[[2, 7]] = -1
        for n_classes in (2, 3):
            with self.subTest(n_classes=n_classes):
                model = GNNClassifier(n_classes, learning_rate=0, early_stopping=False)
                model.fit(self.adjacency, self.adjacency, self.labels, n_epochs=1,
                          validation=0.25, random_state=42)
                mask = model.train_mask
                self.assertTrue(mask.any() and model.val_mask.any())
                self.assertFalse(mask[[2, 7]].any())
                probabilities = model.output_[mask]
                labels = self.labels[mask]
                expected = -np.log(probabilities[np.arange(len(labels)), labels]).mean()
                self.assertAlmostEqual(model.history_['loss'][0], expected)
