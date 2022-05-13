#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for gnn classifier"""

import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph
from sknetwork.gnn.gnn_classifier import GNNClassifier


class TestGNNClassifier(unittest.TestCase):

    def setUp(self) -> None:
        """Test graph for tests."""
        self.adjacency = test_graph()
        self.n = self.adjacency.shape[0]
        self.features = self.adjacency
        self.labels = np.array([0]*5 + [1]*5)

    def test_gnn_classifier_sparse_feat(self):
        gnn = GNNClassifier(self.features.shape[1], 4, 2)
        self.assertTrue(gnn.conv2.activation == 'softmax')
        y_pred = gnn.fit_transform(self.adjacency, self.features, self.labels, test_size=0.2)
        embedding = gnn.conv2.emb
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_dense_feat(self):
        # features not in nparray
        features = self.adjacency.todense()
        gnn = GNNClassifier(features.shape[1], 4, 2)
        with self.assertRaises(TypeError):
            gnn.fit_transform(self.adjacency, features, self.labels, test_size=0.2)

        # features in nparray
        features = np.array(self.adjacency.todense())
        gnn = GNNClassifier(features.shape[1], 4, 2)
        y_pred = gnn.fit_transform(self.adjacency, features, self.labels, test_size=0.2)
        embedding = gnn.conv2.emb
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_optimizer(self):
        opts = ['none', 'Adam']
        for opt in opts:
            gnn = GNNClassifier(self.features.shape[1], 4, 2, opt=opt)
            y_pred = gnn.fit_transform(self.adjacency, self.features, self.labels, test_size=0.2)
            embedding = gnn.conv2.emb
            self.assertTrue(len(y_pred) == self.n)
            self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_loss(self):
        gnn = GNNClassifier(self.features.shape[1], 4, 2)
        with self.assertRaises(ValueError):
            gnn.fit_transform(self.adjacency, self.features, self.labels, loss='toto', test_size=0.2)

    def test_gnn_classifier_1label(self):
        gnn = GNNClassifier(self.features.shape[1], 4, 1)
        y_pred = gnn.fit_transform(self.adjacency, self.features, self.labels, test_size=0.2)
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(np.min(y_pred) >= 0 and np.max(y_pred) <= 1)

    def test_gnn_classifier_masks(self):
        gnn = GNNClassifier(self.features.shape[1], 4, 2)
        with self.assertRaises(ValueError):
            gnn.fit_transform(self.adjacency, self.features, self.labels, train_mask=np.random.randint(2, size=self.n))
        with self.assertRaises(ValueError):
            gnn.fit_transform(self.adjacency, self.features, self.labels, test_mask=np.random.randint(2, size=self.n))
        train_mask = np.random.randint(2, size=self.n).astype(bool)
        test_mask = np.invert(train_mask)
        y_pred = gnn.fit_transform(self.adjacency, self.features, self.labels, train_mask=train_mask,
                                   test_mask=test_mask)
        self.assertTrue(len(y_pred) == self.n)

    def test_gnn_classifier_test_size(self):
        gnn = GNNClassifier(self.features.shape[1], 4, 2)
        with self.assertRaises(ValueError):
            gnn.fit_transform(self.adjacency, self.features, self.labels, test_size=-1)
        with self.assertRaises(ValueError):
            gnn.fit_transform(self.adjacency, self.features, self.labels, test_size=1.5)

    def test_gnn_classifier_shuffle(self):
        gnn = GNNClassifier(self.features.shape[1], 4, 2)
        y_pred = gnn.fit_transform(self.adjacency, self.features, self.labels, test_size=0.2, shuffle=False)
        embedding = gnn.conv2.emb
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_random_state(self):
        gnn = GNNClassifier(self.features.shape[1], 4, 2)
        y_pred = gnn.fit_transform(self.adjacency, self.features, self.labels, test_size=0.2, random_state=42)
        embedding = gnn.conv2.emb
        self.assertTrue(len(y_pred) == self.adjacency.shape[0])
        self.assertTrue(embedding.shape == (self.adjacency.shape[0], 2))

    def test_gnn_classifier_predict(self):
        gnn = GNNClassifier(self.features.shape[1], 4, 2)
        _ = gnn.fit_transform(self.adjacency, self.features, self.labels, test_size=0.2, random_state=42)

        # test result shape for one new node
        new_n = np.random.randint(2, size=self.features.shape[1])
        pred_label = gnn.predict(new_n)
        self.assertTrue(len(pred_label) == 1)

        # test result shape for several new nodes
        new_n = np.random.randint(2, size=(5, self.features.shape[1]))
        pred_labels = gnn.predict(new_n)
        print(pred_labels.shape)
        self.assertTrue(pred_labels.shape == (5,))

        # test invalid format for new nodes
        new_n = [1] * self.features.shape[1]
        with self.assertRaises(TypeError):
            gnn.predict(new_n)
