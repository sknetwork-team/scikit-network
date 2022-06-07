#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for gnn classifier"""

import unittest

import numpy as np
from scipy import sparse

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
        gnn = GNNClassifier('GCNConv', 2, 'Softmax')
        self.assertTrue(gnn.conv1.activation == 'softmax')
        y_pred = gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=0.2)
        embedding = gnn.embedding_
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_dense_feat(self):
        # features not in nparray
        features = self.adjacency.todense()
        gnn = GNNClassifier('GCNConv', 2, 'Softmax')
        with self.assertRaises(TypeError):
            gnn.fit_predict(self.adjacency, features, self.labels, val_size=0.2)

        # features in nparray
        features = np.array(self.adjacency.todense())
        gnn = GNNClassifier('GCNConv', 2, 'Softmax')
        y_pred = gnn.fit_predict(self.adjacency, features, self.labels, val_size=0.2)
        embedding = gnn.embedding_
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_optimizer(self):
        opts = ['None', 'Adam']
        for opt in opts:
            gnn = GNNClassifier('GCNConv', 2, 'Softmax', opt=opt)
            y_pred = gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=0.2)
            embedding = gnn.embedding_
            self.assertTrue(len(y_pred) == self.n)
            self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_loss(self):
        gnn = GNNClassifier('GCNConv', 2, 'Softmax')
        with self.assertRaises(ValueError):
            gnn.fit_predict(self.adjacency, self.features, self.labels, loss='toto', val_size=0.2)

    def test_gnn_classifier_1label(self):
        gnn = GNNClassifier('GCNConv', 1, 'Relu')
        y_pred = gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=0.2)
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(np.min(y_pred) >= 0 and np.max(y_pred) <= 1)

    def test_gnn_classifier_masks(self):
        gnn = GNNClassifier('GCNConv', 2, 'Softmax')
        with self.assertRaises(ValueError):
            gnn.fit_predict(self.adjacency, self.features, self.labels)

        train_mask = np.random.randint(2, size=self.n).astype(bool)
        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, train_mask=train_mask)
        self.assertTrue(sum(gnn.train_mask) + sum(gnn.val_mask) + sum(gnn.test_mask) == self.adjacency.shape[0])

        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=0.2)
        self.assertTrue(sum(gnn.train_mask) + sum(gnn.val_mask) + sum(gnn.test_mask) == self.adjacency.shape[0])

        labels = self.labels.copy()
        labels[:2] = -1  # missing labels
        _ = gnn.fit_predict(self.adjacency, self.features, labels, val_size=0.2)
        self.assertTrue(sum(gnn.test_mask) != 0)
        self.assertTrue(sum(gnn.train_mask) + sum(gnn.val_mask) + sum(gnn.test_mask) == self.adjacency.shape[0])

    def test_gnn_classifier_val_size(self):
        gnn = GNNClassifier('GCNConv', 2, 'Softmax')
        with self.assertRaises(ValueError):
            gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=-1)
        with self.assertRaises(ValueError):
            gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=1.5)

    def test_gnn_classifier_shuffle(self):
        gnn = GNNClassifier('GCNConv', 2, 'Softmax')
        y_pred = gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=0.2, shuffle=False)
        embedding = gnn.embedding_
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_random_state(self):
        gnn = GNNClassifier('GCNConv', 2, 'Softmax')
        y_pred = gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=0.2, random_state=42)
        embedding = gnn.embedding_
        self.assertTrue(len(y_pred) == self.adjacency.shape[0])
        self.assertTrue(embedding.shape == (self.adjacency.shape[0], 2))

    def test_gnn_classifier_verbose(self):
        gnn = GNNClassifier('GCNConv', 2, 'Softmax', verbose=True)
        self.assertTrue(isinstance(gnn, GNNClassifier))

    def test_gnn_classifier_predict(self):
        gnn = GNNClassifier('GCNConv', 2, 'Softmax')
        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=0.2, random_state=42)

        # test result shape for one new node
        new_n = sparse.csr_matrix(np.random.randint(2, size=self.features.shape[1]))
        new_feat = new_n.copy()
        pred_label = gnn.predict(new_n, new_feat)
        self.assertTrue(len(pred_label) == 1)

        # test result shape for several new nodes
        new_n = sparse.csr_matrix(np.random.randint(2, size=(5, self.features.shape[1])))
        new_feat = new_n.copy()
        pred_labels = gnn.predict(new_n, new_feat)
        self.assertTrue(pred_labels.shape == (5,))

        # test invalid format for new nodes
        new_n = [1] * self.features.shape[1]
        with self.assertRaises(TypeError):
            gnn.predict(new_n)
