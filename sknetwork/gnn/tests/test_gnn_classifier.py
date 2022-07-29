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
        gnn = GNNClassifier(2, 'GCNConv', 'Softmax')
        self.assertTrue(gnn.conv1.activation == 'softmax')
        y_pred = gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=0.2)
        embedding = gnn.embedding_
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_dense_feat(self):
        # features not in nparray
        features = self.adjacency.todense()
        gnn = GNNClassifier(2)
        with self.assertRaises(TypeError):
            gnn.fit_predict(self.adjacency, features, self.labels, val_size=0.2)

        # features in numpy array
        features = np.array(self.adjacency.todense())
        gnn = GNNClassifier(2, 'GCNConv', 'Softmax')
        y_pred = gnn.fit_predict(self.adjacency, features, self.labels, val_size=0.2)
        embedding = gnn.embedding_
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_optimizer(self):
        opts = ['None', 'Adam']
        for opt in opts:
            gnn = GNNClassifier(2, 'GCNConv', optimizer=opt)
            y_pred = gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=0.2)
            embedding = gnn.embedding_
            self.assertTrue(len(y_pred) == self.n)
            self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_softmax(self):
        n_labels = len(set(self.labels))
        gnn = GNNClassifier([5, n_labels], 'GCNConv', ['Softmax', 'Softmax'])
        y_pred = gnn.fit_predict(self.adjacency, self.features, self.labels)
        self.assertTrue(len(y_pred) == self.n)

    def test_gnn_classifier_norm(self):
        n_labels = len(set(self.labels))
        gnn = GNNClassifier([5, n_labels], 'GCNConv', activations=['Relu', 'Softmax'],
                            normalizations=['left', 'both'])
        y_pred = gnn.fit_predict(self.adjacency, self.features, self.labels)
        self.assertTrue(len(y_pred) == self.n)

    def test_gnn_classifier_loss(self):
        gnn = GNNClassifier(2, 'GCNConv', 'Softmax')
        with self.assertRaises(ValueError):
            gnn.fit_predict(self.adjacency, self.features, self.labels, loss='toto', val_size=0.2)

    def test_gnn_classifier_1label(self):
        gnn = GNNClassifier(1, 'GCNConv', 'Relu')
        y_pred = gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=0.2)
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(np.min(y_pred) >= 0 and np.max(y_pred) <= 1)

    def test_gnn_classifier_masks(self):
        gnn = GNNClassifier(2, 'GCNConv', 'Softmax', early_stopping=False)

        train_mask = np.array([True, True, True, True, True, True, False, False, False, False])
        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, train_mask=train_mask, n_epochs=5)
        self.assertTrue(sum(gnn.train_mask) + sum(gnn.val_mask) + sum(gnn.test_mask) == self.adjacency.shape[0])

        train_mask = np.array([True, True, True, True, True, True, False, False, False, False])
        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, train_mask=train_mask, resample=True,
                            n_epochs=10)
        self.assertTrue(sum(gnn.train_mask) + sum(gnn.val_mask) + sum(gnn.test_mask) == self.adjacency.shape[0])

        val_mask = np.array([False, False, False, False, False, False, True, False, False, False])
        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, train_mask=train_mask, val_mask=val_mask)
        self.assertTrue(sum(gnn.train_mask) + sum(gnn.val_mask) + sum(gnn.test_mask) == self.adjacency.shape[0])

        test_mask = np.array([False, False, False, False, False, False, False, True, True, True])
        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, train_mask=train_mask, val_mask=val_mask,
                            test_mask=test_mask)
        self.assertTrue(sum(gnn.train_mask) + sum(gnn.val_mask) + sum(gnn.test_mask) == self.adjacency.shape[0])

    def test_gnn_classifier_val_size(self):
        gnn = GNNClassifier(2)
        with self.assertRaises(ValueError):
            gnn.fit_predict(self.adjacency, self.features, self.labels, train_size=None, val_size=None, test_size=None)
        with self.assertRaises(ValueError):
            gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=-1)
        with self.assertRaises(ValueError):
            gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=1.5)

        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, train_size=0.6)
        self.assertTrue(sum(gnn.train_mask) + sum(gnn.val_mask) + sum(gnn.test_mask) == self.adjacency.shape[0])

        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, train_size=0.7, val_size=0.1, test_size=0.2)
        self.assertTrue(sum(gnn.train_mask) + sum(gnn.val_mask) + sum(gnn.test_mask) == self.adjacency.shape[0])

        labels = self.labels.copy()
        labels[:2] = -1  # missing labels
        _ = gnn.fit_predict(self.adjacency, self.features, labels, train_size=0.8, val_size=0.1)
        self.assertTrue(sum(gnn.test_mask) != 0)
        self.assertTrue(sum(gnn.train_mask) + sum(gnn.val_mask) + sum(gnn.test_mask) == self.adjacency.shape[0])

    def test_gnn_classifier_dim_output(self):
        gnn = GNNClassifier(2)
        labels = np.arange(len(self.labels))
        with self.assertRaises(ValueError):
            gnn.fit(self.adjacency, self.features, labels)

    def test_gnn_classifier_random_state(self):
        gnn = GNNClassifier(2)
        y_pred = gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=0.2, random_state=42)
        embedding = gnn.embedding_
        self.assertTrue(len(y_pred) == self.adjacency.shape[0])
        self.assertTrue(embedding.shape == (self.adjacency.shape[0], 2))

    def test_gnn_classifier_verbose(self):
        gnn = GNNClassifier(2, verbose=True)
        self.assertTrue(isinstance(gnn, GNNClassifier))

    def test_gnn_classifier_early_stopping(self):
        gnn = GNNClassifier(2, patience=2)
        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, n_epochs=100)
        self.assertTrue(len(gnn.history_['val_accuracy']) < 100)

        gnn = GNNClassifier(2, early_stopping=False)
        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, n_epochs=100)
        self.assertTrue(len(gnn.history_['val_accuracy']) == 100)

    def test_gnn_classifier_predict(self):
        gnn = GNNClassifier(2)
        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, val_size=0.2, random_state=42)

        pred_labels = gnn.predict()
        self.assertTrue(all(pred_labels == gnn.labels_))

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
