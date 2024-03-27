#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in April 2020
@author: Thomas Bonald <bonald@enst.fr>
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
import unittest

from sknetwork.data.test_graphs import *
from sknetwork.linalg import Laplacian, Normalizer, CoNeighbor
from sknetwork.linalg.basics import safe_sparse_dot


class TestOperators(unittest.TestCase):

    def test_laplacian(self):
        for adjacency in [test_graph(), test_disconnected_graph()]:
            n = adjacency.shape[1]
            # regular Laplacian
            laplacian = Laplacian(adjacency)
            self.assertAlmostEqual(np.linalg.norm(laplacian.dot(np.ones(n))), 0)
            # normalized Laplacian
            laplacian = Laplacian(adjacency, normalized_laplacian=True)
            weights = adjacency.dot(np.ones(n))
            self.assertAlmostEqual(np.linalg.norm(laplacian.dot(np.sqrt(weights))), 0)
            # regularization
            regularization = 0.1
            laplacian = Laplacian(adjacency, regularization=regularization, normalized_laplacian=True)
            weights = adjacency.dot(np.ones(n)) + regularization
            self.assertAlmostEqual(np.linalg.norm(laplacian.dot(np.sqrt(weights))), 0)
            # product
            shape = (n, 3)
            self.assertEqual(laplacian.dot(np.ones(shape)).shape, shape)
            self.assertEqual(safe_sparse_dot(laplacian, np.ones(shape)).shape, shape)

    def test_normalizer(self):
        for adjacency in [test_graph(), test_disconnected_graph()]:
            n_row, n_col = adjacency.shape
            # square matrix
            normalizer = Normalizer(adjacency)
            non_zeros = adjacency.dot(np.ones(n_col)) > 0
            self.assertAlmostEqual(np.linalg.norm(normalizer.dot(np.ones(n_col)) - non_zeros), 0)
            # single row
            normalizer = Normalizer(adjacency[1])
            self.assertAlmostEqual(float(normalizer.dot(np.ones(n_col))), 1)
            normalizer = Normalizer(adjacency[2].toarray().ravel())
            self.assertAlmostEqual(float(normalizer.dot(np.ones(n_col))), 1)
            # regularization
            normalizer = Normalizer(adjacency, 1)
            self.assertAlmostEqual(np.linalg.norm(normalizer.dot(np.ones(n_col)) - np.ones(n_row)), 0)

    def test_coneighbors(self):
        biadjacency = test_bigraph()
        operator = CoNeighbor(biadjacency)
        operator.astype('float')
        operator.right_sparse_dot(sparse.eye(operator.shape[1], format='csr'))

        operator1 = CoNeighbor(biadjacency, normalized=False)
        operator2 = CoNeighbor(biadjacency, normalized=False)
        x = np.random.randn(operator.shape[1])
        x1 = (-operator1).dot(x)
        x2 = (operator2 * -1).dot(x)
        x3 = operator1.T.dot(x)
        self.assertAlmostEqual(np.linalg.norm(x1 - x2), 0)
        self.assertAlmostEqual(np.linalg.norm(x2 - x3), 0)
