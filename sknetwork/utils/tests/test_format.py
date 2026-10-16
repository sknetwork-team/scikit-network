#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for format.py"""
import unittest

import numpy as np
from scipy import sparse

from sknetwork.data.test_graphs import *
from sknetwork.utils.format import *
from sknetwork.utils.format import directed2undirected, is_symmetric


class TestFormats(unittest.TestCase):

    def test_directed2undirected_floating_weights(self):
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                adjacency = sparse.csr_matrix(
                    [[0, .25, 0], [.5, 0, .75], [0, .125, 0]], dtype=dtype)
                original = adjacency.copy()
                result = directed2undirected(adjacency)
                expected = [[0, .75, 0], [.75, 0, .875], [0, .875, 0]]
                np.testing.assert_array_equal(result.toarray(), expected)
                np.testing.assert_array_equal(adjacency.toarray(),
                                              original.toarray())
                self.assertTrue(is_symmetric(result))

    def test_directed2undirected_integer_weights(self):
        for dtype in [bool, np.int32, np.int64]:
            with self.subTest(dtype=dtype):
                adjacency = sparse.csr_matrix([[0, 1], [1, 0]], dtype=dtype)
                result = directed2undirected(adjacency)
                np.testing.assert_array_equal(result.toarray(),
                                              [[0, 2], [2, 0]])

    def setUp(self):
        """Basic biadjacency for tests."""
        self.biadjacency = test_bigraph()

    def test_directed2undirected(self):
        adjacency = test_digraph()
        ref = directed2undirected(adjacency)
        self.assertEqual(ref.shape, adjacency.shape)
        self.assertTrue(is_symmetric(ref))

        adjacency = test_graph().astype(bool)
        n = adjacency.shape[0]
        diff = directed2undirected(adjacency, weighted=False) - adjacency
        self.assertEqual(diff.nnz, 0)

        slr = SparseLR(adjacency, [(np.zeros(n), np.zeros(n))])
        self.assertRaises(ValueError, directed2undirected, slr, weighted=False)
        slr = 0.5 * directed2undirected(slr)
        self.assertEqual(slr.shape, (n, n))

        x = np.random.randn(n)
        error = np.linalg.norm(slr.dot(x) - adjacency.dot(x))
        self.assertAlmostEqual(error, 0)

    def test_bipartite2directed(self):
        n_row, n_col = self.biadjacency.shape
        n = n_row + n_col

        directed_graph = bipartite2directed(self.biadjacency)
        self.assertEqual(directed_graph.shape, (n, n))

        slr = SparseLR(self.biadjacency, [(np.ones(n_row), np.ones(n_col))])
        directed_graph = bipartite2directed(slr)
        self.assertTrue(type(directed_graph) == SparseLR)

    def test_bipartite2undirected(self):
        n_row, n_col = self.biadjacency.shape
        n = n_row + n_col

        undirected_graph = bipartite2undirected(self.biadjacency)
        self.assertEqual(undirected_graph.shape, (n, n))
        self.assertTrue(is_symmetric(undirected_graph))

        slr = SparseLR(self.biadjacency, [(np.ones(n_row), np.ones(n_col))])
        undirected_graph = bipartite2undirected(slr)
        self.assertTrue(type(undirected_graph) == SparseLR)

    def test_check(self):
        with self.assertRaises(ValueError):
            check_format(sparse.csr_matrix((3, 4)), allow_empty=False)
        adjacency = check_format(np.array([[0, 2], [2, 3]]))
        self.assertTrue(adjacency.shape == (2, 2))
