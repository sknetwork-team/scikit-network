#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in April 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.linalg import get_norms, normalize


class TestNormalization(unittest.TestCase):

    def test_formats(self):
        n = 5
        mat1 = normalize(np.eye(n))
        mat2 = normalize(sparse.eye(n))

        x = np.random.randn(n)
        self.assertAlmostEqual(np.linalg.norm(mat1.dot(x) - x), 0)
        self.assertAlmostEqual(np.linalg.norm(mat2.dot(x) - x), 0)

        mat1 = np.random.rand(n**2).reshape((n, n))
        mat2 = sparse.csr_matrix(mat1)
        mat1 = normalize(mat1, p=2)
        mat2 = normalize(mat2, p=2)
        self.assertAlmostEqual(np.linalg.norm(mat1.dot(x) - mat2.dot(x)), 0)

        with self.assertRaises(ValueError):
            normalize(mat1, p=3)

    def test_integer_norms(self):
        examples = [(np.int8, [-128, 3]), (np.int32, [50000, 120000]),
                    (np.uint16, [45000, 60000])]
        for dtype, values in examples:
            matrix = np.array([values, [0, 0]], dtype=dtype)
            expected = matrix.astype(float)
            for sparse_input in [False, True]:
                for p in [1, 2]:
                    with self.subTest(dtype=dtype, sparse=sparse_input, p=p):
                        if sparse_input:
                            adjacency = sparse.csr_matrix(matrix)
                        else:
                            adjacency = matrix.copy()
                        original = adjacency.copy()
                        norms = np.linalg.norm(expected, ord=p, axis=1)
                        np.testing.assert_allclose(
                            get_norms(adjacency, p), norms)
                        normalized = normalize(adjacency, p)
                        if sparse.issparse(normalized):
                            normalized = normalized.toarray()
                        np.testing.assert_allclose(
                            normalized[0], expected[0] / norms[0])
                        np.testing.assert_array_equal(normalized[1], [0, 0])
                        if sparse_input:
                            np.testing.assert_array_equal(
                                adjacency.data, original.data)
                            np.testing.assert_array_equal(
                                adjacency.indices, original.indices)
                            np.testing.assert_array_equal(
                                adjacency.indptr, original.indptr)
                        else:
                            np.testing.assert_array_equal(adjacency, original)
                        self.assertEqual(adjacency.dtype, original.dtype)
