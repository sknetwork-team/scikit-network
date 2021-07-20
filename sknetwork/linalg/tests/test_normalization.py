#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.linalg import normalize, CoNeighbor


class TestNormalization(unittest.TestCase):

    def test_formats(self):
        n = 5
        mat1 = normalize(np.eye(n))
        mat2 = normalize(sparse.eye(n))
        mat3 = normalize(CoNeighbor(mat2))

        x = np.random.randn(n)
        self.assertAlmostEqual(np.linalg.norm(mat1.dot(x) - x), 0)
        self.assertAlmostEqual(np.linalg.norm(mat2.dot(x) - x), 0)
        self.assertAlmostEqual(np.linalg.norm(mat3.dot(x) - x), 0)

        mat1 = np.random.rand(n**2).reshape((n, n))
        mat2 = sparse.csr_matrix(mat1)
        mat1 = normalize(mat1, p=2)
        mat2 = normalize(mat2, p=2)
        self.assertAlmostEqual(np.linalg.norm(mat1.dot(x) - mat2.dot(x)), 0)

        with self.assertRaises(NotImplementedError):
            normalize(mat3, p=2)
        with self.assertRaises(NotImplementedError):
            normalize(mat1, p=3)
