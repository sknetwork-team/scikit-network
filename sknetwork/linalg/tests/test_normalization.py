#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in April 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.linalg import normalize


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
