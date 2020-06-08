#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for simplex.py"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.utils.check import is_proba_array
from sknetwork.utils.simplex import projection_simplex


class TestProjSimplex(unittest.TestCase):

    def test_array(self):
        x = np.random.rand(5)
        proj = projection_simplex(x)
        self.assertTrue(is_proba_array(proj))

        x = np.random.rand(4, 3)
        proj = projection_simplex(x)
        self.assertTrue(is_proba_array(proj))

    def test_csr(self):
        x = sparse.csr_matrix(np.ones((3, 3)))
        proj1 = projection_simplex(x)
        proj2 = projection_simplex(x.astype(bool))
        self.assertEqual(0, (proj1-proj2).nnz)

    def test_other(self):
        with self.assertRaises(TypeError):
            projection_simplex('toto')
