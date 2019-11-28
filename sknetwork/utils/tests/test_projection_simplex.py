#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for projection_simplex.py"""

import unittest

import numpy as np

from sknetwork.utils.checks import is_proba_array
from sknetwork.utils.projection_simplex import projection_simplex


# noinspection PyMissingOrEmptyDocstring
class TestProjSimplex(unittest.TestCase):

    def setUp(self):
        self.one_d_array = np.random.rand(5)
        self.two_d_array = np.random.rand(4, 3)

    def test_proj_simplex(self):
        one_d_proj = projection_simplex(self.one_d_array)
        self.assertTrue(is_proba_array(one_d_proj))
        two_d_proj = projection_simplex(self.two_d_array)
        self.assertTrue(is_proba_array(two_d_proj))
