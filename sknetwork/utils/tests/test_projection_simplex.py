#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for projection_simplex.py"""

import unittest
import numpy as np
from sknetwork.utils.checks import is_proba_array
from sknetwork.utils.projection_simplex import projection_simplex


class TestProjSimplex(unittest.TestCase):

    def setUp(self):
        self.oned_array = np.random.rand(5)
        self.twod_array = np.random.rand(4, 3)

    def test_projsimplex(self):
        oned_proj = projection_simplex(self.oned_array)
        self.assertTrue(is_proba_array(oned_proj))
        twod_proj = projection_simplex(self.twod_array)
        self.assertTrue(is_proba_array(twod_proj))
