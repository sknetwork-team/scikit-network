#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for simplex.py"""

import unittest

import numpy as np

from sknetwork.utils.check import is_proba_array
from sknetwork.utils.simplex import projection_simplex


class TestProjSimplex(unittest.TestCase):

    def test_proj_simplex(self):
        x = np.random.rand(5)
        proj = projection_simplex(x)
        self.assertTrue(is_proba_array(proj))

        x = np.random.rand(4, 3)
        proj = projection_simplex(x)
        self.assertTrue(is_proba_array(proj))
