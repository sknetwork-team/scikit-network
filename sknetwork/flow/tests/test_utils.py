#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for utils.py"""
import unittest

from sknetwork.flow import get_residual_graph
from scipy.sparse import csr_matrix

class TestResidual(unittest.TestCase):
    def test_residual(self):
        adjacency = csr_matrix([[0, 2, 3, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 1, 3, 0], [0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0]])
        flow = csr_matrix([[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
        residual = csr_matrix([[0, 1, 3, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 1, 3, 0], [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0]])
        res_out = residual
        print(res_out)
        rows, cols = residual.nonzero()
        for row in rows:
            for col in cols:
                self.assertEquals(residual[row, col], res_out[row, col])
