#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for utils.py"""
import unittest

from sknetwork.flow import get_residual_graph, flow_is_feasible, find_excess
from scipy.sparse import csr_matrix

class TestUtils(unittest.TestCase):
    def test_get_residual_graph(self):
        adjacency = csr_matrix([[0, 2, 3, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 1, 3, 0], [0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0]])
        flow = csr_matrix([[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
        residual = csr_matrix([[0, 1, 3, 0, 0, 0, 0], [1, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0], [0, 1, 0, 0, 0, 3, 0], [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0]])
        res_out = get_residual_graph(adjacency, flow, 0, 6)
        rows, cols = residual.nonzero()
        for i in range(len(rows)):
            row, col = rows[i], cols[i]
            self.assertEquals(residual[row, col], res_out[row, col])
    def test_flow_is_feasible_1(self):
        adjacency = csr_matrix([[0, 2, 3, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 1, 3, 0], [0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0]])
        flow = csr_matrix([[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
        self.assertFalse(flow_is_feasible(adjacency, flow, 0, 6))
    def test_flow_is_feasible_2(self):
        adjacency = csr_matrix([[0, 2, 3, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 1, 3, 0], [0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0]])
        flow = csr_matrix([[0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
        self.assertFalse(flow_is_feasible(adjacency, flow, 0, 6))
    def test_flow_is_feasible_3(self):
        adjacency = csr_matrix([[0, 2, 3, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 1, 3, 0], [0, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0]])
        flow = csr_matrix([[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
        self.assertTrue(flow_is_feasible(adjacency, flow, 0, 6))
    def test_find_excess(self):
        flow = csr_matrix([[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
        self.assertEquals(0, find_excess(flow, 1))
