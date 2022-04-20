#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for metrics.py"""
import unittest

from sknetwork.data import house
from sknetwork.path import get_diameter


class TestMetrics(unittest.TestCase):

    def test_diameter(self):
        adjacency = house()
        with self.assertRaises(ValueError):
            get_diameter(adjacency, 2.5)
