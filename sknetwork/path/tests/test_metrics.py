#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for metrics.py"""
import unittest

from sknetwork.data import house
from sknetwork.path import diameter, eccentricity, radius


class TestMetrics(unittest.TestCase):

    def test_diameter_1(self):
        adjacency = house()
        with self.assertRaises(ValueError):
            diameter(adjacency, 2.5)
    def test_diameter_2(self):
        adjacency = house()
        self.assertEqual(diameter(adjacency), 2)
    def test_eccentricity_1(self):
        adjacency = house()
        self.assertEqual(eccentricity(adjacency, 1), 2)
    def test_radius_1(self):
        adjacency = house()
        self.assertEqual(radius(adjacency), 2)
