#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for metrics.py"""
import unittest

from sknetwork.data import house
from sknetwork.path import get_diameter, get_eccentricity, get_radius



class TestMetrics(unittest.TestCase):

    def test_diameter_1(self):
        adjacency = house()
        with self.assertRaises(ValueError):
            get_diameter(adjacency, 2.5)
    def test_diameter_2(self):
        adjacency = house()
        self.assertEqual(get_diameter(adjacency), 2)
    def test_eccentricity_1(self):
        adjacency = house()
        self.assertEqual(get_eccentricity(adjacency, 1), 2)
    def test_radius_1(self):
        adjacency = house()
        self.assertEqual(get_radius(adjacency), 2)
    def test_radius_2(self):
        adjacency = house()
        self.assertEqual(get_radius(adjacency,[0,1]), 2)

