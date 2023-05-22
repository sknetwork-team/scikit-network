#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for base.py"""
import unittest

from sknetwork.base import Algorithm


class TestBase(unittest.TestCase):

    def setUp(self):
        class NewAlgo(Algorithm):
            """Docstring"""
            def __init__(self, param: int, name: str):
                self.param = param
                self.name = name

            def fit(self):
                """Docstring"""
                pass
        self.algo = NewAlgo(1, 'abc')

    def test_repr(self):
        self.assertEqual(repr(self.algo), "NewAlgo(param=1, name='abc')")

    def test_get_params(self):
        self.assertEqual(len(self.algo.get_params()), 2)

    def test_set_params(self):
        self.algo.set_params({'param': 3})
        self.assertEqual(self.algo.param, 3)

    def test_fit(self):
        stub = Algorithm()
        self.assertRaises(NotImplementedError, stub.fit, None)
