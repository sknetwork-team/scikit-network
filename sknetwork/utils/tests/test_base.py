#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for base.py"""
import unittest

from abc import ABC

from sknetwork.utils.base import Algorithm


class TestBase(unittest.TestCase):

    def test_repr(self):
        class Stub(Algorithm):
            """Docstring"""
            def __init__(self, param: int, name: str):
                self.param = param
                self.name = name

            def fit(self):
                """Docstring"""
                pass

        stub = Stub(1, 'abc')
        self.assertEqual(repr(stub), "Stub(param=1, name='abc')")

    def test_fit(self):
        stub = Algorithm()
        self.assertRaises(NotImplementedError, stub.fit, None)
