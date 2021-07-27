#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for base.py"""
import unittest

from sknetwork.utils.base import Algorithm


class TestBase(unittest.TestCase):

    def test_repr(self):
        class Stub(Algorithm):
            """Docstring"""
            def __init__(self, some_param: int, another_param: str, random_state: int, some_unused_param: int):
                self.some_param = some_param
                self.another_param = another_param
                self.random_state = random_state

            def fit(self):
                """Docstring"""
                pass

        stub = Stub(1, 'abc', 3, 4)
        self.assertEqual(repr(stub), "Stub(some_param=1, another_param='abc')")

    def test_fit(self):
        stub = Algorithm()
        self.assertRaises(NotImplementedError, stub.fit, None)
