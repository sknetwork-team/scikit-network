#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for base.py"""

import unittest

from sknetwork.utils.base import Algorithm


class TestBase(unittest.TestCase):

    def test_repr(self):
        class Stub(Algorithm):
            def __init__(self, some_param: int, another_param: int, random_state: int):
                self.some_param = some_param
                self.another_param = another_param

            def fit(self):
                pass

        stub = Stub(1, 2, 3)
        self.assertEqual(repr(stub), 'Stub(some_param=1, another_param=2)')
