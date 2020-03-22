#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for bunch"""

import unittest

from sknetwork.utils import Bunch


class TestBunch(unittest.TestCase):

    def test_key_error(self):
        bunch = Bunch(a=1, b=2)
        with self.assertRaises(AttributeError):
            # noinspection PyStatementEffect
            bunch.c
