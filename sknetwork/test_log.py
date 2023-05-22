#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for verbose.py"""

import unittest

from sknetwork.log import Log


class TestVerbose(unittest.TestCase):

    def test_prints(self):
        logger = Log(verbose=True)
        logger.print_log('Hello', 42)
        self.assertEqual(str(logger.log), 'Hello 42\n')
