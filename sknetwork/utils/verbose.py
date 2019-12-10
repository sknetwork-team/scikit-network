#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 3, 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""


class Log:
    """
    Log class for easier verbosity features
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.log = ''

    def print(self, *args):
        if self.verbose:
            print(*args)
        self.log += ' '.join(map(str, args)) + '\n'

    def __repr__(self):
        return self.log


class VerboseMixin:
    """
    Mixin class for verbosity
    """
    def __init__(self, verbose: bool = False):
        self.log = Log(verbose)
