#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 3, 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""


class Log:
    """Log class for easier verbosity features"""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.log = ''

    def print(self, *args):
        """Fill log with text."""
        if self.verbose:
            print(*args)
        self.log += ' '.join(map(str, args)) + '\n'

    def __repr__(self):
        return self.log


class VerboseMixin:
    """Mixin class for verbosity"""
    def __init__(self, verbose: bool = False):
        self.log = Log(verbose)

    def _scipy_solver_info(self, info: int):
        """Fill log with scipy info."""
        if info == 0:
            self.log.print('Successful exit.')
        elif info > 0:
            self.log.print('Convergence to tolerance not achieved.')
        else:
            self.log.print('Illegal input or breakdown.')
