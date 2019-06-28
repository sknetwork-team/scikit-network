#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 28, 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""
import inspect


class Algorithm:
    """
    Base class for all algorithms
    """

    def __repr__(self):
        output = self.__class__.__name__ + '('
        signature = inspect.signature(self.__class__.__init__)
        arguments = [arg.name for arg in signature.parameters.values() if arg.name != 'self']
        for p in arguments:
            try:
                val = self.__dict__[p]
            except KeyError:
                continue
            if type(val) == str:
                val = "'" + val + "'"
            else:
                val = str(self.__dict__[p])
            output += p + '=' + val + ', '
        return output[:-2] + ')'
