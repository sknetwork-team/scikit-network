#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in June 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""
import inspect


class Algorithm:
    """Base class for all algorithms.
    """
    def __repr__(self):
        params_exclude = ['self', 'random_state', 'verbose']
        output = self.__class__.__name__ + '('
        signature = inspect.signature(self.__class__.__init__)
        params = [param.name for param in signature.parameters.values() if param.name not in params_exclude]
        for param in params:
            try:
                value = self.__dict__[param]
            except KeyError:
                continue
            if type(value) == str:
                value = "'" + value + "'"
            else:
                value = str(value)
            output += param + '=' + value + ', '
        if output[-1] != '(':
            return output[:-2] + ')'
        else:
            return output + ')'

    def fit(self, *args, **kwargs):
        """Fit algorithm to data."""
        raise NotImplementedError
