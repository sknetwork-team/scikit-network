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
    def get_params(self):
        """Get parameters as dictionary.

        Returns
        -------
        params : dict
            Parameters of the algorithm.
        """
        signature = inspect.signature(self.__class__.__init__)
        params_exclude = ['self', 'random_state', 'verbose']
        params = dict()
        for param in signature.parameters.values():
            name = param.name
            if name not in params_exclude:
                try:
                    value = self.__dict__[name]
                except KeyError:
                    continue
                params[name] = value
        return params

    def set_params(self, params: dict) -> 'Algorithm':
        """Set parameters of the algorithm.

        Parameters
        ----------
        params : dict
            Parameters of the algorithm.

        Returns
        -------
        self : :class:`Algorithm`
        """
        valid_params = self.get_params()
        if type(params) is not dict:
            raise ValueError('The parameters must be given as a dictionary.')
        for name, value in params.items():
            if name not in valid_params:
                raise ValueError(f'Invalid parameter: {name}.')
            setattr(self, name, value)
        return self

    def __repr__(self):
        params_string = []
        for name, value in self.get_params().items():
            if type(value) == str:
                value = "'" + value + "'"
            else:
                value = str(value)
            params_string.append(name + '=' + value)
        return self.__class__.__name__ + '(' + ', '.join(params_string) + ')'

    def fit(self, *args, **kwargs):
        """Fit algorithm to data."""
        raise NotImplementedError
