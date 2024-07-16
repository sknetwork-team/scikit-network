#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2023
@author: Thomas Bonald <bonald@enst.fr>
"""


class Dataset(dict):
    """Container object for datasets.
    Dictionary-like object that exposes its keys as attributes.
    >>> dataset = Dataset(name='dataset')
    >>> dataset['name']
    'dataset'
    >>> dataset.name
    'dataset'
    """
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


# alias for Dataset
Bunch = Dataset

