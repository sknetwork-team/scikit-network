#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""


def check_norm(norm: str):
    """Check if normalization is known."""
    if norm not in ['both']:
        raise ValueError('Unknown norm parameter.')
