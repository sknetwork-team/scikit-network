#!/usr/bin/env python3
# coding: utf-8
"""
Created on Jun 2020
@author: Victor Manach <victor.manach@telecom-paris.fr>
@author: Rémi Jaylet <remi.jaylet@telecom-paris.fr>
"""

from sknetwork.embedding.base import BaseEmbedding

class Force_atlas(BaseEmbedding):

    def __init__(self, strength: float = None, n_iter: int = 50, tol: float = 1e-4):
        super(Force_atlas, self).__init__()
        self.strength = strength
        self.n_iter = n_iter
        self.tol = tol

    
