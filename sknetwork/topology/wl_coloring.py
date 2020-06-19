#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 19, 2020
@author:
@author:
"""
from typing import Union, Optional

import numpy as np


class WLColoring():
    """Weisefeler-Lehman algorithm for coloring/labeling graphs in order to check similarity.

    * Graphs
    * Digraphs

    Parameters
    ----------
    modularity : str
        Which objective function to maximize. Can be ``'dugue'``, ``'newman'`` or ``'potts'``.
    tol_optimization :
        Minimum increase in the objective function to enter a new optimization pass.


    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.

    Example
    -------
    >>> from sknetwork.topology import WLColoring
    >>> from sknetwork.data import karate_club
    >>> wlcoloring = WLColoring()
    >>> adjacency = karate_club()
    >>> labels = wlcoloring.fit_transform(adjacency)


    References
    ----------
    >>> https://people.mpi-inf.mpg.de/~mehlhorn/ftp/genWLpaper.pdf

    """

    def __init__(self):
        self.labels_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'WLColoring':
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`WLColoring`
        """  ############################

        return self

