#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vincent tang <vin.tang@gmail.com>
"""
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path


def _get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]


def betweenness_centrality(entry: Union[sparse.csr_matrix, np.ndarray]) -> dict:
    """Computes betweenness for adjacency matrix."""
    
    D, Pr = shortest_path(entry, directed=True, method='FW', return_predecessors=True)
    n_vertex = len(entry)    

    # find shortest path for each pair source/target vertices
    betweenness = {}
    for s in range(n_vertex):
        for t in range(n_vertex):
            path = _get_path(Pr, s, t)
            
            # we're interested only if there is at least one vertex between source and target vertices
            if len(path) > 2:  
                for v in path[1:-1]:
                    if v not in betweenness.keys():
                        betweenness[v] = 1
                    else:
                        betweenness[v] += 1
    return betweenness