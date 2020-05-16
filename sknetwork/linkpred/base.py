#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC
from typing import Union, Iterable, Tuple

import numpy as np

from sknetwork.utils.base import Algorithm


class BaseLinkPred(Algorithm, ABC):
    """Base class for link prediction algorithms."""

    def predict_node(self, node: int):
        """Prediction for a single node."""
        raise NotImplementedError

    def predict_nodes(self, nodes: np.ndarray):
        """Prediction for multiple nodes."""
        preds = []
        for node in nodes:
            preds.append(self.predict_node(node))
        return np.array(preds)

    def predict_edge(self, source: int, target: int):
        """Prediction for a single edge."""
        raise NotImplementedError

    def predict_edges(self, edges: np.ndarray):
        """Prediction for a list of edges."""
        preds = []
        for edge in edges:
            i, j = edge[0], edge[1]
            preds.append(self.predict_edge(i, j))
        return preds

    def predict(self, query: Union[int, Iterable, Tuple]):
        """Compute similarity scores"""
        if np.issubdtype(type(query), np.integer):
            return self.predict_node(query)
        if isinstance(query, Tuple):
            return self.predict_edge(query[0], query[1])
        if isinstance(query, list):
            query = np.array(query)
        if isinstance(query, np.ndarray):
            if query.ndim == 1:
                return self.predict_nodes(query)
            elif query.ndim == 2:
                return self.predict_edges(query)
            else:
                raise ValueError("Query not understood.")
        else:
            raise ValueError("Query not understood.")

    def fit_predict(self, adjacency, query):
        """Fit algorithm to data and compute scores for requested edges."""
        self.fit(adjacency)
        self.predict(query)
