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

    def _predict_base(self, source: int, targets: Iterable):
        """Prediction for a single node and multiple targets"""
        raise NotImplementedError

    def _predict_node(self, node: int):
        """Prediction for a single node."""
        raise NotImplementedError

    def _predict_nodes(self, nodes: np.ndarray):
        """Prediction for multiple nodes."""
        preds = []
        for node in nodes:
            preds.append(self._predict_node(node))
        return np.array(preds)

    def _predict_edge(self, source: int, target: int):
        """Prediction for a single edge."""
        return self._predict_base(source, [target])[0]

    def _predict_edges(self, edges: np.ndarray):
        """Prediction for a list of edges."""
        preds = []
        for edge in edges:
            i, j = edge[0], edge[1]
            preds.append(self._predict_edge(i, j))
        return np.array(preds)

    def predict(self, query: Union[int, Iterable, Tuple]):
        """Compute similarity scores.

        Parameters
        ----------
        query : int, list, array or Tuple
            * If int i, return the similarities s(i, j) for all j.
            * If list or array integers, return s(i, j) for i in query, for all j as array.
            * If tuple (i, j), return the similarity s(i, j).
            * If list of tuples or array of shape (n_queries, 2), return s(i, j) for (i, j) in query as array.

        Returns
        -------
        predictions : int, float or array
            The prediction scores.
        """
        if np.issubdtype(type(query), np.integer):
            return self._predict_node(query)
        if isinstance(query, Tuple):
            return self._predict_edge(query[0], query[1])
        if isinstance(query, list):
            query = np.array(query)
        if isinstance(query, np.ndarray):
            if query.ndim == 1:
                return self._predict_nodes(query)
            elif query.ndim == 2 and query.shape[1] == 2:
                return self._predict_edges(query)
            else:
                raise ValueError("Query not understood.")
        else:
            raise ValueError("Query not understood.")

    def fit_predict(self, adjacency, query):
        """Fit algorithm to data and compute scores for requested edges."""
        self.fit(adjacency)
        return self.predict(query)
