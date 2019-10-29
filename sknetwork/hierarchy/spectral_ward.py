#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding import Spectral
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format, is_symmetric
from sknetwork.utils.ward import Ward


class SpectralWard(Algorithm):
    """Pipeline for spectral Ward hierarchical clustering.

    Parameters
    ----------
    embedding_dimension:
        Dimension of the embedding on which to apply the hierarchical clustering.
    l2normalization:
        If ``True``, each row of the embedding is projected onto the L2-sphere before hierarchical clustering.

    Attributes
    ----------
    dendrogram_:
        Dendrogram.

    """

    def __init__(self, embedding_dimension: int = 16, l2normalization: bool = True):
        self.embedding_dimension = embedding_dimension
        self.l2normalization = l2normalization

        self.dendrogram_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'SpectralWard':
        """Apply embedding method followed by hierarchical clustering to the graph.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`SpectralWard`

        """
        adjacency = check_format(adjacency)
        if not is_symmetric(adjacency):
            raise ValueError('The adjacency is not symmetric.')

        spectral = Spectral(self.embedding_dimension).fit(adjacency)
        embedding = spectral.embedding_

        if self.l2normalization:
            norm = np.linalg.norm(embedding, axis=1)
            norm[norm == 0.] = 1
            embedding /= norm[:, np.newaxis]

        ward = Ward()
        ward.fit(embedding)

        self.dendrogram_ = ward.dendrogram_

        return self
