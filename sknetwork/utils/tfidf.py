#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in February 2023
@author: Thomas Bonald <thomas.bonald&telecom-paris.fr>
"""
import numpy as np
from scipy import sparse

from sknetwork.linalg import normalize
from sknetwork.utils import get_degrees


def get_tfidf(count_matrix: sparse.csr_matrix):
    """Get the tf-idf from a count matrix in sparse format.

    Parameters
    ----------
    count_matrix : sparse.csr_matrix
        Count matrix, shape (n_documents, n_words).

    Returns
    -------
    tf_idf : sparse.csr_matrix
        tf-idf matrix, shape (n_documents, n_words).

    References
    ----------
    https://en.wikipedia.org/wiki/Tfidf
    """
    n_documents, n_words = count_matrix.shape
    tf = normalize(count_matrix)
    freq = get_degrees(count_matrix > 0, transpose=True)
    idf = np.zeros(n_words)
    idf[freq > 0] = np.log(n_documents / freq[freq > 0])
    tf_idf = tf.dot(sparse.diags(idf))
    return tf_idf
