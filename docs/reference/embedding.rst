.. _embedding:

Embedding
*********

.. currentmodule:: sknetwork


Spectral
--------
.. automodule:: sknetwork.embedding

.. autoclass:: sknetwork.embedding.spectral.SpectralEmbedding
    :members:

SVD
---

.. autoclass:: sknetwork.embedding.gsvd.GSVDEmbedding
    :members:

Metrics
-------
.. autofunction:: sknetwork.embedding.metrics.dot_modularity

.. autofunction:: sknetwork.embedding.metrics.hscore


Randomized methods
------------------
.. autofunction:: sknetwork.embedding.randomized_matrix_factorization.randomized_range_finder

.. autofunction:: sknetwork.embedding.randomized_matrix_factorization.randomized_svd

.. autofunction:: sknetwork.embedding.randomized_matrix_factorization.randomized_eig
