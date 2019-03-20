.. _embedding:

Embedding
*********

.. currentmodule:: sknetwork


Spectral embeddings
-------------------
.. automodule:: sknetwork.embedding

.. autoclass:: sknetwork.embedding.gsvd.GSVDEmbedding
    :members:

.. autoclass:: sknetwork.embedding.spectral.SpectralEmbedding
    :members:

Metrics
-------
.. autofunction:: sknetwork.embedding.metrics.dot_modularity

.. autofunction:: sknetwork.embedding.metrics.hscore


Randomized Methods
------------------
.. autofunction:: sknetwork.embedding.randomized_matrix_factorization.randomized_range_finder

.. autofunction:: sknetwork.embedding.randomized_matrix_factorization.randomized_svd

.. autofunction:: sknetwork.embedding.randomized_matrix_factorization.randomized_eig
