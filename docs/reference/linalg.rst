.. _linalg:

Linalg
******

.. currentmodule:: sknetwork

Submodule for linear Algebra.

Sparse + Low Rank structure
---------------------------

.. autoclass:: sknetwork.linalg.SparseLR
    :members:

Solvers
-------

.. autoclass:: sknetwork.linalg.LanczosEig
    :members:

.. autoclass:: sknetwork.linalg.HalkoEig
    :members:

Randomized methods
------------------

.. autofunction:: sknetwork.linalg.randomized_matrix_factorization.randomized_range_finder

.. autofunction:: sknetwork.linalg.randomized_svd

.. autofunction:: sknetwork.linalg.randomized_eig
