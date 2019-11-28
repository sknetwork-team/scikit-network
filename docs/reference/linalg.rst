.. _linalg:

Linear algebra
**************

.. currentmodule:: sknetwork

Module for linear Algebra.

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

.. autoclass:: sknetwork.linalg.LanczosSVD
    :members:

.. autoclass:: sknetwork.linalg.HalkoSVD
    :members:

Randomized methods
------------------

.. autofunction:: sknetwork.linalg.randomized_matrix_factorization.randomized_range_finder

.. autofunction:: sknetwork.linalg.randomized_svd

.. autofunction:: sknetwork.linalg.randomized_eig

Miscellaneous
-------------

.. autofunction:: sknetwork.linalg.diag_pinv
