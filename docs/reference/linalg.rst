.. _linalg:

Linear algebra
**************

Tools of linear algebra.

Polynomes
---------

.. autoclass:: sknetwork.linalg.Polynome

Sparse + Low Rank
-----------------

.. autoclass:: sknetwork.linalg.SparseLR

Operators
---------

.. autoclass:: sknetwork.linalg.Regularizer

.. autoclass:: sknetwork.linalg.RegularizedLaplacian

.. autoclass:: sknetwork.linalg.NormalizedAdjacencyOperator

.. autoclass:: sknetwork.linalg.CoNeighborOperator

Solvers
-------

.. autoclass:: sknetwork.linalg.LanczosEig

.. autoclass:: sknetwork.linalg.HalkoEig

.. _lanczossvd:
.. autoclass:: sknetwork.linalg.LanczosSVD

.. _halkosvd:
.. autoclass:: sknetwork.linalg.HalkoSVD

.. autofunction:: sknetwork.linalg.ppr_solver.get_pagerank

Randomized methods
------------------

.. autofunction:: sknetwork.linalg.randomized_methods.randomized_range_finder

.. autofunction:: sknetwork.linalg.randomized_svd

.. autofunction:: sknetwork.linalg.randomized_eig

Miscellaneous
-------------

.. autofunction:: sknetwork.linalg.diag_pinv

.. autofunction:: sknetwork.linalg.normalize
