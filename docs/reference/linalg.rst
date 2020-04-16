.. _linalg:

Linear algebra
**************

Tools of linear algebra.

Sparse + Low Rank structure
---------------------------

.. autoclass:: sknetwork.linalg.SparseLR

Solvers
-------

.. autoclass:: sknetwork.linalg.LanczosEig

.. autoclass:: sknetwork.linalg.HalkoEig

.. autoclass:: sknetwork.linalg.LanczosSVD

.. autoclass:: sknetwork.linalg.HalkoSVD

Randomized methods
------------------

.. autofunction:: sknetwork.linalg.randomized_methods.randomized_range_finder

.. autofunction:: sknetwork.linalg.randomized_svd

.. autofunction:: sknetwork.linalg.randomized_eig

Miscellaneous
-------------

.. autofunction:: sknetwork.linalg.diag_pinv

.. autofunction:: sknetwork.linalg.normalize
