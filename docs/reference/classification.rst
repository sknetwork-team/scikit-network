.. _classification:

Classification
**************

Node classification algorithms.

The attribute ``labels_``  assigns a label to each node of the graph.

PageRank
--------
.. autoclass:: sknetwork.classification.PageRankClassifier

.. autoclass:: sknetwork.classification.BiPageRankClassifier

Diffusion
---------
.. autoclass:: sknetwork.classification.DiffusionClassifier

.. autoclass:: sknetwork.classification.BiDiffusionClassifier

Dirichlet
---------
.. autoclass:: sknetwork.classification.DirichletClassifier

.. autoclass:: sknetwork.classification.BiDirichletClassifier

Propagation
-----------
.. autoclass:: sknetwork.classification.Propagation

.. autoclass:: sknetwork.classification.BiPropagation

Nearest neighbors
-----------------
.. autoclass:: sknetwork.classification.KNN

.. autoclass:: sknetwork.classification.BiKNN

Metrics
-------
.. autofunction:: sknetwork.classification.accuracy_score
