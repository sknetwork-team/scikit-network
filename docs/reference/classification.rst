.. _classification:

Classification
**************

.. currentmodule:: sknetwork

This module contains node classification algorithms.

The attribute ``labels_``  assigns a label to each node of the graph.

PageRank
--------
.. autoclass:: sknetwork.classification.PageRankClassifier

.. autoclass:: sknetwork.classification.BiPageRankClassifier

.. autoclass:: sknetwork.classification.CoPageRankClassifier

Diffusion
---------
.. autoclass:: sknetwork.classification.DiffusionClassifier

.. autoclass:: sknetwork.classification.BiDiffusionClassifier

Nearest neighbors
-----------------
.. autoclass:: sknetwork.classification.KNN

.. autoclass:: sknetwork.classification.BiKNN
