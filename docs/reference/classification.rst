.. _classification:

Classification
**************

Node classification algorithms.

The attribute ``labels_``  assigns a label to each node of the graph.

PageRank
--------
.. autoclass:: sknetwork.classification.PageRankClassifier

Diffusion
---------
.. autoclass:: sknetwork.classification.DiffusionClassifier

Propagation
-----------
.. autoclass:: sknetwork.classification.Propagation

Nearest neighbors
-----------------
.. autoclass:: sknetwork.classification.KNN

Metrics
-------
.. autofunction:: sknetwork.classification.get_accuracy_score

.. autofunction:: sknetwork.classification.get_f1_score

.. autofunction:: sknetwork.classification.get_f1_scores

.. autofunction:: sknetwork.classification.get_average_f1_score

.. autofunction:: sknetwork.classification.get_confusion_matrix

