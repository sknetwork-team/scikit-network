.. _classification:

Classification
**************

Node classification algorithms.

The attribute ``labels_``  gives the label of each node of the graph.

Diffusion
---------
.. autoclass:: sknetwork.classification.DiffusionClassifier

Nearest neighbors
-----------------
.. autoclass:: sknetwork.classification.NNClassifier

Propagation
-----------
.. autoclass:: sknetwork.classification.Propagation

PageRank
--------
.. autoclass:: sknetwork.classification.PageRankClassifier

Metrics
-------
.. autofunction:: sknetwork.classification.get_accuracy_score

.. autofunction:: sknetwork.classification.get_f1_score

.. autofunction:: sknetwork.classification.get_f1_scores

.. autofunction:: sknetwork.classification.get_average_f1_score

.. autofunction:: sknetwork.classification.get_confusion_matrix

