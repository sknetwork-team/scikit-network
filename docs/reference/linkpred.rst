.. _linkpred:

Link prediction
***************

Link prediction algorithms.

The method ``predict`` assigns a scores to edges.


First order methods
-------------------
.. autoclass:: sknetwork.linkpred.CommonNeighbors

.. autoclass:: sknetwork.linkpred.JaccardIndex

.. autoclass:: sknetwork.linkpred.SaltonIndex

.. autoclass:: sknetwork.linkpred.SorensenIndex

.. autoclass:: sknetwork.linkpred.HubPromotedIndex

.. autoclass:: sknetwork.linkpred.HubDepressedIndex

.. autoclass:: sknetwork.linkpred.AdamicAdar

.. autoclass:: sknetwork.linkpred.ResourceAllocation

.. autoclass:: sknetwork.linkpred.PreferentialAttachment


Post-processing
---------------
.. autofunction:: sknetwork.linkpred.is_edge

.. autofunction:: sknetwork.linkpred.whitened_sigmoid
