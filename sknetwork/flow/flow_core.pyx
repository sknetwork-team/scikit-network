# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# cython: infer_types=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
Created on July 7, 2022.
@author: Henry L. Carscadden <hcarscad@gmail.com>
"""
cimport cython
from scipy import sparse

@cython.boundscheck(False)
@cython.wraparound(False)
def push(residual: sparse.csr_matrix, int curr_node, int src, int sink,
          int [:] heights, int [:] excess_flow):
          """
          This function completes the push phase of the push relabel algorithm.
          """
          cdef int residual_amt
          cdef long dest_node
          cdef int push_amt

          for dest_node in residual.indices[residual.indptr[curr_node]:residual.indptr[curr_node + 1]]:
              # Push step
              if heights[curr_node] > heights[dest_node]:
                  # Find the residual edge.
                  residual_amt = residual[curr_node, dest_node]
                  # Find the amount to push.
                  push_amt = min(residual_amt, excess_flow[curr_node])
                  # perform push
                  residual[curr_node, dest_node] -= push_amt
                  residual[dest_node, curr_node] += push_amt
                  # Update excesses
                  excess_flow[curr_node] -= push_amt
                  if dest_node != sink and dest_node != src:
                      excess_flow[dest_node] += push_amt
                  if push_amt > 0:
                      return True
          return False
