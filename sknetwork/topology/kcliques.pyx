# distutils: language = c++
# cython: language_level=3
"""
Created on Jun 3, 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""
cimport numpy as np
import numpy as np
from scipy import sparse

cimport cython

from sknetwork.topology.kcore import CoreDecomposition

ctypedef np.int_t int_type_t
ctypedef np.uint8_t bool_type_t

@cython.boundscheck(False)
@cython.wraparound(False)

#------ Wrapper for integer array -------

cdef class IntArray:
	
	def __cinit__(self, int n):
		self.arr.reserve(n)
		
	def __getitem__(self, int key) -> int:
		return self.arr[key]
		
	def __setitem__(self, int key, int val) -> None:
		self.arr[key] = val
		
# ----- Collections of arrays used by our listing algorithm -----

cdef class ListingBox:
	
	def __cinit__(self, int n_nodes, int[:] indptr, short k):
		self.initBox(n_nodes, indptr, k)
		
	#Building the special graph structure
	cdef void initBox(self, int n_nodes, int[:] indptr, short k):
		cdef int i
		cdef int max_deg = 0
		
		cdef IntArray deg
		cdef IntArray sub
		
		cdef np.ndarray[int, ndim=1] ns
		cdef np.ndarray[short, ndim=1] lab
		
		lab = np.full((n_nodes,), k, dtype=np.int16)
		
		deg = IntArray.__new__(IntArray, n_nodes)
		sub = IntArray.__new__(IntArray, n_nodes)
		
		for i in range(n_nodes):
			deg[i] = indptr[i+1] - indptr[i]
			max_deg = max(deg[i], max_deg)
			sub[i] = i
		
		self.ns = np.empty((k+1,), dtype=np.int32)
		self.ns[k] = n_nodes
	
		self.degres = np.empty((k+1,), dtype=object)
		self.subs = np.empty((k+1,), dtype=object)
		
		self.degres[k] = deg
		self.subs[k] = sub
		
		for i in range(2, k):
			deg = IntArray.__new__(IntArray, n_nodes)
			sub = IntArray.__new__(IntArray, max_deg)
			self.degres[i] = deg
			self.subs[i] = sub
			
		self.lab = lab

# ---------------------------------------------------------------

cdef long listing_rec(int n_nodes, int[:] indptr, int[:] indices, short l, ListingBox box):
	
	cdef long nb_cliques
	cdef int i, j, k
	cdef int u, v, w
	cdef int cd
	cdef IntArray sub_l, degre_l
	
	nb_cliques = 0
	
	#if l < 2:
	#	raise ValueError("l should not be inferior to 2")
		
	if l == 2:
		degre_l = box.degres[2]
		sub_l = box.subs[2]
		for i in range(box.ns[2]):
			j = sub_l[i]
			nb_cliques += degre_l[j]
			#cd = indptr[j] + degre_l[j]
			#for j in range(indptr[j], cd):
			#	nb_cliques += 1
			
		return nb_cliques
	
	cdef IntArray deg_prevs, sub_prev
	sub_l = box.subs[l]
	sub_prev = box.subs[l-1]
	degre_l = box.degres[l]
	deg_prev = box.degres[l-1]
	for i in range(box.ns[l]):
		u = sub_l[i]
		box.ns[l-1] = 0;
		cd = indptr[u] + degre_l[u]
		for j in range(indptr[u], cd):
			v = indices[j]
			if box.lab[v] == l:
				box.lab[v] = l-1
				#box.subs[l-1][ns[l-1]++] = v
				sub_prev[box.ns[l-1]] = v
				box.ns[l-1] += 1
				deg_prev[v] = 0
		
		for j in range(box.ns[l-1]):
			v = sub_prev[j]
			cd = indptr[v] + degre_l[v]
			k = indptr[v]
			while k < cd:
				w = indices[k]
				if box.lab[w] == l-1:
					deg_prev[v] += 1
				else:
					cd -= 1
					#indices[k--] = indices[--cd]
					indices[k] = indices[cd]
					k -= 1
					indices[cd] = w;
				
				k += 1
				
		nb_cliques += listing_rec(n_nodes, indptr, indices, l-1, box)
		for j in range(box.ns[l-1]):
			v = sub_prev[j]
			box.lab[v] = l
		
	return nb_cliques
	

cdef long fit_core(int n_nodes, int n_edges, int[:] indptr, int[:] indices, int[:] cores, short l):
	
	cdef vector[int] indexation
	cdef int i, j, k
	
	indexation.reserve(n_nodes)
	for i in range(n_nodes):
		indexation[cores[i]] = i
		
	cdef int e
	cdef int[:] row, column
	cdef np.ndarray[bool_type_t, ndim=1] data
	
	e = n_edges // 2
	row = np.empty((e,), dtype=np.int32)
	column = np.empty((e,), dtype=np.int32)
	
	e = 0
	for i in range(n_nodes):
		for k in range(indptr[i], indptr[i+1]):
			j = indices[k]
			if indexation[i] < indexation[j]:
				row[e] = indexation[i]
				column[e] = indexation[j]
				e += 1					 
	
	row = row[:e]
	column = column[:e]
	data = np.ones((e,), dtype=bool)
	dag = sparse.csr_matrix((data, (row, column)), (n_nodes, n_nodes), dtype=bool)
	#dag.has_sorted_indices = True
	#dag.sort_indices():
	return listing_rec(n_nodes, dag.indptr, dag.indices, l, ListingBox.__new__(ListingBox, n_nodes, dag.indptr, l))
	

class CliqueListing():
	"""Clique listing algorithm which creates a DAG and list all cliques on it.

	* Graphs

	Attributes
	----------
	nb_cliques : int
		Number of cliques

	Example
	-------
	>>> from sknetwork.clustering import CliqueListing
	>>> from sknetwork.data import karate_club
	>>> tri = CliqueListing()
	>>> adjacency = karate_club()
	>>> nb = tri.fit_transform(adjacency)
	4
	"""
	
	def __init__(self):
		self.nb_cliques = 0
		#self.printing = printing
		
		
	def fit(self, adjacency : sparse.csr_matrix, k : int) -> 'CliqueListing':
		""" k-cliques listing.

		Parameters
		----------
		adjacency:
			Adjacency matrix of the graph.
		k:
			k value of cliques to list

		Returns
		-------
		 self: :class:`CliqueListing`
		"""
		
		core = CoreDecomposition()
		core.fit(adjacency)
		
		self.nb_cliques = fit_core(adjacency.shape[0], adjacency.nnz, adjacency.indptr,\
			adjacency.indices, core.ordered, k)
		
		return self
		
	def fit_transform(self, *args, **kwargs) -> int:
		""" Fit algorithm to the data and return the number of cliques. Same parameters as the ``fit`` method.

		Returns
		-------
		nb_cliques : int
			Number of k-cliques.
		"""
		self.fit(*args, **kwargs)
		return self.nb_cliques
		
		
