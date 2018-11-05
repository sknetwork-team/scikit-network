from model.graph import Graph
from utils.lazy_property import cached_property, DeleteCacheMixin
from collections import deque


class BFS_algo(DeleteCacheMixin):
	"""
	TODO use DeleteCacheMixin?
	"""

	def __init__(self, graph):
		self._graph = graph
		self.parent = {}


	def iter_edge(self, s):
		"""
		BFS algorithm from source s

		Example
		-------
		>>>graph = { 0 : [1,4,5],
	          1 : [0,2,4],
	          2 : [1,3],
	          3 : [2,4],
	          4 : [0,1,3],
	          5 : [0]
	        }
		>>>G = Graph(graph)
		>>>BFS = BFS_algo(G)
		>>>for i in BFS.iter_edge(1):
			print(i)
		(1, 0)
		(1, 2)
		(1, 4)
		(0, 4)
		(0, 5)
		(2, 3)
		(4, 3)
		"""
		q = deque([s])
		self._discovered = {s}
		processed = set()
		while q:
			v = q[0]
			processed.add(v)
			for k in self._graph._graph_dict[v]:
				if (k not in processed) or self._graph._directed:
					yield v , k
				if (k not in self._discovered):
					q.append(k)
					self._discovered.add(k)
			q.popleft()

	#@cached_property
	def tree(self,s):
		self.parent = {i:-1 for i in self._graph._graph_dict.keys()}
		for (v,k) in self.iter_edge(s):
			if k not in self._discovered:
				self.parent[k] = int(v)
		#return self.parent


