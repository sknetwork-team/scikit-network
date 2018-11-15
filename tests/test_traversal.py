import unittest

from sknetwork.model.graph import Graph
from sknetwork.loader.loaders import load_from_mtx
from sknetwork.traversal.breath_first_search import BreathFirstSearchAlgo
from sknetwork.traversal.depth_first_search import DepthFirstSearchAlgo

graph = { 0 : [1,4,5],
	          1 : [0,2,4],
	          2 : [1,3],
	          3 : [2,4],
	          4 : [0,1,3],
	          5 : [0]
	        }
g = Graph(graph)

res = [1, (1, 0), (1, 2), (1, 4), 0, (0, 4), (0, 5), 2, (2, 3), 4, (4, 3), 5, 3]

class MyTestCase(unittest.TestCase):
    def test_bfsiterator(self):
        bfs = BreathFirstSearchAlgo(g)
        for (j,i) in enumerate(bfs.iterator(1, process_vertex_early=True, process_edge=True)):
            self.assertEqual(i, res[j])


if __name__ == '__main__':
    unittest.main()
