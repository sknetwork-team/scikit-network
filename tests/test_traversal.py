import unittest
import sys

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

resbfs = [1, (1, 0), (1, 2), (1, 4), 0, (0, 4), (0, 5), 2, (2, 3), 4, (4, 3), 5, 3]
resdfs = [1, (1, 0), 0, (0, 4),	4, (4, 1), (4, 3), 3, (3, 2), 2, (2, 1), (0, 5), 5]
comp = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2}

class MyTestCase(unittest.TestCase):
    def test_bfsiterator(self):
        bfs = BreathFirstSearchAlgo(g)
        for (j,i) in enumerate(bfs.iterator(1, process_vertex_early=True, process_edge=True)):
            self.assertEqual(i, resbfs[j])

    def test_find_path_bfs(self):
        bfs = BreathFirstSearchAlgo(g)
        self.assertEqual(bfs.find_path_BFS(3,5),[3, 4, 0, 5])

    def test_connectedcomponents(self):
        g.add_vertex(6, verbose=False)
        bfs = BreathFirstSearchAlgo(g)
        self.assertEqual(bfs.connected_components(), (2, [0,6]))
        for i in range(7):
            self.assertEqual(bfs.component[i],comp[i])

    def test_dfsiterator(self):
        dfs = DepthFirstSearchAlgo(g)
        for (j,i) in enumerate(dfs.iterator(1, process_vertex_early=True, process_edge=True)):
            self.assertEqual(i, resdfs[j])


if __name__ == '__main__':
    unittest.main()
