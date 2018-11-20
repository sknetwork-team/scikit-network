from sknetwork.model.graph import Graph

import os
import os.path
import zipfile

from six.moves import urllib

urls = {'polblogs.mtx': 'http://nrvis.com/download/data/dimacs10/polblogs.zip'}

class Dataset():
    def __init__(self, file_name, directed = True, download = True, root='./data/'):
        self.graph = Graph()
        self.graph.clear()
        self.graph._directed = directed
        self.root = root

        if file_name not in urls.keys():
            raise RuntimeError('Unknown dataset')

        self.make_dir()

        file_path = os.path.join(root, file_name)

        if download:
            self.download(file_name, file_path)

        if file_name[-3:] == 'mtx':
            #print(file_path)
            self.load_from_mtx(file_path)

    def make_dir(self):
        #root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def download(self, file_name, file_path):
        if not os.path.exists(file_path):
            if urls[file_name][-4:] == '.zip':
                urllib.request.urlretrieve(urls[file_name], file_path[:-3]+'zip')
                try:
                    zip_ref = zipfile.ZipFile(file_path[:-3]+'zip', 'r')
                    zip_ref.extract(file_name, self.root)
                    zip_ref.close()
                except ValueError:
                    print('Pb with the zip file')
                return file_path
            else:
                print('Not implemented')

    def load_from_mtx(self, path, comments="%", delimiter=None):
        """
        load a graph in mtx format see http://networkrepository.com/mtx-matrix-market-format.html

        Inputs
        ------
        path : str
            where the file is stored

        Returns
        -------
        a graph

        Example
        -------
        >>> polblogs = Dataset('polblogs.mtx')
        loading graph with 1490 nodes and 19025 edges.
        number of nodes: 1224 (expected 1490); number of edges: 19025 (expected 19025)

        Note
        ----
        Useful repo : https://sparse.tamu.edu/SNAP

        """

        first_line = True
        with open(path, 'rt') as f:
            for line in f:
                p=line.find(comments)
                if p>=0:
                    line = line[:p]
                if not len(line):
                    continue
                # split line, should have 2 or more
                s=line.strip().split(delimiter)
                if len(s)<2:
                    continue
                if first_line:
                    try:
                        n = s.pop(0)
                        m = s.pop(0)
                        e = s.pop(0)
                        if n != m:
                            print("check the number of nodes")
                        else:
                            print("loading graph with %s nodes and %s edges." %(n,e))
                            first_line = False
                    except:
                        raise ValueError("First line should contain num_nodes num_nodes num_edges.")
                    continue
                u=s.pop(0)
                v=s.pop(0)
                d=s
                self.graph.add_edge(int(u), int(v), direct=True)

        self.graph.basics()
        print("number of nodes: %d (expected %s); number of edges: %d (expected %s)" % (self.graph.n_vertices, n, self.graph.n_edges, e))

