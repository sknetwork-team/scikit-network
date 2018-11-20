from sknetwork.model.graph import Graph

import os
import os.path
import zipfile
import tarfile

from six.moves import urllib

urls = {'polblogs.mtx': 'http://nrvis.com/download/data/dimacs10/polblogs.zip',
        'ca-CondMat.mtx': 'https://sparse.tamu.edu/MM/SNAP/ca-CondMat.tar.gz'}

class Dataset():
    def __init__(self, file_name, directed = True, download = True, root='./data/'):
        self.graph = Graph()
        self.graph.clear()
        self.graph._directed = directed
        self.root = root

        if file_name not in urls.keys():
            raise RuntimeError('Unknown dataset')

        self.make_dir()

        self.file_path = os.path.join(root, file_name)

        if download:
            self.download(file_name)

        if file_name[-3:] == 'mtx':
            #print(file_path)
            self.load_from_mtx(self.file_path)

    def make_dir(self):
        #root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def download(self, file_name):
        if not os.path.exists(self.file_path):
            if urls[file_name][-4:] == '.zip':
                urllib.request.urlretrieve(urls[file_name], self.file_path[:-3]+'zip')
                try:
                    zip_ref = zipfile.ZipFile(self.file_path[:-3]+'zip', 'r')
                    zip_ref.extract(file_name, self.root)
                    zip_ref.close()
                except ValueError:
                    print('Pb with the zip file')
            elif urls[file_name][-6:] == 'tar.gz':
                urllib.request.urlretrieve(urls[file_name], self.file_path[:-3] + 'tar.gz')
                try:
                    tar = tarfile.open(self.file_path[:-3] + 'tar.gz', "r:gz")
                    for member in tar.getmembers():
                        #print(member.name)
                        l = len(file_name)
                        if member.name[-l:] == file_name:
                            tar.extract(member, path=self.root)
                            self.file_path = os.path.join(self.root,member.name)
                except ValueError:
                    print('Pb with the tar.gz file')
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

        >>> ca = Dataset('ca-CondMat.mtx')
        loading graph with 23133 nodes and 93497 edges.
        number of nodes: 23133 (expected 23133); number of edges: 93497 (expected 93497)

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

