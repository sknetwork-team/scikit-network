
import os
import os.path
import zipfile
import tarfile

from six.moves import urllib
from scipy.sparse import csr_matrix

urls = {}


class Dataset:

    def __init__(self, file_name, directed=True, download=True, root='./data/'):
        self.root = root

        if file_name not in urls.keys():
            raise RuntimeError('Unknown dataset')

        self.make_dir()

        self.file_path = os.path.join(root, file_name)

        if download:
            self.download(file_name)

        if file_name[-3:] == 'mtx':
            self.load_from_mtx(self.file_path)

    def make_dir(self):
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
                    print('Problem with the zip file')
            elif urls[file_name][-6:] == 'tar.gz':
                urllib.request.urlretrieve(urls[file_name], self.file_path[:-3] + 'tar.gz')
                try:
                    tar = tarfile.open(self.file_path[:-3] + 'tar.gz', "r:gz")
                    for member in tar.getmembers():
                        file_length = len(file_name)
                        if member.name[-file_length:] == file_name:
                            tar.extract(member, path=self.root)
                            self.file_path = os.path.join(self.root, member.name)
                except ValueError:
                    print('Problem with the tar.gz file')
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

        Note
        ----
        Useful repo : https://sparse.tamu.edu/SNAP

        """

        rows = []
        cols = []
        vals = []
        first_line = True
        with open(path, 'rt') as f:
            for line in f:
                p = line.find(comments)
                if p >= 0:
                    line = line[:p]
                if not len(line):
                    continue
                s = line.strip().split(delimiter)
                if len(s) < 2:
                    continue
                if first_line:
                    try:
                        n = s.pop(0)
                        m = s.pop(0)
                        e = s.pop(0)
                        if n != m:
                            print("check the number of nodes")
                        else:
                            print("loading graph with %s nodes and %s edges." % (n, e))
                            first_line = False
                    except:
                        raise ValueError("First line must contain num_nodes num_nodes num_edges.")
                    continue
                u = s.pop(0)
                v = s.pop(0)
                rows.append(int(u))
                cols.append(int(v))
                vals.append(1)

        return csr_matrix((vals, (rows, cols)))
