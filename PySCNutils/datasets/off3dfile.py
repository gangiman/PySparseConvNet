import os
import numpy as np


class Off3DFile(object):
    """Add some helpful methods to Off files
    """
    file_path = None

    def __init__(self, file_path):
        if os.path.exists(file_path):
            self.file_path = file_path
        else:
            raise IOError("File {} doesn't exists!".format(file_path))

    def read_off(self):
        with open(self.file_path, 'r') as _file:
            if 'OFF' != _file.readline().strip():
                raise ValueError('Not a valid OFF header')
            n_verts, n_faces, n_dontknow = tuple(
                [int(s) for s in _file.readline().strip().split(' ')])
            verts = []
            for i_vert in range(n_verts):
                verts.append(
                    [float(s) for s in _file.readline().strip().split(' ')])
            faces = []
            for i_face in range(n_faces):
                faces.append(
                    [int(s) for s in _file.readline().strip().split(' ')][1:])
            return np.array(verts), np.array(faces)

    def plot_triangles(self, ax=None, title=''):
        import matplotlib.pyplot as plt
        if not title:
            title = self.file_path.split('/')[-1]
        if ax is None:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
        verts_np, patches = self.read_off()
        # ax.scatter(x, y, z, zdir='z', c= 'red')
        ax.plot_trisurf(verts_np[:, 0], verts_np[:, 1], verts_np[:, 2],
                        triangles=patches, lw=0)
        ax.set_title(title)
