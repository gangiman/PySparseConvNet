import numpy as np
import os
os.environ["ETS_TOOLKIT"] = "qt4"
from mayavi import mlab

from PySparseConvNet import Off3DPicture

class Off3DFile(object):
    """Add some helpful methods to Off files
    """
    file_path = None

    def __init__(self, file_path):
        if os.path.exists(file_path):
            self.file_path = file_path
            self.vertices, self.faces = self.read_off()
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

    def voxelize(self, ss, rs):
        pic = Off3DPicture(self.file_path, rs)
        pairs, _ = pic.codifyInputData(ss)
        list_of_coordinates = []
        for key_id, feature_idx in pairs:
            list_of_coordinates.append((
                (key_id / ss / ss) % ss,
                (key_id / ss) % ss,
                key_id % ss
            ))
        del pic
        return zip(*list_of_coordinates)

    def plot_shape(self, kind, spatialSize=None, renderSize=None,
                   features=None):
        """ outline=True  - box edges
            title=True - print kind of picture
            **kwargs # one of: vertA, faceA, features, x, y, z, spatialSize
        """
        f = mlab.figure(bgcolor=(1, 1, 1))
        if kind == 'solid':
            vertA, faceA = self.vertices, self.faces
            mlab.triangular_mesh(vertA[:, 0], vertA[:, 1], vertA[:, 2], faceA)
        elif kind == 'transparent':
            vertA, faceA = self.vertices, self.faces
            mlab.triangular_mesh(vertA[:,0], vertA[:, 1], vertA[:, 2], faceA,
                                 opacity=0.1)
        elif kind == 'wireframe':
            vertA, faceA = self.vertices, self.faces
            mlab.triangular_mesh(vertA[:, 0], vertA[:, 1], vertA[:, 2], faceA,
                                 representation='wireframe')
        elif kind == "Bool":
            x, y, z = map(np.array, self.voxelize(spatialSize, renderSize))
            assert len(x) == len(y) == len(z)
            N = len(x)
            scalars = np.arange(N) # Key point: set an integer for each point
            colors = np.zeros((N, 4), dtype=np.uint8)
            colors[:, -1] = 255  # No transparency
            if features is not None:
                features = features.ravel()
                colors[:, 0] = 255
                colors[:, 1] = (
                255 * (1 - features / np.max(features))).astype(np.uint8)
                colors[:, 2] = (
                255 * (1 - features / np.max(features))).astype(np.uint8)
            else:
                colors[:, 0] = 0
                colors[:, 1] = 255
                colors[:, 2] = 0

            pts = mlab.quiver3d(
                x-spatialSize/2,
                y-spatialSize/2+0.5,
                z-spatialSize/2+0.5,
                np.ones(N), np.zeros(N), np.zeros(N),
                scalars=scalars, mode='cube', scale_factor=0.7, line_width=10)
            pts.glyph.color_mode = 'color_by_scalar'
            try:
                pts.module_manager.scalar_lut_manager.lut.table = colors
            except:
                pass
            mlab.draw()

        return f
