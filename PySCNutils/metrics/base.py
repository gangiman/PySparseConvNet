from collections import OrderedDict
from scipy.spatial.distance import cosine
import numpy as np
import shelve
import os

try:
    from PySparseConvNet import Off3DPicture
    from PySparseConvNet import SparseDataset
except ImportError:
    print("PySparseConvNet doesn't imports")
    raise


def get_feature_extractor(network, weights_path, _layer=17,
                          renderSize=40):
    weights_folder = os.path.dirname(weights_path)
    weights_file = os.path.basename(weights_path)
    prefix, epoch = weights_file.split('_epoch')
    epoch = epoch.strip('-.cnn')
    network.loadWeights(
        os.path.join(weights_folder, prefix), int(epoch))

    def wraped(pic_path):
        pic = Off3DPicture(pic_path, renderSize)
        loi = network.layer_activations(pic)
        return np.array(loi[_layer]['features'])

    return wraped


class Metric(object):

    name = None
    caching = True

    def __call__(self, *args):
        x, y = args
        try:
            assert all(os.path.exists(_arg) for _arg in args)
        except AssertionError:
            print(args)
            raise
        if x == y:
            return 0
        else:
            return self.compute_metric(x, y)

    def compute_metric(self, x, y):
        raise NotImplementedError()


class NNMetric(Metric):

    dist = OrderedDict((
        ('L2', lambda x, y: np.linalg.norm(x-y)),
        ('cos', cosine)
    ))

    def __init__(self, network, weights_path=None, norm='cos', _layer=17,
                 render_size=40, cache_dir='/tmp'):
        self.name = os.path.basename(weights_path).rstrip('.cnn')
        self.metric = self.dist[norm]
        self.unique_hash = "{}_{}_{}_{}".format(self.name, norm, _layer,
                                                render_size)

        self.extractor = get_feature_extractor(network, weights_path,
                                               _layer=_layer,
                                               renderSize=render_size)
        if not os.path.isdir(cache_dir):
            raise IOError("No such directory '{}'".format(cache_dir))
        self.cache_file = os.path.join(cache_dir,
                                       self.unique_hash + '.db')
        self.cache = shelve.open(self.cache_file, writeback=True)

    def __del__(self):
        self.cache.close()

    def get_vector(self, off_path):
        if off_path not in self.cache:
            feature_vector = self.extractor(off_path)
            feature_vector /= np.sqrt((feature_vector**2).sum())
            self.cache[off_path] = feature_vector
        return self.cache[off_path]

    def compute_metric(self, *args):
        return self.metric(*map(self.get_vector, args))
