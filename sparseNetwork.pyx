from _SparseConvNet cimport SparseConvNet
from _SparseConvNet cimport VLEAKYRELU
from _SparseConvNet cimport TRAINBATCH
from _SparseConvNet cimport TESTBATCH
from _SparseConvNet cimport UNLABELEDBATCH
from _SparseConvNet cimport NetworkInNetworkLayer
from _SparseConvNet cimport OffSurfaceModelPicture
from _SparseConvNet cimport Picture
from _SparseConvNet cimport SpatiallySparseDataset
from _SparseConvNet cimport activation
from _SparseConvNet cimport pd_report

from _SparseConvNet cimport Bool
from _SparseConvNet cimport ScalarArea
from _SparseConvNet cimport AreaNormal
from _SparseConvNet cimport Quadform
from _SparseConvNet cimport Eigenvalues
from _SparseConvNet cimport QFoverSA
from _SparseConvNet cimport EVoverSA
from _SparseConvNet cimport AngularDefect

from _SparseConvNet cimport nFeaturesPerVoxel_set

from libcpp.string cimport string
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc

from _SparseConvNet cimport SparseGrid
from _SparseConvNet cimport SparseGridIter
from libcpp.vector cimport vector
from libcpp cimport bool

import numpy as np
cimport numpy as np

from copy import deepcopy
from pprint import pprint



cdef class SparseNetwork:
    """create a network object, configure layers, threads and dimensionality of input
    """
    cdef SparseConvNet* net
    cdef list layers
    cdef int dimension
    cdef int nInputFeatures
    cdef int nClasses

    def __cinit__(self, int dimension, int nInputFeatures, int nClasses,
                  int cudaDevice=-1, int nTop=1, int nThreads=10):
        """Initializing Network.

        dimension - number of input dimension
        nInputFeatures number of features in one cell of the grid
        """
        self.layers = []
        self.net = new SparseConvNet(dimension, nInputFeatures,
                                     nClasses, cudaDevice, nTop, nThreads)
        self.dimension = dimension
        self.nInputFeatures = nInputFeatures
        self.nClasses = nClasses

    def __dealloc__(self):
        del self.net

    def processDataset(self, SparseDataset dataset, int batchSize=100,
                       float learningRate=0, float momentum=0.99):
        """
        """
        return self.net.processDataset(deref(dataset.ssd), batchSize, learningRate, momentum)

    def processDatasetRepeatTest(self, SparseDataset dataset, batchSize=100, nReps=12,
                                 predictionsFilename="",
                                 confusionMatrixFilename=""):
        """
        """
        cdef vector[pd_report] reports = \
            self.net.processDatasetRepeatTest(deref(dataset.ssd),
                                          batchSize, nReps,
                                          predictionsFilename,
                                          confusionMatrixFilename)

        return reports


    def addLeNetLayerMP(self, nFeatures, filterSize, filterStride, poolSize,
                        poolStride, activationFn, dropout, minActiveInputs=1):
        """
        int nFeatures
        int filterSize
        int filterStride
        int poolSize
        int poolStride
        ActivationFunction activationFn
        float dropout
        int minActiveInputs
        """
        if activationFn == 'VLEAKYRELU':
            _activationFn = VLEAKYRELU
        else:
            raise ValueError("Unknown activation function!")
        self.net.addLeNetLayerMP(nFeatures, filterSize, filterStride, poolSize,
                        poolStride, _activationFn, dropout, minActiveInputs)
        self.layers.append({
            'type': 'ConvolutionalLayer',
            'filterSize': filterSize,
            'filterStride':  filterStride,

        })
        self.layers.append({
            'type': 'LearntLayer',
            'activationFn': activationFn
        })
        if poolSize > 1:
            self.layers.append({
                'type': 'MaxPoolingLayer',
                'poolSize': poolSize,
                'poolStride': poolStride
            })

    def addSoftmaxLayer(self):
        """
        """
        self.net.addSoftmaxLayer()
        self.layers.append({
            'type': 'LearntLayer',
            'activationFn': 'SOFTMAX'
        })

    def loadWeights(self, baseName, epoch, momentum=False,
                    firstNlayers=1000000):
        """
        """
        self.net.loadWeights(baseName, epoch, momentum, firstNlayers)

    def saveWeights(self, baseName, epoch, momentum=False):
        """
        """
        self.net.saveWeights(baseName, epoch, momentum)

    def get_layers(self):
        return self.layers

    def get_weights(self):
        cdef int layer_id
        cdef dict layer_dict
        cdef NetworkInNetworkLayer* layer
        layers_with_weights = deepcopy(self.layers)
        for layer_id, layer_dict in enumerate(layers_with_weights):
            if layer_dict['type'] == 'LearntLayer':
                layer = <NetworkInNetworkLayer*>self.net.cnn.get().layers[layer_id]
                layer_dict['nFeaturesIn'] = layer.nFeaturesIn
                layer_dict['nFeaturesOut'] = layer.nFeaturesOut
                layer_dict['weights'] = {
                    "W": np.zeros(layer.nFeaturesIn * layer.nFeaturesOut, dtype=np.float64),
                    "B": np.zeros(layer.nFeaturesOut, dtype=np.float64)
                }
                layer_dict['weights']['W'][:] = layer.W.hVector()[:]
                layer_dict['weights']['B'][:] = layer.B.hVector()[:]
        return layers_with_weights

    def predict(self, SparseDataset dataset):
        cdef vector[vector[float]] prediction_matrix
        np_matrix = np.zeros((dataset.ssd.pictures.size(), self.nClasses), dtype=np.float64)
        prediction_matrix = self.net.cnn.get().predict(deref(dataset.ssd))
        np_matrix[...] = prediction_matrix
        return np_matrix

    def layer_activations(self, Off3DPicture picture):
        cdef vector[activation] interfaces
        cdef SparseDataset dataset = SparseDataset("-", 'UNLABELEDBATCH', 1, 1)
        dataset.add_picture(picture)
        interfaces = self.net.cnn.get().layer_activations(deref(dataset.ssd))
        return interfaces

cdef char* _train = 'TRAINBATCH'
cdef char* _test = 'TESTBATCH'
cdef char* _unlabeled = 'UNLABELEDBATCH'

def convert_feature_set(frozeset features):
    _feature_kind = set()
    if "Bool" in features:
        _feature_kind.add(Bool)
    if "ScalarArea" in features:
        _feature_kind.add(ScalarArea)
    if "AreaNormal" in features:
        _feature_kind.add(AreaNormal)
    if "Quadform" in features:
        _feature_kind.add(Quadform)
    if "Eigenvalues" in features:
        _feature_kind.add(Eigenvalues)
    if "QFoverSA" in features:
        _feature_kind.add(QFoverSA)
    if "EVoverSA" in features:
        _feature_kind.add(EVoverSA)
    if "AngularDefect" in features:
        _feature_kind.add(AngularDefect)
    if not _feature_kind:
        raise ValueError("Need to select at least one feature extractor!")
    return _feature_kind

cdef class SparseDataset:
    """A collection of Off3DPicture objects, that can be repeated and augmented
    """
    cdef SpatiallySparseDataset* ssd
    cdef string name

    def __cinit__(self, string name, string _type, int nClasses, frozeset features={'Bool'}):
        self.ssd = new SpatiallySparseDataset()
        _feature_kind = convert_feature_set(features)
        self.ssd.feature_kind = _feature_kind
        if _type == _train:
            self.ssd.type = TRAINBATCH
        elif _type == _test:
            self.ssd.type = TESTBATCH
        elif _type == _unlabeled:
            self.ssd.type = UNLABELEDBATCH
        else:
            raise ValueError('Unknown type of batch!')

        self.ssd.name = name
        self.ssd.nFeatures = nFeaturesPerVoxel_set(_feature_kind)
        self.ssd.nClasses = nClasses

    def summary(self):
        self.ssd.summary()

    @property
    def nClasses(self):
        return self.ssd.nClasses

    @property
    def name(self):
        return self.ssd.name

    def __dealloc__(self):
        del self.ssd

    def repeatSamples(self, int nreps):
        self.ssd.repeatSamples(nreps)

    def add_picture(self, Off3DPicture picture):
        picture.pic.feature_kind = self.ssd.feature_kind
        self.ssd.pictures.push_back(<Picture*>picture.pic)


cdef class Off3DPicture:
    """wraps “.off” mesh objects, can voxelise them based on some parameters
    """
    cdef OffSurfaceModelPicture* pic
    cdef SparseGrid grid
    cdef vector[float] features
    cdef int nSpatialSites

    def __cinit__(self, string filename, int renderSize, int label=-1,frozeset features={'Bool'}):

        _feature_kind = convert_feature_set(features)
        self.nSpatialSites = 0
        self.pic = new OffSurfaceModelPicture(filename, renderSize, label, _feature_kind)
        if load:
            self.pic.loadPicture()

    def codifyInputData(self, int spatialSize):
        if not self.pic.is_loaded:
            self.pic.loadPicture()
        self.pic.normalize()
        self.features.resize(0)
        self.pic.codifyInputData(self.grid, self.features,
                                 self.nSpatialSites, spatialSize)
        print("Size of SparseGrid {0}".format(self.grid.mp.size()))
        print("Size of features {0}".format(self.features.size()))
        print("nSpatialSites = {0}".format(self.nSpatialSites))
        cdef list pairs = []
        cdef SparseGridIter it = self.grid.mp.begin()
        while it != self.grid.mp.end():
            pairs.append(deref(it))
            inc(it)
        return pairs, self.features
