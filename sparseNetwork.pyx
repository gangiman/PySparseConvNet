from _SparseConvNet cimport SparseConvNet
from _SparseConvNet cimport VLEAKYRELU
from _SparseConvNet cimport TRAINBATCH
from _SparseConvNet cimport TESTBATCH
from _SparseConvNet cimport UNLABELEDBATCH
from _SparseConvNet cimport NetworkInNetworkLayer
from _SparseConvNet cimport OffSurfaceModelPicture
from _SparseConvNet cimport Picture
from _SparseConvNet cimport SpatiallySparseDataset
from _SparseConvNet cimport SpatiallySparseBatchInterface
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
    cdef SparseConvNet* net
    cdef list layers
    cdef int dimension
    cdef int nInputFeatures
    cdef int nClasses

    def __cinit__(self, int dimension, int nInputFeatures, int nClasses,
                  int cudaDevice=-1, int nTop=1, int nThreads=1):
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
        self.net.processDataset(deref(dataset.ssd), batchSize, learningRate, momentum)

    def processDatasetRepeatTest(self, SparseDataset dataset, batchSize=100, nReps=12,
                                 predictionsFilename="",
                                 confusionMatrixFilename=""):
        """
        """
        self.net.processDatasetRepeatTest(deref(dataset.ssd),
                                          batchSize, nReps,
                                          predictionsFilename,
                                          confusionMatrixFilename)


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

    def layer_activations(self, SparseDataset dataset):
        cdef vector[SpatiallySparseBatchInterface] interfaces
        interfaces = self.net.cnn.get().layer_activations(deref(dataset.ssd))
        cdef list list_of_interfaces = []
        cdef list pairs = []
        cdef SparseGridIter it
        for _i in range(interfaces.size()):
            it = interfaces[_i].grids[0].mp.begin()
            pairs = []
            while it != interfaces[_i].grids[0].mp.end():
                pairs.append(deref(it))
                inc(it)
            dict_of_activations = {
            "layer": _i,
        "grid_size": interfaces[_i].grids[0].mp.size(),
     "feature_size": interfaces[_i].sub.features.size(),
    "nSpatialSites": interfaces[_i].nSpatialSites,
      "spatialSize": interfaces[_i].spatialSize,
        "nFeatures": interfaces[_i].nFeatures
            }
            pprint(dict_of_activations)
            dict_of_activations["sparse_grid"] = deepcopy(pairs)
            dict_of_activations["features"] = np.zeros(interfaces[_i].sub.features.size(), dtype=np.float64)
            dict_of_activations["features"][...] = interfaces[_i].sub.features.hVector()
            list_of_interfaces.append(dict_of_activations)

        return list_of_interfaces


cdef char* _train = 'TRAINBATCH'
cdef char* _test = 'TESTBATCH'
cdef char* _unlabeled = 'UNLABELEDBATCH'

cdef class SparseDataset:
    cdef SpatiallySparseDataset* ssd
    cdef string name

    def __cinit__(self, string name, string _type, int nFeatures, int nClasses):
        self.ssd = new SpatiallySparseDataset()
        if _type == _train:
            self.ssd.type = TRAINBATCH
        elif _type == _test:
            self.ssd.type = TESTBATCH
        elif _type == _unlabeled:
            self.ssd.type = UNLABELEDBATCH
        else:
            raise ValueError('Unknown type of batch!')
        self.ssd.name = name
        self.ssd.nFeatures = nFeatures
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
        self.ssd.pictures.push_back(<Picture*>picture.pic)


cdef class Off3DPicture:
    cdef OffSurfaceModelPicture* pic
    cdef SparseGrid grid
    cdef vector[float] features
    cdef int nSpatialSites

    def __cinit__(self, string filename, int renderSize, int label=-1, bool load=False):
        self.nSpatialSites = 0
        self.pic = new OffSurfaceModelPicture(filename, renderSize, label)
        if load:
            self.pic.loadPicture()

    #def __dealloc__(self):
    #    del self.pic
    #     # del self.grid
    #     # del self.features

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
