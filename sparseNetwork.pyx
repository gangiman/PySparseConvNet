from _SparseConvNet cimport SparseConvNet
from _SparseConvNet cimport SpatiallySparseBatch
from _SparseConvNet cimport BatchProducer

from _SparseConvNet cimport NOSIGMOID
from _SparseConvNet cimport RELU
from _SparseConvNet cimport LEAKYRELU
from _SparseConvNet cimport VLEAKYRELU
from _SparseConvNet cimport TANH
from _SparseConvNet cimport SOFTMAX
from _SparseConvNet cimport PRELU
from _SparseConvNet cimport SIGMOID

from _SparseConvNet cimport TRAINBATCH
from _SparseConvNet cimport TESTBATCH
from _SparseConvNet cimport UNLABELEDBATCH

from _SparseConvNet cimport NetworkInNetworkLayer
from _SparseConvNet cimport OffSurfaceModelPicture
from _SparseConvNet cimport VoxelPicture
from _SparseConvNet cimport Picture
from _SparseConvNet cimport SpatiallySparseDataset
from _SparseConvNet cimport activation
from _SparseConvNet cimport pd_report

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

from libcpp.vector cimport vector


cdef class SparseBatch:
    cdef SpatiallySparseBatch* ssb
    cdef int batchSize
    cdef list sampleNumbers
    cdef list labels

    def __cinit__(self):
        pass

    def _construct(self):
        self.batchSize = self.ssb.batchSize
        self.sampleNumbers = self.ssb.sampleNumbers
        self.labels = self.ssb.labels.hVector()[:]

    def get_batchSize(self):
        return self.batchSize

    def get_sampleNumbers(self):
        return self.sampleNumbers

    def get_labels(self):
        return self.labels

    # def __dealloc__(self):
    #     del self.ssb


cdef class SparseNetwork:
    """create a network object, configure layers, threads and dimensionality of input
    """
    cdef SparseConvNet* net
    cdef list layers
    cdef int dimension
    cdef int nInputFeatures
    cdef int nClasses
    cdef int input_spatial_size

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
        self.input_spatial_size = -1

    def __dealloc__(self):
        del self.net

    def batch_generator(self, SparseDataset dataset, batchSize,
                        output_spatial_size=1):
        if self.input_spatial_size < 0:
            self.input_spatial_size = self.net.cnn.get().\
                computeInputSpatialSize(output_spatial_size)
        cdef BatchProducer* bp = new BatchProducer(
            deref(self.net.cnn.get()),
            deref(dataset.ssd),
            self.input_spatial_size,
            batchSize)
        cdef SpatiallySparseBatch *batch = bp.nextBatch()
        while batch != NULL:
            sb = SparseBatch()
            sb.ssb = batch
            sb._construct()
            yield sb
            batch = bp.nextBatch()

    def computeInputSpatialSize(self):
        return self.net.cnn.get().computeInputSpatialSize(1)

    def processDataset(self, SparseDataset dataset, int batchSize=100,
                       float learningRate=0, float momentum=0.99):
        """
        """
        return self.net.processDataset(deref(dataset.ssd), batchSize, learningRate, momentum)

    def processBatchForward(self, SparseBatch batch):
        cdef activation _activation
        _activation = self.net.cnn.get().processBatchForward(deref(batch.ssb))
        last_layer_activation = dict()
        bs = batch.get_batchSize()
        last_layer_activation['grid_size'] = _activation.grid_size
        last_layer_activation['spatialSize'] = _activation.spatialSize
        last_layer_activation['nFeatures'] = _activation.nFeatures
        last_layer_activation['features'] = np.zeros(_activation.features.size(),
                                                     dtype=np.float64)
        last_layer_activation['features'][:] = _activation.features[:]
        last_layer_activation['features'] = last_layer_activation['features'].reshape((bs, _activation.nFeatures))
        return last_layer_activation

    def processBatchBackward(self, SparseBatch batch, np.ndarray dfeatures,
                             float learningRate=0, float momentum=0.99):
        self.net.cnn.get().processBatchBackward(deref(batch.ssb),learningRate,
                             momentum, dfeatures.ravel().tolist())

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

    @staticmethod
    def _get_activation_fn(fn_name):
        if fn_name == 'NOSIGMOID':
            return NOSIGMOID
        elif fn_name == 'RELU':
            return RELU
        elif fn_name == 'VLEAKYRELU':
            return VLEAKYRELU
        elif fn_name == 'LEAKYRELU':
            return LEAKYRELU
        elif fn_name == 'TANH':
            return TANH
        elif fn_name == 'SOFTMAX':
            return SOFTMAX
        elif fn_name == 'PRELU':
            return PRELU
        elif fn_name == 'SIGMOID':
            return SIGMOID
        else:
            raise ValueError("Unknown activation function!")

    def addConvolutionalLayer(self, nFeatures, filterSize, filterStride,
                              activationFn = 'RELU', dropout = 0.0,
                              minActiveInputs = 1, poolingToFollow = 1.0):
        _activationFn = self._get_activation_fn(activationFn)
        self.net.cnn.get().addConvolutionalLayer(nFeatures, filterSize,
                                                 filterStride,_activationFn,
                                                 dropout,
                                                 minActiveInputs,
                                                 poolingToFollow)
        if filterSize > 1:
            self.layers.append({
                'type': 'ConvolutionalLayer',
                'filterSize': filterSize,
                'filterStride':  filterStride,

            })
        self.layers.append({
            'type': 'LearntLayer',
            'activationFn': activationFn
        })


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
        _activationFn = self._get_activation_fn(activationFn)
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

    def get_num_of_paramenters(self):
        num_of_paramenters = 0
        layers_with_weights = self.get_weights()
        for layer_dict in layers_with_weights:
            if layer_dict['type'] == 'LearntLayer':
                num_of_paramenters += (layer_dict['nFeaturesIn'] + 1) * \
                    layer_dict['nFeaturesOut']
        return num_of_paramenters


    def layer_activations(self, Off3DPicture picture):
        if self.input_spatial_size < 0:
            self.input_spatial_size = self.net.cnn.get().\
                computeInputSpatialSize(1)
        cdef vector[activation] interfaces
        cdef SparseDataset dataset = SparseDataset("-", 'UNLABELEDBATCH', 1, 1)
        dataset.add_picture(picture)
        interfaces = self.net.cnn.get().layer_activations(deref(dataset.ssd))
        return interfaces

cdef char* _train = 'TRAINBATCH'
cdef char* _test = 'TESTBATCH'
cdef char* _unlabeled = 'UNLABELEDBATCH'

cdef class SparseDataset:
    """A collection of Off3DPicture objects, that can be repeated and augmented
    """
    cdef SpatiallySparseDataset* ssd
    cdef string type

    def __cinit__(self, string name, string _type, int nFeatures, int nClasses,
                  shuffle=True):
        self.ssd = new SpatiallySparseDataset()
        self.ssd.do_shuffle = shuffle
        self.type = _type
        if _type == _train:
            self.ssd.type = TRAINBATCH
        elif _type == _test:
            self.ssd.type = TESTBATCH
        elif _type == _unlabeled:
            self.ssd.type = UNLABELEDBATCH
        else:
            raise ValueError("Unknown type of batch! Must be "
                             "'TRAINBATCH' or 'TESTBATCH' or 'UNLABELEDBATCH'")

        self.ssd.name = name
        self.ssd.nFeatures = nFeatures
        self.ssd.nClasses = nClasses

    def summary(self):
        print("Name:            {}".format(self.ssd.name))
        print("Type:            {}".format(self.type))
        print("nFeatures:       {}".format(self.ssd.nFeatures))
        print("nPictures:       {}".format(self.ssd.pictures.size()))
        print("nClasses:        {}".format(self.ssd.nClasses))
        count = [0] * int(self.ssd.nClasses)
        for i in range(self.ssd.pictures.size()):
            count[self.ssd.pictures[i].label] += 1
        if self.ssd.type != UNLABELEDBATCH:
            print("nPictures/class: {}".format(", ".join(map(str, count))))

    @property
    def nClasses(self):
        return self.ssd.nClasses

    @property
    def nFeatures(self):
        return self.ssd.nFeatures

    @property
    def nSamples(self):
        return self.ssd.pictures.size()

    @property
    def name(self):
        return self.ssd.name

    def __dealloc__(self):
        del self.ssd

    def repeatSamples(self, int nreps):
        self.ssd.repeatSamples(nreps)

    def add_picture(self, Off3DPicture picture):
        self.ssd.pictures.push_back(<Picture*>picture.pic)

    def add_voxel_picture(self, PyVoxelPicture picture):
        self.ssd.pictures.push_back(<Picture*>picture.pic)


cdef class Off3DPicture:
    """wraps '.off' mesh objects, can voxelise them based on some parameters
    """
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
        cdef list pairs = []
        cdef SparseGridIter it = self.grid.mp.begin()
        while it != self.grid.mp.end():
            pairs.append(deref(it))
            inc(it)
        return pairs, self.features


cdef class PyVoxelPicture:
    """wraps VoxelPicture
    """
    cdef VoxelPicture* pic
    cdef SparseGrid grid
    cdef vector[float] features
    cdef int nSpatialSites

    def __cinit__(self, np.ndarray[long, mode="c", ndim=2] indices,
                     np.ndarray[double, mode="c", ndim=2] input_features,
                           int renderSize, int label=-1, int n_features=1):

        self.nSpatialSites = 0
        self.pic = new VoxelPicture(indices, input_features, renderSize, label, n_features)


    #def __dealloc__(self):
    #    del self.pic
    #     # del self.grid
    #     # del self.features

    def codifyInputData(self, int spatialSize):
        self.features.resize(0)
        self.pic.codifyInputData(self.grid, self.features,
                                 self.nSpatialSites, spatialSize)
        cdef list pairs = []
        cdef SparseGridIter it = self.grid.mp.begin()
        while it != self.grid.mp.end():
            pairs.append(deref(it))
            inc(it)
        return pairs, self.features
