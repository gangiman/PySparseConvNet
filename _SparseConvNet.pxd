from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.memory cimport unique_ptr
from libc.stdint cimport int64_t
from libcpp.utility cimport pair
from libc.stdint cimport uint64_t
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "SparseConvNet/SpatiallySparseLayer.h":
    cdef cppclass SpatiallySparseLayer:
        pass

cdef extern from "SparseConvNet/SparseConvNetCUDA.h":
    cdef cppclass SparseConvNetCUDA:
        vector[SpatiallySparseLayer*] layers
        vector[vector[float]] predict(SpatiallySparseDataset &dataset)
        vector[activation] layer_activations(SpatiallySparseDataset &dataset)

    struct pd_report:
        float errorRate
        float nll
        int MegaMultiplyAdds_per_sample
        long time
        int GigaMultiplyAdds_per_s
        int rate

    struct activation:
        long grid_size
        int feature_size
        int nSpatialSites
        int spatialSize
        int nFeatures
        vector[float] features
        vector[pair[int64_t, int]] sparse_grid

cdef extern from "SparseConvNet/types.h":
    cdef enum ActivationFunction:
        NOSIGMOID,
        RELU,
        VLEAKYRELU,
        LEAKYRELU,
        TANH,
        SOFTMAX,
        PRELU,
        SIGMOID

    cdef enum PicturePreprocessing:
        ROTATE_Z_AXIS,
        ROTATE_GENERAL

    cdef enum batchType:
        TRAINBATCH,
        TESTBATCH,
        UNLABELEDBATCH,
        RESCALEBATCH

    cdef char[] batchTypeNames

cdef extern from "SparseConvNet/SparseConvNet.h":
    cdef cppclass SparseConvNet:
        SparseConvNet(int dimension, int nInputFeatures, int nClasses, int pciBusID, int nTop, int nThreads) except +
        unique_ptr[SparseConvNetCUDA] cnn
        void addLeNetLayerMP(int nFeatures, int filterSize, int filterStride,
                           int poolSize, int poolStride,
                           ActivationFunction activationFn,
                           float dropout, int minActiveInputs)
        void addLeNetLayerPOFMP(int nFeatures, int filterSize, int filterStride,
                              int poolSize, float fmpShrink,
                              ActivationFunction activationFn,
                              float dropout, int minActiveInputs)
        void addLeNetLayerROFMP(int nFeatures, int filterSize, int filterStride,
                              int poolSize, float fmpShrink,
                              ActivationFunction activationFn,
                              float dropout, int minActiveInputs)
        void addLeNetLayerPDFMP(int nFeatures, int filterSize, int filterStride,
                              int poolSize, float fmpShrink,
                              ActivationFunction activationFn,
                              float dropout, int minActiveInputs)
        void addLeNetLayerRDFMP(int nFeatures, int filterSize, int filterStride,
                              int poolSize, float fmpShrink,
                              ActivationFunction activationFn,
                              float dropout, int minActiveInputs)
        void addTerminalPoolingLayer(int poolSize)
        void addSoftmaxLayer()
        void addIndexLearnerLayer()
        pd_report processDataset(SpatiallySparseDataset &dataset, int batchSize,
                           float learningRate, float momentum)
                           # PicturePreprocessing preprocessing_type)
        vector[pd_report] processDatasetRepeatTest(SpatiallySparseDataset &dataset,
                                    int batchSize, int nReps,
                                    string predictionsFilename,
                                    string confusionMatrixFilename)
                                    # PicturePreprocessing preprocessing_type)
        float processIndexLearnerDataset(SpatiallySparseDataset &dataset,
                                       int batchSize,
                                       float learningRate,
                                       float momentum)
        void processDatasetDumpTopLevelFeatures(SpatiallySparseDataset &dataset,
                                              int batchSize, int reps)
        void loadWeights(string baseName, int epoch, bool momentum,
                       int firstNlayers)
        void saveWeights(string baseName, int epoch, bool momentum)
        void calculateInputRegularizingConstants(SpatiallySparseDataset dataset)


cdef extern from "SparseConvNet/vectorCUDA.h":
    cdef cppclass vectorCUDA[T]:
        T *d_vec
        vector[T] vec
        vector[T] &hVector()
        unsigned int size()


cdef extern from "SparseConvNet/ConvolutionalLayer.h":
    cdef cppclass ConvolutionalLayer:
        int fs
        int inSpatialSize
        int outSpatialSize
        int filterSize
        int filterStride
        int dimension
        int nFeaturesIn
        int nFeaturesOut
        int minActiveInputs
        # normally 1, <=ipow(filterSize,dimension)
        int calculateInputSpatialSize(int outputSpatialSize)


cdef extern from "SparseConvNet/NetworkInNetworkLayer.h":
    cdef cppclass NetworkInNetworkLayer:
        vectorCUDA[float] W  # Weights
        vectorCUDA[float] MW # momentum
        vectorCUDA[float] w  # shrunk versions
        vectorCUDA[float] dw # For backprop
        vectorCUDA[float] B  # Weights
        vectorCUDA[float] MB # momentum
        vectorCUDA[float] b  # shrunk versions
        vectorCUDA[float] db # For backprop
        ActivationFunction fn
        int nFeaturesIn
        int nFeaturesOut
        float dropout
        int calculateInputSpatialSize(int outputSpatialSize)

cdef extern from "SparseConvNet/MaxPoolingLayer.h":
    cdef cppclass MaxPoolingLayer:
        int inSpatialSize
        int outSpatialSize
        int poolSize
        int poolStride
        int dimension
        int sd
        int calculateInputSpatialSize(int outputSpatialSize)


cdef extern from "<functional>" namespace "std":
    cdef cppclass hash[T]:
        pass
    cdef cppclass equal_to[T]:
        pass


# From
# https://github.com/syllog1sm/cython-sparsehash/blob/a94041cf5b8560cb62dd82d9b0467fca869a6b77/sparsehash/dense_hash_map.pxd

cdef extern from "google/dense_hash_map" namespace "google":
    cdef cppclass dense_hash_map[K, D, V, N]:
        K& key_type
        D& data_type
        pair[K, D]& value_type
        uint64_t size_type
        cppclass iterator:
            pair[K, D]& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        iterator begin()
        iterator end()
        uint64_t size()
        uint64_t max_size()
        bint empty()
        uint64_t bucket_count()
        uint64_t bucket_size(uint64_t i)
        uint64_t bucket(K& key)
        double max_load_factor()
        void max_load_vactor(double new_grow)
        double min_load_factor()
        double min_load_factor(double new_grow)
        void set_resizing_parameters(double shrink, double grow)
        void resize(uint64_t n)
        void rehash(uint64_t n)
        dense_hash_map()
        dense_hash_map(uint64_t n)
        void swap(dense_hash_map&)
        pair[iterator, bint] insert(pair[K, D]) nogil
        void set_empty_key(K&)
        void set_deleted_key(K& key)
        void clear_deleted_key()
        void erase(iterator pos)
        uint64_t erase(K& k)
        void erase(iterator first, iterator last)
        void clear()
        void clear_no_resize()
        pair[iterator, iterator] equal_range(K& k)
        D& operator[](K&) nogil


ctypedef dense_hash_map[int64_t, int, hash[int64_t], equal_to[int64_t]] SparseGridMap
ctypedef dense_hash_map[int64_t, int, hash[int64_t], equal_to[int64_t]].iterator SparseGridIter

cdef extern from "SparseConvNet/SparseGrid.h":
    cdef cppclass SparseGrid:
        int backgroundCol
        SparseGridMap mp
        SparseGrid()


cdef extern from "SparseConvNet/Rng.h":
    cdef cppclass RNG:
        pass

cdef extern from "<armadillo>" namespace "arma":
    cdef cppclass mat:
        pass


cdef extern from "SparseConvNet/Picture.h":
    cdef cppclass Picture:
        Picture(int label)
        void codifyInputData(SparseGrid &grid, vector[float] &features,
                             int &nSpatialSites, int spatialSize)
        Picture *distort(RNG &rng, batchType type)
        void loadPicture()
        bool is_loaded
        int label # -1 for unknown


cdef extern from "SparseConvNet/Off3DFormatPicture.h":
    cdef cppclass OffSurfaceModelPicture:
        mat points
        vector[vector[int]] surfaces # Will assume all surfaces are triangles for now
        int renderSize
        string picture_path
        bool is_loaded
        int label # -1 for unknown
        OffSurfaceModelPicture(string filename, int renderSize, int label_)
        void loadPicture()
        void normalize() # Fit centrally in the cube [-scale_n/2,scale_n/2]^3
        void random_rotation(RNG &rng)
        void jiggle(RNG &rng, float alpha)
        void affineTransform(RNG &rng, float alpha)
        void codifyInputData(SparseGrid &grid, vector[float] &features,
                             int &nSpatialSites, int spatialSize)
        Picture *distort(RNG &rng, batchType type)



cdef extern from "SparseConvNet/SpatiallySparseDataset.h":
    cdef cppclass SpatiallySparseDataset:
        string name
        # RNG rng
        # string header
        vector[Picture *] pictures
        int nFeatures
        int nClasses
        batchType type
        void summary()
        void shuffle()
        SpatiallySparseDataset extractValidationSet(float p)
        void subsetOfClasses(vector[int] activeClasses)
        SpatiallySparseDataset subset(int n)
        SpatiallySparseDataset balancedSubset(int n)
        void repeatSamples(int reps)


cdef extern from "SparseConvNet/SpatiallySparseBatchInterface.h":
    cdef cppclass SpatiallySparseBatchSubInterface:
        vectorCUDA[float] features

    cdef cppclass SpatiallySparseBatchInterface:
        SpatiallySparseBatchSubInterface *sub
        int nFeatures # Features per spatial location
        vectorCUDA[int] featuresPresent # Not dropped out features per spatial location
        int nSpatialSites # Total active spatial locations within the
        int spatialSize   # spatialSize x spatialSize grid
        vector[SparseGrid] grids # batchSize vectors of maps storing info on


