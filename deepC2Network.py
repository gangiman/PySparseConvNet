from __future__ import print_function

from PySparseConvNet import SparseNetwork
from PySparseConvNet import Off3DPicture
from PySparseConvNet import SparseDataset

import os
import math


def create_DeepC2Network(dimension, n_layers, n_filters_multiplier, fn, nInputFeatures,
                         nClasses, dropout, nThreads=1):
    sparse_net = SparseNetwork(dimension, nInputFeatures, nClasses, nThreads=nThreads)
    for i in range(n_layers + 1):
        sparse_net.addLeNetLayerMP(
            (i + 1) * n_filters_multiplier,
            2,
            1,
            3 if (i < n_layers) else 1,
            2 if (i < n_layers) else 1,
            fn,
            dropout * i * 1.0 / n_layers)
    sparse_net.addSoftmaxLayer()
    return sparse_net


def custom_DeepC2Network(dimension, n_filters_multiplier,
                         fsizes, fstrides, pool_sizes, pool_strides,
                         fn, nInputFeatures,
                         nClasses, dropout, nThreads=1):
    sparse_net = SparseNetwork(dimension, nInputFeatures, nClasses, nThreads=nThreads)
    n_layers = len(fsizes)
    assert(len(fsizes) == len(fstrides))
    assert(len(fsizes) == len(pool_sizes))
    assert(len(fsizes) == len(pool_strides))
    for i in xrange(n_layers + 1):
        sparse_net.addLeNetLayerMP(
            (i + 1) * n_filters_multiplier,
            fsizes[i] if i < n_layers else 2,
            fstrides[i] if i < n_layers else 1,
            pool_sizes[i] if i < n_layers else 1,
            pool_strides[i] if i < n_layers else 1,
            fn,
            dropout * i * 1.0 / n_layers)
    sparse_net.addSoftmaxLayer()
    return sparse_net



def create_dC2():
    nFeatures = 1
    nClasses = 40
    return create_DeepC2Network(3, 5, 32, 'VLEAKYRELU', nFeatures, nClasses, 0.0)

def load_and_get_weights(deepC2):
    baseName = "SparseConvNet/weights/ModelNet"
    epoch = 20
    deepC2.loadWeights(baseName, epoch)
    return deepC2.get_weights()


def load_3d_off():
    path = "SparseConvNet/Data/ModelNet/airplane/train/airplane_0511.off"
    print("Creating Off3DPicture object")
    picture = Off3DPicture(path, 40)
    print("Codifying...")
    pairs, features = picture.codifyInputData(126)
    print ("done")
    return pairs


def generate_modelnet_dataset(full=False, limit=-1):
    number_of_features = 1
    renderSize = 40
    if full:
        data_folder = "SparseConvNet/Data/ModelNet/"
    else:
        data_folder = "SparseConvNet/Data/_ModelNet/"
    class_folders = os.listdir(data_folder)
    class_folders.sort()
    number_of_classes = len(class_folders)
    sparse_dataset = SparseDataset("ModelNet (Train subset)", 'TRAINBATCH',
                                   number_of_features, number_of_classes)
    for class_id, folder in enumerate(class_folders):
        dirpath = os.path.join(data_folder, folder, 'train')
        for _count, filename in enumerate(os.listdir(dirpath)):
            if _count > limit > 0:
                break
            sparse_dataset.add_picture(Off3DPicture(
                os.path.join(dirpath, filename), renderSize, label=class_id))
    # sparse_dataset.repeatSamples(10)
    return sparse_dataset


def learn_simple_network(full=False, batchSize=10, limit=1, epoch=2):
    network = create_dC2()
    print("Created network")
    dataset = generate_modelnet_dataset(full=full, limit=limit)
    dataset.summary()
    print("Created dataset {0}".format(dataset.name))
    for epoch in xrange(1, epoch):
        learning_rate = 0.003 * math.exp(-0.05 / 2 * epoch)
        # print("epoch {0}, lr={1} ".format(epoch, learning_rate), end='')
        network.processDataset(dataset, batchSize=batchSize,
                               learningRate=learning_rate)
