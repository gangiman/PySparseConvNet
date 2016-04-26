from __future__ import print_function

from PySparseConvNet import SparseNetwork
from PySparseConvNet import Off3DPicture
from PySparseConvNet import SparseDataset

import os
import math


def create_DeepC2Network(dimension, l, k, fn, nInputFeatures, nClasses, p, nThreads=1):
    sparse_net = SparseNetwork(dimension, nInputFeatures, nClasses, nThreads=nThreads)
    for i in range(l + 1):
        sparse_net.addLeNetLayerMP(
            (i + 1) * k, 2, 1, 3 if (i < l) else 1, 2 if (i < l) else 1, fn,
            p * i * 1.0 / l)
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

def generate_modelnet_dataset(batch_mode='TRAINBATCH'):
    number_of_features = 1
    renderSize = 40
    data_folder = "/media/toshiba/shape_retrieval_datasets/ModelNet/ModelNet40/"

    class_folders = os.listdir(data_folder)
    class_folders.sort()
    number_of_classes = len(class_folders)
    if batch_mode == 'TRAINBATCH':
        sparse_dataset = SparseDataset("ModelNet (Train subset)", 'TRAINBATCH',
                                       number_of_features, number_of_classes)
        max_n_pictures = 80
        batch_folder = 'train'
    else:
        sparse_dataset = SparseDataset("ModelNet (Test subset)", 'TESTBATCH',
                                       number_of_features, number_of_classes)
        max_n_pictures = 20
        batch_folder = 'test'

    for class_id, folder in enumerate(class_folders):
        dirpath = os.path.join(data_folder, folder, batch_folder)
        for filename in os.listdir(dirpath)[:max_n_pictures]:
            sparse_dataset.add_picture(Off3DPicture(
                os.path.join(dirpath, filename), renderSize, label=class_id))
    print(sparse_dataset.summary())
    return sparse_dataset


def learn_simple_network(full=False, batchSize=10, limit=1, epoch=2):
    network = create_dC2()
    print("Created network")
    dataset = generate_modelnet_dataset(full=full, limit=limit)
    dataset.summary()
    print("Created dataset {0}".format(dataset.name))
    for epoch in xrange(1, epoch):
        learning_rate = 0.003 * math.exp(-0.05 / 2 * epoch)
        print("epoch {0}, lr={1} ".format(epoch, learning_rate), end='')
        network.processDataset(dataset, batchSize=batchSize,
                               learningRate=learning_rate)
