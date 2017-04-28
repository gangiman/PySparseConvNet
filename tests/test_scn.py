from __future__ import print_function
import unittest
import sys

from PySparseConvNet import SparseNetwork
from PySparseConvNet import SparseDataset
from PySparseConvNet import Off3DPicture

import os
import numpy as np

from PySCNutils.networks import create_dC2


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
        learning_rate = 0.003 * np.exp(-0.05 / 2 * epoch)
        # print("epoch {0}, lr={1} ".format(epoch, learning_rate), end='')
        network.processDataset(dataset, batchSize=batchSize,
                               learningRate=learning_rate)


class TestHighLevelLogic(unittest.TestCase):

    def test_dC2_creation_and_loading(self):
        network = create_dC2()
        self.assertEqual(type(network), SparseNetwork)
        layers = load_and_get_weights(network)
        self.assertEqual(len(layers), 18)

    def test_Off_file_loading(self):
        pairs = load_3d_off()
        self.assertTrue(len(pairs) > 0)

    def test_dataset_creation(self):
        with self.assertRaises(ValueError):
            SparseDataset("Testing DataSet", "FakeBatch", 1, 2)
        print("Successfully caught Value Exception.")
        ds = SparseDataset("Testing DataSet", "TRAINBATCH", 1, 2)
        self.assertEqual(ds.name, "Testing DataSet")


class TestTraining(unittest.TestCase):

    def test_simple_training(self):
        learn_simple_network(limit=-1)

    def test_predict(self):
        unlabeled_dataset = SparseDataset("One pic", 'UNLABELEDBATCH', 1, 1)
        network = create_dC2()
        num_of_inputs = 5
        nClasses = 40
        renderSize = 40
        test_file = ('SparseConvNet/Data/ModelNet/night_stand/'
                     'train/night_stand_0180.off')
        for i in range(num_of_inputs):
            unlabeled_dataset.add_picture(Off3DPicture(test_file, renderSize))
        matrix_of_preds = network.predict(unlabeled_dataset)
        self.assertEqual(matrix_of_preds.shape, (num_of_inputs, nClasses))

    def test_forward_backward_pass(self):
        import numpy as np
        batchSize = 10
        nInputFeatures = 1
        nClasses = 40
        p = 0.0
        dimension = 3
        l = 5
        k = 32
        fn = 'VLEAKYRELU'
        network = SparseNetwork(dimension, nInputFeatures, nClasses)
        for i in range(l + 1):
            network.addLeNetLayerMP(
                (i + 1) * k, 2, 1, 3 if (i < l) else 1, 2 if (i < l) else 1, fn,
                p * i * 1.0 / l)

        print("Created network")

        dataset = generate_modelnet_dataset(full=True, limit=1)

        dataset.summary()

        learning_rate = 0.003

        for _bid, batch in enumerate(network.batch_generator(dataset, batchSize)):
            print("Processing batch {}".format(_bid))
            activation = network.processBatchForward(batch)
            print("Forward pass Done!")
            ft = activation['features']
            delta_features = ft - np.random.random(ft.shape)
            network.processBatchBackward(
                batch, delta_features, learning_rate)
            print("Backward pass Done!")


class TestDataExtraction(unittest.TestCase):

    def test_layer_activation(self):
        network = create_dC2()
        network.loadWeights('SparseConvNet/weights/ModelNet_10_repeat_bs100_nthrd10/ModelNet', 200)
        lois = [
            network.layer_activations(
                Off3DPicture(
                    'SparseConvNet/Data/ModelNet/car/test/car_0216.off', 40)),
            network.layer_activations(
                Off3DPicture(
                    'SparseConvNet/Data/ModelNet/sink/test/sink_0133.off', 40))
        ]
        self.assertEqual(len(lois[0]), 19)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        unittest.main()
    else:
        if sys.argv[1] == '0':
            suite = unittest.TestSuite()
            suite.addTest(TestTraining("test_simple_training"))
        elif sys.argv[1] == '1':
            suite = unittest.TestSuite()
            suite.addTest(TestTraining("test_forward_backward_pass"))
        else:
            exit()
    runner = unittest.TextTestRunner()
    runner.run(suite)
