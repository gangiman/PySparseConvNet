from __future__ import print_function
import unittest

from PySparseConvNet import SparseNetwork
from PySparseConvNet import SparseDataset

from deepC2Network import create_dC2
from deepC2Network import load_and_get_weights
from deepC2Network import load_3d_off
from deepC2Network import generate_modelnet_dataset

import math

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
        network = create_dC2()
        print("Created network")
        dataset = generate_modelnet_dataset()
        dataset.summary()
        print("Created dataset {0}".format(dataset.name))
        for epoch in xrange(1, 2):
            learning_rate = 0.003 * math.exp(-0.05 / 2 * epoch)
            print("epoch {0}, lr={1} ".format(epoch, learning_rate),end='')
            network.processDataset(
                dataset, batchSize=10, learningRate=learning_rate)

if __name__ == '__main__':
    unittest.main()
