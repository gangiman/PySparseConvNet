from __future__ import print_function
import unittest

from PySparseConvNet import SparseNetwork
from PySparseConvNet import SparseDataset
from PySparseConvNet import Off3DPicture


from deepC2Network import create_dC2
from deepC2Network import load_and_get_weights
from deepC2Network import load_3d_off
from deepC2Network import learn_simple_network

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
            SparseDataset("Testing DataSet", "FakeBatch", 2)
        print("Successfully caught Value Exception.")
        ds = SparseDataset("Testing DataSet", "TRAINBATCH", 2)
        self.assertEqual(ds.name, "Testing DataSet")


class TestTraining(unittest.TestCase):

    def test_simple_training(self):
        learn_simple_network(limit=-1)

    def test_predict(self):
        unlabeled_dataset = SparseDataset("One pic", 'UNLABELEDBATCH', 1)
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
        self.assertEqual(len(lois[1]), 19)

if __name__ == '__main__':
    unittest.main()
