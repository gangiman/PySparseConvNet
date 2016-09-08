from deepC2Network import create_dC2
from deepC2Network import generate_modelnet_dataset
import math
import unittest


class TestTrainingOfBigDataset(unittest.TestCase):
    def test_big_dataset_training(self):
        batchSize = 10
        epoch = 2
        network = create_dC2()
        print("Created network")
        dataset = generate_modelnet_dataset(full=False, limit=1)
        dataset.repeatSamples(200)
        dataset.summary()
        print("Created dataset {0}".format(dataset.name))
        for epoch in xrange(1, epoch):
            learning_rate = 0.003 * math.exp(-0.05 / 2 * epoch)
            # print("epoch {0}, lr={1} ".format(epoch, learning_rate), end='')
            network.processDataset(dataset, batchSize=batchSize,
                                   learningRate=learning_rate)


if __name__ == '__main__':
    unittest.main()
