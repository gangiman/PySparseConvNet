# coding: utf-8
import os
import numpy as np
import gc
from random import shuffle
from glob import glob
from tqdm import tqdm


try:
    from PySparseConvNet import Off3DPicture
    from PySparseConvNet import SparseDataset
except ImportError:
    print("PySparseConvNet doesn't imports")
    raise


def train(ds, network):
    """
    Not finished yet
    :param ds:
    :param network:
    :return:
    """
    def generate_train_test_dataset(self, renderSize=40, do_shuffle=True):
        if self.sparse_test_dataset is not None:
            del self.sparse_test_dataset
        if self.sparse_train_dataset is not None:
            del self.sparse_train_dataset
        gc.collect()
        self.sparse_train_dataset = SparseDataset(
            "SHREC16 (Train subset)", 'TRAINBATCH', self.number_of_features,
            self.number_of_classes)
        self.sparse_test_dataset = SparseDataset(
            "SHREC16 (Test subset)", 'TESTBATCH', self.number_of_features,
            self.number_of_classes)

        #         labels = []
        for _class_id, _class_name in enumerate(self.classes):
            #             offs = list(classes[_class_id][:])
            train_samples = glob(
                os.path.join(self.train_dir, _class_name, '*.off'))
            test_samples = glob(
                os.path.join(self.test_dir, _class_name, '*.off'))
            if do_shuffle:
                shuffle(train_samples)
                shuffle(test_samples)
            else:
                train_samples.sort()
                test_samples.sort()
            for _off in train_samples:
                self.sparse_train_dataset.add_picture(
                    Off3DPicture(_off, renderSize, label=_class_id))
            for _off in test_samples:
                self.sparse_test_dataset.add_picture(
                    Off3DPicture(_off, renderSize, label=_class_id))

        return self.sparse_test_dataset, self.sparse_train_dataset, self.classes

    raise NotImplementedError()

    batchSize = 50

    baseName = "weights/SHREC16"


    testSet, trainSet, labels = ds.generate_train_test_dataset()

    train_log = []
    test_log = {}

    for epoch in tqdm(range(101)):
        train_report = network.processDataset(trainSet, batchSize,
                                          0.003 * np.exp(-0.05 / 2 * epoch))
        train_log.append(train_report)
        tqdm.write(
            "Train err: {errorRate:.4}, nll: {nll:.4}".format(**train_report))
        if epoch > 0 and epoch % 5 == 0:
            network.saveWeights(baseName, epoch)
            test_reports = network.processDatasetRepeatTest(testSet, batchSize, 3)
            test_log[epoch] = test_reports
            tqdm.write("Test mean err: {0:.4}, nll: {1:.4}".format(
                sum(_r['errorRate'] for _r in test_reports) / 3.,
                sum(_r['nll'] for _r in test_reports) / 3.))

