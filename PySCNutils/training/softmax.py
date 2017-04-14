# coding: utf-8
import numpy as np
from tqdm import tqdm
from itertools import islice
from itertools import izip
from functools import partial
from itertools import groupby
from operator import itemgetter
from random import shuffle
import os

try:
    from PySparseConvNet import Off3DPicture
    from PySparseConvNet import SparseDataset
    from PySparseConvNet import SparseNetwork
except ImportError:
    print("PySparseConvNet doesn't imports")
    raise


def generate_train_test_dataset(dataset, renderSize=40, do_shuffle=True):
    sparse_train_dataset = SparseDataset("{} (Train subset)".format(
        dataset.name), 'TRAINBATCH', 1, dataset.class_count)
    sparse_test_dataset = SparseDataset("{} (Test subset)".format(
        dataset.name), 'TESTBATCH', 1, dataset.class_count)

    test_samples = []
    for _k, _it in groupby(
            sorted(dataset.search_map_to_class.items(), key=itemgetter(1)),
            itemgetter(1)):
        test_samples.append(map(lambda x: dataset.search_to_file(x[0]), _it))

    for _class_id, (train_samples, test_samples) in enumerate(
            zip(dataset.classes, test_samples)):
        if do_shuffle:
            shuffle(train_samples)
            shuffle(test_samples)
        else:
            train_samples.sort()
            test_samples.sort()

        for _off in train_samples:
            sparse_train_dataset.add_picture(
                Off3DPicture(_off, renderSize, label=_class_id))
        for _off in test_samples:
            sparse_test_dataset.add_picture(
                Off3DPicture(_off, renderSize, label=_class_id))

    return sparse_test_dataset, sparse_train_dataset, dataset.class_labels


def nop(*args, **kwargs):
    pass

DEFAULT_EPOCH_LIMIT = 100


def train(ds, network, experiment_hash,
          batch_size=150, test_every_n_epochs=5,
          lr_policy=nop,
          momentum_policy=nop,
          render_size=40, weights_dir='./weights',
          epoch=0, epoch_limit=None,
          train_iteration_hook=nop,
          test_iteration_hook=nop):
    ds.summary()
    weights_temp = os.path.join(weights_dir, experiment_hash)

    if epoch_limit is None:
        epoch_limit = DEFAULT_EPOCH_LIMIT

    testSet, trainSet, labels = ds.generate_train_test_dataset(
        renderSize=render_size)

    for _ in tqdm(xrange(epoch_limit),
                  total=epoch_limit, unit="epoch"):
        learning_rate = lr_policy(epoch)
        momentum = momentum_policy(epoch)
        train_report = network.processDataset(trainSet, batch_size,
                                              learningRate=learning_rate,
                                              momentum=momentum)
        train_iteration_hook(train_report=train_report)
        if epoch > 0 and epoch % test_every_n_epochs == 0:
            network.saveWeights(weights_temp, epoch)
            test_reports = network.processDatasetRepeatTest(testSet,
                                                            batch_size, 3)
            test_iteration_hook(_network=network,
                                learning_rate=learning_rate,
                                momentum=momentum,
                                epoch=epoch,
                                weights_path="{}_epoch-{}.cnn".format(
                                     weights_temp, epoch),
                                test_reports=test_reports)
        epoch += 1
