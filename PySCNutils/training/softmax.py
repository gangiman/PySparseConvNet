# coding: utf-8
from tqdm import tqdm
import os


def nop(*args, **kwargs):
    pass


def train(ds, network, experiment_hash,
          batch_size=150, test_every_n_epochs=5,
          lr_policy=nop,
          momentum_policy=nop,
          dataset_gen_args=(),
          weights_dir='./weights',
          epoch=0, epoch_limit=100,
          train_iteration_hook=nop,
          test_iteration_hook=nop):
    ds.summary()
    weights_temp = os.path.join(weights_dir, experiment_hash)

    testSet, trainSet, labels = ds.generate_train_test_dataset(
        **dict(dataset_gen_args))

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
