# coding: utf-8
import numpy as np
from tqdm import tqdm
from itertools import islice
from itertools import izip
from functools import partial
import os

try:
    from PySparseConvNet import Off3DPicture
    from PySparseConvNet import SparseDataset
    from PySparseConvNet import SparseNetwork
except ImportError:
    print("PySparseConvNet doesn't imports")
    raise


def get_functions(norm_type='cosine'):
    """
    function linear: triplet_loss = ReLU(L + norm(g,p) - norm(g,n))
    function logarithmic: triplet_loss = ln(1 + ReLU(norm(g,p)/(norm(g,n) + L))
    norm - L2 / cos
    g (ground) - some sample, p (positive) - farthest same class sample in batch,
    n - (negative) closest not same class sample in batch
    :param norm_type:
    :return:
    """
    from autograd import grad
    import autograd.numpy as _np

    def cos_norm(x, y):
        return 1 - _np.dot(x, y)/_np.sqrt(_np.dot(x, x) * _np.dot(y, y))

    def l2_norm(x, y):
        xn = x/_np.sqrt((x * x).sum())
        yn = y/_np.sqrt((y * y).sum())
        return ((xn - yn) ** 2).sum()

    if norm_type == 'cosine':
        norm = cos_norm
    elif norm_type == 'L2':
        norm = l2_norm

    def relu(x):
        return _np.maximum(0, x)

    def _linear_triplet_loss(trip):
        L = 0.1
        g, p, n = trip
        return relu(L + norm(g, p) - norm(g, n))

    return _linear_triplet_loss, grad(_linear_triplet_loss), norm


def generate_network(dimension=3, l=5, k=32, fn='VLEAKYRELU', nInputFeatures=1,
                     nClasses=50, p=0.0):
    network = SparseNetwork(dimension, nInputFeatures, nClasses)
    for i in range(l + 1):
        network.addLeNetLayerMP(
            (i + 1) * k, 2, 1, 3 if (i < l) else 1, 2 if (i < l) else 1, fn,
            p * i * 1.0 / l)
    return network


def train(ds, network, batch_size=150, test_every_n_batches=100,
          unique_classes_in_batch=5, lr_decay_rate=0.025, pair_taking_method=0,
          render_size=40, weights_dir='./weights',
          in_batch_sample_selection=False):
    linear_triplet_loss, ltl_grad, norm = get_functions()
    ds.summary()
    gen = ds.generate_triplets(batch_size=batch_size,
                               unique_classes_in_batch=unique_classes_in_batch,
                               method=pair_taking_method)

    weights_temp = os.path.join(weights_dir, '{}_triplet'.format(ds.name))
    print('Taking {} batches in to dataset'.format(test_every_n_batches))
    epoch = 0

    total_number_of_epochs = int(np.ceil(sum(map(
        lambda l: len(l) * (len(l) - 1), ds.classes
    )) / (batch_size / 3.0) / test_every_n_batches))

    for _ in tqdm(xrange(total_number_of_epochs),
                  total=total_number_of_epochs, unit="epoch"):
        train_ds = SparseDataset(
            ds.name + " train", 'TRAINBATCH', 1, ds.class_count, shuffle=False)
        ranges_for_all = []
        for batch_samples, ranges in tqdm(islice(gen, test_every_n_batches),
                                          leave=False,
                                          desc="Creating dataset"):
            ranges_for_all.append(ranges)
            for _sample in batch_samples:
                train_ds.add_picture(Off3DPicture(_sample, render_size))
        if not ranges_for_all:
            break
        batch_gen = network.batch_generator(train_ds, batch_size)
        learning_rate = 0.003 * np.exp(- lr_decay_rate * epoch)
        for batch, _ranges in tqdm(izip(batch_gen, ranges_for_all),
                                   leave=False, unit='batch',
                                   total=test_every_n_batches):
            activation = network.processBatchForward(batch)
            feature_vectors = activation['features']
            delta = np.zeros_like(feature_vectors)
            batch_loss = []
            for _offset, _range in zip(3 * np.cumsum([0] + _ranges)[:-1], _ranges):
                if in_batch_sample_selection:
                    one_class_ids = np.arange(2 * _range) + _offset
                    other_class_ids = np.arange(2 * _range, 3 * _range) + _offset
                    while one_class_ids.any():
                        anchor = one_class_ids[0]
                        positive_id = np.apply_along_axis(
                            partial(norm, feature_vectors[anchor]), 1,
                            feature_vectors[one_class_ids[1:]]).argmax() + 1
                        negative_id = np.apply_along_axis(
                            partial(norm, feature_vectors[anchor]), 1,
                            feature_vectors[other_class_ids]).argmin()
                        triplet_slice = [anchor,
                                         one_class_ids[positive_id],
                                         other_class_ids[negative_id]]
                        one_class_ids = np.delete(one_class_ids, [0, positive_id])
                        other_class_ids = np.delete(other_class_ids, negative_id)
                        delta[triplet_slice] = ltl_grad(
                            feature_vectors[triplet_slice])
                        batch_loss.append(linear_triplet_loss(
                            feature_vectors[triplet_slice]))
                else:
                    for _i in range(_range):
                        triplet_slice = _offset + (np.arange(3) * _range) + _i
                        delta[triplet_slice] = ltl_grad(feature_vectors[triplet_slice])
                        batch_loss.append(linear_triplet_loss(feature_vectors[triplet_slice]))
            # batch_iteration_hook(batch_loss=batch_loss,
            # learning_rate=learning_rate,epoch=epoch
            tqdm.write("Triplet loss for batch {}".format(sum(batch_loss)))
            network.processBatchBackward(batch, delta,
                                         learningRate=learning_rate)
        # epoch_iteration_hook(
        network.saveWeights(weights_temp, epoch)
        epoch += 1
        del train_ds
