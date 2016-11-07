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


def get_functions(norm_type='cosine', margin=0.1, norm_for_l2=True):
    """
    function linear: triplet_loss = ReLU(L + norm(g,p) - norm(g,n))
    function logarithmic: triplet_loss = ln(1 + ReLU(norm(g,p)/(norm(g,n) + L))
    g (ground) - some sample, p (positive) - same class sample,
    n - (negative) not same class sample.

    :param norm_type: - 'L2' or 'cosine'
    :param margin: float
    :param norm_for_l2:
    :return:
    """
    from autograd import grad
    import autograd.numpy as _np

    def cos_norm(x, y):
        return 1 - _np.dot(x, y)/_np.sqrt(_np.dot(x, x) * _np.dot(y, y))

    def l2_norm(x, y):
        if norm_for_l2:
            xn = x/_np.sqrt((x * x).sum())
            yn = y/_np.sqrt((y * y).sum())
        else:
            xn, yn = x, y
        return ((xn - yn) ** 2).sum()

    if norm_type == 'cosine':
        norm = cos_norm
    elif norm_type == 'L2':
        norm = l2_norm

    def relu(x):
        return _np.maximum(0, x)

    def _linear_triplet_loss(trip):
        g, p, n = trip
        return relu(margin + norm(g, p) - norm(g, n))

    return _linear_triplet_loss, grad(_linear_triplet_loss), norm


def generate_network(dimension=3, l=5, k=32, fn='VLEAKYRELU', nInputFeatures=1,
                     nClasses=50, p=0.0, cudaDevice=-1):
    network = SparseNetwork(dimension, nInputFeatures, nClasses,
                            cudaDevice=cudaDevice)
    for i in range(l + 1):
        network.addLeNetLayerMP(
            (i + 1) * k, 2, 1, 3 if (i < l) else 1, 2 if (i < l) else 1, fn,
            p * i * 1.0 / l)
    return network


def generate_wide_network(dimension=3, l=5, filter_mult=None, fn='VLEAKYRELU',
                          nInputFeatures=1, nClasses=50, p=0.0, cudaDevice=-1):
    assert len(filter_mult) == l + 1
    network = SparseNetwork(dimension, nInputFeatures, nClasses,
                            cudaDevice=cudaDevice)
    """ for l = 5, k = 32
    sparse_net.addLeNetLayerMP(32, 2, 1, 3, 2, 'VLEAKYRELU', 0.0)
    sparse_net.addLeNetLayerMP(64, 2, 1, 3, 2, 'VLEAKYRELU', 0.0)
    sparse_net.addLeNetLayerMP(96, 2, 1, 3, 2, 'VLEAKYRELU', 0.0)
    sparse_net.addLeNetLayerMP(128, 2, 1, 3, 2, 'VLEAKYRELU', 0.0)
    sparse_net.addLeNetLayerMP(160, 2, 1, 3, 2, 'VLEAKYRELU', 0.0)
    sparse_net.addLeNetLayerMP(192, 2, 1, 1, 1, 'VLEAKYRELU', 0.0)
    """
    for i, fm in enumerate(filter_mult):
        network.addLeNetLayerMP(
            fm,
            2, 1, 3 if (i < l) else 1, 2 if (i < l) else 1, fn,
            p * i * 1.0 / l)
    return network


def nop(*args, **kwargs):
    pass


def get_decreasing_weights(l, fall='line'):
    if l == 1:
        return np.array([1.])
    line_range = np.arange(1, l - 1, dtype=np.float)[::-1]
    if fall == 'log':
        log_range = np.log(line_range + 1)
    elif fall == 'line':
        log_range = line_range
    else:
        raise Exception("fuuu")
    log_range /= log_range[0] if log_range.any() else 1
    full_weights = np.hstack((
            np.array((1., 1.)),
            log_range
        ))
    return full_weights / full_weights.sum()


def weighted_sampling_of_best(arr, best=None):
    l = len(arr)
    if l == 1:
        return 0
    if best == 'min':
        sorted_arr_ids = arr.argsort()  # from smallest to biggest
    elif best == 'max':
        sorted_arr_ids = arr.argsort()[::-1]  # from smallest to biggest
    else:
        raise AssertionError('mode is not min/max.')
    return np.random.choice(sorted_arr_ids, 1, p=get_decreasing_weights(l))[0]


def train(ds, network, experiment_hash,
          batch_size=150, test_every_n_batches=100,
          unique_classes_in_batch=5,
          lr_policy=nop,
          momentum_policy=nop,
          pair_taking_method=0,
          render_size=40, weights_dir='./weights',
          in_batch_sample_selection=False, norm_type='cosine', L=1,
          epoch=0, epoch_limit=None,
          batch_iteration_hook=nop, epoch_iteration_hook=nop,
          pairs_limit=None):
    linear_triplet_loss, ltl_grad, norm = get_functions(norm_type=norm_type,
                                                        margin=L)
    ds.summary()
    gen = ds.generate_triplets(batch_size=batch_size,
                               unique_classes_in_batch=unique_classes_in_batch,
                               method=pair_taking_method, limit=pairs_limit)

    weights_temp = os.path.join(weights_dir, experiment_hash)
    print('Taking {} batches in to dataset'.format(test_every_n_batches))
    if epoch_limit is None:
        if pairs_limit is None:
            total_pairs_num = sum(map(lambda l: len(l) * (len(l) - 1),
                                      ds.classes))
        else:
            total_pairs_num = ds.get_limit(pairs_limit) * ds.class_count
        total_number_of_epochs = int(np.ceil(
            total_pairs_num
            / (batch_size / 3.0) / test_every_n_batches))
    else:
        total_number_of_epochs = epoch_limit
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
        learning_rate = lr_policy(epoch)
        momentum = momentum_policy(epoch)
        for bid, (batch, _ranges) in tqdm(
                enumerate(izip(batch_gen, ranges_for_all)),
                leave=False, unit='batch', total=test_every_n_batches):
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
                        positive_id = weighted_sampling_of_best(
                            np.apply_along_axis(
                                partial(norm, feature_vectors[anchor]), 1,
                                feature_vectors[one_class_ids[1:]]),
                            best='max')
                        positive_id += 1
                        negative_id = weighted_sampling_of_best(
                            np.apply_along_axis(
                                partial(norm, feature_vectors[anchor]), 1,
                                feature_vectors[other_class_ids]), best='min')
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
            batch_iteration_hook(
                batch_loss=batch_loss,
                epoch=epoch,
                bid=bid
            )
            network.processBatchBackward(batch, delta,
                                         learningRate=learning_rate,
                                         momentum=momentum)
        network.saveWeights(weights_temp, epoch)
        epoch_iteration_hook(_network=network,
                             learning_rate=learning_rate,
                             momentum=momentum,
                             epoch=epoch,
                             weights_path="{}_epoch-{}.cnn".format(
                                 weights_temp, epoch))
        epoch += 1
        del train_ds
