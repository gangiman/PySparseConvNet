import PySparseConvNet as pyscn


def generate_network(dimension=3, l=5, k=32, fn='VLEAKYRELU', nInputFeatures=1,
                     nClasses=50, p=0.0, cudaDevice=-1):
    network = pyscn.SparseNetwork(
        dimension, nInputFeatures, nClasses, cudaDevice=cudaDevice)
    for i in range(l + 1):
        network.addLeNetLayerMP(
            (i + 1) * k, 2, 1, 3 if (i < l) else 1, 2 if (i < l) else 1, fn,
            p * i * 1.0 / l)
    return network


def generate_wide_network(dimension=3, l=5, filter_mult=None, fn='VLEAKYRELU',
                          nInputFeatures=1, nClasses=50, p=0.0, cudaDevice=-1):
    assert len(filter_mult) == l + 1
    network = pyscn.SparseNetwork(
        dimension, nInputFeatures, nClasses, cudaDevice=cudaDevice)
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


def create_DeepC2Network(dimension, n_layers, n_filters_multiplier, fn, nInputFeatures,
                         nClasses, dropout, nThreads=1):
    sparse_net = pyscn.SparseNetwork(dimension, nInputFeatures, nClasses, nThreads=nThreads)
    for i in range(n_layers + 1):
        sparse_net.addLeNetLayerMP(
            (i + 1) * n_filters_multiplier,
            2,
            1,
            3 if (i < n_layers) else 1,
            2 if (i < n_layers) else 1,
            fn,
            dropout * i * 1.0 / n_layers)
    sparse_net.addSoftmaxLayer()
    return sparse_net


def custom_DeepC2Network(dimension, n_filters_multiplier,
                         fsizes, fstrides, pool_sizes, pool_strides,
                         fn, nInputFeatures,
                         nClasses, dropout, nThreads=1):
    sparse_net = pyscn.SparseNetwork(dimension, nInputFeatures, nClasses, nThreads=nThreads)
    n_layers = len(fsizes)
    assert(len(fsizes) == len(fstrides))
    assert(len(fsizes) == len(pool_sizes))
    assert(len(fsizes) == len(pool_strides))
    for i in xrange(n_layers + 1):
        sparse_net.addLeNetLayerMP(
            (i + 1) * n_filters_multiplier,
            fsizes[i] if i < n_layers else 2,
            fstrides[i] if i < n_layers else 1,
            pool_sizes[i] if i < n_layers else 1,
            pool_strides[i] if i < n_layers else 1,
            fn,
            dropout * i * 1.0 / n_layers)
    sparse_net.addSoftmaxLayer()
    return sparse_net



def create_dC2():
    nFeatures = 1
    nClasses = 40
    return create_DeepC2Network(3, 5, 32, 'VLEAKYRELU', nFeatures, nClasses, 0.0)
