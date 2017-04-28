from __future__ import absolute_import
try:
    import PySparseConvNet as pscn
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(__file__, '..'))
    import PySparseConvNet as pscn

import numpy as np
import unittest
from PySCNutils.networks import create_dC2


def create_dummy_sparse_indicies_and_features(spatial_size=6, sparcity=.2,
                                              n_features=1):
    total_voxels = np.power(spatial_size, 3)
    active_voxels = int(np.ceil(sparcity * total_voxels))
    voxels_ids = np.unique(np.random.randint(
        0, high=total_voxels, size=(active_voxels,)))
    voxels_ids.sort()
    total_voxels = voxels_ids.shape[0]
    features = np.r_[
        np.zeros((n_features, 1), dtype=np.float),
        np.random.random((n_features * total_voxels, 1))
    ]

    # size of 3-d tensor, all sides are equal
    return convert_pairs_and_features_to_map(
        list(zip(voxels_ids, np.arange(1, total_voxels + 1))),
        features.reshape(-1).tolist(),
        spatial_size,
        n_features
    )


def convert_pairs_and_features_to_map(
        pairs, features, spatial_size, n_features):
    features = np.asarray(features)
    indices = np.zeros((len(pairs), 3), dtype=np.int)
    output_features = np.zeros((len(pairs), n_features), dtype=np.float)
    for i, (key_id, feature_idx) in enumerate(pairs):
        indices[i, :] = (
            (key_id / spatial_size / spatial_size) % spatial_size,
            (key_id / spatial_size) % spatial_size,
            key_id % spatial_size)
        output_features[i, :] = features[
            feature_idx * (1 + np.arange(n_features))]

    return indices, output_features


def get_test_voxel_picture(spatial_size=126):
    ind, feat = create_dummy_sparse_indicies_and_features(
        spatial_size=spatial_size, sparcity=0.05)
    return pscn.PyVoxelPicture(ind, feat, spatial_size)


class TestVoxelPicture(unittest.TestCase):

    def test_constructor_from_row_matrix(self):
        # indices - an array of shape (num_points, 3),
        #   its columns are indices x,y,z
        indices = np.array([
            [0, 0, 0],
            [1, 0, 5],
            [3, 4, 2],
            [5, 5, 5]
        ], dtype=np.int)
        # size of 3-d tensor, all sides are equal
        spatial_size = 6
        n_features = 1
        # features of size (num_points, num_features)
        # in this case num_features=1
        features = np.ones((indices.shape[0], 1), dtype=np.float)
        # creating a picture object
        pic = pscn.PyVoxelPicture(indices, features, spatial_size)
        # extracting indices and features must be the same
        # as ones it was created from
        returned_indices, returned_features = pic.codifyInputData(spatial_size)
        sparse_indicies, sparse_features = convert_pairs_and_features_to_map(
            returned_indices, returned_features, spatial_size, n_features)
        self.assertEqual(set(map(tuple, sparse_indicies.tolist())),
                         set(map(tuple, indices.tolist())))
        np.testing.assert_almost_equal(features, sparse_features)

    def test_layers_activation_for_voxel_picture(self):
        pic = get_test_voxel_picture()
        ds = pscn.SparseDataset('test_ds', 'UNLABELEDBATCH', 1, 40)
        ds.add_voxel_picture(pic)
        net = create_dC2()
        lois = net.layer_activations_for_dataset(ds)
        self.assertEqual(len(lois), 19)

    def test_batch_processing_for_voxel_picture(self):
        pic = get_test_voxel_picture()
        ds = pscn.SparseDataset('test_ds', 'TRAINBATCH', 1, 40)
        ds.add_voxel_picture(pic)
        net = create_dC2()
        batch_gen = net.batch_generator(ds, 1)
        batch = next(batch_gen)
        batch_output = net.processBatchForward(batch)
        self.assertEqual(batch_output['spatialSize'], 1)
