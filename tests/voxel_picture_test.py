from __future__ import absolute_import
try:
    from PySparseConvNet import PyVoxelPicture
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(__file__, '..'))

import numpy as np
import unittest


def convert_pairs_and_features_to_map(pairs, features, ss, nFeatures):
    features = np.asarray(features)
    dict_of_sparse_tensor = {}
    for key_id, feature_idx in pairs:
        dict_of_sparse_tensor[(
                (key_id / ss / ss) % ss,
                (key_id / ss) % ss,
                key_id % ss
            )] = features[feature_idx * (1 + np.arange(nFeatures))]

    return np.array([
            (_x, _y, _z) + tuple(f)
            for (_x, _y, _z), f in dict_of_sparse_tensor.items()
        ])


class TestVoxelPicture(unittest.TestCase):
    def test_current_implementation(self):
        n_features = 1
        spatial_size = 3

        tensor = np.zeros((3, 3, 3), dtype=np.int)
        tensor[0, 0, 0] = 1
        tensor[1, 0, 1] = 1
        tensor[2, 2, 2] = 1

        pic = PyVoxelPicture(tensor.ravel().tolist(), spatial_size)
        pairs, feat = pic.codifyInputData(spatial_size)
        sparse_row_matrix = convert_pairs_and_features_to_map(
            pairs, feat, spatial_size, n_features)
        self.assertTrue((0 <= sparse_row_matrix[:, :3]).all())
        self.assertTrue(
            (sparse_row_matrix[:, :3] < spatial_size).all())
        self.assertTrue((0 <= sparse_row_matrix[:, 3]).all())
        self.assertTrue((sparse_row_matrix[:, 3] == 1).all())

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
        # features of size (num_points, num_features)
        # in this case num_features=1
        features = np.ones((indices.shape[0], 1), dtype=np.int)
        # creating a picture object
        pic = PyVoxelPicture(indices=indices, features=features,
                             spatial_size=spatial_size)
        # extracting indices and features must be the same
        # as ones it was created from
        returned_indices, returned_features = pic.codifyInputData(spatial_size)
        np.testing.assert_almost_equal(returned_indices, indices)
        np.testing.assert_almost_equal(returned_features, features)

    # def test_constructor_from_dense_tensor(self):
    #     # tensor - an array of shape
    #     # (spatial_size, spatial_size, spatial_size)
    #     # spatial_size = 3
    #     tensor = np.zeros((3, 3, 3), dtype=np.int)
    #     tensor[0, 0, 0] = 1
    #     tensor[1, 0, 1] = 1
    #     tensor[2, 2, 2] = 1
    #     # np.where(tensor.ravel() == 1)
    #     pic = PyVoxelPicture(dense_tensor=tensor)
    #     # extracting indexes and features must be the same
    #     # as ones it was created from
    #     returned_indexes, returned_features = pic.codifyInputData()
    #
    #     np.testing.assert_almost_equal(returned_indexes, indexes)
    #     np.testing.assert_almost_equal(
    #         returned_features, np.ones(indexes.shape[0], dtype=np.int))
