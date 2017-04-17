from __future__ import absolute_import
try:
    from PySparseConvNet import PyVoxelPicture
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(__file__, '..'))
    from PySparseConvNet import PyVoxelPicture

import numpy as np
import unittest


def convert_pairs_and_features_to_map(pairs, features, ss, nFeatures):
    features = np.asarray(features)
    indices = np.zeros((len(pairs), 3), dtype=np.int)
    output_features = np.zeros((len(pairs), nFeatures), dtype=np.float)
    for i, (key_id, feature_idx) in enumerate(pairs):
        indices[i, :] = (
                (key_id / ss / ss) % ss,
                (key_id / ss) % ss,
                key_id % ss)
        output_features[i, :] = features[
            feature_idx * (1 + np.arange(nFeatures))]

    return indices, output_features


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
        pic = PyVoxelPicture(indices, features, spatial_size)
        # extracting indices and features must be the same
        # as ones it was created from
        returned_indices, returned_features = pic.codifyInputData(spatial_size)
        sparse_indicies, sparse_features = convert_pairs_and_features_to_map(
            returned_indices, returned_features, spatial_size, n_features)
        self.assertEqual(set(map(tuple, sparse_indicies.tolist())),
                         set(map(tuple, indices.tolist())))
        np.testing.assert_almost_equal(features, sparse_features)
