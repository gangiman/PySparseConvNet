# coding: utf-8

import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gc

import PySparseConvNet as pyscn


def do_sampling(tensors, n_samples):
    if n_samples <= tensors.shape[0]:
        ids = np.arange(0, n_samples, dtype=np.int)
    else:
        ids = np.r_[
            np.arange(0, tensors.shape[0], dtype=np.int),
            np.random.permutation(tensors.shape[0])[
            :(n_samples - tensors.shape[0])]
        ]
    return tensors[ids]


class ADNIDataset:
    """
    ls /media/toshiba/shared_storage/libfun/ADNI2_masked
    wc -l /media/toshiba/shared_storage/libfun/clean_meta_full.csv
    
    in csv change `/home/mount/neuro-x01-hdd/` to
    `/media/toshiba/shared_storage/libfun/`

    http://ida.loni.usc.edu/collaboration/access/appLicense.jsp

    The Image Data Archive at the Laboratory of Neuro Imaging (IDA) provides a
    safe repository for medical imaging data. LONI seeks to improve the
    understanding of the brain in health and disease through the development of
    algorithms and approaches for the comprehensive and quantitative mapping of
    its structure and function.
    """
    sparse_train_dataset = None
    sparse_test_dataset = None
    name = 'ADNI MRI Alzheimer detection dataset'
    BASEDIR = '/media/toshiba/shared_storage/libfun/'
    number_of_features = 1
    sampling_methods = {
        'upsampling': max,
        'downsampling': min,
        'meansampling': lambda x: int(np.round(np.mean(x)))
    }

    def __init__(self, labels=None, threshold=0.01, sampling=None):
        self._validate_sampling(sampling)
        self.sampling = sampling
        self.threshold = threshold
        path_to_csv = os.path.join(self.BASEDIR, 'clean_meta_full.csv')
        self.metadata = pd.read_csv(path_to_csv)
        self.metadata.Path = self.metadata.Path.str.replace(
            '/home/mount/neuro-x01-hdd/',
            '/media/toshiba/shared_storage/libfun/')
        if labels is not None:
            self.metadata = self.metadata[self.metadata.Label.isin(labels)]
        self.labels = sorted(self.metadata.Label.unique().tolist())
        self.number_of_classes = len(self.labels)
        self._validate_labels(labels)
        self.classes = []

    def summary(self):
        print("{} dataset wrapper object:".format(self.name))
        print("Number of classes {}".format(self.number_of_classes))
        print("Number of files {}".format(sum(self.get_count_of_classes())))

    def get_count_of_classes(self):
        return self.metadata.Label.value_counts()

    def _validate_labels(self, classes):
            assert (set(classes).issubset(set(self.metadata.Label.unique()))
                    or classes is None), ("classes should be None to take all"
                                          " classes or subset of all labels of"
                                          " dataset")

    def _validate_sampling(self, sampling):
        assert sampling in self.sampling_methods.keys() or sampling is None

    def _create_voxel_picture_from_tensor(self, tensor, label, spatial_size):
        x, y, z = np.where(tensor > self.threshold)
        c = tensor[x, y, z]
        indices = np.c_[x, y, z]
        features = c[:, np.newaxis].astype(np.float)

        return pyscn.PyVoxelPicture(indices, features,
                                    spatial_size, label=label)

    def generate_train_test_dataset(self, proportion=.3, do_shuffle=True,
                                    spatial_size=126):

        for label in self.labels:
            smc_mask = (self.metadata.Label == label).values

            data_for_label = np.zeros((smc_mask.sum(),
                                       110, 110, 110), dtype='float32')

            for it, im in tqdm(enumerate(self.metadata[smc_mask].Path.values),
                               total=smc_mask.sum(),
                               desc='Reading MRI to memory',
                               leave=False):
                mx = nib.load(im).get_data()\
                    .max(axis=0).max(axis=0).max(axis=0)
                data_for_label[it, :, :, :] = np.array(
                    nib.load(im).get_data()) / mx
            self.classes.append(data_for_label)

        if self.sparse_test_dataset is not None:
            del self.sparse_test_dataset
        if self.sparse_train_dataset is not None:
            del self.sparse_train_dataset
        gc.collect()

        self.sparse_train_dataset = pyscn.SparseDataset(
            "ADNI (Train subset)", 'TRAINBATCH',
            self.number_of_features, self.number_of_classes)
        self.sparse_test_dataset = pyscn.SparseDataset(
            "ADNI (Test subset)", 'TESTBATCH', self.number_of_features,
            self.number_of_classes)

        if self.sampling in self.sampling_methods.keys():
            n_samples_in_train = self.sampling_methods[self.sampling](
                int(np.ceil((1-proportion) * _class.shape[0]))
                for _class in self.classes)
        else:
            n_samples_in_train = None

        for _class_id, _class_data in enumerate(self.classes):
            if do_shuffle:
                np.random.shuffle(_class_data)
            n_samples_test = int(np.ceil(proportion * _class_data.shape[0]))
            test_samples = _class_data[:n_samples_test]
            if n_samples_in_train is not None:
                train_samples = do_sampling(_class_data[n_samples_test:],
                                            n_samples_in_train)
            else:
                train_samples = _class_data[n_samples_test:]

            for _sample in train_samples:
                self.sparse_train_dataset.add_voxel_picture(
                    self._create_voxel_picture_from_tensor(_sample, _class_id,
                                                           spatial_size))
            for _sample in test_samples:
                self.sparse_test_dataset.add_picture(
                    self._create_voxel_picture_from_tensor(_sample, _class_id,
                                                           spatial_size))

        return self.sparse_test_dataset, self.sparse_train_dataset, self.labels

    @staticmethod
    def plot_slices(tensor):
        fig, ax = plt.subplots(3, 3, figsize=(13, 13))
        ids = np.asarray(np.ceil(np.linspace(10, 80, 9)), dtype=int)

        for _id, _ax in zip(ids, ax.ravel()):
            _ax.imshow(tensor[_id, :, :])
            _ax.set_title('slice #{}'.format(_id))
