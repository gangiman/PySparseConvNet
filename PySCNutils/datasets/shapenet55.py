from .classificationdataset import ClassificationDataset
from .retrievaldataset import RetrievalDataset
from . import DATASET_FOLDER
import os
from glob import glob


class SHREC16_dataset(ClassificationDataset, RetrievalDataset):
    name = 'ShapeNet55'
    BASEDIR = os.path.join(DATASET_FOLDER,
               'SHREC16_3D_SHAPE_RETRIEVAL_SHAPENET55')
    train_dir = os.path.join(BASEDIR, 'train')
    test_dir = os.path.join(BASEDIR, 'val')
    is_pre_split = True

    def __init__(self, validation_limit=None):
        """
        """
        self.name = 'ShapeNet55'
        self.class_labels = os.listdir(self.train_dir)
        self.class_labels.sort()
        self.classes = [
            glob(os.path.join(self.train_dir, _class_label, "*.off"))
            for _class_label in self.class_labels
        ]

        self.search_map_to_class = {}
        self._num_of_samples_per_class = {}
        for _class_folder in os.listdir(self.test_dir):
            samples_for_class = glob(os.path.join(
                    self.test_dir, _class_folder, '*.off'))[:validation_limit]
            self._num_of_samples_per_class[_class_folder] = len(
                samples_for_class)
            self.search_map_to_class.update(
                (fpath, self.class_labels.index(_class_folder))
                for fpath in samples_for_class)

        self.query_to_label = self.search_map_to_class.__getitem__
        self.search_to_label = self.search_map_to_class.__getitem__

    def get_all_queries(self):
        return list(self.search_map_to_class.keys())

    def query_to_file(self, x, **kwargs):
        return x

    def search_to_file(self, x, **kwargs):
        return x

    def get_search_set_for_query_sample(self, query_sample):
            search_set = list(self.search_map_to_class.keys())
            search_set.remove(query_sample)
            return search_set

    def num_relevant_samples_for_query(self, query):
        return self._num_of_samples_per_class[
            query.split('/')[-2]
        ] - 1
