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

    def __init__(self):
        """
        """
        self.name = 'ShapeNet55'
        self.class_labels = os.listdir(self.train_dir)
        self.class_labels.sort()
        self.classes = [
            glob(os.path.join(self.train_dir, _class_label, "*.off"))
            for _class_label in self.class_labels
        ]
