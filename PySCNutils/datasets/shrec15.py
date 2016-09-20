# coding: utf-8
from .classificationdataset import ClassificationDataset
from . import DATASET_FOLDER
import collections
import os


def parse_cla(cla_file):
    classifier = collections.OrderedDict()
    with open(cla_file) as f:
        if "PSB" not in f.readline():
            raise IOError("file must be \"cla\" format file.")
        # n_class, n_data = map(int, f.readline().split(' '))
        while True:
            line = f.readline()
            if line == '':
                break
            split_line = line.split(' ')
            if len(split_line) == 3:
                name, parent_name, n = split_line
                if int(n) > 0:
                    ids = [int(f.readline()) for i in xrange(int(n))]
                    classifier.setdefault(name, ids)
    return classifier


class SHREC2015_dataset(ClassificationDataset):
    BASEDIR = os.path.join(DATASET_FOLDER, 'shrec15-non-rigid')

    def __init__(self):
        self.name = "SHREC2015"
        parsed_cla = parse_cla(os.path.join(
            self.BASEDIR, 'SHREC15_Non-rigid_ToolKit/test.cla'
        ))
        self.class_labels = list(parsed_cla.keys())
        self.class_labels.sort()
        self.classes = [
            map(
                os.path.join(self.BASEDIR, "SHREC15NonRigidTestDB/T{}.off"
                             ).format, parsed_cla[_class_name])
            for _class_name in self.class_labels]
