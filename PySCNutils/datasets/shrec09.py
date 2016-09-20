from .retrievaldataset import RetrievalDataset
from .classificationdataset import ClassificationDataset
from . import DATASET_FOLDER
from scipy.io import loadmat
import os
from glob import glob


class SHREC09(ClassificationDataset, RetrievalDataset):
    BASEDIR = os.path.join(
        DATASET_FOLDER,
        'SHREC_2009_Shape_Retrieval_Contest_of_Partial_3D_Models/')

    samples_per_class = 18
    name = 'SHREC_2009'
    class_labels = [
        'Bird', 'Fish', 'NonFlyingInsect', 'FlyingInsect', 'Biped',
        'Quadruped', 'ApartmentHouse', 'Skyscraper', 'SingleHouse',
        'Bottle', 'Cup', 'Glasses', 'HandGun', 'SubmachineGun',
        'MusicalInstrument', 'Mug', 'FloorLamp', 'DeskLamp', 'Sword',
        'Cellphone', 'DeskPhone', 'Monitor', 'Bed', 'NonWheelChair',
        'WheelChair', 'Sofa', 'RectangleTable', 'RoundTable', 'Bookshelf',
        'HomePlant', 'Tree', 'Biplane', 'Helicopter', 'Monoplane', 'Rocket',
        'Ship', 'Motorcycle', 'Car', 'MilitaryVehicle', 'Bicycle'
    ]

    def __init__(self):
        def convert_filepath_to_num(pth):
            filename = pth.split('/')[-1]
            x = int(filename.lstrip('D0').rstrip('.of'))
            return x
        _all_samples = sorted(
            glob(os.path.join(self.BASEDIR, 'TargetModels/*.off')),
            key=convert_filepath_to_num)
        self.classes = list(zip(*[iter(_all_samples)]*18))

        self.num_of_relevant_samples = self.samples_per_class

        self.search_map_to_class = {
            __sample.split('/')[-1].rstrip('.of'): _num
            for _num, _list_of_samples in enumerate(self.classes)
            for __sample in _list_of_samples
        }
        self.search_to_label = self.search_map_to_class.__getitem__
        self.query_labels = {}
        for fname in ['range_query_labels', 'parts_query_labels']:
            rql = loadmat(os.path.join(self.BASEDIR,
                'evaluate_rank_lists_code/{}.mat'.format(fname)))
            _query_labels = rql[fname]
            self.query_labels.update({
                _query_labels[i, 0][0]: int(_query_labels[i, 1][0]) - 1
                for i in xrange(_query_labels.shape[0])
            })
        self.query_to_label = self.query_labels.__getitem__

    def query_to_file(self, query, _type=None):
        if _type is None:
            _type = {
                'R': 'scan_hr',
                'P': 'parts'
            }[query[0]]
        return os.path.join(
            self.BASEDIR,
            {
                'parts': 'partial_query_models/{}.off',
                'scan_hr': 'range_query_models_high_resolution/{}.off',
                'scan_lr': 'range_query_models_low_resolution/{}.off'
            }[_type].format(query)
        )

    def search_to_file(self, search):
        return os.path.join(
            self.BASEDIR,
            'TargetModels/{}.off'.format(search)
        )

    def get_search_set_for_query_sample(self, query_sample):
        return list(self.search_map_to_class.keys())

