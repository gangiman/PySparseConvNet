import os

if os.path.exists('/media/toshiba'):
    DEFAULT_DATASET_FOLDER = '/media/toshiba/shape_retrieval_datasets/'
else:
    DEFAULT_DATASET_FOLDER = '/media/hdd/shape_retrieval_datasets/'

DATASET_FOLDER = os.environ.get('DATASET_FOLDER', DEFAULT_DATASET_FOLDER)
