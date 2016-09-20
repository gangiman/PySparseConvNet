import os

DEFAULT_DATASET_FOLDER = '/media/toshiba/shape_retrieval_datasets/'

DATASET_FOLDER = os.environ.get('DATASET_FOLDER', DEFAULT_DATASET_FOLDER)
