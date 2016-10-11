import sys

WORK_DIR = '/var/workplace/PySparseConvNet/'
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

try:
    from PySparseConvNet import Off3DPicture
    from PySparseConvNet import SparseDataset
except ImportError:
    print("PySparseConvNet doesn't imports")
    # raise
