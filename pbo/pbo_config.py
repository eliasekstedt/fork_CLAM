
import os

def create_dirs(*dirpaths):
    for dirpath in dirpaths:
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)

csvDir = 'csv/'
mrxsDir = '../CLAM/data0_mrxs/' #.mrxs
dpath_patchset = 'data/data1_expatch/' #.h5
dpath_vectors = 'data/data2_vectors/' #.h5 + .pt

create_dirs(
    dpath_patchset,
    dpath_vectors,
)

from types import SimpleNamespace
cfg = SimpleNamespace()
cfg.dpath_mrxs = mrxsDir
cfg.dpath_patchset = dpath_patchset
cfg.fpath_map_patient_info = os.path.join(csvDir, 'patient_info.csv')
cfg.fpath_map_patchset = os.path.join(csvDir, 'map_patchset.csv')
cfg.dpath_vectorsDir = os.path.join(dpath_vectors, 'pt_files/')




