
import os

def create_dirs(*dirpaths):
    for dirpath in dirpaths:
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)

dpath_csvRoot = 'csv/'
dpath_dataRoot = 'data/'

from types import SimpleNamespace
cfg = SimpleNamespace()

cfg.dpath_mrxsRoot = '../CLAM/data0_mrxs/' #.mrxs WSI
cfg.fpath_map_patient_info = os.path.join(dpath_csvRoot, 'patient_info.csv')
cfg.fpath_map_patchset = os.path.join(dpath_csvRoot, 'map_patchset.csv')

cfg.dpath_patchRoot = os.path.join(dpath_dataRoot, 'data1_expatch/') #.h5
cfg.dpath_patchset = os.path.join(cfg.dpath_patchRoot, 'patches')
cfg.dpath_patchset_masks = os.path.join(cfg.dpath_patchRoot, 'masks')
cfg.dpath_patchset_stitch = os.path.join(cfg.dpath_patchRoot, 'stitches')

cfg.dpath_featuresRoot = os.path.join(dpath_dataRoot, 'data2_features/') #.h5 + .pt
cfg.dpath_features_pt = os.path.join(cfg.dpath_featuresRoot, 'pt_files/')
cfg.dpath_features_h5 = os.path.join(cfg.dpath_featuresRoot, 'h5_files/')

create_dirs(
    cfg.dpath_patchRoot,
    cfg.dpath_patchset,
    cfg.dpath_patchset_masks,
    cfg.dpath_patchset_stitch,
    cfg.dpath_featuresRoot,
    cfg.dpath_features_pt,
    cfg.dpath_features_h5,
)

cfg.fpath_fexmodel = 'pbo_model/pbo_res18.ckpt'
cfg.fexparam_batch_size=256
cfg.fexparam_patch_size=224
cfg.fexparam_slide_extension='.mrxs'
cfg.fexparam_no_auto_skip=False
