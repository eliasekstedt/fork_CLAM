
from pathlib import Path
import os

def create_dirs(*dirpaths):
    for dirpath in dirpaths:
        if type(dirpath) == type(Path):
            dirpath.mkdir(exist_ok=True)
        else:
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)


from types import SimpleNamespace
cfg = SimpleNamespace()

cfg.dpath_csvRoot = 'csv/'
cfg.dpath_dataRoot = 'data/'

cfg.dpath_mrxsRoot = '../CLAM/data0_mrxs/'

cfg.dpath_patchRoot = os.path.join(cfg.dpath_dataRoot, 'data1_excoords/') #.h5
cfg.dpath_patchcoords = os.path.join(cfg.dpath_patchRoot, 'coords')
cfg.dpath_patchcoords_masks = os.path.join(cfg.dpath_patchRoot, 'masks')
cfg.dpath_patchcoords_stitch = os.path.join(cfg.dpath_patchRoot, 'stitches')

cfg.dpath_featuresRoot = os.path.join(cfg.dpath_dataRoot, 'data2_features/') #.h5 + .pt

cfg.dpath_features_pt = os.path.join(cfg.dpath_featuresRoot, 'features_pt/')
cfg.dpath_features_h5 = os.path.join(cfg.dpath_featuresRoot, 'features_h5/')

cfg.dpath_classifierRoot = os.path.join(cfg.dpath_dataRoot, 'data3_mil')
cfg.dpath_milFolds = os.path.join(cfg.dpath_csvRoot, 'milFolds')

cfg.dpath_patch2encode_samples = Path('diagnostics/')

create_dirs(
    cfg.dpath_csvRoot,
    cfg.dpath_dataRoot,
    cfg.dpath_patchRoot,
    cfg.dpath_patchcoords,
    cfg.dpath_patchcoords_masks,
    cfg.dpath_patchcoords_stitch,
    cfg.dpath_featuresRoot,
    cfg.dpath_features_pt,
    cfg.dpath_features_h5,
    cfg.dpath_classifierRoot,
    cfg.dpath_milFolds,
    cfg.dpath_patch2encode_samples,
)

cfg.fpath_map_approved = Path('...')
cfg.fpath_map_patient_info = os.path.join(cfg.dpath_csvRoot, 'patient_info.csv')
cfg.fpath_map_patchcoords = os.path.join(cfg.dpath_csvRoot, 'map_patchcoords.csv')
cfg.fpath_map_fold_0 = os.path.join(cfg.dpath_milFolds, 'map_fold_0.csv')
cfg.fpath_map_fold_1 = os.path.join(cfg.dpath_milFolds, 'map_fold_1.csv')

# generate patches parameters
#cfg.mthresh = 7
#cfg.close = 8
#cfg.a_t = 1260
#cfg.a_h = 10
#cfg.max_holes = 800

# feauture extraction
cfg.fpath_Xmodel = os.path.join(cfg.dpath_dataRoot, 'Xmodel.ckpt')

cfg.fexparam_batch_size = 512
cfg.fexparam_patch_size = 224
cfg.fexparam_slide_extension = '.mrxs'
cfg.fexparam_no_auto_skip = False

cfg.tag = 'no_more leaks'
cfg.hparam = {
    'dropout': 0.0,
    'batch_size':1,
    'learning_rate':0.1,
    'weight_decay':1e-6,
    'nr_epochs':120,
}
cfg.augm = {
    'placeholder':True
}

cfg.state_dict = ''